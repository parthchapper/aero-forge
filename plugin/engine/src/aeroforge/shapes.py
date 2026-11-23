"""Point geometry of sections, bodies and propellers (pure numpy, metres)."""

from itertools import pairwise

import numpy as np

from aeroforge.airfoil import naca_outline
from aeroforge.constants import number, numbers
from aeroforge.facet_query import inside_faceted, sample_points
from aeroforge.models import Body, Design, Propulsor, Section, Surface, instance_names

X_HAT = np.array([1.0, 0.0, 0.0])
MIRROR_Y = np.array([1.0, -1.0, 1.0])
FLIP_BLADE = np.array([1.0, -1.0, -1.0])


def section_normal(cant_deg: float) -> np.ndarray:
    """Unit thickness axis of a section canted cant_deg about x (+z at 0°)."""
    c = np.radians(cant_deg)
    return np.array([0.0, -np.sin(c), np.cos(c)])


def section_incidence_deg(surface: Surface, section: Section) -> float:
    """Nose-up turn of one section about its leading edge [deg]: the surface incidence plus the
    section's own twist (the one place both are combined: outlines, aero stations, internals)."""
    return surface.incidence_deg + section.twist_deg


def section_outline(section: Section, surface: Surface, n_per_side: int) -> np.ndarray:
    """Closed airfoil outline of one section [m], shape (2 n - 1, 3)."""
    xy = naca_outline(surface.camber_frac, surface.camber_loc_frac, surface.tc, n_per_side)
    normal = section_normal(section.cant_deg)
    # incidence turns the section nose-up about its leading edge (asb WingXSec twist convention)
    i = np.radians(section_incidence_deg(surface, section))
    chord_dir = np.cos(i) * X_HAT - np.sin(i) * normal
    thick_dir = np.sin(i) * X_HAT + np.cos(i) * normal
    return np.asarray(section.le_m) + section.chord_m * (
        xy[:, :1] * chord_dir + xy[:, 1:2] * thick_dir
    )


def surface_points(surface: Surface, n_per_side: int, spacing_m: float) -> np.ndarray:
    """Outline points of the right/centre instance, blended between sections every spacing_m [m],
    plus the trailing edge (both ends of the outline) every spacing_m / sample_te_refine: on a
    swept panel the spanwise step is a longer x gap there, where a pusher hub slides against it."""
    outlines = [section_outline(s, surface, n_per_side) for s in surface.sections]
    refine = int(number("sample_te_refine"))
    blocks = []
    for a, b, sa, sb in zip(
        outlines[:-1], outlines[1:], surface.sections[:-1], surface.sections[1:], strict=True
    ):
        steps = max(1, int(np.ceil(np.linalg.norm(np.subtract(sb.le_m, sa.le_m)) / spacing_m)))
        w = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None, None]
        blocks.append(((1 - w) * a + w * b).reshape(-1, 3))
        edge = np.linspace(0.0, 1.0, steps * refine, endpoint=False)[:, None, None]
        blocks.append(((1 - edge) * a[[0, -1]] + edge * b[[0, -1]]).reshape(-1, 3))
    return np.vstack([*blocks, outlines[-1]])


def _cross_yz(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """yz-plane cross product over the last axis."""
    return u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]


def in_outline(xy: np.ndarray, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Mask of unit-chord points (xc, yc) inside the outline xy."""
    split = len(xy) // 2
    upper, lower = xy[: split + 1][::-1], xy[split:]
    inside = (xc >= 0.0) & (xc <= 1.0)
    return (
        inside
        & (yc <= np.interp(xc, upper[:, 0], upper[:, 1]))
        & (yc >= np.interp(xc, lower[:, 0], lower[:, 1]))
    )


def _inside_panel(a: Section, b: Section, xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Mask of pts inside the ruled loft from section a to b.

    A loft point is L(f) + xc C(f) x_hat + yc M(f) with L, C, M linear in f, so the spanwise
    fraction f of a point solves a quadratic (its yz offset from L(f) is parallel to M(f)).
    """
    l0 = np.asarray(a.le_m)
    dl = np.asarray(b.le_m) - l0
    m0 = a.chord_m * section_normal(a.cant_deg)
    dm = b.chord_m * section_normal(b.cant_deg) - m0
    q = pts - l0
    qa = -_cross_yz(dl, dm)
    qb = _cross_yz(q, dm) - _cross_yz(dl, m0)
    qc = _cross_yz(q, m0)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -(qb + np.copysign(np.sqrt(qb * qb - 4.0 * qa * qc), qb)) / 2.0
        candidates = (root / qa, qc / root)
    out = np.zeros(len(pts), dtype=bool)
    for f in candidates:
        ok = np.isfinite(f) & (f >= 0.0) & (f <= 1.0)
        fs = np.where(ok, f, 0.0)[:, None]
        r = q - fs * dl
        m = m0 + fs * dm
        chord = a.chord_m + fs[:, 0] * (b.chord_m - a.chord_m)
        yc = (r[:, 1] * m[:, 1] + r[:, 2] * m[:, 2]) / np.einsum("ij,ij->i", m, m)
        out |= ok & in_outline(xy, r[:, 0] / chord, yc)
    return out


def inside_loft(
    sections: tuple[Section, ...], xy: np.ndarray, pts: np.ndarray, mirrored: bool
) -> np.ndarray:
    """Mask of pts inside the ruled loft through sections (and its y-mirror when mirrored)."""
    out = np.zeros(len(pts), dtype=bool)
    probes = (pts, pts * MIRROR_Y) if mirrored else (pts,)
    for a, b in pairwise(sections):
        for p in probes:
            out |= _inside_panel(a, b, xy, p)
    return out


def body_radius_frac(body: Body, t: np.ndarray) -> np.ndarray:
    """Local size / max size at length fraction t: elliptic nose, cosine boat-tail."""
    t = np.clip(t, 0.0, 1.0)
    nose = np.sqrt(np.clip(1.0 - (1.0 - t / body.nose_frac) ** 2, 0.0, None))
    u = np.clip((t - (1.0 - body.tail_frac)) / body.tail_frac, 0.0, 1.0)
    tail = body.tail_radius_frac + (1.0 - body.tail_radius_frac) * (1.0 + np.cos(np.pi * u)) / 2
    return np.where(t < body.nose_frac, nose, np.where(u > 0.0, tail, 1.0))


def body_axis_z(body: Body, t: np.ndarray) -> np.ndarray:
    """Axis height [m] at length fractions t: level to the tail cone, then rising at
    tail_upsweep_deg (0 leaves the axis exactly at the nose height)."""
    run = np.clip(np.clip(t, 0.0, 1.0) - (1.0 - body.tail_frac), 0.0, None) * body.length_m
    return body.nose_m[2] + run * np.tan(np.radians(body.tail_upsweep_deg))


def unit_ring(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin) of n_points angles, exactly mirror-symmetric in both axes (n_points % 4 == 0)."""
    k = np.arange(n_points)
    theta = 2.0 * np.pi * k / n_points
    c, s = np.cos(theta), np.sin(theta)
    half, neg = (n_points // 2 - k) % n_points, (-k) % n_points
    c = (c - c[half]) / 2
    c = (c + c[neg]) / 2
    s = (s + s[half]) / 2
    s = (s - s[neg]) / 2
    return c, s


def body_rings(body: Body, t: np.ndarray, n_points: int) -> np.ndarray:
    """Super-ellipse sections at length fractions t, shape (len(t), n_points, 3) [m]."""
    c, s = unit_ring(n_points)
    power = 2.0 / body.section_exp
    r = body_radius_frac(body, t)[:, None]
    y = body.half_width_m * r * (np.sign(c) * np.abs(c) ** power)
    z = body.half_height_m * r * (np.sign(s) * np.abs(s) ** power)
    x = np.broadcast_to((t * body.length_m)[:, None], y.shape)
    ring = np.asarray(body.nose_m) + np.stack([x, y, z], axis=-1)
    ring[..., 2] += (body_axis_z(body, t) - body.nose_m[2])[:, None]
    return ring


def body_stations(n_stations: int) -> np.ndarray:
    """Length fractions clustered at both ends, including 0 and 1."""
    return (1.0 - np.cos(np.linspace(0.0, np.pi, n_stations))) / 2.0


def body_points(body: Body, n_stations: int, n_ring: int) -> np.ndarray:
    """Surface points of the right/centre instance [m]."""
    return body_rings(body, body_stations(n_stations), n_ring).reshape(-1, 3)


def inside_body(body: Body, points: np.ndarray, tol_m: float = 1e-9) -> np.ndarray:
    """Mask of points inside the right/centre instance of body."""
    rel = points - np.asarray(body.nose_m)
    if body.facets is not None:
        return inside_faceted(body.facets, rel, tol_m)
    t = rel[:, 0] / body.length_m
    rel[:, 2] -= body_axis_z(body, t) - body.nose_m[2]
    r = body_radius_frac(body, t)
    a = body.half_width_m * r + tol_m
    b = body.half_height_m * r + tol_m
    with np.errstate(divide="ignore", invalid="ignore"):
        level = (
            np.abs(rel[:, 1] / a) ** body.section_exp + np.abs(rel[:, 2] / b) ** body.section_exp
        )
    in_length = (rel[:, 0] >= -tol_m) & (rel[:, 0] <= body.length_m + tol_m)
    return in_length & (level <= 1.0)


def axis_sign(prop: Propulsor) -> float:
    """+1 when the spinner points aft (pusher), -1 when forward (tractor)."""
    return 1.0 if prop.pusher else -1.0


def blade_rings(prop: Propulsor, n_points: int) -> list[np.ndarray]:
    """Elliptic blade sections of the +z blade of the right/centre instance [m]."""
    d = prop.diameter_m
    radius = d / 2
    root_r = number("blade_root_spinner_frac") * number("spinner_radius_frac") * d
    chord_root = number("blade_chord_frac") * d
    pitch = np.radians(number("blade_pitch_deg"))
    chord_dir = np.array([np.sin(pitch), np.cos(pitch), 0.0])
    thick_dir = np.array([np.cos(pitch), -np.sin(pitch), 0.0])
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)[:, None]
    rings = []
    for frac in numbers("blade_stations_frac"):
        r = root_r + frac * (radius - root_r)
        chord = chord_root * (1.0 + frac * (number("blade_tip_chord_frac") - 1.0))
        half_t = 0.5 * chord * number("blade_thickness_frac")
        ring = 0.5 * chord * np.cos(theta) * chord_dir + half_t * np.sin(theta) * thick_dir
        rings.append(np.asarray(prop.hub_m) + ring + np.array([0.0, 0.0, r]))
    return rings


def prop_points(prop: Propulsor, n_points: int) -> np.ndarray:
    """Blade, spinner and shaft points of the right/centre instance [m]."""
    hub = np.asarray(prop.hub_m)
    blade = np.vstack(blade_rings(prop, n_points))
    other = hub + (blade - hub) * FLIP_BLADE
    tip = hub + axis_sign(prop) * number("spinner_length_frac") * prop.diameter_m * X_HAT
    return np.vstack([blade, other, tip, hub, np.asarray(prop.attach_m)])


def _body_samples(body: Body, spacing_m: float, n_ring: int) -> np.ndarray:
    """Surface samples of a body: rings of the smooth law, or vertices and face-edge points."""
    if body.facets is not None:
        return np.asarray(body.nose_m) + sample_points(body.facets, spacing_m)
    return body_points(body, _cosine_stations(body.length_m, spacing_m), n_ring)


def _cosine_stations(length_m: float, spacing_m: float) -> int:
    """Cosine-clustered station count whose widest gap (mid-body) is at most spacing_m."""
    return int(np.ceil(np.pi * length_m / (2.0 * spacing_m))) + 1


def component_points(design: Design, include_props: bool) -> dict[str, np.ndarray]:
    """Sampled points per part instance (mirrored instances included) [m]."""
    n_af = int(number("sample_airfoil_points_per_side"))
    spacing = number("sample_spacing_m")
    ring = int(number("sample_body_ring_points"))
    raw: list[tuple[str, bool, np.ndarray]] = []
    raw += [(s.name, s.mirrored, surface_points(s, n_af, spacing)) for s in design.surfaces]
    raw += [(b.name, b.mirrored, _body_samples(b, spacing, ring)) for b in design.bodies]
    ring = int(number("body_ring_points"))
    if include_props:
        raw += [(p.name, p.mirrored, prop_points(p, ring)) for p in design.propulsors]
    out: dict[str, np.ndarray] = {}
    for name, mirrored, pts in raw:
        names = instance_names(name, mirrored)
        out[names[-1]] = pts
        if mirrored:
            out[names[0]] = pts * MIRROR_Y
    return out
