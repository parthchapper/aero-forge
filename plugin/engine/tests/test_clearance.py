"""Prop clearance (SPEC §5 inv 7) and wing-box fit (inv 5) with samplers independent of the
decoder's: denser, offset stations and a brute-force loft inversion instead of the quadratic."""

from itertools import pairwise

import numpy as np
import pytest

from aeroforge.airfoil import wing_outline
from aeroforge.constants import constant, number
from aeroforge.genome import decode
from aeroforge.models import Body, Box, Design, Requirements, Section, Surface
from aeroforge.shapes import body_radius_frac

SPAN_STEP_M = 0.004  # decoder samples every 8 mm
AIRFOIL_POINTS = 41  # decoder uses 16 per side
BODY_STEP_M = 0.004
BODY_RING = 96  # decoder uses 48
LOFT_F_STEPS = 801  # sign-change scan, then linear refinement
BOX_LATTICE = 7  # decoder checks 5 per axis
BELLY_VS_WHEELS_GENOMES = 60
MIRROR = np.array([1.0, -1.0, 1.0])
X_HAT = np.array([1.0, 0.0, 0.0])


def _normal(cant_deg: float) -> np.ndarray:
    c = np.radians(cant_deg)
    return np.array([0.0, -np.sin(c), np.cos(c)])


def _outline(sec: Section, xy: np.ndarray) -> np.ndarray:
    # the section's twist turns it nose-up about its leading edge (decode leaves the surface
    # incidence at zero); written out here, not shapes.section_incidence_deg
    i, normal = np.radians(sec.twist_deg), _normal(sec.cant_deg)
    chord_dir = np.cos(i) * X_HAT - np.sin(i) * normal
    thick_dir = np.sin(i) * X_HAT + np.cos(i) * normal
    return np.asarray(sec.le_m) + sec.chord_m * (xy[:, :1] * chord_dir + xy[:, 1:] * thick_dir)


def _dense(xy: np.ndarray) -> np.ndarray:
    """A polygon's edges cut into AIRFOIL_POINTS / 2 parts each (a polygon's long edges would hide
    a disc crossing them); a smooth outline is dense already."""
    t = np.linspace(0.0, 1.0, AIRFOIL_POINTS // 2, endpoint=False)[None, :, None]
    return ((1 - t) * xy[:-1, None, :] + t * xy[1:, None, :]).reshape(-1, 2)


def _outlines(s: Surface) -> list[np.ndarray]:
    xy = wing_outline(s.camber_frac, s.camber_loc_frac, s.tc, AIRFOIL_POINTS, s.airfoil_facets)
    xy = _dense(xy) if s.airfoil_facets else xy
    return [_outline(sec, xy) for sec in s.sections]


def _surface_points(s: Surface, x_window_m: tuple[float, float]) -> np.ndarray:
    """Loft points of panels reaching into the x window (a loft point lies between its two
    outlines, so a panel whose outlines miss the window has no point in it)."""
    blocks = [np.empty((0, 3))]
    for (a, b), (oa, ob) in zip(pairwise(s.sections), pairwise(_outlines(s)), strict=True):
        both = np.vstack([oa, ob])
        if both[:, 0].max() < x_window_m[0] or both[:, 0].min() > x_window_m[1]:
            continue
        n = int(np.ceil(np.linalg.norm(np.subtract(b.le_m, a.le_m)) / SPAN_STEP_M)) + 1
        f = np.concatenate([[0.0], (np.arange(n) + 0.5) / n, [1.0]])[:, None, None]
        blocks.append(((1 - f) * oa + f * ob).reshape(-1, 3))
    return np.vstack(blocks)


def _facet_points(b: Body, x_window_m: tuple[float, float]) -> np.ndarray:
    """Points on every flat face of a faceted body (fan triangles, barycentric grid of about
    BODY_STEP_M) inside the x window."""
    assert b.facets is not None
    v = np.asarray(b.facets.vertices_m) + np.asarray(b.nose_m)
    blocks = []
    for face in b.facets.faces:
        for i in range(1, len(face) - 1):
            a, p, q = v[face[0]], v[face[i]], v[face[i + 1]]
            n = max(
                int(np.ceil(max(np.linalg.norm(p - a), np.linalg.norm(q - a)) / BODY_STEP_M)), 1
            )
            u, w = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
            keep = (u + w <= 1.0).ravel()
            blocks.append(a + u.ravel()[keep, None] * (p - a) + w.ravel()[keep, None] * (q - a))
    pts = np.vstack(blocks)
    return pts[(pts[:, 0] >= x_window_m[0]) & (pts[:, 0] <= x_window_m[1])]


def _body_points(b: Body, x_window_m: tuple[float, float]) -> np.ndarray:
    """Body points at stations inside the x window."""
    if b.facets is not None:
        return _facet_points(b, x_window_m)
    t_lo, t_hi = np.clip((np.asarray(x_window_m) - b.nose_m[0]) / b.length_m, 0.0, 1.0)
    n = int(np.ceil((t_hi - t_lo) * b.length_m / BODY_STEP_M)) + 1
    t = t_lo + (t_hi - t_lo) * np.concatenate([[0.0], (np.arange(n) + 0.5) / n, [1.0]])
    theta = (np.arange(BODY_RING) + 0.5) * 2 * np.pi / BODY_RING
    r = body_radius_frac(b, t)[:, None]
    c, s = np.cos(theta), np.sin(theta)
    p = 2.0 / b.section_exp
    y = b.half_width_m * r * np.sign(c) * np.abs(c) ** p
    z = b.half_height_m * r * np.sign(s) * np.abs(s) ** p
    x = np.broadcast_to(t[:, None] * b.length_m, y.shape)
    # the tail cone's axis rises linearly at tail_upsweep_deg (written out, not shapes.body_axis_z)
    rise = np.clip(t - (1.0 - b.tail_frac), 0.0, None) * b.length_m
    z = z + (rise * np.tan(np.radians(b.tail_upsweep_deg)))[:, None]
    return (np.asarray(b.nose_m) + np.stack([x, y, z], axis=-1)).reshape(-1, 3)


def _structure_points(d: Design, x_window_m: tuple[float, float]) -> np.ndarray:
    parts = [(_surface_points(s, x_window_m), s.mirrored) for s in d.surfaces]
    parts += [(_body_points(b, x_window_m), b.mirrored) for b in d.bodies]
    return np.vstack([q for p, m in parts for q in ((p, p * MIRROR) if m else (p,))])


def _body_low_z(b: Body) -> float:
    """Lowest point of a body: the smooth rule (widest station) or, for a flat-panel body, its
    polyhedron when that goes deeper (a coarse body went 8 mm below the rule, U3 audit)."""
    rule = b.nose_m[2] - b.half_height_m
    if b.facets is None:
        return rule
    return min(rule, float(np.asarray(b.facets.vertices_m)[:, 2].min()))


def _lowest_z(d: Design) -> float:
    """Lowest structure point: a ruled loft is linear between outlines, so its extremes lie on
    the section outlines; a body's lowest ring point is at its widest station."""
    z_surfaces = min(float(o[:, 2].min()) for s in d.surfaces for o in _outlines(s))
    z_bodies = [_body_low_z(b) for b in d.bodies]
    return min([z_surfaces, *z_bodies])


def _discs(d: Design) -> list[tuple[np.ndarray, float]]:
    hubs = [(np.asarray(p.hub_m), p.diameter_m / 2, p.mirrored) for p in d.propulsors]
    return [(h * k, r) for h, r, m in hubs for k in ((1.0, MIRROR) if m else (1.0,))]


def test_prop_discs_clear_structure_by_the_margin(decoded: list[Design]) -> None:
    margin = number("prop_clearance_m")
    for d in decoded:
        for hub, r in _discs(d):
            pts = _structure_points(d, (hub[0] - margin, hub[0] + margin))
            rho = np.hypot(pts[:, 1] - hub[1], pts[:, 2] - hub[2])
            assert not np.any((rho <= r + margin) & (np.abs(pts[:, 0] - hub[0]) < margin))


def _ground_line_applies(brief: Requirements) -> bool:
    recoveries = constant("constants", "ground_line_recoveries")
    assert isinstance(recoveries, list)
    return brief.recovery in recoveries


def test_props_clear_the_ground_line_for_wheels_and_skid(
    decoded: list[Design], brief: Requirements
) -> None:
    if not _ground_line_applies(brief):
        pytest.skip(f"{brief.recovery} recovery has no ground line (CHANGELOG Deviations)")
    margin = number("prop_clearance_m")
    for d in decoded:
        ground = _lowest_z(d)
        assert all(hub[2] - r >= ground + margin for hub, r in _discs(d))


def test_belly_recovery_keeps_props_on_their_mount_height(
    genomes: np.ndarray, req: Requirements
) -> None:
    assert req.recovery == "belly"
    wheels = req.model_copy(update={"recovery": "wheels"})
    lower = 0
    for g in genomes[:BELLY_VS_WHEELS_GENOMES]:
        belly_z = [p.hub_m[2] - p.attach_m[2] for p in decode(g, req).propulsors]
        wheels_z = [p.hub_m[2] - p.attach_m[2] for p in decode(g, wheels).propulsors]
        assert all(b <= w for b, w in zip(belly_z, wheels_z, strict=True))
        lower += any(b < w for b, w in zip(belly_z, wheels_z, strict=True))
    assert lower > 0


def _lattice(box: Box) -> np.ndarray:
    u = np.linspace(-0.5, 0.5, BOX_LATTICE)
    grid = np.stack(np.meshgrid(u, u, u, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(box.centre_m) + grid * np.asarray(box.size_m)


def in_outline(xy: np.ndarray, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    split = len(xy) // 2
    upper, lower = xy[: split + 1][::-1], xy[split:]
    z_hi = np.interp(xc, upper[:, 0], upper[:, 1])
    z_lo = np.interp(xc, lower[:, 0], lower[:, 1])
    return (xc >= 0) & (xc <= 1) & (yc <= z_hi) & (yc >= z_lo)


def _inside_panel_brute(a: Section, b: Section, xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Scan the spanwise fraction f for zeros of the yz cross product, refine, test the section."""
    f = np.linspace(0.0, 1.0, LOFT_F_STEPS)

    def at(ff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = ff[..., None]
        le = (1 - w) * np.asarray(a.le_m) + w * np.asarray(b.le_m)
        m = (1 - w) * a.chord_m * _normal(a.cant_deg) + w * b.chord_m * _normal(b.cant_deg)
        return le, m, (1 - ff) * a.chord_m + ff * b.chord_m

    le, m, _ = at(f)
    rel = pts[:, None, :] - le[None]
    cross = rel[..., 1] * m[None, :, 2] - rel[..., 2] * m[None, :, 1]
    pi, fi = np.nonzero(np.sign(cross[:, :-1]) * np.sign(cross[:, 1:]) <= 0)
    c0, c1 = cross[pi, fi], cross[pi, fi + 1]
    t = np.where(c0 == c1, 0.0, c0 / np.where(c0 == c1, 1.0, c0 - c1))
    le_r, m_r, chord_r = at(f[fi] + t * (f[fi + 1] - f[fi]))
    r = pts[pi] - le_r
    yc = np.einsum("ij,ij->i", r, m_r) / np.einsum("ij,ij->i", m_r, m_r)
    out = np.zeros(len(pts), dtype=bool)
    np.logical_or.at(out, pi, in_outline(xy, r[:, 0] / chord_r, yc))
    return out


def _inside_wing(s: Surface, pts: np.ndarray) -> np.ndarray:
    xy = wing_outline(s.camber_frac, s.camber_loc_frac, s.tc, 40, s.airfoil_facets)
    out = np.zeros(len(pts), dtype=bool)
    for a, b in pairwise(s.sections):
        # A loft point lies between its two outlines, so its y is within their y range.
        y = np.concatenate([_outline(a, xy)[:, 1], _outline(b, xy)[:, 1]])
        for p in (pts, pts * MIRROR):
            near = (p[:, 1] >= y.min()) & (p[:, 1] <= y.max())
            out[near] |= _inside_panel_brute(a, b, xy, p[near])
    return out


def test_wing_boxes_fit_the_real_loft_across_their_width(decoded: list[Design]) -> None:
    for d in decoded:
        for box in d.boxes:
            if box.host == "surface0":
                assert _inside_wing(d.surfaces[0], _lattice(box)).all(), box.name
