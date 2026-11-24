"""Internals (SPEC §10): hardpoints placed from Design fields, completeness and containment.

Surface items are laid out in their section frame (unit-chord coordinates), so containment is an
exact 2-D test against the section outline; body items are tested against the analytic body."""

from itertools import pairwise
from typing import Literal

import numpy as np

from aeroforge.airfoil import naca_outline
from aeroforge.constants import number, numbers
from aeroforge.models import Body, Design, Frozen, Recovery, Section, Surface, Vec3
from aeroforge.shapes import (
    MIRROR_Y,
    X_HAT,
    body_axis_z,
    in_outline,
    inside_body,
    inside_loft,
    section_incidence_deg,
    section_normal,
)

Item = Literal["spar", "joiner", "hinge", "servo_tray", "mount", "bay", "hatch", "recovery"]
IDENTITY: tuple[Vec3, Vec3, Vec3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ORIGIN: Vec3 = (0.0, 0.0, 0.0)


class Internal(Frozen):
    """One internal part (right/centre instance; mirrored ones get a `_L` twin).

    tube: path_m polyline, od_m / id_m. box: centre_m, size_m (length, width, height along axes),
    hollow open_top (tray) or ring (outline frame) with wall_m. disc: centre_m, axes[0] normal,
    size_m (thickness, radius, 0), holes_m bolt centres. Containment: section_uv (unit-chord points
    that must lie in `surface`'s outline) or, without a surface, the part inside any of `within`.
    """

    name: str
    item: Item
    owner: str
    mirrored: bool
    shape: Literal["tube", "box", "disc"]
    path_m: tuple[Vec3, ...] = ()
    od_m: float = 0.0
    id_m: float = 0.0
    centre_m: Vec3 = ORIGIN
    size_m: Vec3 = ORIGIN
    axes: tuple[Vec3, Vec3, Vec3] = IDENTITY
    hollow: Literal["solid", "open_top", "ring"] = "solid"
    wall_m: float = 0.0
    holes_m: tuple[Vec3, ...] = ()
    hole_d_m: float = 0.0
    surface: str | None = None
    section_uv: tuple[tuple[float, float], ...] = ()
    within: tuple[str, ...] = ()


def _vec(a: np.ndarray) -> Vec3:
    """numpy 3-vector as a plain float tuple."""
    return (float(a[0]), float(a[1]), float(a[2]))


def _unit(a: np.ndarray) -> np.ndarray:
    """a / |a|."""
    return a / np.linalg.norm(a)


def outline(s: Surface) -> np.ndarray:
    """Unit-chord outline of the surface's airfoil (same for every section)."""
    return naca_outline(
        s.camber_frac, s.camber_loc_frac, s.tc, int(number("airfoil_points_per_side"))
    )


def mid_half(xy: np.ndarray, xc: float) -> tuple[float, float]:
    """(camber-line y, half thickness) / chord at chord station xc."""
    split = len(xy) // 2
    upper, lower = xy[: split + 1][::-1], xy[split:]
    yu = float(np.interp(xc, upper[:, 0], upper[:, 1]))
    yl = float(np.interp(xc, lower[:, 0], lower[:, 1]))
    return (yu + yl) / 2, (yu - yl) / 2


def _frame(s: Surface, sec: Section) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(leading edge, chord direction, thickness direction) of a section, incidence included."""
    normal = section_normal(sec.cant_deg)
    i = np.radians(section_incidence_deg(s, sec))
    return (
        np.asarray(sec.le_m),
        np.cos(i) * X_HAT - np.sin(i) * normal,
        np.sin(i) * X_HAT + np.cos(i) * normal,
    )


def _point(s: Surface, sec: Section, xc: float, yc: float) -> np.ndarray:
    """Point at unit-chord (xc, yc) of a section [m] (the loft's own parametrisation)."""
    le, cd, td = _frame(s, sec)
    return le + sec.chord_m * (xc * cd + yc * td)


def _station(s: Surface, f: float) -> tuple[int, float]:
    """(structural panel index, blend weight) at fraction f of the structural LE path length."""
    les = np.array([sec.le_m for sec in s.sections[: s.n_structural]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(les, axis=0), axis=1))])
    d = float(np.clip(f, 0.0, 1.0)) * cum[-1]
    k = int(min(np.searchsorted(cum, d, side="right") - 1, len(cum) - 2))
    return k, float((d - cum[k]) / (cum[k + 1] - cum[k]))


def _blend(s: Surface, f: float, xc: float, yc: float) -> tuple[np.ndarray, float]:
    """Loft point at structural fraction f and unit-chord (xc, yc) [m], and the local chord [m]."""
    k, w = _station(s, f)
    a, b = s.sections[k], s.sections[k + 1]
    p = (1 - w) * _point(s, a, xc, yc) + w * _point(s, b, xc, yc)
    return p, (1 - w) * a.chord_m + w * b.chord_m


def _ring_uv(xc: float, yc: float, r_m: float, chords: list[float]) -> list[tuple[float, float]]:
    """Unit-chord ring of radius r_m around (xc, yc) at each chord."""
    t = np.linspace(0.0, 2 * np.pi, int(number("containment_ring_points")), endpoint=False)
    return [
        (float(xc + r_m * np.cos(a) / c), float(yc + r_m * np.sin(a) / c))
        for c in chords
        for a in t
    ]


def _spar_items(s: Surface) -> list[Internal]:
    """Spar tube through the structural sections plus a joiner rod at the root and each break."""
    xy, skin = outline(s), number("spar_skin_thickness_m")
    xs = s.spar.chord_frac
    ym = mid_half(xy, xs)[0]
    structural = s.sections[: s.n_structural]
    path = [_point(s, sec, xs, ym) for sec in structural]
    chords = [sec.chord_m for sec in structural]
    common = {"owner": s.name, "mirrored": s.mirrored, "shape": "tube", "surface": s.name}
    items = [
        Internal(
            name=f"spar_{s.name}",
            item="spar",
            path_m=tuple(map(_vec, path)),
            od_m=s.spar.od_m,
            id_m=s.spar.id_m,
            section_uv=tuple(_ring_uv(xs, ym, s.spar.od_m / 2 + skin, chords)),
            **common,
        )
    ]
    dirs = [_unit(b - a) for a, b in pairwise(path)]
    half = number("joiner_length_m") / 2
    for k in range(len(path) - 1):
        axis = dirs[0] if k == 0 else _unit(dirs[k - 1] + dirs[k])
        items.append(
            Internal(
                name=f"joiner_{s.name}_{k}",
                item="joiner",
                path_m=(_vec(path[k] - half * axis), _vec(path[k] + half * axis)),
                od_m=s.spar.id_m,
                section_uv=tuple(_ring_uv(xs, ym, s.spar.id_m / 2, [chords[k]])),
                **common,
            )
        )
    return items


def _hinge(s: Surface) -> Internal | None:
    """Hinge-line marker rod over the control span (diameter capped to the local thickness)."""
    if s.control is None:
        return None
    xy = outline(s)
    xh = 1.0 - s.control.chord_frac
    ym, half = mid_half(xy, xh)
    (p0, c0), (p1, c1) = (
        _blend(s, f, xh, ym) for f in (s.control.span_start_frac, s.control.span_end_frac)
    )
    d = min(number("hinge_marker_d_m"), number("hinge_fill_frac") * 2 * half * min(c0, c1))
    return Internal(
        name=f"hinge_{s.name}",
        item="hinge",
        owner=s.name,
        mirrored=s.mirrored,
        shape="tube",
        path_m=(_vec(p0), _vec(p1)),
        od_m=d,
        surface=s.name,
        section_uv=tuple(_ring_uv(xh, ym, d / 2, [c0, c1])),
    )


def _tray_in_surface(s: Surface, f: float) -> Internal | None:
    """Servo tray between spar and hinge at structural fraction f, None when it does not fit."""
    if s.control is None:
        return None
    xy, skin = outline(s), number("spar_skin_thickness_m")
    length, width, height = numbers("servo_tray_m")
    span = float(
        np.linalg.norm(np.subtract(s.sections[s.n_structural - 1].le_m, s.sections[0].le_m))
    )
    dw = width / 2 / span
    c = min(_blend(s, g, 0.0, 0.0)[1] for g in (max(f - dw, 0.0), f, min(f + dw, 1.0)))
    front = s.spar.chord_frac + s.spar.od_m / 2 / c
    back = 1.0 - s.control.chord_frac
    xcen = (front + back) / 2
    ym = mid_half(xy, xcen)[0]
    hu, hv = length / 2 / c, (height / 2 + skin) / c
    uv = [(xcen + a * hu, ym + b * hv) for a in (-1, 1) for b in (-1, 1)]
    u, v = np.array(uv).T
    if 2 * hu > back - front or not bool(in_outline(xy, u, v).all()):
        return None
    k = _station(s, f)[0]
    _, cd, td = _frame(s, s.sections[k])
    axes = (_vec(cd), _vec(_unit(np.cross(td, cd))), _vec(td))
    return Internal(
        name=f"servo_{s.name}",
        item="servo_tray",
        owner=s.name,
        mirrored=s.mirrored,
        shape="box",
        centre_m=_vec(_blend(s, f, xcen, ym)[0]),
        size_m=(length, width, height),
        axes=axes,
        hollow="open_top",
        wall_m=number("servo_tray_wall_m"),
        surface=s.name,
        section_uv=tuple(uv),
    )


def _tray_in_body(s: Surface, body: Body) -> Internal:
    """Servo tray in the host body beside the root chord (pushrod drive): a thin surface."""
    length, width, height = numbers("servo_tray_m")
    root = s.sections[0]
    # a mirrored surface on the centre body puts its right tray beside the centreline; on a pod
    # (mirrored itself) the right tray sits on the right pod's axis
    y = body.nose_m[1] + (width if s.mirrored and not body.mirrored else 0.0)
    x = root.le_m[0] + root.chord_m / 2
    z = float(body_axis_z(body, np.array([(x - body.nose_m[0]) / body.length_m]))[0])
    centre = (x, y, z)
    return Internal(
        name=f"servo_{s.name}",
        item="servo_tray",
        owner=s.name,
        mirrored=s.mirrored,
        shape="box",
        centre_m=centre,
        size_m=(length, width, height),
        hollow="open_top",
        wall_m=number("servo_tray_wall_m"),
        within=(body.name,),
    )


def _servo(s: Surface, bodies: dict[str, Body]) -> Internal | None:
    """Servo tray at mid control span, else at the root section, else in the host body."""
    if s.control is None:
        return None
    mid = (s.control.span_start_frac + s.control.span_end_frac) / 2
    for f in (mid, 0.0):
        tray = _tray_in_surface(s, f)
        if tray is not None:
            return tray
    if s.host in bodies:
        return _tray_in_body(s, bodies[s.host])
    # No room anywhere: keep the mid-span tray so containment reports the misfit.
    return _forced_tray(s, mid)


def _forced_tray(s: Surface, f: float) -> Internal:
    """Mid-span tray with its section corners recorded even though they do not fit."""
    xy = outline(s)
    length, width, height = numbers("servo_tray_m")
    _, c = _blend(s, f, 0.0, 0.0)
    xcen = (s.spar.chord_frac + 1.0 - s.control.chord_frac) / 2 if s.control else 0.5
    ym = mid_half(xy, xcen)[0]
    uv = tuple(
        (xcen + a * length / 2 / c, ym + b * height / 2 / c) for a in (-1, 1) for b in (-1, 1)
    )
    return Internal(
        name=f"servo_{s.name}",
        item="servo_tray",
        owner=s.name,
        mirrored=s.mirrored,
        shape="box",
        centre_m=_vec(_blend(s, f, xcen, ym)[0]),
        size_m=(length, width, height),
        hollow="open_top",
        wall_m=number("servo_tray_wall_m"),
        surface=s.name,
        section_uv=uv,
    )


def probe_points(it: Internal) -> np.ndarray:
    """Outer section points of a 3-D internal: tube rings, box corners, disc face rings [m]."""
    n = int(number("containment_ring_points"))
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)[:, None]
    ax = np.asarray(it.axes)
    if it.shape == "box":
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) / 2
        return np.asarray(it.centre_m) + (signs * np.asarray(it.size_m)) @ ax
    if it.shape == "disc":
        thick, radius = it.size_m[0], it.size_m[1]
        ring = radius * (np.cos(t) * ax[1] + np.sin(t) * ax[2])
        c = np.asarray(it.centre_m)
        return np.vstack([c + s * thick / 2 * ax[0] + ring for s in (-1, 1)])
    path = np.asarray(it.path_m)
    rings = []
    for k, p in enumerate(path):
        d = _unit(path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)])
        u = _unit(np.cross(d, [0.0, 0.0, 1.0])) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        rings.append(p + it.od_m / 2 * (np.cos(t) * u + np.sin(t) * np.cross(d, u)))
    return np.vstack(rings)


def inside_part(design: Design, name: str, pts: np.ndarray) -> np.ndarray:
    """Mask of pts inside the outer mould of component `name` (body or surface, both instances)."""
    for b in design.bodies:
        if b.name == name:
            hit = inside_body(b, pts)
            return hit | inside_body(b, pts * MIRROR_Y) if b.mirrored else hit
    for s in design.surfaces:
        if s.name == name:
            return inside_loft(s.sections, outline(s), pts, s.mirrored, s.incidence_deg)
    raise KeyError(f"no component {name!r}")


def contained(design: Design, it: Internal) -> bool:
    """Every section point of the internal lies inside its outer mould (analytic)."""
    if it.surface is not None:
        s = next(x for x in design.surfaces if x.name == it.surface)
        u, v = np.array(it.section_uv).T
        return bool(in_outline(outline(s), u, v).all())
    pts = probe_points(it)
    mask = np.zeros(len(pts), dtype=bool)
    for name in it.within:
        mask |= inside_part(design, name, pts)
    return bool(mask.all())


def surface_items(design: Design) -> list[Internal]:
    """Spars, joiners, hinge markers and servo trays."""
    bodies = {b.name: b for b in design.bodies}
    items: list[Internal] = []
    for s in design.surfaces:
        items += _spar_items(s)
        items += [x for x in (_hinge(s), _servo(s, bodies)) if x is not None]
    return items


def expected_names(design: Design, recovery: Recovery) -> list[str]:
    """Names SPEC §10 requires: spar + joiners per surface, hinge + tray per control, mount per
    propulsor (its prop part carries the spinner), frame + hatch per bay, one recovery item."""
    names = [f"recovery_{recovery}"]
    for s in design.surfaces:
        names += [f"spar_{s.name}"] + [f"joiner_{s.name}_{k}" for k in range(s.n_structural - 1)]
        names += [f"hinge_{s.name}", f"servo_{s.name}"] if s.control is not None else []
    names += [f"mount_{p.name}" for p in design.propulsors]
    names += [f"{kind}_{b.name}" for b in design.boxes for kind in ("bay", "hatch")]
    return sorted(names)


def missing(design: Design, recovery: Recovery, items: tuple[Internal, ...]) -> list[str]:
    """Completeness: required internals absent from items."""
    have = {it.name for it in items}
    return [n for n in expected_names(design, recovery) if n not in have]
