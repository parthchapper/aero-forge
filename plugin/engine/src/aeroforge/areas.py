"""Top-down and frontal shadow areas of a design (novelty plan step 9).

Pure numpy, no CAD. Each view sweeps scan lines along y; on every line each part contributes one
interval along the other axis (x from above, z from the front), computed exactly. The union of
the intervals on a line is integrated over y, so overlapping parts count once. The only
discretisation error is the spacing of the scan lines along y, never a pixel edge along the
thin axis, which matters for frontal wing bands a centimetre thick.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import hypot

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_query import _hull_2d, cell_projection
from aeroforge.models import Body, Design
from aeroforge.shapes import body_axis_z, body_radius_frac, body_stations, section_normal
from aeroforge.style import panels

Intervals = tuple[
    np.ndarray, np.ndarray
]  # (lo, hi), shape (parts, scan lines); empty = (inf, -inf)


@dataclass(frozen=True)
class Areas:
    """Shadow areas of a design [m^2] and the body's share of the top-down shadow."""

    top_m2: float
    frontal_m2: float
    body_top_share: float


def _none(n_lines: int) -> Intervals:
    """No parts: an empty (0, n_lines) interval pair."""
    return np.empty((0, n_lines)), np.empty((0, n_lines))


def _polygon_intervals(polys: np.ndarray, s: np.ndarray) -> Intervals:
    """Interval of each polygon along its second coordinate on every scan line s.

    polys has shape (parts, corners, 2) with columns (scan coordinate, interval coordinate); a
    polygon is convex or nearly so, and an interval is the span of its edge crossings. Each
    polygon is evaluated only on the scan lines inside its own extent.
    """
    lo = np.full((len(polys), len(s)), np.inf)
    hi = np.full((len(polys), len(s)), -np.inf)
    for k, poly in enumerate(polys):
        i0, i1 = np.searchsorted(s, [poly[:, 0].min(), poly[:, 0].max()])
        sub = s[i0:i1]
        p, q = poly, np.roll(poly, -1, axis=0)
        ds = q[:, 0] - p[:, 0]
        ok = ds != 0.0
        t = (sub[None, :] - p[:, :1]) / np.where(ok, ds, 1.0)[:, None]
        hit = ok[:, None] & (t >= 0.0) & (t <= 1.0)
        v = p[:, 1:] + t * (q[:, 1] - p[:, 1])[:, None]
        lo[k, i0:i1] = np.where(hit, v, np.inf).min(axis=0)
        hi[k, i0:i1] = np.where(hit, v, -np.inf).max(axis=0)
    return lo, hi


def _mirrored(polys: list[np.ndarray], mirrored: bool) -> list[np.ndarray]:
    """polys plus their y -> -y copies when the part is mirrored."""
    if not mirrored:
        return polys
    return [*polys, *(p * np.array([-1.0, 1.0]) for p in polys)]


def _top_polygons(d: Design) -> np.ndarray:
    """Planform quads (y, x) of every surface panel, mirrored instances included."""
    out: list[np.ndarray] = []
    for s in d.surfaces:
        quads = [p.pts_m[[0, 1, 3, 2]][:, [1, 0]] for p in panels(s)]
        out += _mirrored(quads, s.mirrored)
    return np.array(out).reshape(-1, 4, 2)


def _front_polygons(d: Design) -> np.ndarray:
    """Frontal quads (y, z): between two sections, each as its thickness segment along the
    section normal (tc x chord), so a panel adds its canted height and its thickness."""
    out: list[np.ndarray] = []
    min_len = number("style_panel_min_m")
    for s in d.surfaces:
        quads = []
        for a, b in pairwise(s.sections):
            if hypot(b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]) < min_len:
                continue
            ends = []
            for sec in (a, b):
                half = s.tc * sec.chord_m / 2.0 * section_normal(sec.cant_deg)[1:]
                centre = np.array(sec.le_m[1:])
                ends.append((centre - half, centre + half))
            quads.append(np.array([ends[0][0], ends[0][1], ends[1][1], ends[1][0]]))
        out += _mirrored(quads, s.mirrored)
    return np.array(out).reshape(-1, 4, 2)


def _body_centres(d: Design) -> list[tuple[Body, float]]:
    """Every body instance as (body, centre y [m])."""
    out = []
    for b in d.bodies:
        y0 = b.nose_m[1]
        out += [(b, y0), (b, -y0)] if b.mirrored else [(b, y0)]
    return out


def _faceted_shadow(b: Body, y0: float, s: np.ndarray, columns: tuple[int, int]) -> Intervals:
    """Intervals of a faceted body: its convex cells projected onto two body-frame coordinates (the
    union of the cell shadows is the solid's shadow), placed at the body's nose and y0."""
    polys = cell_projection(b.facets, columns) + np.array([y0, b.nose_m[columns[1]]])
    return _polygon_intervals(polys, s)


def _body_top(b: Body, y0: float, s: np.ndarray) -> Intervals:
    """x interval of the planform |y - y0| <= half_width x radius(t), by inverting the nose
    (elliptic) and tail (cosine boat-tail) laws of shapes.body_radius_frac."""
    if b.facets is not None:
        return _faceted_shadow(b, y0, s, (1, 0))
    i0, i1 = np.searchsorted(s, [y0 - b.half_width_m, y0 + b.half_width_m])
    q = np.abs(s[i0:i1] - y0) / b.half_width_m
    q = np.minimum(q, 1.0)
    t_lo = b.nose_frac * (1.0 - np.sqrt(1.0 - q**2))
    c = np.clip((q - b.tail_radius_frac) / (1.0 - b.tail_radius_frac), 0.0, 1.0)
    t_tail = (1.0 - b.tail_frac) + b.tail_frac * np.arccos(2.0 * c - 1.0) / np.pi
    t_hi = np.where(q <= b.tail_radius_frac, 1.0, t_tail)
    lo_all, hi_all = np.full((1, len(s)), np.inf), np.full((1, len(s)), -np.inf)
    lo_all[0, i0:i1] = b.nose_m[0] + b.length_m * t_lo
    hi_all[0, i0:i1] = b.nose_m[0] + b.length_m * t_hi
    return lo_all, hi_all


def _body_front(b: Body, y0: float, s: np.ndarray) -> Intervals:
    """z interval of the super-ellipse sections sampled along x, joined into one span per line
    (consecutive sections overlap, so their union is that span)."""
    if b.facets is not None:
        return _faceted_shadow(b, y0, s, (1, 2))
    i0, i1 = np.searchsorted(s, [y0 - b.half_width_m, y0 + b.half_width_m])
    r = body_radius_frac(b, body_stations(int(number("area_body_stations"))))
    a, h = b.half_width_m * r[:, None], b.half_height_m * r[:, None]
    u = np.abs(s[i0:i1] - y0)[None, :] / np.where(a > 0.0, a, 1.0)
    inside = (a > 0.0) & (u <= 1.0)
    half = h * (1.0 - np.where(inside, u, 0.0) ** b.section_exp) ** (1.0 / b.section_exp)
    z0 = body_axis_z(b, body_stations(int(number("area_body_stations"))))[:, None]
    lo_all, hi_all = np.full((1, len(s)), np.inf), np.full((1, len(s)), -np.inf)
    lo_all[0, i0:i1] = np.where(inside, z0 - half, np.inf).min(axis=0)
    hi_all[0, i0:i1] = np.where(inside, z0 + half, -np.inf).max(axis=0)
    return lo_all, hi_all


def _stack(parts: list[Intervals], n_lines: int) -> Intervals:
    """Concatenate interval pairs along the parts axis."""
    parts = [*parts, _none(n_lines)]
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def _union_length(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Total length of the union of the intervals on each scan line."""
    if lo.shape[0] == 0:
        return np.zeros(lo.shape[1])
    order = np.argsort(lo, axis=0)
    lo_s, hi_s = np.take_along_axis(lo, order, 0), np.take_along_axis(hi, order, 0)
    reach = np.maximum.accumulate(hi_s, axis=0)
    prev = np.vstack([np.full((1, lo.shape[1]), -np.inf), reach[:-1]])
    return np.clip(hi_s - np.maximum(lo_s, prev), 0.0, None).sum(axis=0)


def _scan_lines(
    polys: np.ndarray, centres: list[tuple[Body, float]], n_lines: int | None = None
) -> tuple[np.ndarray, float]:
    """Scan-line positions (midpoints of equal strips spanning every part) and strip width [m];
    n_lines defaults to area_scan_lines."""
    lows = [polys[..., 0].min()] if len(polys) else []
    highs = [polys[..., 0].max()] if len(polys) else []
    lows += [y0 - b.half_width_m for b, y0 in centres]
    highs += [y0 + b.half_width_m for b, y0 in centres]
    if not lows:
        return np.zeros(0), 0.0
    n = int(number("area_scan_lines")) if n_lines is None else n_lines
    width = (max(highs) - min(lows)) / n
    return min(lows) + (np.arange(n) + 0.5) * width, width


def _shadow(polys: np.ndarray, d: Design, front: bool) -> tuple[float, float]:
    """(body-only area, total area) of the shadow in one view [m^2]."""
    centres = _body_centres(d)
    s, width = _scan_lines(polys, centres)
    if width == 0.0:
        return 0.0, 0.0
    body_of = _body_front if front else _body_top
    bodies = _stack([body_of(b, y0, s) for b, y0 in centres], len(s))
    surfaces = _polygon_intervals(polys, s)
    both = _stack([bodies, surfaces], len(s))
    return (
        float(_union_length(*bodies).sum() * width),
        float(_union_length(*both).sum() * width),
    )


def top_area_m2(d: Design) -> float:
    """Area [m^2] of the design's shadow on the x-y plane, seen from above; overlaps count once."""
    return _shadow(_top_polygons(d), d, front=False)[1]


def frontal_area_m2(d: Design) -> float:
    """Area [m^2] of the design's shadow on the y-z plane, seen from the front."""
    return _shadow(_front_polygons(d), d, front=True)[1]


def body_top_share(d: Design) -> float:
    """Body shadow area / total shadow area seen from above: 1 for a bare body, 0 without one."""
    body, total = _shadow(_top_polygons(d), d, front=False)
    return body / total if total > 0.0 else 0.0


def areas(d: Design) -> Areas:
    """All three measures of a design in one pass over the top view."""
    body, total = _shadow(_top_polygons(d), d, front=False)
    frontal = _shadow(_front_polygons(d), d, front=True)[1]
    return Areas(total, frontal, body / total if total > 0.0 else 0.0)


@dataclass(frozen=True)
class Shares:
    """The body's share of the shadow seen from above, from the side and from the front, and the
    part of the wing shadow seen from above that lies inside the body's (all in [0, 1])."""

    top: float
    side: float
    front: float
    blend: float


def _pad(hulls: list[np.ndarray]) -> np.ndarray:
    """Hulls as one array (parts, corners, 2); shorter hulls repeat their last corner."""
    if not hulls:
        return np.zeros((0, 3, 2))
    width = max(len(h) for h in hulls)
    return np.array([np.vstack([h, np.repeat(h[-1:], width - len(h), axis=0)]) for h in hulls])


def _side_surfaces(d: Design) -> np.ndarray:
    """Side-view (x, z) hull of every surface panel: its four corners, each stretched by half the
    local thickness (tc x chord) along the section normal. Mirrored halves coincide."""
    out = []
    for s in d.surfaces:
        for p in panels(s):
            le_a, te_a, le_b, te_b = p.pts_m
            pts = []
            for pt, chord in ((le_a, te_a[0] - le_a[0]), (le_b, te_b[0] - le_b[0])):
                half = s.tc * chord * abs(np.cos(np.radians(p.cant_deg))) / 2.0
                pts += [(pt[0], pt[2] - half), (pt[0], pt[2] + half)]
                pts += [(pt[0] + chord, pt[2] - half), (pt[0] + chord, pt[2] + half)]
            out.append(_hull_2d(np.array(pts)))
    return _pad(out)


def _side_bodies(d: Design) -> np.ndarray:
    """Side-view (x, z) outline of every body: the cells of a flat-panel body projected, the upper
    and lower z of a smooth one along its stations (x-monotone, so the span per line is exact)."""
    out = []
    for b in d.bodies:
        if b.facets is not None:
            cells = cell_projection(b.facets, (0, 2)) + np.array([b.nose_m[0], b.nose_m[2]])
            out += list(cells)
            continue
        t = body_stations(int(number("area_body_stations")))
        x = b.nose_m[0] + b.length_m * t
        z0, h = body_axis_z(b, t), b.half_height_m * body_radius_frac(b, t)
        out.append(np.vstack([np.column_stack([x, z0 + h]), np.column_stack([x, z0 - h])[::-1]]))
    return _pad(out)


def _side_shadow(d: Design, n_lines: int) -> tuple[float, float]:
    """(body-only area, total area) of the shadow on the x-z plane [m^2]."""
    bodies_p, surfaces_p = _side_bodies(d), _side_surfaces(d)
    both = [p for p in (bodies_p, surfaces_p) if len(p)]
    lo = min(p[..., 0].min() for p in both)
    width = (max(p[..., 0].max() for p in both) - lo) / n_lines
    if width <= 0.0:
        return 0.0, 0.0
    s = lo + (np.arange(n_lines) + 0.5) * width
    bodies = _polygon_intervals(bodies_p, s) if len(bodies_p) else _none(n_lines)
    surfaces = _polygon_intervals(surfaces_p, s) if len(surfaces_p) else _none(n_lines)
    total = _union_length(*_stack([bodies, surfaces], n_lines)).sum() * width
    return float(_union_length(*bodies).sum() * width), float(total)


def _top_or_front(d: Design, front: bool, n_lines: int) -> tuple[float, float, float]:
    """(body-only, surfaces-only, total) shadow area of the top or front view [m^2]."""
    polys = _front_polygons(d) if front else _top_polygons(d)
    centres = _body_centres(d)
    s, width = _scan_lines(polys, centres, n_lines)
    if width == 0.0:
        return 0.0, 0.0, 0.0
    body_of = _body_front if front else _body_top
    bodies = _stack([body_of(b, y0, s) for b, y0 in centres], len(s))
    surfaces = _polygon_intervals(polys, s)
    total = _union_length(*_stack([bodies, surfaces], len(s))).sum() * width
    return (
        float(_union_length(*bodies).sum() * width),
        float(_union_length(*surfaces).sum() * width),
        float(total),
    )


def _ratio(part: float, whole: float) -> float:
    return float(np.clip(part / whole, 0.0, 1.0)) if whole > 0.0 else 0.0


def shares(d: Design) -> Shares:
    """Body share of the top, side and front shadows and the wing top shadow hidden in the body
    (overlap = body + wings - union), on desc2_scan_lines scan lines per view."""
    n = int(number("desc2_scan_lines"))
    top_body, top_wing, top_all = _top_or_front(d, False, n)
    front_body, _, front_all = _top_or_front(d, True, n)
    side_body, side_all = _side_shadow(d, n)
    return Shares(
        _ratio(top_body, top_all),
        _ratio(side_body, side_all),
        _ratio(front_body, front_all),
        _ratio(top_body + top_wing - top_all, top_wing),
    )
