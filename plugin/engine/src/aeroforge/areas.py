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


def _body_top(b: Body, y0: float, s: np.ndarray) -> Intervals:
    """x interval of the planform |y - y0| <= half_width x radius(t), by inverting the nose
    (elliptic) and tail (cosine boat-tail) laws of shapes.body_radius_frac."""
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


def _scan_lines(polys: np.ndarray, centres: list[tuple[Body, float]]) -> tuple[np.ndarray, float]:
    """Scan-line positions (midpoints of equal strips spanning every part) and strip width [m]."""
    lows = [polys[..., 0].min()] if len(polys) else []
    highs = [polys[..., 0].max()] if len(polys) else []
    lows += [y0 - b.half_width_m for b, y0 in centres]
    highs += [y0 + b.half_width_m for b, y0 in centres]
    if not lows:
        return np.zeros(0), 0.0
    n = int(number("area_scan_lines"))
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
