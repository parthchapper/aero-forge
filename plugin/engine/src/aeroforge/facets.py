"""Faceted (flat-panel) bodies: section polygons, polyhedron, panel plan, containment (SPEC §4, §5).

Pure numpy, metres, the body's own frame (nose at the origin, x aft, y right, z up). A body is S
stations along x; each station is the same convex K-gon scaled evenly about the axis, so every
face joining two stations is a flat trapezoid. Where the polygon changes proportions (a shape
break) the face is split into two flat triangles. The nose and tail close with flat faces.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from aeroforge.constants import number
from aeroforge.facet_ends import FacetError, apex_points, end_faces, end_panel_count
from aeroforge.models import BodyShape, Polyhedron

NOSES = ("point", "chisel", "blunt")
TAILS = ("point", "knife", "flat")


@dataclass(frozen=True)
class PanelPlan:
    """Counts that fix the panel number: N = K (S - 1) + K B + n_nose + n_tail."""

    sides: int
    stations: int
    breaks: int
    nose: str
    tail: str


def break_stations(plan: PanelPlan, max_x_frac: float) -> list[int]:
    """Stations a shape break may squash, tail end first: only stations other than the widest one
    (squashing that one would flatten the whole body)."""
    t = station_fractions(plan, max_x_frac)
    ends = [len(t) - 1, 0]
    return [i for i in ends if abs(t[i] - max_x_frac) > number("facet_geom_tol_m")]


def effective_plan(shape: BodyShape) -> PanelPlan:
    """The plan the genes ask for, repaired: blunt / flat ends need a station of their own and a
    break needs a station aft of the widest one."""
    need = 1 + (shape.nose == "blunt") + (shape.tail == "flat")
    plan = PanelPlan(shape.sides, max(shape.stations, need), 0, shape.nose, shape.tail)
    wants = shape.tail_flatten >= number("facet_break_flatten_min")
    ends = break_stations(plan, shape.max_section_x_frac)
    tail_end = bool(ends) and ends[0] == plan.stations - 1
    return replace(plan, breaks=int(wants and tail_end))


def _outline(shape: BodyShape) -> list[tuple[str, float, float]]:
    """Corners (name, y, z) of the right half of the convex outline, ridge to belly, in a box of
    half-extent 1: roof ridge, roof / side shoulder, chine, lower waist, belly (or keel)."""
    eps = number("facet_corner_min_frac")
    zc = float(
        np.clip(shape.chine_z_frac, -number("facet_chine_limit"), number("facet_chine_limit"))
    )
    tau = np.radians(shape.side_tumblehome_deg)
    cap = 90.0 - shape.side_tumblehome_deg - number("facet_roof_margin_deg")
    theta = np.radians(min(shape.roof_pitch_deg, cap))
    det = np.cos(theta + tau)
    t = (np.cos(tau) - np.sin(tau) * (1.0 - zc)) / det
    s = ((1.0 - zc) * np.cos(theta) - np.sin(theta)) / det
    pts = [("ridge", 0.0, 1.0)]
    if t > eps and s > eps and t * np.cos(theta) < 1.0 - eps:
        pts.append(("shoulder", float(t * np.cos(theta)), float(1.0 - t * np.sin(theta))))
    pts.append(("chine", 1.0, zc))
    flare = min(
        np.radians(shape.lower_flare_deg), np.arctan(number("facet_flare_margin") / (1 + zc))
    )
    drop = (1.0 + zc) / 2.0
    if flare > eps:
        pts.append(("waist", float(1.0 - drop * np.tan(flare)), float(zc - drop)))
    belly = shape.belly_flat_frac * (1.0 - (1.0 + zc) * np.tan(flare))
    pts.append(("belly", float(belly), -1.0) if belly >= eps else ("keel", 0.0, -1.0))
    return pts


def _spread(total: float, at: list[float], count: int) -> list[float]:
    """at plus count arclength positions, each halving the longest gap of the outline."""
    out = sorted(at)
    for _ in range(count):
        edges = [0.0, *out, total]
        gap = int(np.argmax(np.diff(edges)))
        out = sorted([*out, (edges[gap] + edges[gap + 1]) / 2.0])
    return out


def _turns(d: np.ndarray) -> np.ndarray:
    """Clockwise turn [rad] at every outline vertex given unit segment directions d: the interior
    ones, and at the two axis ends twice the segment's angle to the horizontal (the mirror half
    continues it)."""
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    inner = np.arctan2(-cross, (d[:-1] * d[1:]).sum(axis=1))
    first = 2.0 * np.arctan2(-d[0, 1], d[0, 0])
    last = 2.0 * np.arctan2(abs(d[-1, 1]), abs(d[-1, 0]))
    return np.concatenate([[first], inner, [last]])


def _outline_points(xy: np.ndarray, arc: np.ndarray, s: list[float]) -> np.ndarray:
    """Points of the outline at arclengths s. A point that is not at a true corner is pushed
    outward on a parabola over its straight run (sagitta bounded by the turns at the run's ends),
    so the polygon is strictly convex: no spread corner is collinear with its neighbours."""
    d = np.diff(xy, axis=0)
    d = d / np.hypot(*d.T)[:, None]
    turns = _turns(d)
    corner = turns > number("facet_corner_min_frac")
    c_arc, c_turn = arc[corner], turns[corner]
    out = []
    for si in s:
        i = max(int(np.searchsorted(c_arc, si, side="right")) - 1, 0)
        s0, t0 = c_arc[i], c_turn[i]
        s1, t1 = (c_arc[i + 1], c_turn[i + 1]) if i + 1 < len(c_arc) else (2 * arc[-1] - s0, t0)
        sag = number("facet_bulge_frac") * (s1 - s0) * min(t0, t1)
        push = sag * 4.0 * (si - s0) * (s1 - si) / (s1 - s0) ** 2
        seg = min(int(np.searchsorted(arc, si, side="right")) - 1, len(d) - 1)
        base = np.array([np.interp(si, arc, xy[:, 0]), np.interp(si, arc, xy[:, 1])])
        out.append(base + push * np.array([-d[seg, 1], d[seg, 0]]))
    return np.array(out)


def section_polygon(shape: BodyShape, sides: int) -> np.ndarray:
    """Convex, left-right symmetric K-gon (y, z), shape (K, 2), in a box of half-extent 1.

    Chine, shoulder, belly and waist are corners when K allows (that order); a roof ridge, then
    a keel, take the corners on the axis. The remaining corners are spread evenly along the
    outline.
    """
    chain = _outline(shape)
    on_axis = chain[-1][0] == "keel"
    n_axis = 1 if sides % 2 else (2 if on_axis else 0)
    pairs = (sides - n_axis) // 2
    off_axis = [p for p in chain if p[0] not in ("ridge", "keel")]
    forced = sorted(off_axis, key=lambda p: ("chine", "shoulder", "belly", "waist").index(p[0]))
    kept = {p[0] for p in forced[:pairs]}
    line = [(y, z) for _, y, z in chain] + ([] if on_axis else [(0.0, -1.0)])
    xy = np.array(line)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    at = [float(arc[i]) for i, p in enumerate(chain) if p[0] in kept]
    right = _outline_points(xy, arc, _spread(float(arc[-1]), at, pairs - len(kept)))
    top = [[0.0, 1.0]] if n_axis else []
    bottom = [[0.0, -1.0]] if n_axis == 2 else []
    poly = np.array([*top, *right, *bottom, *(right[::-1] * np.array([-1.0, 1.0]))])
    if len(poly) != sides:
        raise FacetError(f"section polygon has {len(poly)} corners, not {sides}")
    z_lo, z_hi = float(poly[:, 1].min()), float(poly[:, 1].max())
    # centred and scaled to its bounding box so the body axis is inside it and the half-width and
    # half-height of the body are the polygon's real extents
    return np.column_stack(
        [
            poly[:, 0] / np.abs(poly[:, 0]).max(),
            (poly[:, 1] - (z_hi + z_lo) / 2) / ((z_hi - z_lo) / 2),
        ]
    )


def _radius_frac(t: np.ndarray, plan: PanelPlan, shape: BodyShape) -> np.ndarray:
    """Local size / max size at length fraction t: elliptic nose to the widest station, cosine
    boat-tail after it."""
    m, end = (
        shape.max_section_x_frac,
        max(shape.tail_end_size_frac, number("facet_tail_end_min_frac")),
    )
    blunt = number("facet_blunt_nose_size_frac") if plan.nose == "blunt" else 0.0
    u = np.clip((t - m) / (1.0 - m), 0.0, 1.0)
    if shape.linear_edges:  # strict alignment: one straight edge on each side of the widest station
        return np.where(t < m, blunt + (1.0 - blunt) * t / m, end + (1.0 - end) * (1.0 - u))
    nose = blunt + (1.0 - blunt) * np.sqrt(np.clip(1.0 - (1.0 - t / m) ** 2, 0.0, None))
    return np.where(t < m, nose, end + (1.0 - end) * (1.0 + np.cos(np.pi * u)) / 2.0)


def station_fractions(plan: PanelPlan, max_x_frac: float) -> np.ndarray:
    """Length fractions of the stations: the widest at max_x_frac, the others spread each side of
    it in proportion to the length there; a blunt nose / flat tail has a station at its end."""
    lo_b, lo_a = int(plan.nose == "blunt"), int(plan.tail == "flat")
    n_b = int(np.clip(round((plan.stations - 1) * max_x_frac), lo_b, plan.stations - 1 - lo_a))
    n_a = plan.stations - 1 - n_b
    k = np.arange(n_b)
    before = max_x_frac * (k / n_b if lo_b else (k + 1) / (n_b + 1))
    k = np.arange(1, n_a + 1)
    after = max_x_frac + (1 - max_x_frac) * (k / n_a if lo_a else k / (n_a + 1))
    return np.concatenate([before, [max_x_frac], after])


def _outward(face: tuple[int, ...], verts: np.ndarray, inside: np.ndarray) -> tuple[int, ...]:
    """face with its loop turned so the Newell normal points away from the point inside."""
    p = verts[list(face)]
    normal = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
    return face if float(normal @ (p.mean(axis=0) - inside)) >= 0.0 else face[::-1]


def panel_count(plan: PanelPlan, polygon: np.ndarray) -> int:
    """N = K (S - 1 + B) + n_nose + n_tail; a chisel or knife adds a triangle where its two fans
    meet, so such an end has K + 0..2 panels (Deviation from the plan's K, CHANGELOG)."""
    lateral = plan.sides * (plan.stations - 1 + plan.breaks)
    return lateral + end_panel_count(polygon, plan.nose) + end_panel_count(polygon, plan.tail)


def panel_plan(count: int, wish: PanelPlan, shape: BodyShape) -> tuple[PanelPlan, bool]:
    """The plan with `count` panels nearest the wish (sides, stations, breaks, ends), and whether
    the count is exact; when none is, the plan with the nearest count."""
    best: tuple[tuple[object, ...], PanelPlan] | None = None
    for k in range(3, int(number("facet_max_sides")) + 1):
        poly = section_polygon(shape, k)
        ends = {s: end_panel_count(poly, s) for s in (*NOSES, *TAILS)}
        for s in range(1, int(number("facet_max_stations")) + 1):
            for nose, tail in product(NOSES, TAILS):
                if s < 1 + (nose == "blunt") + (tail == "flat"):
                    continue
                fits = len(break_stations(PanelPlan(k, s, 0, nose, tail), shape.max_section_x_frac))
                for b in range(fits + 1):
                    n = k * (s - 1 + b) + ends[nose] + ends[tail]
                    miss = abs(k - wish.sides) + abs(s - wish.stations) + abs(b - wish.breaks)
                    miss += (nose != wish.nose) + (tail != wish.tail)
                    key = (abs(n - count), miss, k, s, b, nose, tail)
                    if best is None or key < best[0]:
                        best = (key, PanelPlan(k, s, b, nose, tail))
    if best is None:
        raise FacetError("no panel plan exists")
    return best[1], best[0][0] == 0


def axis_rise_m(
    t: np.ndarray,
    shape: BodyShape,
    length_m: float,
    upsweep_deg: float,
    rise_x_frac: float | None = None,
) -> np.ndarray:
    """Height [m] of the axis at length fractions t: level to the widest station (or to rise_x_frac
    when given), then rising at upsweep_deg. A station is shifted as a whole, so every face stays
    flat; between two stations the axis is straight, which is exact only at the stations
    themselves (axis_rise_between_m is the body's axis anywhere)."""
    start = shape.max_section_x_frac if rise_x_frac is None else rise_x_frac
    run = np.clip(t - start, 0.0, None) * length_m
    return run * np.tan(np.radians(upsweep_deg))


def axis_rise_between_m(
    t: np.ndarray,
    shape: BodyShape,
    length_m: float,
    upsweep_deg: float,
    rise_x_frac: float | None = None,
) -> np.ndarray:
    """Height [m] of the body's axis at length fractions t, as the flat-panel body has it: the rise
    of each station (axis_rise_m) joined by straight lines, so a rise that starts between two
    stations tilts the whole cell it starts in."""
    st = station_fractions(effective_plan(shape), shape.max_section_x_frac)
    knots = np.concatenate([[0.0], st, [1.0]])
    rise = axis_rise_m(knots, shape, length_m, upsweep_deg, rise_x_frac)
    return np.interp(t, knots, rise)


def _rings(
    plan: PanelPlan,
    shape: BodyShape,
    length_m: float,
    axes_m: tuple[float, float],
    upsweep_deg: float,
    rise_x_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Station rings, shape (S, K, 3) in the body frame, and each station's squash factor."""
    t = station_fractions(plan, shape.max_section_x_frac)
    radius = _radius_frac(t, plan, shape)
    squash = np.zeros(len(t))
    flatten = shape.tail_flatten
    if flatten < number("facet_break_flatten_min"):
        flatten = number("facet_break_flatten_default")  # a plan asks for a break the genes did not
    for i in break_stations(plan, shape.max_section_x_frac)[: plan.breaks]:
        squash[i] = flatten
    poly = section_polygon(shape, plan.sides)
    y = axes_m[0] * radius[:, None] * poly[None, :, 0]
    z = axes_m[1] * radius[:, None] * (1.0 - squash[:, None]) * poly[None, :, 1]
    z = z + axis_rise_m(t, shape, length_m, upsweep_deg, rise_x_frac)[:, None]
    x = np.broadcast_to((t * length_m)[:, None], y.shape)
    return np.stack([x, y, z], axis=-1), squash


def _lateral(
    verts: np.ndarray, ring_a: int, ring_b: int, k: int, broken: bool
) -> list[tuple[int, ...]]:
    """Faces joining the rings starting at vertex ring_a and ring_b: a trapezoid per edge, or at
    a shape break the lateral facets of the hull of both rings (2 K triangles)."""
    if not broken:
        return [
            (ring_a + j, ring_a + (j + 1) % k, ring_b + (j + 1) % k, ring_b + j) for j in range(k)
        ]
    idx = [*range(ring_a, ring_a + k), *range(ring_b, ring_b + k)]
    hull = ConvexHull(verts[idx])
    sides = [
        tuple(idx[i] for i in tri) for tri in hull.simplices if 0 < sum(i < k for i in tri) < 3
    ]
    if len(sides) != 2 * k:
        raise FacetError("shape break between stations is not a clean 2 K triangle skin")
    return sides


def _closure(style: str, verts: np.ndarray, ring: int, k: int, apex: int) -> list[tuple[int, ...]]:
    """Faces closing the end ring starting at vertex `ring`: a cap, or the end faces to the apex
    vertices starting at `apex` (indices mapped to the body's vertex numbers)."""
    if style in ("blunt", "flat"):
        return [tuple(range(ring, ring + k))]
    n_apex = 1 if style == "point" else 2
    local = end_faces(verts[ring : ring + k], verts[apex : apex + n_apex])
    remap = [*range(ring, ring + k), *range(apex, apex + n_apex)]
    return [tuple(remap[i] for i in face) for face in local]


def body_polyhedron(
    plan: PanelPlan,
    shape: BodyShape,
    length_m: float,
    axes_m: tuple[float, float],
    upsweep_deg: float = 0.0,
    rise_x_frac: float | None = None,
) -> Polyhedron:
    """Closed solid of flat faces for a plan: station rings scaled to the half-width and
    half-height axes_m, the nose at the origin, length_m long. Every face is turned outward
    about the centre of the convex cell it bounds."""
    rings, squash = _rings(plan, shape, length_m, axes_m, upsweep_deg, rise_x_frac)
    s, k = rings.shape[:2]
    no_apex = np.zeros((0, 3))
    nose_pts = no_apex if plan.nose == "blunt" else apex_points(plan.nose, rings[0], 0.0)
    if plan.tail == "flat":
        tail_pts = no_apex
    else:  # the apex sits on the axis at the end, above the level ring by the rise gained since
        rise = axis_rise_m(
            np.array([station_fractions(plan, shape.max_section_x_frac)[-1], 1.0]),
            shape,
            length_m,
            upsweep_deg,
            rise_x_frac,
        )
        tail_pts = apex_points(plan.tail, rings[-1] - np.array([0.0, 0.0, rise[0]]), length_m)
        tail_pts = tail_pts + np.array([0.0, 0.0, rise[1]])
    verts = np.vstack([rings.reshape(-1, 3), nose_pts, tail_pts])
    ring = [list(range(i * k, (i + 1) * k)) for i in range(s)]
    nose_at = s * k
    tail_at = nose_at + len(nose_pts)
    faces: list[tuple[int, ...]] = []

    def emit(new: list[tuple[int, ...]], cell: list[int]) -> None:
        centre = verts[cell].mean(axis=0)
        faces.extend(_outward(f, verts, centre) for f in new)

    nose_cell = ring[0] + (list(range(nose_at, tail_at)) if len(nose_pts) else ring[1])
    emit(_closure(plan.nose, verts, 0, k, nose_at), nose_cell)
    for i in range(s - 1):
        broken = bool(squash[i] != squash[i + 1])
        emit(_lateral(verts, i * k, (i + 1) * k, k, broken), ring[i] + ring[i + 1])
    tail_cell = ring[-1] + (list(range(tail_at, len(verts))) if len(tail_pts) else ring[-2])
    emit(_closure(plan.tail, verts, (s - 1) * k, k, tail_at), tail_cell)
    return Polyhedron(
        vertices_m=tuple((float(x), float(y), float(z)) for x, y, z in verts), faces=tuple(faces)
    )


@lru_cache(maxsize=512)
def facet_body(
    shape: BodyShape,
    length_m: float,
    axes_m: tuple[float, float],
    upsweep_deg: float = 0.0,
    rise_x_frac: float | None = None,
) -> Polyhedron:
    """Polyhedron of the plan the shape genes ask for (repaired by effective_plan)."""
    return body_polyhedron(effective_plan(shape), shape, length_m, axes_m, upsweep_deg, rise_x_frac)
