"""Facet coarseness of a flat-panel body (SPEC §6, U3b): straight nose and tail edges, and as few
section corners as the smallest crease angle and the sheet budget ask for.

One pass on the sized centre body (body_align.next_shape): the radius law becomes linear (the
nose and tail are then each one sheet per corner), and the section polygon loses corners while a
corner turns by less than the smallest crease or the body has more sheets than coarse_body_panels
_max. A step moves vertices by at most coarse_move_frac of the body length; a step over that is
skipped and noted. Pure; metres, degrees."""

import numpy as np

from aeroforge.coarse import lerp
from aeroforge.constants import number
from aeroforge.facet_decimate import min_crease_deg
from aeroforge.facet_pieces import PieceError, pieces
from aeroforge.facets import _radius_frac, effective_plan, facet_body, section_polygon
from aeroforge.models import Body, BodyShape

MIN_CORNERS = 3  # a section polygon keeps at least a triangle


def min_turn_deg(polygon: np.ndarray) -> float:
    """Smallest turn [deg] at a corner of a closed convex polygon (rows are corners)."""
    return float(_turns_deg(polygon).min())


def _to_boundary(points: np.ndarray, polygon: np.ndarray) -> float:
    """Largest distance from any of points to the closed polygon's boundary."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    worst = 0.0
    for p in points:
        t = np.clip(((p - a) * ab).sum(axis=1) / (ab * ab).sum(axis=1), 0.0, 1.0)
        worst = max(worst, float(np.hypot(*(a + t[:, None] * ab - p).T).min()))
    return worst


def polygon_move_m(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance [m] between the boundaries of two section polygons, read at the corners."""
    return max(_to_boundary(a, b), _to_boundary(b, a))


def _scaled(shape: BodyShape, k: int, body: Body) -> np.ndarray:
    return section_polygon(shape, k) * np.array([body.half_width_m, body.half_height_m])


def _linear_move_m(shape: BodyShape, body: Body) -> float:
    """Largest radius change [m] of the nose and tail laws going linear."""
    t = np.linspace(0.0, 1.0, int(number("body_ring_points")))
    plan = effective_plan(shape)
    lin = shape.model_copy(update={"linear_edges": True})
    dr = np.abs(_radius_frac(t, plan, shape) - _radius_frac(t, plan, lin))
    return float(dr.max()) * max(body.half_width_m, body.half_height_m)


def _sheets(shape: BodyShape, body: Body) -> int:
    """Physical sheets of the body with this shape, 0 when they cannot be counted."""
    axes = (body.half_width_m, body.half_height_m)
    poly = facet_body(shape, body.length_m, axes, body.tail_upsweep_deg, body.rise_x_frac)
    try:
        return len(pieces(poly))
    except PieceError:
        return 0


def _turns_deg(polygon: np.ndarray) -> np.ndarray:
    """Turn [deg] at every corner of a closed convex polygon (rows are corners)."""
    d = np.roll(polygon, -1, axis=0) - polygon
    ang = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees(np.abs((np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi))


def _right_rows(polygon: np.ndarray) -> list[int]:
    """Rows of the polygon's right half off the axis, ridge to belly (the droppable corners)."""
    return [i for i, y in enumerate(polygon[:, 0]) if y > 1e-9][: len(polygon) // 2]


def _dropping(
    cur: BodyShape, old: np.ndarray, body: Body, alive: list[int], tests: tuple[bool, float, float]
) -> BodyShape | None:
    """cur with its flattest droppable corner (and mirror) dropped, None when nothing may go:
    every wanted corner is over the cap, or none is wanted. tests = (over budget, smallest
    crease, cap [m])."""
    over, least, cap_m = tests
    poly = _scaled(cur, cur.sides, body)
    turn, rows = _turns_deg(poly), _right_rows(poly)
    if len(poly) - 2 < MIN_CORNERS:
        return None
    for j in sorted(range(len(rows)), key=lambda r: turn[rows[r]]):
        if not over and turn[rows[j]] >= least:
            return None
        trial = cur.model_copy(update={"dropped": tuple(sorted([*cur.dropped, alive[j]]))})
        moved = polygon_move_m(old, _scaled(trial, trial.sides, body))
        if moved <= cap_m and len(poly) - 2 >= MIN_CORNERS:
            alive.pop(j)
            return trial
    return None


def _sides(shape: BodyShape, body: Body, c: float, cap_m: float) -> tuple[BodyShape, list[str]]:
    """shape without the section corners it can lose: the flattest corner of the gene polygon
    goes first (with its mirror) while a corner turns by less than the smallest crease or the body
    has more sheets than the budget, within cap_m of the gene polygon."""
    old, least = _scaled(shape, shape.sides, body), min_crease_deg(c)
    budget = int(number("coarse_body_panels_max"))
    corners = round(lerp(c, "coarse_body_sides_loose", "coarse_body_sides_tight"))
    cur, alive, blocked = shape, list(range(len(_right_rows(old)))), False
    while True:
        poly = _scaled(cur, cur.sides, body)
        n = _sheets(cur, body)
        over = n > budget or len(poly) > corners
        if not over and _turns_deg(poly).min() >= least:
            break
        nxt = _dropping(cur, old, body, alive, (over, least, cap_m))
        if nxt is None:
            blocked = True
            break
        cur = nxt
    return _body_notes(shape, cur, body, c, cap_m, (old, n, blocked))


def _body_notes(
    shape: BodyShape, out: BodyShape, body: Body, c: float, cap_m: float, seen: tuple
) -> tuple[BodyShape, list[str]]:
    """Notes of the corner decimation: what was dropped and what the cap kept."""
    old, n, blocked = seen
    poly, least, notes = _scaled(out, out.sides, body), min_crease_deg(c), []
    if out.dropped:
        k0, k1 = len(old), len(poly)
        move = polygon_move_m(old, poly) / body.length_m
        notes.append(
            f"coarseness {c:.2f}: body section {k0} -> {k1} corners (flattest dropped first; "
            f"largest vertex move {move:.1%} of the body length, "
            f"cap {number('coarse_move_frac'):.0%})"
        )
    turn = float(_turns_deg(poly).min())
    if blocked and turn < least:
        notes.append(
            f"coarseness {c:.2f}: body section corners turn by as little as {turn:.1f} deg "
            f"(smallest crease {least:.1f}): dropping them would move vertices over the cap, "
            "merge blocked by the cap"
        )
    if blocked and n > int(number("coarse_body_panels_max")):
        notes.append(
            f"coarseness {c:.2f}: body has {n} sheets, over the budget (blocked by the cap)"
        )
    return out, notes


def coarsen(shape: BodyShape, body: Body, c: float) -> tuple[BodyShape, tuple[str, ...]]:
    """The sized body's shape made coarser, and a note per change or skipped change."""
    cap_m = number("coarse_move_frac") * body.length_m
    notes: list[str] = []
    if not shape.linear_edges:
        move = _linear_move_m(shape, body)
        if move <= cap_m:
            shape = shape.model_copy(update={"linear_edges": True})
            notes.append(
                f"coarseness {c:.2f}: straight body nose and tail edges (largest radius move "
                f"{move / body.length_m:.1%} of the body length)"
            )
        else:
            notes.append(
                f"coarseness {c:.2f}: curved body nose and tail kept: straight edges would move "
                f"the skin {move / body.length_m:.1%} of the body length, over the cap"
            )
    shape, more = _sides(shape, body, c, cap_m)
    return shape, tuple(notes + more)
