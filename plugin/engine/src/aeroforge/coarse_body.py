"""Facet coarseness of a flat-panel body (SPEC §6, U3b): straight nose and tail edges, and as few
section corners as the smallest crease angle and the sheet budget ask for.

One pass on the sized centre body (body_align.next_shape): the radius law becomes linear (the
nose and tail are then each one sheet per corner), and the section polygon loses corners while a
corner turns by less than the smallest crease or the body has more sheets than coarse_body_panels
_max. A step moves vertices by at most coarse_move_frac of the body length; a step over that is
skipped and noted. Pure; metres, degrees."""

import numpy as np

from aeroforge.coarse import lerp, min_crease_deg
from aeroforge.constants import number
from aeroforge.facet_pieces import PieceError, pieces
from aeroforge.facets import _radius_frac, effective_plan, facet_body, section_polygon
from aeroforge.models import Body, BodyShape


def min_turn_deg(polygon: np.ndarray) -> float:
    """Smallest turn [deg] at a corner of a closed convex polygon (rows are corners)."""
    d = np.roll(polygon, -1, axis=0) - polygon
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.abs((np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi))
    return float(turn.min())


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


def _sides(shape: BodyShape, body: Body, c: float, cap_m: float) -> tuple[BodyShape, list[str]]:
    """shape with the section corners the smallest crease and the sheet budget leave."""
    k0 = shape.sides
    old, least = _scaled(shape, k0, body), min_crease_deg(c)
    want = min(k0, round(lerp(c, "coarse_body_sides_loose", "coarse_body_sides_tight")))
    ladder = [
        k for k in range(want, 2, -1) if polygon_move_m(old, _scaled(shape, k, body)) <= cap_m
    ]
    sharp = [k for k in ladder if min_turn_deg(_scaled(shape, k, body)) >= least]
    k = sharp[0] if sharp else (ladder[0] if ladder else k0)
    out = shape.model_copy(update={"sides": k})
    budget = int(number("coarse_body_panels_max"))
    while (n := _sheets(out, body)) > budget and k > 3 and (k - 1) in ladder:
        k -= 1
        out = shape.model_copy(update={"sides": k})
    notes = []
    if k != k0:
        move = polygon_move_m(old, _scaled(shape, k, body)) / body.length_m
        notes.append(
            f"coarseness {c:.2f}: body section {k0} -> {k} corners (largest vertex move "
            f"{move:.1%} of the body length, cap {number('coarse_move_frac'):.0%})"
        )
    turn = min_turn_deg(_scaled(out, k, body))
    if turn < least:
        notes.append(
            f"coarseness {c:.2f}: body section corners turn by as little as {turn:.1f} deg "
            f"(smallest crease {least:.1f}): fewer corners would move vertices over the cap, "
            "merge skipped"
        )
    if n > budget:
        notes.append(f"coarseness {c:.2f}: body has {n} sheets, over the budget of {budget}")
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
