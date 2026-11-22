"""Pure body shift (user Deviation, step 3 gate r3): body0 moves along x relative to the surfaces.

Roots on body0 keep the spacing they have on the unshifted body (the anchor); the end of the
body that a root would run off grows instead, within body_grow_max_ref_frac x ref length. Body
growth keeps fineness, so a grown body is also fatter (size_centre_body).
"""

from dataclasses import dataclass

import numpy as np

from aeroforge.constants import number
from aeroforge.layout import size_centre_body
from aeroforge.models import Body, Family, Requirements

GROWTH_TOL_M = 1e-9  # growth changes below this are round-off between repair passes
CAP_BISECT_ITER = 60


@dataclass(frozen=True)
class Anchor:
    """Unshifted body length the roots are spaced on and the x offset added to them [m]."""

    length_m: float
    offset_m: float


@dataclass(frozen=True)
class Growth:
    """Front and rear body growth [m] and the shift actually applied [m]."""

    front_m: float
    rear_m: float
    shift_m: float


def centre_body(
    req: Requirements,
    fam: Family,
    length_frac: float,
    min_length_m: float,
    g: Growth,
    holds_boxes: bool = True,
) -> tuple[Body, Anchor | None]:
    """Centre body grown for the shift, and the root anchor (None without a shift)."""
    base = size_centre_body(req, fam, length_frac, min_length_m, holds_boxes)
    if g.shift_m == 0.0:
        return base, None
    grown = size_centre_body(
        req, fam, length_frac, base.length_m + g.front_m + g.rear_m, holds_boxes
    )
    return grown, Anchor(base.length_m, g.front_m - g.shift_m)


def anchored_x_m(root_x_frac: float, chord_m: float, anchor: Anchor) -> float:
    """Root mid-chord x from the body nose: its unshifted spot plus the anchor offset [m]."""
    half = chord_m / 2
    unshifted = float(np.clip(root_x_frac * anchor.length_m, half, anchor.length_m - half))
    return unshifted + anchor.offset_m


def _needed(
    roots: list[tuple[float, float]], length_m: float, shift_m: float
) -> tuple[float, float]:
    """(front, rear) growth [m] keeping every (root_x_frac, chord) on the body after shift_m."""
    front = rear = 0.0
    for frac, chord in roots:
        half = chord / 2
        rel = float(np.clip(frac * length_m, half, length_m - half)) - shift_m
        front = max(front, half - rel)
        rear = max(rear, rel + half - length_m)
    return front, rear


def growth(
    roots: list[tuple[float, float]], anchor_length_m: float, shift_m: float, req: Requirements
) -> Growth:
    """Growth for the requested shift; the shift shrinks toward 0 when the body cap binds."""
    cap = number("body_grow_max_ref_frac") * req.ref_length_m

    def total(s: float) -> float:
        return anchor_length_m + sum(_needed(roots, anchor_length_m, s))

    applied = shift_m
    if total(shift_m) > cap:
        ok, bad = 0.0, shift_m
        for _ in range(CAP_BISECT_ITER):
            mid = (ok + bad) / 2
            ok, bad = (mid, bad) if total(mid) <= cap else (ok, mid)
        applied = ok
    front, rear = _needed(roots, anchor_length_m, applied)
    return Growth(front, rear, applied)
