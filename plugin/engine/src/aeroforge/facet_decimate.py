"""Vertex decimation of faceted outlines (SPEC §6, U3b): flattest corner first.

A coarse part keeps a subset of the corners of its gene polygon: the corner with the largest
interior angle (the smallest turn, the flattest crease) goes first, until every interior corner
turns by at least the smallest crease angle and the segment budget is met, as long as the
polygon stays within `cap` of the gene polygon (measured cumulatively against the gene polygon,
not step by step). A corner whose removal would exceed the cap stays; the call reports that it
was blocked. Pure numpy; the unit is whatever the points are in.

The facet code packs a wing section's gene segment count k0 and its coarseness c into one int
(k0 + STEP x round(100 c)); c = 0 is the plain count, so every U3 consumer keeps working."""

from dataclasses import dataclass

import numpy as np

from aeroforge.constants import number

STEP = 10


def pack(k0: int, c: float) -> int:
    """Facet code of k0 gene segments per side at coarseness c (c = 0: k0 itself)."""
    return k0 + STEP * round(100.0 * c) if k0 and c > 0.0 else k0


def unpack(code: int) -> tuple[int, float]:
    """(gene segments per side, coarseness) of a facet code."""
    return code % STEP, (code // STEP) / 100.0


def min_crease_deg(c: float) -> float:
    """Smallest crease between adjacent sheets allowed at coarseness c."""
    return c * number("coarse_min_crease_deg")


def segment_limit(c: float) -> int:
    """Most segments per side a wing section keeps at coarseness c."""
    return round(
        (1.0 - c) * number("coarse_wing_facets_loose") + c * number("coarse_wing_facets_tight")
    )


def turns_deg(pts: np.ndarray) -> np.ndarray:
    """Turn angle [deg] at every interior vertex of a polyline (n, 2)."""
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.abs(np.degrees(np.diff(ang)))


@dataclass(frozen=True)
class Decimated:
    """Result of decimating one polyline: the kept vertex indices, whether the cap blocked a
    wanted removal, and the largest distance of the gene polyline from the kept one."""

    keep: tuple[int, ...]
    blocked: bool
    move: float


def _move(pts: np.ndarray, keep: list[int]) -> float:
    """Largest distance (in y, the polyline is a function of x) between pts and its subset."""
    sub = pts[keep]
    return float(np.abs(pts[:, 1] - np.interp(pts[:, 0], sub[:, 0], sub[:, 1])).max())


def decimate(
    pts: np.ndarray, least_deg: float, max_segments: int, cap: float, min_points: int = 2
) -> Decimated:
    """Subset of the polyline pts (x increasing) whose interior corners all turn by at least
    least_deg and with at most max_segments segments, flattest corner removed first, as far as
    the cap allows; never fewer than min_points vertices."""
    keep = list(range(len(pts)))
    while True:
        turn = turns_deg(pts[keep])
        over = len(keep) - 1 > max_segments
        if len(turn) == 0 or not (over or turn.min() < least_deg):
            return Decimated(tuple(keep), False, _move(pts, keep))
        blocked = Decimated(tuple(keep), True, _move(pts, keep))
        for j in np.argsort(turn):
            if not over and turn[j] >= least_deg:
                return blocked
            trial = keep[: j + 1] + keep[j + 2 :]
            if len(trial) >= min_points and _move(pts, trial) <= cap:
                keep = trial
                break
        else:
            return blocked
