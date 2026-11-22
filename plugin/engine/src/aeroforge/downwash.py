"""Tier-1 mutual interference of lifting surfaces from horseshoe vortices (SPEC §8 tier 1).

AeroBuildup sums surfaces as if each flew alone, so a tail never sees the wing's downwash; that
left tier 1's neutral point 15 % MAC aft of the VLM's on the conventional. Each lifting surface j
is replaced by one horseshoe of span pi/4 b_j (elliptic loading) and circulation L_j / (rho V b');
every other lifting surface k then sees the area-averaged vertical velocity at its quarter-chord
stations as a change of angle of attack.
"""

from itertools import pairwise

import numpy as np

from aeroforge.airplane import aero_sections, mac_m, quarter_mac_x_m
from aeroforge.constants import number
from aeroforge.models import Design, Surface
from aeroforge.style import wing_surfaces

X_HAT = np.array([1.0, 0.0, 0.0])


def _segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, core_m: float) -> np.ndarray:
    """Velocity per unit circulation at points_m p from the vortex segment a → b (core-smoothed)."""
    r1, r2, r0 = p - a, p - b, b - a
    cross = np.cross(r1, r2)
    denom = (cross**2).sum(axis=1) + (core_m * np.linalg.norm(r0)) ** 2
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    dot = r1 @ r0 / n1 - r2 @ r0 / n2
    return cross * (dot / (4 * np.pi * denom))[:, None]


def _trailing(p: np.ndarray, start: np.ndarray, core_m: float) -> np.ndarray:
    """Velocity per unit circulation from a vortex leaving start along +x to infinity."""
    r = p - start
    cross = np.cross(X_HAT, r)
    rho2 = (cross**2).sum(axis=1) + core_m**2
    cos = r[:, 0] / np.linalg.norm(r, axis=1)
    return cross * ((1.0 + cos) / (4 * np.pi * rho2))[:, None]


def _bound(s: Surface) -> tuple[np.ndarray, float]:
    """Centre of the bound vortex (quarter-MAC x, mean quarter-chord z) and its half span [m]."""
    secs = aero_sections(s)
    z = np.mean([c.le_m[2] for c in secs])
    half = np.pi / 4 * max(abs(c.le_m[1]) for c in secs)
    return np.array([quarter_mac_x_m(s), 0.0, z]), half


def horseshoe_w(s: Surface, points_m: np.ndarray) -> np.ndarray:
    """Vertical velocity per unit circulation [1/m] at points_m from the horseshoe of s."""
    centre, half = _bound(s)
    core = number("downwash_core_mac_frac") * mac_m(s)
    offset = np.array([0.0, half, 0.0])
    left, right = centre - offset, centre + offset
    v = _segment(points_m, left, right, core) + _trailing(points_m, right, core)
    v -= _trailing(points_m, left, core)
    return v[:, 2]


def _stations(s: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Quarter-chord points_m of the right/centre instance, downwash_stations_per_panel strips per
    aero panel at their midpoints, and the strip areas."""
    n = int(number("downwash_stations_per_panel"))
    u = (np.arange(n) + 0.5) / n
    pts, areas = [], []
    for a, b in pairwise(aero_sections(s)):
        le = np.asarray(a.le_m) + u[:, None] * (np.asarray(b.le_m) - np.asarray(a.le_m))
        chord = a.chord_m + u * (b.chord_m - a.chord_m)
        pts.append(le + np.outer(0.25 * chord, X_HAT))
        width = np.hypot(b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]) / n
        areas.append(width * chord)
    return np.vstack(pts), np.concatenate(areas)


def influence(d: Design) -> np.ndarray:
    """C[j, k]: area-mean vertical velocity on surface k per unit circulation of surface j [1/m];
    zero rows and columns for fins and on the diagonal."""
    n = len(d.surfaces)
    lifting = {s.name for s in wing_surfaces(d)}
    c = np.zeros((n, n))
    for k, sk in enumerate(d.surfaces):
        if sk.name not in lifting:
            continue
        pts, areas = _stations(sk)
        for j, sj in enumerate(d.surfaces):
            if j != k and sj.name in lifting:
                c[j, k] = float((horseshoe_w(sj, pts) * areas).sum() / areas.sum())
    return c


def bound_span_m(s: Surface) -> float:
    """Horseshoe span pi/4 b of a lifting surface [m]."""
    return 2 * _bound(s)[1]
