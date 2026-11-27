"""Aerodynamic stand-in for polygon wing sections (SPEC §8, U3).

NeuralFoil fits a smooth Kulfan section to whatever coordinates it is given, so a polygon's corners
are smoothed away or, on many shapes, drop its confidence below 0.1 (CHANGELOG Decisions U3). A
faceted surface therefore enters AeroBuildup as its smooth parent airfoil, and this module adds the
profile drag of the corners as a separate, assumed increment (constant facet_cd_per_turn_sq)."""

import numpy as np

from aeroforge.airfoil import corner_turn_sq_sum, wing_outline
from aeroforge.constants import number
from aeroforge.models import Design, Surface


def section_cd_increment(s: Surface) -> float:
    """Profile-drag coefficient increment of a faceted surface's section (0 when smooth)."""
    if not s.airfoil_facets:
        return 0.0
    xy = wing_outline(
        s.camber_frac,
        s.camber_loc_frac,
        s.tc,
        int(number("airfoil_points_per_side")),
        s.airfoil_facets,
    )
    return number("facet_cd_per_turn_sq") * corner_turn_sq_sum(xy)


def facet_drag_area_m2(d: Design) -> float:
    """Sum over faceted surfaces of the section Cd increment x the surface's planform area [m^2]."""
    return sum(section_cd_increment(s) * s.area_m2 for s in d.surfaces)


def facet_profile_drag_n(d: Design, speeds_ms: np.ndarray, rho_kg_m3: float) -> np.ndarray:
    """Added profile drag [N] at each speed: q x facet_drag_area_m2."""
    return 0.5 * rho_kg_m3 * np.asarray(speeds_ms, dtype=float) ** 2 * facet_drag_area_m2(d)
