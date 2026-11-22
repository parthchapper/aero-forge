"""NACA 4-digit section outlines and thickness through aerosandbox."""

from functools import lru_cache

import aerosandbox as asb
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates


@lru_cache(maxsize=4096)
def naca_outline(
    camber_frac: float, camber_loc_frac: float, tc: float, n_per_side: int
) -> np.ndarray:
    """Unit-chord outline [x/c, y/c], upper trailing edge → leading edge → lower trailing edge."""
    coords = get_NACA_coordinates(
        max_camber=camber_frac,
        camber_loc=camber_loc_frac,
        thickness=tc,
        n_points_per_side=n_per_side,
    )
    coords.setflags(write=False)
    return coords


@lru_cache(maxsize=4096)
def local_thickness_frac(
    camber_frac: float, camber_loc_frac: float, tc: float, x_over_c: float, n_per_side: int
) -> float:
    """Section thickness / chord at chord station x_over_c."""
    outline = np.array(naca_outline(camber_frac, camber_loc_frac, tc, n_per_side))
    airfoil = asb.Airfoil(name="section", coordinates=outline)
    return float(np.ravel(airfoil.local_thickness(x_over_c))[0])
