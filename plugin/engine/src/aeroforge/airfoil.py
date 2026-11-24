"""NACA 4-digit section outlines and thickness through aerosandbox."""

from functools import lru_cache

import aerosandbox as asb
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates

from aeroforge.constants import number


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
def facet_outline(
    camber_frac: float, camber_loc_frac: float, tc: float, n_per_side: int, facets: int
) -> np.ndarray:
    """Unit-chord polygon [x/c, y/c] of `facets` flat segments per side, vertices on the NACA
    outline at x/c = (i / facets)^airfoil_facet_x_power, one sharp vertex at each end; same order
    as naca_outline (upper trailing edge -> leading edge -> lower trailing edge, the trailing-edge
    vertex repeated at both ends), 2 facets + 1 points."""
    xy = naca_outline(camber_frac, camber_loc_frac, tc, n_per_side)
    split = len(xy) // 2
    upper, lower = xy[: split + 1][::-1], xy[split:]
    xs = (np.arange(facets + 1) / facets) ** number("airfoil_facet_x_power")
    yu = np.interp(xs, upper[:, 0], upper[:, 1])
    yl = np.interp(xs, lower[:, 0], lower[:, 1])
    if camber_frac == 0.0:  # a symmetric section is exactly symmetric (centreline fins mirror)
        yl = -yu
    yu[[0, -1]] = yl[[0, -1]] = (yu[[0, -1]] + yl[[0, -1]]) / 2.0
    out = np.vstack([np.column_stack([xs, yu])[::-1], np.column_stack([xs, yl])[1:]])
    out.setflags(write=False)
    return out


def wing_outline(
    camber_frac: float, camber_loc_frac: float, tc: float, n_per_side: int, facets: int = 0
) -> np.ndarray:
    """The wing section outline every consumer reads: the smooth NACA outline, or (facets > 0) its
    polygon (one outline, so CAD, thickness, spar room, box room and containment agree)."""
    if facets:  # one canonical polygon whatever the caller's sampling density
        return facet_outline(
            camber_frac, camber_loc_frac, tc, int(number("airfoil_points_per_side")), facets
        )
    return naca_outline(camber_frac, camber_loc_frac, tc, n_per_side)


def corner_turn_sq_sum(outline: np.ndarray) -> float:
    """Sum over the polygon's corners, trailing edge excluded (a smooth section has that corner
    too), of the squared turn angle [rad^2] of its boundary."""
    pts = outline[:-1]  # the trailing-edge vertex is repeated at the end
    edge = np.roll(pts, -1, axis=0) - pts
    ang = np.arctan2(edge[:, 1], edge[:, 0])
    turn = np.abs((ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi)  # turn at vertex i
    return float((turn[1:] ** 2).sum())


@lru_cache(maxsize=4096)
def local_thickness_frac(
    camber_frac: float,
    camber_loc_frac: float,
    tc: float,
    x_over_c: float,
    n_per_side: int,
    facets: int = 0,
) -> float:
    """Section thickness / chord at chord station x_over_c (polygon when facets > 0)."""
    outline = np.array(wing_outline(camber_frac, camber_loc_frac, tc, n_per_side, facets))
    if facets:
        split = len(outline) // 2
        upper, lower = outline[: split + 1][::-1], outline[split:]
        return float(
            np.interp(x_over_c, upper[:, 0], upper[:, 1])
            - np.interp(x_over_c, lower[:, 0], lower[:, 1])
        )
    airfoil = asb.Airfoil(name="section", coordinates=outline)
    return float(np.ravel(airfoil.local_thickness(x_over_c))[0])
