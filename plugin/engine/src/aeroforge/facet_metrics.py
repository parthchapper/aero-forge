"""Size measures of a faceted body read from its flat faces, for consumers that once read the
smooth envelope (style, host ends, host heights). Pure numpy, body frame, metres."""

import numpy as np

from aeroforge.facet_query import _hull_2d, slice_points, slice_section
from aeroforge.models import Polyhedron


def largest_section_half_extents_m(poly: Polyhedron) -> tuple[float, float]:
    """(half width, half height) [m] of the widest and of the tallest station (the polygon's own
    extents: the tail upsweep shifts stations whole, so it does not add to the height)."""
    v = np.asarray(poly.vertices_m)
    layers = [v[v[:, 0] == x] for x in np.unique(v[:, 0])]
    width = max(float(np.ptp(layer[:, 1])) for layer in layers)
    height = max(float(np.ptp(layer[:, 2])) for layer in layers)
    return width / 2.0, height / 2.0


def vertical_extent_m(poly: Polyhedron) -> float:
    """Total height [m] of the solid, tail rise included."""
    return float(np.ptp(np.asarray(poly.vertices_m)[:, 2]))


def end_radius_m(poly: Polyhedron, x_m: float) -> float:
    """Half of the smaller side [m] of the cross-section at x_m (body frame): 0 where the solid
    closes to a point or an edge, the cap's radius where it ends in a flat face."""
    width, height, _, _ = slice_section(poly, x_m)
    return min(width, height) / 2.0


def inradius_on_axis_m(poly: Polyhedron, x_m: float) -> float:
    """Radius [m] of the largest circle centred on the axis (y = 0, the middle of the section's
    height) that lies inside the cross-section at x_m (body frame); 0 where the solid closes to a
    point or an edge. This is what a round firewall plate at x_m can have."""
    pts = slice_points(poly, x_m)
    if len(pts) < 3:
        return 0.0
    hull = _hull_2d(pts)
    if len(hull) < 3:
        return 0.0
    centre = np.array([0.0, (pts[:, 1].max() + pts[:, 1].min()) / 2.0])
    edge = np.roll(hull, -1, axis=0) - hull
    rel = centre - hull
    dist = np.abs(edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]) / np.linalg.norm(edge, axis=1)
    return float(dist.min())
