"""Flat-panel wings (SPEC §4, §6, U3): the ruled loft between polygon sections as a closed
polyhedron of flat faces, in the world frame of the surface's right / centre instance [m].

Between two sections each polygon edge sweeps a quad. It is flat only when both of its edges are
parallel (same cant, same twist, straight taper); otherwise it is split into two flat triangles
along its shorter diagonal, so every face is planar to facet_geom_tol_m (invariant 11)."""

from functools import lru_cache

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_query import volume_m3
from aeroforge.models import Polyhedron, Surface
from aeroforge.shapes import section_outline


def _flat(p: np.ndarray) -> bool:
    """The four corners p (4, 3) lie in one plane, within facet_geom_tol_m."""
    n = np.cross(p[1] - p[0], p[2] - p[0])
    length = float(np.linalg.norm(n))
    if length == 0.0:
        return False
    return abs(float((p[3] - p[0]) @ n)) / length <= number("facet_geom_tol_m")


def _strip(v: np.ndarray, a: int, b: int, k: int) -> list[tuple[int, ...]]:
    """Faces between the rings of k corners starting at vertex a and b: a quad per polygon edge,
    two triangles (shorter diagonal) where the quad is not flat."""
    out: list[tuple[int, ...]] = []
    for i in range(k):
        j = (i + 1) % k
        quad = (a + i, a + j, b + j, b + i)
        if _flat(v[list(quad)]):
            out.append(quad)
        elif np.linalg.norm(v[quad[0]] - v[quad[2]]) <= np.linalg.norm(v[quad[1]] - v[quad[3]]):
            out += [quad[:3], (quad[0], quad[2], quad[3])]
        else:
            out += [(quad[0], quad[1], quad[3]), quad[1:]]
    return out


@lru_cache(maxsize=256)
def wing_polyhedron(surface: Surface) -> Polyhedron:
    """Closed solid of flat faces of a faceted surface (airfoil_facets > 0): each section's
    polygon is a ring of 2 facets corners; rings are joined by _strip, the end rings are caps."""
    n_af = int(number("airfoil_points_per_side"))
    rings = [section_outline(s, surface, n_af)[:-1] for s in surface.sections]
    k = len(rings[0])
    v = np.vstack(rings)
    faces: list[tuple[int, ...]] = [tuple(reversed(range(k)))]
    for j in range(len(rings) - 1):
        faces += _strip(v, j * k, (j + 1) * k, k)
    faces.append(tuple(range((len(rings) - 1) * k, len(rings) * k)))
    poly = Polyhedron(
        vertices_m=tuple((float(x), float(y), float(z)) for x, y, z in v), faces=tuple(faces)
    )
    if volume_m3(poly) >= 0.0:
        return poly
    return poly.model_copy(update={"faces": tuple(tuple(reversed(f)) for f in poly.faces)})


def is_faceted(surface: Surface) -> bool:
    """The surface has polygon sections."""
    return surface.airfoil_facets > 0
