"""Flat-panel wings (SPEC §4, §6, U3): the ruled loft between polygon sections as a closed
polyhedron of flat faces, in the world frame of the surface's right / centre instance [m].

Between two sections each polygon edge sweeps a quad. It is flat only when both of its edges are
parallel (same cant, same twist, straight taper); otherwise it is split into two flat triangles
along its shorter diagonal, so every face is planar to facet_geom_tol_m (invariant 11)."""

from functools import lru_cache

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_decimate import min_crease_deg, unpack
from aeroforge.facet_planar import Planar, _normal, planarize
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


def _mergeable(rings: list[np.ndarray], k: int) -> list[tuple[int, ...]]:
    """Every strip quad of rings of k corners (vertex-index tuples, in _strip's corner order)."""
    return [
        (j * k + i, j * k + (i + 1) % k, (j + 1) * k + (i + 1) % k, (j + 1) * k + i)
        for j in range(len(rings) - 1)
        for i in range(k)
    ]


def _part_length(surface: Surface) -> float:
    """Length [m] of a wing instance: the largest extent of its section corners (the span
    along its own direction, or the chord of a stubby one)."""
    pts = np.array([s.le_m for s in surface.sections])
    ext = pts.max(axis=0) - pts.min(axis=0)
    chord = max(s.chord_m for s in surface.sections)
    return float(max(np.linalg.norm(ext), chord))


def _edges_kept(v0: np.ndarray, v1: np.ndarray, k: int) -> bool:
    """No ring edge or strip edge of the moved vertices is under half or over twice its length."""
    n = len(v0)
    pairs = [(i, (i + 1) % k + (i // k) * k) for i in range(n)]  # around a ring
    pairs += [(i, i + k) for i in range(n - k)]  # along a strip
    a, b = np.array(pairs).T
    old = np.linalg.norm(v0[a] - v0[b], axis=1)
    new = np.linalg.norm(v1[a] - v1[b], axis=1)
    return bool(((new >= 0.5 * old) & (new <= 2.0 * old + 1e-12)).all())


def _joints(
    v: np.ndarray, quads: list[tuple[int, ...]], k: int, c: float
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Pairs of quads of consecutive strips at one polygon edge whose planes meet at less than the
    smallest crease of coarseness c (a break nobody can see: made coplanar if the cap allows)."""
    least = np.radians(min_crease_deg(c))
    out = []
    for q1, q2 in zip(quads, quads[k:], strict=False):
        n1, n2 = (_normal(v[list(q)]) for q in (q1, q2))
        if np.arccos(np.clip(n1 @ n2, -1.0, 1.0)) < least:
            out.append((q1, q2))
    return out


@lru_cache(maxsize=256)
def wing_geometry(surface: Surface) -> tuple[Polyhedron, Planar | None]:
    """Closed solid of flat faces of a faceted surface (airfoil_facets > 0): each section's
    polygon is a ring of corners; rings are joined by _strip, the end rings are caps. At facet
    coarseness > 0 the vertices first move (facet_planar, within coarse_move_frac of the largest
    chord) so that as many twisted quads as possible are flat instead of two triangles."""
    n_af = int(number("airfoil_points_per_side"))
    rings = [section_outline(s, surface, n_af)[:-1] for s in surface.sections]
    k = len(rings[0])
    v, planar = np.vstack(rings), None
    _, c = unpack(surface.airfoil_facets)
    if c > 0.0 and len(rings) > 1:
        fixed = np.zeros(v.size, dtype=bool)
        if surface.mirrored:  # the root stays in its plane so the two halves meet flush
            fixed[1 : 3 * k : 3] = True
        part = number("coarse_move_frac") * _part_length(surface)
        local = number("coarse_move_width_frac") * np.repeat(
            [s.chord_m for s in surface.sections], k
        )
        caps = [tuple(range(k)), tuple(range((len(rings) - 1) * k, len(rings) * k))]
        quads = _mergeable(rings, k)
        joints = _joints(v, quads, k, c)
        for shrink in (1.0, 0.5, 0.25):  # a merge that squeezes an edge of a small tip section
            planar = planarize(v, quads, caps, fixed, np.minimum(part, local) * shrink, joints)
            if _edges_kept(v, planar.vertices, k):
                v = planar.vertices
                break
        else:
            planar = None
    faces: list[tuple[int, ...]] = [tuple(reversed(range(k)))]
    for j in range(len(rings) - 1):
        faces += _strip(v, j * k, (j + 1) * k, k)
    faces.append(tuple(range((len(rings) - 1) * k, len(rings) * k)))
    poly = Polyhedron(
        vertices_m=tuple((float(x), float(y), float(z)) for x, y, z in v), faces=tuple(faces)
    )
    if volume_m3(poly) < 0.0:
        poly = poly.model_copy(update={"faces": tuple(tuple(reversed(f)) for f in poly.faces)})
    return poly, planar


def wing_polyhedron(surface: Surface) -> Polyhedron:
    """Closed solid of flat faces of a faceted surface (see wing_geometry)."""
    return wing_geometry(surface)[0]


def is_faceted(surface: Surface) -> bool:
    """The surface has polygon sections."""
    return surface.airfoil_facets > 0
