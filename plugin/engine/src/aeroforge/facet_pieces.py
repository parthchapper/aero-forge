"""Physical pieces of a flat-panel body: neighbouring faces that lie in one plane are one sheet
(the hull triangulation splits a planar quad in two, a vertical side continues across stations).
Pure numpy; every piece is one polygon with at least three corners."""

from dataclasses import dataclass

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_query import face_areas_m2
from aeroforge.models import Polyhedron


class PieceError(ValueError):
    """A group of coplanar faces does not close to one simple outline."""


@dataclass(frozen=True)
class Piece:
    """One sheet: its source faces, outline (vertex indices, outward counter-clockwise), the piece
    across each outline edge, unit normal and area [m^2]."""

    faces: tuple[int, ...]
    loop: tuple[int, ...]
    across: tuple[int, ...]
    normal: np.ndarray
    area_m2: float


def _normals(poly: Polyhedron) -> np.ndarray:
    v = np.asarray(poly.vertices_m)
    out = []
    for face in poly.faces:
        p = v[list(face)]
        n = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def _coplanar(poly: Polyhedron, normals: np.ndarray, i: int, j: int) -> bool:
    """Faces i and j lie in one plane: every corner of j is within tolerance of i's plane."""
    v = np.asarray(poly.vertices_m)
    if normals[i] @ normals[j] <= 0.0:
        return False
    p = v[list(poly.faces[j])] - v[poly.faces[i][0]]
    return bool(np.abs(p @ normals[i]).max() <= number("facet_geom_tol_m"))


def _groups(poly: Polyhedron, normals: np.ndarray) -> list[list[int]]:
    """Faces joined across shared edges while they stay coplanar (union-find), in face order."""
    parent = list(range(len(poly.faces)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner: dict[tuple[int, int], int] = {}
    for k, face in enumerate(poly.faces):
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            other = owner.get((b, a))
            if other is not None and _coplanar(poly, normals, k, other):
                parent[find(k)] = find(other)
            owner[(a, b)] = k
    groups: dict[int, list[int]] = {}
    for k in range(len(poly.faces)):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def _outline(poly: Polyhedron, group: list[int]) -> list[tuple[int, int]]:
    """Directed boundary edges of a face group chained into one loop (edge i ends where edge
    i + 1 starts)."""
    edges = {
        (a, b)
        for k in group
        for a, b in zip(poly.faces[k], poly.faces[k][1:] + poly.faces[k][:1], strict=True)
    }
    boundary = {e for e in edges if (e[1], e[0]) not in edges}
    start: dict[int, tuple[int, int]] = {}
    for e in boundary:
        if e[0] in start:
            raise PieceError("coplanar faces touch at a point: the outline is not simple")
        start[e[0]] = e
    first = min(boundary)
    loop, edge = [first], start[first[1]]
    while edge != first:
        loop.append(edge)
        edge = start[edge[1]]
    if len(loop) != len(boundary):
        raise PieceError("coplanar faces leave a hole or two outlines")
    return loop


def _straight(v: np.ndarray, a: int, b: int, c: int) -> bool:
    """Corner b lies on the segment a-c's line (turn angle below the geometry tolerance)."""
    u, w = v[b] - v[a], v[c] - v[b]
    cross = float(np.linalg.norm(np.cross(u, w)))
    return cross <= number("facet_geom_tol_m") * float(np.linalg.norm(u) + np.linalg.norm(w))


def pieces(poly: Polyhedron) -> list[Piece]:
    """The sheets of the solid. A corner is dropped from a sheet's outline when the outline runs
    straight through it and the same neighbouring sheet lies across both edges."""
    v = np.asarray(poly.vertices_m)
    normals, areas = _normals(poly), face_areas_m2(poly)
    groups = _groups(poly, normals)
    piece_of = {k: i for i, g in enumerate(groups) for k in g}
    face_across: dict[tuple[int, int], int] = {}
    for k, face in enumerate(poly.faces):
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            face_across[(a, b)] = k
    out = []
    for g in groups:
        edges = _outline(poly, g)
        across = [piece_of[face_across[(b, a)]] for a, b in edges]
        loop = [a for a, _ in edges]
        keep = [
            i
            for i in range(len(loop))
            if across[i - 1] != across[i]
            or not _straight(v, loop[i - 1], loop[i], loop[(i + 1) % len(loop)])
        ]
        # the edge leaving a kept corner runs to the next kept corner: its neighbour is shared
        out.append(
            Piece(
                faces=tuple(g),
                loop=tuple(loop[i] for i in keep),
                across=tuple(across[i] for i in keep),
                normal=normals[g[0]],
                area_m2=float(sum(areas[k] for k in g)),
            )
        )
    return out
