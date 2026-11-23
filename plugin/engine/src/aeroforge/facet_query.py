"""Queries on a faceted body's polyhedron: containment, box fit, planarity, closure, samples
(SPEC §5 invariants 5, 11, 12). Pure numpy, body frame, metres."""

from functools import cache

import numpy as np

from aeroforge.constants import number
from aeroforge.models import Polyhedron


def _normal(points: np.ndarray) -> np.ndarray:
    """Unit Newell normal of a polygon loop (outward for the loop order of a polyhedron face)."""
    n = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    return n / np.linalg.norm(n)


@cache
def _cells(poly: Polyhedron) -> tuple[tuple[float, float, np.ndarray, np.ndarray], ...]:
    """Convex cells between consecutive x layers: (x_lo, x_hi, outward normals, offsets d with
    the inside n . p <= d) of the faces that span the two layers. Cap faces bound a layer only."""
    v = np.asarray(poly.vertices_m)
    tol = number("facet_geom_tol_m")
    groups: dict[tuple[float, float], list[tuple[np.ndarray, float]]] = {}
    for face in poly.faces:
        p = v[list(face)]
        lo, hi = float(p[:, 0].min()), float(p[:, 0].max())
        if hi - lo > tol:
            n = _normal(p)
            groups.setdefault((lo, hi), []).append((n, float(n @ p[0])))
    return tuple(
        (lo, hi, np.array([n for n, _ in g]), np.array([d for _, d in g]))
        for (lo, hi), g in sorted(groups.items())
    )


def inside_faceted(poly: Polyhedron, points: np.ndarray, tol_m: float = 1e-9) -> np.ndarray:
    """Mask of points (body frame) inside the solid: inside every flat face of some convex cell."""
    out = np.zeros(len(points), dtype=bool)
    for lo, hi, normals, offsets in _cells(poly):
        in_x = (points[:, 0] >= lo - tol_m) & (points[:, 0] <= hi + tol_m)
        out |= in_x & np.all(points @ normals.T - offsets <= tol_m, axis=1)
    return out


def fit_scale(poly: Polyhedron, corners: np.ndarray) -> float:
    """Smallest even scale k of the cross-section (y and z, x fixed) that holds every corner (body
    frame): inside a cell, a corner needs k >= (n_yz . p_yz) / (d - n_x x) for each face with
    n_yz . p_yz > 0. inf when a corner lies outside every cell's x range or on a closing apex."""
    tol = number("facet_geom_tol_m")
    best = np.full(len(corners), np.inf)
    for lo, hi, normals, offsets in _cells(poly):
        in_x = (corners[:, 0] >= lo - tol) & (corners[:, 0] <= hi + tol)
        a = corners[:, 1:] @ normals[:, 1:].T
        b = offsets[None, :] - corners[:, :1] * normals[None, :, 0]
        ok = b > tol
        need = np.where(a > tol, np.where(ok, a / np.where(ok, b, 1.0), np.inf), 0.0)
        best = np.minimum(best, np.where(in_x, need.max(axis=1), np.inf))
    return float(best.max())


def axis_half_height_m(poly: Polyhedron, x_lo_m: float, x_hi_m: float) -> float:
    """Smallest half-height [m] of the solid on its symmetry plane y = 0 over x_lo_m..x_hi_m (body
    frame). The band is linear between stations, so its smallest value is at the chord ends or a
    station; at a station the cell on either side gives the same band."""
    tol = number("facet_geom_tol_m")
    cells = _cells(poly)
    inner = [lo for lo, _, _, _ in cells if x_lo_m < lo < x_hi_m]
    best = np.inf
    for x in (x_lo_m, *inner, x_hi_m):
        band = 0.0
        for lo, hi, normals, offsets in cells:
            if not lo - tol <= x <= hi + tol:
                continue
            room = offsets - normals[:, 0] * x
            high, low = normals[:, 2] > tol, normals[:, 2] < -tol
            up, down = room[high] / normals[high, 2], room[low] / normals[low, 2]
            if len(up) and len(down):
                band = max(band, float(up.min() - down.max()) / 2.0)
        best = min(best, band)
    return float(best)


def planarity_m(poly: Polyhedron) -> float:
    """Largest distance [m] of any face vertex from its face's plane (SPEC §5 invariant 11)."""
    v = np.asarray(poly.vertices_m)
    worst = 0.0
    for face in poly.faces:
        p = v[list(face)]
        worst = max(worst, float(np.abs((p - p[0]) @ _normal(p)).max()))
    return worst


def is_closed(poly: Polyhedron) -> bool:
    """Every edge shared by exactly two faces and V - E + F = 2 (SPEC §5 invariant 12)."""
    edges: dict[tuple[int, int], int] = {}
    for face in poly.faces:
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    v = len(poly.vertices_m)
    return all(n == 2 for n in edges.values()) and v - len(edges) + len(poly.faces) == 2


def _split_long_edges(tris: np.ndarray, spacing_m: float) -> np.ndarray:
    """Triangles (n, 3, 3) bisected across their longest edge until none exceeds sqrt(2)
    spacing_m (a right isosceles leaf of legs spacing_m: as dense as a grid of that spacing).
    Skinny faces get few leaves: the count follows area / spacing^2, not edge length."""
    done: list[np.ndarray] = []
    while len(tris):
        edges = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=2)  # edge i: i -> i + 1
        longest = edges.argmax(axis=1)
        big = edges.max(axis=1) > np.sqrt(2.0) * spacing_m
        done.append(tris[~big])
        tris, longest = tris[big], longest[big]
        rows = np.arange(len(tris))
        v0 = tris[rows, longest]
        v1 = tris[rows, (longest + 1) % 3]
        v2 = tris[rows, (longest + 2) % 3]
        mid = (v0 + v1) / 2.0
        tris = np.concatenate([np.stack([v0, mid, v2], axis=1), np.stack([mid, v1, v2], axis=1)])
    return np.concatenate(done) if done else np.zeros((0, 3, 3))


def sample_points(poly: Polyhedron, spacing_m: float) -> np.ndarray:
    """Points covering every face, vertices and edges included: no point of a face is farther
    than spacing_m from one (faces fan-triangulated, then bisected; body frame)."""
    v = np.asarray(poly.vertices_m)
    tris = np.array(
        [[v[f[0]], v[f[i]], v[f[i + 1]]] for f in poly.faces for i in range(1, len(f) - 1)]
    )
    pts = np.vstack([v, _split_long_edges(tris, spacing_m).reshape(-1, 3)])
    # fan triangulation differs between a face and its mirror image, so keep the right half and
    # mirror it: the sample set is then exactly symmetric about y = 0 (SPEC §5 invariant 1)
    right = pts[pts[:, 1] > 0.0]
    out = np.vstack([pts[pts[:, 1] == 0.0], right, right * np.array([1.0, -1.0, 1.0])])
    return out


def face_projection(poly: Polyhedron, columns: tuple[int, int]) -> np.ndarray:
    """Every face projected onto two body-frame coordinates, shape (F, max corners, 2); shorter
    faces repeat their last corner."""
    v = np.asarray(poly.vertices_m)[:, list(columns)]
    width = max(len(f) for f in poly.faces)
    out = np.empty((len(poly.faces), width, 2))
    for i, face in enumerate(poly.faces):
        p = v[list(face)]
        out[i] = np.vstack([p, np.repeat(p[-1:], width - len(p), axis=0)])
    return out
