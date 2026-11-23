"""Queries on a faceted body's polyhedron: containment, box fit, planarity, closure, samples
(SPEC §5 invariants 5, 11, 12). Pure numpy, body frame, metres."""

from functools import lru_cache
from itertools import pairwise

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_ends import FacetError
from aeroforge.models import Polyhedron

ASB_AREA_POWER = 1.8717618013591173  # aerosandbox FuselageXSec.xsec_area: area = w h / (s^-p + 1)


def _normal(points: np.ndarray) -> np.ndarray:
    """Unit Newell normal of a polygon loop (outward for the loop order of a polyhedron face)."""
    n = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    return n / np.linalg.norm(n)


CELLS_CACHE_SIZE = 256  # one decode builds about 8 polyhedra: bounded so a long search stays flat


@lru_cache(maxsize=CELLS_CACHE_SIZE)
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


def inside_faceted(poly: Polyhedron, points: np.ndarray, tol_m: float | None = None) -> np.ndarray:
    """Mask of points (body frame) inside the solid: inside every flat face of some convex cell,
    within tol_m [m] (default facet_geom_tol_m)."""
    tol_m = number("facet_geom_tol_m") if tol_m is None else tol_m
    out = np.zeros(len(points), dtype=bool)
    for lo, hi, normals, offsets in _cells(poly):
        in_x = (points[:, 0] >= lo - tol_m) & (points[:, 0] <= hi + tol_m)
        out |= in_x & np.all(points @ normals.T - offsets <= tol_m, axis=1)
    return out


def fit_scale(poly: Polyhedron, corners: np.ndarray) -> float:
    """Smallest even scale k of the cross-section (y and z, x fixed) that holds every corner (body
    frame): inside a cell, a corner needs k >= (n_yz . p_yz) / (d - n_x x) for each face with
    n_yz . p_yz > 0. inf when a corner lies outside every cell's x range or on a closing apex.
    FacetError for no corners or a non-finite corner (a NaN would read as "does not fit")."""
    if corners.size == 0 or not np.all(np.isfinite(corners)):
        raise FacetError("fit_scale needs at least one corner and every coordinate finite")
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


def _hull_2d(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (monotone chain, collinear points dropped). Plain
    loops: for a few dozen points this is several times faster than a Qhull call."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = [tuple(row) for row in pts[order]]

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for c in seq:
            while len(out) >= 2 and (
                (out[-1][0] - out[-2][0]) * (c[1] - out[-2][1])
                - (out[-1][1] - out[-2][1]) * (c[0] - out[-2][0])
                <= 0.0
            ):
                out.pop()
            out.append(c)
        return out[:-1]

    return np.array(half(p) + half(p[::-1]))


def cell_projection(poly: Polyhedron, columns: tuple[int, int]) -> np.ndarray:
    """Every convex cell between consecutive x layers projected onto two body-frame coordinates:
    the hull of its vertices, shape (cells, max corners, 2), counter-clockwise; shorter hulls
    repeat their last corner. The union of the cell shadows is the solid's shadow, with a polygon
    per station gap instead of one per face."""
    v = np.asarray(poly.vertices_m)
    layers = np.unique(v[:, 0])
    hulls = [
        _hull_2d(v[(v[:, 0] >= lo) & (v[:, 0] <= hi)][:, list(columns)])
        for lo, hi in pairwise(layers)
    ]
    width = max(len(h) for h in hulls)
    return np.array([np.vstack([h, np.repeat(h[-1:], width - len(h), axis=0)]) for h in hulls])


def _triangles(poly: Polyhedron) -> tuple[np.ndarray, np.ndarray]:
    """Fan triangles of every face, shape (T, 3, 3), and the face index of each (body frame)."""
    v = np.asarray(poly.vertices_m)
    tris, owner = [], []
    for k, face in enumerate(poly.faces):
        for i in range(1, len(face) - 1):
            tris.append([v[face[0]], v[face[i]], v[face[i + 1]]])
            owner.append(k)
    return np.array(tris), np.array(owner)


def face_areas_m2(poly: Polyhedron) -> np.ndarray:
    """Area [m^2] of every face, in face order."""
    tris, owner = _triangles(poly)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2.0
    return np.bincount(owner, weights=area, minlength=len(poly.faces))


def surface_area_m2(poly: Polyhedron) -> float:
    """Exact wetted area [m^2]: the sum of the flat faces."""
    return float(face_areas_m2(poly).sum())


def area_centroid_m(poly: Polyhedron) -> np.ndarray:
    """(x, y, z) [m] of the centroid of the skin (area-weighted), body frame: z follows the
    tail upsweep."""
    tris, _ = _triangles(poly)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2.0
    return (area[:, None] * tris.mean(axis=1)).sum(axis=0) / area.sum()


def volume_m3(poly: Polyhedron) -> float:
    """Exact enclosed volume [m^3] (divergence theorem over the outward fan triangles)."""
    t, _ = _triangles(poly)
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def _edges(poly: Polyhedron) -> np.ndarray:
    """Unique vertex-index pairs of the polyhedron, shape (E, 2)."""
    pairs = {
        (min(a, b), max(a, b)) for f in poly.faces for a, b in zip(f, f[1:] + f[:1], strict=True)
    }
    return np.array(sorted(pairs))


def slice_points(poly: Polyhedron, x_m: float) -> np.ndarray:
    """(y, z) points [m] where the polyhedron's edges meet the plane x = x_m (body frame); the
    cross-section is their convex hull (sections are convex, SPEC §5)."""
    tol = number("facet_geom_tol_m")
    v = np.asarray(poly.vertices_m)
    x_m = float(np.clip(x_m, v[:, 0].min(), v[:, 0].max()))
    a, b = (v[c] for c in _edges(poly).T)
    flat = np.abs(a[:, 0] - x_m) + np.abs(b[:, 0] - x_m) <= 2.0 * tol
    cross = ((a[:, 0] - x_m) * (b[:, 0] - x_m) <= 0.0) & (np.abs(a[:, 0] - b[:, 0]) > tol)
    dx = np.where(cross, b[:, 0] - a[:, 0], 1.0)
    t = np.clip((x_m - a[:, 0]) / dx, 0.0, 1.0)[:, None]
    hit = (a + t * (b - a))[cross][:, 1:]
    return np.vstack([hit, a[flat][:, 1:], b[flat][:, 1:]])


def slice_section(poly: Polyhedron, x_m: float) -> tuple[float, float, float, float]:
    """(width, height, area, centre z) [m, m, m^2, m] of the cross-section at x = x_m: the
    bounding box of its hull and the hull's area (zeros where the solid closes)."""
    p = slice_points(poly, x_m)
    if len(p) < 3:
        return 0.0, 0.0, 0.0, float(p[:, 1].mean()) if len(p) else 0.0
    mid = p.mean(axis=0)
    ring = p[np.argsort(np.arctan2(p[:, 1] - mid[1], p[:, 0] - mid[0]), kind="stable")]
    area = (
        abs(
            float(
                np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
            )
        )
        / 2.0
    )
    lo, hi = p.min(axis=0), p.max(axis=0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1]), area, float((hi[1] + lo[1]) / 2.0)


def equivalent_exponent(width_m: float, height_m: float, area_m2: float) -> float:
    """Super-ellipse exponent whose asb FuselageXSec area (w h / (s^-1.8717618 + 1)) is area_m2:
    1 is the diamond (half the box), the table's maximum the near-rectangle."""
    top = number("equiv_section_exp_max")
    if area_m2 <= 0.0 or width_m * height_m <= 0.0:
        return 1.0
    fill = width_m * height_m / area_m2 - 1.0
    if fill <= 0.0:
        return top
    return float(np.clip(fill ** (-1.0 / ASB_AREA_POWER), 1.0, top))
