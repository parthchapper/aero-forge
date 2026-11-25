"""Planarising a faceted wing (SPEC §6, U3b): merge the two triangles of a slightly twisted strip.

A ruled strip between two sections that differ in cant or twist has non-planar quads; the loft
splits each into two flat triangles with a crease along the diagonal. Under facet coarseness that
crease is a defect, so the vertices move the least possible (a Gauss-Newton minimum-norm step on
the planarity residuals of every strip quad and the two end caps) until every quad is flat. A quad
whose merge would take any vertex further than `cap_m` is given up one at a time (the least
mergeable first) and stays two triangles: the cap blocks it, and the call reports how many.
Pure numpy; metres."""

from dataclasses import dataclass

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_pieces import pieces
from aeroforge.facet_query import planarity_m
from aeroforge.models import Polyhedron

NEWTON_STEPS = 15
MAX_TRIES = 12  # greedy merge attempts after the all-at-once attempt failed (bounds the cost)
RCOND = 1e-12  # eigenvalues of J J^T under this share of the largest are dependent constraints


@dataclass(frozen=True)
class Planar:
    """Outcome: the moved vertices, the quads (as vertex-index tuples) now flat, the quads the
    cap kept as triangles, the largest vertex move [m], and the neighbouring-quad pairs made
    coplanar (joints)."""

    vertices: np.ndarray
    merged: frozenset[tuple[int, ...]]
    blocked: int
    move_m: float
    joined: int = 0


def _det(p: np.ndarray) -> float:
    """Signed volume x 6 of the tetrahedron p[0..3]: zero when the four points are coplanar."""
    return float(np.linalg.det(np.array([p[1] - p[0], p[2] - p[0], p[3] - p[0]])))


def _height(p: np.ndarray) -> float:
    """Distance of p[3] from the plane of p[0..2] [m]."""
    n = np.cross(p[1] - p[0], p[2] - p[0])
    length = float(np.linalg.norm(n))
    return abs(_det(p)) / length if length > 0.0 else 0.0


def _tetras(quads: list[tuple[int, ...]], caps: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Index 4-tuples whose coplanarity is wanted: every quad, and for each cap ring the first
    three corners with every other corner."""
    out = list(quads)
    for ring in caps:
        out += [(ring[0], ring[1], ring[2], r) for r in ring[3:]]
    return out


def _residual_jacobian(x: np.ndarray, tet: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    """Residuals (planarity determinants) and their Jacobian w.r.t. the flattened vertices."""
    v = x.reshape(-1, 3)
    idx = np.asarray(tet, dtype=int).reshape(-1, 4)
    a, b, c = (v[idx[:, k]] - v[idx[:, 0]] for k in (1, 2, 3))
    g1, g2, g3 = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    res = (a * g1).sum(axis=1)
    jac = np.zeros((len(idx), len(v), 3))
    rows = np.arange(len(idx))
    for k, g in enumerate((-(g1 + g2 + g3), g1, g2, g3)):
        np.add.at(jac, (rows, idx[:, k]), g)
    return res, jac.reshape(len(idx), -1)


def _solve(v: np.ndarray, tet: list[tuple[int, ...]], fixed: np.ndarray) -> np.ndarray | None:
    """Vertices nearest to v (minimum-norm corrections) with every tetra in tet coplanar, over the
    dofs not fixed (boolean per flattened coordinate); None when Newton does not converge. Each
    step is the minimum-norm solution of the linearised constraints through the eigen-
    decomposition of J J^T (a pseudo-inverse: dependent constraints drop out)."""
    x = v.reshape(-1).copy()
    free = ~fixed
    start = None
    for step_no in range(NEWTON_STEPS):
        res, jac = _residual_jacobian(x, tet)
        size = float(np.abs(res).max(initial=0.0))
        if size < number("planar_det_tol"):
            return x.reshape(-1, 3)
        if start is None:
            start = size
        if not np.isfinite(size) or (step_no >= 4 and size > 1e-2 * start):
            return None
        jf = jac[:, free]
        try:
            w, q = np.linalg.eigh(jf @ jf.T)
        except np.linalg.LinAlgError:
            return None
        keep = w > RCOND * w.max()  # truncated pseudo-inverse: dependent constraints drop out
        x[free] += jf.T @ (q[:, keep] @ ((q[:, keep].T @ -res) / w[keep]))
    res, _ = _residual_jacobian(x, tet)
    return x.reshape(-1, 3) if np.abs(res).max(initial=0.0) < number("planar_det_tol") else None


def _normal(p: np.ndarray) -> np.ndarray:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    return n / np.linalg.norm(n)


def _joint_height(v: np.ndarray, q1: tuple[int, ...], q2: tuple[int, ...]) -> float:
    """Largest distance [m] of q2's far corners from q1's plane."""
    a0, a1, _, b0 = q1
    n = _normal(v[[a0, a1, b0]])
    far = [i for i in q2 if i not in q1]
    return float(max(abs((v[i] - v[a0]) @ n) for i in far))


def planarize(
    v: np.ndarray,
    quads: list[tuple[int, ...]],
    caps: list[tuple[int, ...]],
    fixed: np.ndarray,
    cap_m: float | np.ndarray,
    joints: list[tuple[tuple[int, ...], tuple[int, ...]]] = (),
) -> Planar:
    """Move the vertices v (n, 3) so that as many quads as the cap allows are flat, and the pairs
    of neighbouring quads in `joints` (strip j and j + 1, same corner) are coplanar. caps are the
    end rings (kept planar), fixed the boolean mask of flattened coordinates that may not move.
    Each demand is the distance [m] its vertices must travel to the plane; the largest are given
    up first (a quad goes back to two triangles, a joint back to a crease) until the largest
    vertex move is within cap_m (one number, or one per vertex)."""
    groups: list[tuple[float, list[tuple[int, ...]], tuple[tuple[int, ...], ...]]] = [
        (_height(v[list(q)]), [q], (q,)) for q in quads
    ]
    for q1, q2 in joints:
        h = _joint_height(v, q1, q2)
        tets = [(q1[0], q1[1], q1[3], i) for i in q2 if i not in q1]
        groups.append((h, tets, (q1, q2)))
    groups.sort(key=lambda g: g[0])
    while True:
        out = _solve(v, _tetras([t for g in groups for t in g[1]], caps), fixed)
        step = np.linalg.norm(out - v, axis=1) if out is not None else np.full(len(v), np.inf)
        moved = float(step.max())
        flat = frozenset(q for g in groups if len(g[2]) == 1 for q in g[2])
        if out is not None and bool((step <= cap_m).all()):
            return Planar(out, flat, len(quads) - len(flat), moved, len(groups) - len(flat))
        if not groups:
            return Planar(v, frozenset(), len(quads), 0.0, 0)
        gone = groups.pop()  # the group that travels furthest goes back to its triangles / crease
        if len(gone[2]) == 1:  # a quad that goes also takes its joints
            groups = [g for g in groups if len(g[2]) == 1 or gone[2][0] not in g[2]]


def _face_normals(poly: Polyhedron) -> np.ndarray:
    """Unscaled Newell normals of the faces (length = twice the area)."""
    v = np.asarray(poly.vertices_m)
    out = []
    for face in poly.faces:
        p = v[list(face)]
        out.append(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0))
    return np.array(out)


def _faces_kept(old: Polyhedron, new: Polyhedron) -> bool:
    """No face of new has collapsed, flipped or turned far: each keeps at least half its area and
    its normal within 60 deg of the old one's."""
    a, b = _face_normals(old), _face_normals(new)
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if (lb < 0.5 * la).any():
        return False
    return bool(((a * b).sum(axis=1) >= 0.5 * la * lb).all())


@dataclass(frozen=True)
class Merged:
    """Outcome of merge_sheets: how many neighbouring sheet pairs were made coplanar, how many
    the cap kept as creases, and the largest vertex move [m]."""

    merged: int
    blocked: int
    move_m: float


def _basis(v: np.ndarray, loop: tuple[int, ...]) -> tuple[int, int, int]:
    """Three corners of a sheet's vertices spanning a large triangle (the loop may hold collinear
    corners, so the first three can be a bad plane): the first, the one furthest from it, and the
    one furthest from the line through them."""
    pts = v[list(loop)]
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    d = pts[i1] - pts[i0]
    i2 = int(np.argmax(np.linalg.norm(np.cross(pts - pts[i0], d / np.linalg.norm(d)), axis=1)))
    return loop[i0], loop[i1], loop[i2]


def _moved(poly: Polyhedron, v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(poly.vertices_m) - v, axis=1).max())


def _accept(
    poly: Polyhedron, v: np.ndarray, tets: list[tuple[int, ...]], fixed: np.ndarray, cap_m: float
) -> Polyhedron | None:
    """poly with the vertices solved for tets if that stays within cap_m, every face flat and no
    face collapsed or turned over; else None."""
    out = _solve(v, tets, fixed)
    if out is None or float(np.linalg.norm(out - v, axis=1).max()) > cap_m:
        return None
    trial = poly.model_copy(
        update={"vertices_m": tuple((float(x), float(y), float(z)) for x, y, z in out)}
    )
    ok = _faces_kept(poly, trial) and planarity_m(trial) <= number("facet_geom_tol_m") * 0.5
    return trial if ok else None


def merge_sheets(
    poly: Polyhedron, least_deg: float, cap_m: float, keep_x: bool = True
) -> tuple[Polyhedron, Merged]:
    """poly with every pair of neighbouring sheets that meet at less than least_deg made coplanar
    where no vertex has to move more than cap_m, every sheet kept flat. keep_x: vertices keep
    their x, so a body's station rings stay in their planes. The pairs that would move vertices
    furthest are given up first. Nothing else changes: faces and vertex numbers are the same."""
    sheets = pieces(poly)
    v = np.asarray(poly.vertices_m, dtype=float)
    # every vertex of a sheet's faces (an interior vertex of a merged sheet is not on its loop)
    own = [tuple(sorted({i for f in s.faces for i in poly.faces[f]})) for s in sheets]
    basis = [_basis(v, loop) for loop in own]
    base = [(*basis[k], r) for k, loop in enumerate(own) for r in loop if r not in basis[k]]
    cos = np.cos(np.radians(least_deg))
    groups = []
    for i, a in enumerate(sheets):
        for j in a.across:
            b = sheets[j]
            if j <= i or not 0.0 <= a.normal @ b.normal < 1.0 or a.normal @ b.normal < cos:
                continue
            o = basis[i]
            far = [r for r in own[j] if r not in own[i]]
            tets = [(*o, r) for r in far]
            demand = max(abs((v[r] - v[o[0]]) @ _normal(v[list(o)])) for r in far)
            groups.append((demand, tets, ((i, j),)))
    groups.sort(key=lambda g: g[0])
    fixed = np.zeros(v.size, dtype=bool)
    if keep_x:
        fixed[0::3] = True
    if not groups:
        return poly, Merged(0, 0, 0.0)
    best, taken, tets = poly, 0, list(base)
    whole = _accept(poly, v, [*tets, *(t for g in groups for t in g[1])], fixed, cap_m)
    if whole is not None:  # every merge holds together: one solve
        return whole, Merged(len(groups), 0, _moved(whole, v))
    for _, group, _ in groups[:MAX_TRIES]:  # smallest demands first; each joins if the whole holds
        trial = _accept(poly, v, [*tets, *group], fixed, cap_m)
        if trial is not None:
            best, taken, tets = trial, taken + 1, [*tets, *group]
    return best, Merged(taken, len(groups) - taken, _moved(best, v))
