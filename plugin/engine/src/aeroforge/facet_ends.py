"""Closing the ends of a faceted body with flat faces (pyramid, chisel or knife edge, cap)."""

import numpy as np

from aeroforge.constants import number


class FacetError(ValueError):
    """A faceted body cannot be built from the request (impossible plan or non-convex cell)."""


def _supports(plane: np.ndarray, others: np.ndarray) -> bool:
    """All others lie on one side of the plane through the three points (a hull facet)."""
    n = np.cross(plane[1] - plane[0], plane[2] - plane[0])
    length = float(np.linalg.norm(n))
    if length == 0.0:
        return False
    d = (others - plane[0]) @ (n / length)
    tol = number("facet_geom_tol_m")
    return bool(np.all(d <= tol) or np.all(d >= -tol))


def end_faces(ring: np.ndarray, apex: np.ndarray) -> list[tuple[int, ...]]:
    """Flat faces closing ring (K, 3) to apex (1 or 2 points); ring indices 0..K-1, apex after.

    One apex: a pyramid. Two (an edge A-B): each ring edge joins the apex end that makes a hull
    facet, or both when parallel to the apex edge (a trapezoid); a triangle with both ends fills
    each vertex where the two fans meet.
    """
    k = len(ring)
    if len(apex) == 1:
        return [(j, (j + 1) % k, k) for j in range(k)]
    pts, tol = np.vstack([ring, apex]), number("facet_geom_tol_m")
    axis = apex[1] - apex[0]
    kinds: list[str] = []
    faces: list[tuple[int, ...]] = []
    for a in range(k):
        b = (a + 1) % k
        edge = pts[b] - pts[a]
        if np.linalg.norm(np.cross(edge, axis)) <= tol * np.linalg.norm(axis):
            kinds.append("Q")
            faces.append((a, b, k + 1, k) if edge @ axis > 0 else (a, b, k, k + 1))
            continue
        for end, kind in ((k, "A"), (k + 1, "B")):
            others = [i for i in range(k + 2) if i not in (a, b, end)]
            if _supports(pts[[a, b, end]], pts[others]):
                kinds.append(kind)
                faces.append((a, b, end))
                break
        else:
            raise FacetError("no hull facet closes a ring edge to the apex edge")
    faces += [(j, k, k + 1) for j in range(k) if {kinds[j - 1], kinds[j]} == {"A", "B"}]
    return faces


def apex_points(style: str, ring: np.ndarray, x_m: float) -> np.ndarray:
    """Apex point(s) at x_m closing ring (K, 3), on the height of the ring's middle: a point, a
    horizontal edge (chisel) or a vertical edge (knife) sized from the ring."""
    z_mid = float(ring[:, 2].max() + ring[:, 2].min()) / 2.0
    if style == "point":
        return np.array([[x_m, 0.0, z_mid]])
    if style == "chisel":
        w = number("facet_chisel_width_frac") * float(np.abs(ring[:, 1]).max())
        return np.array([[x_m, w, z_mid], [x_m, -w, z_mid]])
    h = number("facet_knife_height_frac") * float(ring[:, 2].max() - ring[:, 2].min()) / 2.0
    return np.array([[x_m, 0.0, z_mid + h], [x_m, 0.0, z_mid - h]])


def end_panel_count(polygon: np.ndarray, style: str) -> int:
    """Flat faces closing one end of a body whose end polygon is `polygon` (y, z)."""
    if style in ("blunt", "flat"):
        return 1
    ring = np.column_stack([np.ones(len(polygon)), polygon])
    return len(end_faces(ring, apex_points(style, ring, 0.0)))
