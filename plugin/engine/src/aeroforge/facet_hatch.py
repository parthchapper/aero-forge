"""Hatch outlines on flat-panel bodies: the roof panel that carries a bay's hatch and the largest
rectangle of the bay footprint that stays on it at the hatch height (SPEC §10). Pure numpy, body
frame, metres."""

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_query import _normal
from aeroforge.models import Polyhedron


def in_convex(poly_xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Mask of pts (n, 2) inside the convex polygon poly_xy (either winding, edges included)."""
    edge = np.roll(poly_xy, -1, axis=0) - poly_xy
    rel = pts[:, None, :] - poly_xy[None, :, :]
    side = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    tol = number("facet_geom_tol_m")
    return np.all(side >= -tol, axis=1) | np.all(side <= tol, axis=1)


def roof_panels(poly: Polyhedron) -> list[tuple[np.ndarray, np.ndarray]]:
    """(outline xy [m, body frame], plane (n_x, n_y, n_z, d)) of the upward-facing faces: the roof
    panels; the plane is n . p = d with n the unit outward normal."""
    v = np.asarray(poly.vertices_m)
    tol = number("facet_geom_tol_m")
    out = []
    for f in poly.faces:
        p = v[list(f)]
        n = _normal(p)
        if n[2] > tol:
            out.append((p[:, :2], np.append(n, float(n @ p[0]))))
    return out


def _clip_below_plane(outline: np.ndarray, plane: np.ndarray, z_m: float) -> np.ndarray:
    """The part of a roof panel's outline where the panel surface is at least z_m high: the
    plane is n . p = d, so its height at (x, y) is (d - n_x x - n_y y) / n_z."""
    nx, ny, nz, d = plane
    out: list[np.ndarray] = []
    margin = (d - nx * outline[:, 0] - ny * outline[:, 1]) / nz - z_m
    for k, (a, b) in enumerate(zip(outline, np.roll(outline, -1, axis=0), strict=True)):
        ma, mb = margin[k], margin[(k + 1) % len(outline)]
        if ma >= 0.0:
            out.append(a)
        if (ma >= 0.0) != (mb >= 0.0):
            out.append(a + (b - a) * ma / (ma - mb))
    return np.array(out).reshape(-1, 2)


def _clip(poly_xy: np.ndarray, axis: int, value: float, keep_above: bool) -> np.ndarray:
    """Convex polygon cut by the line coordinate[axis] = value (Sutherland-Hodgman)."""
    sign = 1.0 if keep_above else -1.0
    out: list[np.ndarray] = []
    for a, b in zip(poly_xy, np.roll(poly_xy, -1, axis=0), strict=True):
        da, db = sign * (a[axis] - value), sign * (b[axis] - value)
        if da >= 0.0:
            out.append(a)
        if (da >= 0.0) != (db >= 0.0):
            out.append(a + (b - a) * da / (da - db))
    return np.array(out).reshape(-1, 2)


def _area_xy(p: np.ndarray) -> float:
    """Area of a convex polygon [m^2] (0 for fewer than 3 corners)."""
    if len(p) < 3:
        return 0.0
    return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))) / 2.0


def _centroid_xy(p: np.ndarray) -> np.ndarray:
    """Area centroid of a convex polygon (the vertex mean when it has no area)."""
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < number("facet_geom_tol_m") ** 2:
        return p.mean(axis=0)
    return np.array(
        [((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]
    ) / (6.0 * area)


def _hatch_on_panel(
    panel: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, bool] | None:
    """(centre, half sizes, scale, corners lie in the panel) of the hatch rectangle on one panel:
    the footprint (lo..hi) cut back to the panel, centred on the cut's centroid, shrunk until its
    corners lie in the panel. None when the panel does not touch the footprint."""
    cut = panel
    for axis in (0, 1):
        cut = _clip(_clip(cut, axis, lo[axis], True), axis, hi[axis], False)
    if _area_xy(cut) <= 0.0:
        return None
    mid = _centroid_xy(cut)
    half = np.minimum((cut.max(axis=0) - cut.min(axis=0)) / 2.0, np.minimum(mid - lo, hi - mid))
    s = rect_scale_in_panel(panel, (float(mid[0]), float(mid[1])), (float(half[0]), float(half[1])))
    sign = np.array([[a, b] for a in (-1, 1) for b in (-1, 1)])
    return mid, half * s, s, bool(in_convex(panel, mid + half * s * sign).all())


def hatch_rect_xy(
    poly: Polyhedron,
    centre_xy: tuple[float, float],
    half_xy: tuple[float, float],
    z_m: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(centre, half sizes) [m, body frame] of the hatch rectangle for a bay footprint: on the roof
    panel (only its part high enough to hold a hatch at z_m, when given) that gives the largest
    hatch with its corners in the panel, else on the panel covering most of the footprint, cut back
    to its edges and shrunk about the cut's centroid. Unchanged without a roof panel."""
    lo = np.asarray(centre_xy) - np.asarray(half_xy)
    hi = np.asarray(centre_xy) + np.asarray(half_xy)
    best, best_key = None, (False, False, 0.0)
    for outline, plane in roof_panels(poly):
        panel = outline if z_m is None else _clip_below_plane(outline, plane, z_m)
        found = _hatch_on_panel(panel, lo, hi) if len(panel) >= 3 else None
        if found is None:
            continue
        mid, half, scale, inside = found
        key = (
            inside and scale >= number("hatch_panel_min_frac"),
            inside,
            float(4.0 * half[0] * half[1]),
        )
        if best is None or key > best_key:
            best, best_key = (mid, half), key
    if best is None:
        return centre_xy, half_xy
    mid, half = best
    return (float(mid[0]), float(mid[1])), (float(half[0]), float(half[1]))


def rect_scale_in_panel(
    panel_xy: np.ndarray, centre_xy: tuple[float, float], half_xy: tuple[float, float]
) -> float:
    """Largest scale in [hatch_panel_floor_frac, 1] of the rectangle (half sizes half_xy about
    centre_xy) whose four corners lie in the panel outline: the hatch ends on the panel's edges.
    Scales under hatch_panel_min_frac are taken only when no larger one fits (a sliver of panel)."""
    sign = np.array([[a, b] for a in (-1, 1) for b in (-1, 1)])
    centre, half = np.asarray(centre_xy), np.asarray(half_xy)

    def fits(s: float) -> bool:
        return bool(in_convex(panel_xy, centre + s * half * sign).all())

    lo = number("hatch_panel_floor_frac")
    if fits(1.0):
        return 1.0
    if not fits(lo):
        return lo
    hi = 1.0
    for _ in range(int(number("outline_search_steps"))):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo * (1.0 - number("hatch_panel_edge_margin_frac"))  # stop just short of the edge
