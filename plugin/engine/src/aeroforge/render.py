"""Numpy z-buffer rasteriser: orthographic views, Lambert shading, depth-edge outline -> PNG."""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from aeroforge.constants import constant, number, numbers
from aeroforge.mesh import PartMesh

FRAGMENT_BUDGET = 1_000_000  # candidate pixels per vectorised chunk (memory bound, not a tunable)


class RenderError(ValueError):
    """Unknown view or empty scene."""


@dataclass(frozen=True)
class Scene:
    """Flattened triangles: vertices, faces, vertex normals, face colours and part ids."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    colours: np.ndarray
    part_ids: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    """v / |v| along the last axis (zero vectors stay zero)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def camera(view: str) -> np.ndarray:
    """Rows (right, up, towards-camera) of an orthographic view."""
    views = constant("constants", "render_views")
    if not isinstance(views, dict) or view not in views:
        raise RenderError(f"unknown view {view!r}")
    d = _unit(np.array(views[view]["dir"], dtype=float))
    right = _unit(np.cross(np.array(views[view]["up"], dtype=float), d))
    return np.stack([right, np.cross(d, right), d])


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (smooth within each CAD face)."""
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(acc, faces[:, k], face_n)
    return _unit(acc)


def _palette() -> dict[str, list[float]]:
    """RGB colours by part kind plus background and outline."""
    palette = constant("constants", "render_colours")
    if not isinstance(palette, dict):
        raise RenderError("render_colours must be a mapping")
    return palette


def flatten(meshes: list[PartMesh]) -> Scene:
    """Concatenate part meshes with per-face colour by part kind."""
    if not meshes:
        raise RenderError("nothing to render")
    palette = _palette()
    verts, faces, normals, colours, ids = [], [], [], [], []
    offset = 0
    for i, m in enumerate(meshes):
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        normals.append(_vertex_normals(m.vertices, m.faces))
        colours.append(np.tile(np.array(palette[m.kind], dtype=float), (len(m.faces), 1)))
        ids.append(np.full(len(m.faces), i + 1))
        offset += len(m.vertices)
    return Scene(
        np.vstack(verts),
        np.vstack(faces),
        np.vstack(normals),
        np.vstack(colours),
        np.concatenate(ids),
    )


def _fragments(tx: np.ndarray, ty: np.ndarray, width: int) -> tuple[np.ndarray, ...]:
    """(pixel index, face index, barycentric weights) of pixel centres inside each triangle."""
    x0 = np.clip(np.ceil(tx.min(1) - 0.5), 0, width).astype(np.int64)
    x1 = np.clip(np.floor(tx.max(1) - 0.5), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.ceil(ty.min(1) - 0.5), 0, width).astype(np.int64)
    y1 = np.clip(np.floor(ty.max(1) - 0.5), -1, width - 1).astype(np.int64)
    area2 = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (
        ty[:, 1] - ty[:, 0]
    )
    size = np.maximum(x1 - x0 + 1, y1 - y0 + 1)
    live = (size > 0) & (np.abs(area2) > 1e-12)
    out: list[list[np.ndarray]] = [[], [], []]
    lower = 0
    block = 1
    scan_min = int(number("render_scan_min_px"))
    big = np.flatnonzero(live & (size > scan_min))
    rows = y1[big] - y0[big] + 1
    for group in np.split(big, np.flatnonzero(np.diff(np.cumsum(rows) // FRAGMENT_BUDGET)) + 1):
        if len(group):
            for part, arr in zip(out, _scan(group, tx, ty, area2, x0, x1, y0, width), strict=True):
                part.append(arr)
    while lower < scan_min:
        sel = np.flatnonzero(live & (size > lower) & (size <= block))
        step = max(1, FRAGMENT_BUDGET // (block * block))
        for start in range(0, len(sel), step):
            for part, arr in zip(
                out,
                _chunk(sel[start : start + step], block, tx, ty, area2, x0, x1, y0, y1, width),
                strict=True,
            ):
                part.append(arr)
        lower, block = block, block * 2
    if not out[0]:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    return np.concatenate(out[0]), np.concatenate(out[1]), np.concatenate(out[2])


def _chunk(
    idx: np.ndarray,
    block: int,
    tx: np.ndarray,
    ty: np.ndarray,
    area2: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    width: int,
) -> tuple[np.ndarray, ...]:
    """Fragments of triangles idx whose pixel bounding boxes fit in block x block."""
    offs = np.arange(block)
    px = x0[idx, None, None] + offs[None, None, :]
    py = y0[idx, None, None] + offs[None, :, None]
    cx, cy = px + 0.5, py + 0.5
    ax, ay = tx[idx][:, :, None, None], ty[idx][:, :, None, None]
    a2 = area2[idx][:, None, None]
    w0 = ((ax[:, 1] - cx) * (ay[:, 2] - cy) - (ax[:, 2] - cx) * (ay[:, 1] - cy)) / a2
    w1 = ((ax[:, 2] - cx) * (ay[:, 0] - cy) - (ax[:, 0] - cx) * (ay[:, 2] - cy)) / a2
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    inside &= (px <= x1[idx, None, None]) & (py <= y1[idx, None, None])
    t, j, i = np.nonzero(inside)
    pix = py[t, j, 0] * width + px[t, 0, i]
    bary = np.stack([w0[t, j, i], w1[t, j, i], w2[t, j, i]], axis=1)
    return pix, idx[t], bary


def _bary(
    t: np.ndarray, px: np.ndarray, py: np.ndarray, tx: np.ndarray, ty: np.ndarray, a2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(inside mask, barycentric weights) of pixel centres (px, py) in triangles t."""
    cx, cy = px + 0.5, py + 0.5
    ax, ay = tx[t], ty[t]
    w0 = ((ax[:, 1] - cx) * (ay[:, 2] - cy) - (ax[:, 2] - cx) * (ay[:, 1] - cy)) / a2[t]
    w1 = ((ax[:, 2] - cx) * (ay[:, 0] - cy) - (ax[:, 0] - cx) * (ay[:, 2] - cy)) / a2[t]
    w2 = 1.0 - w0 - w1
    return (w0 >= 0) & (w1 >= 0) & (w2 >= 0), np.stack([w0, w1, w2], axis=1)


def _scan(
    idx: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    area2: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    width: int,
) -> tuple[np.ndarray, ...]:
    """Fragments of large triangles by scanline: each row tests only the pixels between its edge
    crossings (+-1 px), with the same inside test as _chunk; long thin facets (tubes, ruled
    panels) would otherwise test their whole bounding box."""
    rows = np.maximum(np.floor(ty[idx].max(1) - 0.5).clip(-1, width - 1) - y0[idx] + 1, 0)
    rows = rows.astype(np.int64)
    t = np.repeat(idx, rows)
    y = y0[t] + np.arange(len(t)) - np.repeat(np.cumsum(rows) - rows, rows)
    cy = (y + 0.5)[:, None]
    ax, ay = tx[t], ty[t]
    bx, by = np.roll(ax, -1, axis=1), np.roll(ay, -1, axis=1)
    crosses = ((ay - cy) * (by - cy) <= 0) & (ay != by)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = ax + (cy - ay) * (bx - ax) / (by - ay)
    lo = np.where(crosses, xs, np.inf).min(axis=1)
    hi = np.where(crosses, xs, -np.inf).max(axis=1)
    ok = np.isfinite(lo) & np.isfinite(hi)
    xa = np.maximum(np.ceil(np.where(ok, lo, 0.0) - 0.5) - 1, x0[t]).astype(np.int64)
    xb = np.minimum(np.floor(np.where(ok, hi, -1.0) - 0.5) + 1, x1[t]).astype(np.int64)
    n = np.where(ok, np.maximum(xb - xa + 1, 0), 0)
    tt = np.repeat(t, n)
    px = np.repeat(xa, n) + np.arange(len(tt)) - np.repeat(np.cumsum(n) - n, n)
    py = np.repeat(y, n)
    inside, bary = _bary(tt, px, py, tx, ty, area2)
    return py[inside] * width + px[inside], tt[inside], bary[inside]


def _outline(
    depth_m: np.ndarray, ids: np.ndarray, threshold_m: float, thickness_px: int
) -> np.ndarray:
    """Mask of pixels on a silhouette, part boundary or depth jump, dilated to thickness_px."""
    edge = np.zeros(ids.shape, dtype=bool)
    for axis in (0, 1):
        diff_id = np.diff(ids, axis=axis) != 0
        diff_z = np.abs(np.diff(np.where(ids > 0, depth_m, 0.0), axis=axis)) > threshold_m
        jump = diff_id | diff_z
        lead = [slice(None)] * 2
        lead[axis] = slice(0, -1)
        edge[tuple(lead)] |= jump
    for _ in range(thickness_px - 1):
        grown = edge.copy()
        grown[1:] |= edge[:-1]
        grown[:, 1:] |= edge[:, :-1]
        edge = grown
    return edge


def render_rgb(meshes: list[PartMesh], view: str, px: int) -> np.ndarray:
    """Shaded, outlined uint8 image (px, px, 3) of the parts from a named view."""
    ss = int(number("render_supersample"))
    width = px * ss
    scene = flatten(meshes)
    cam = camera(view)
    proj = scene.vertices @ cam.T
    lo, hi = proj.min(0), proj.max(0)
    scale = (1 - 2 * number("render_margin_frac")) * width / max(hi[0] - lo[0], hi[1] - lo[1])
    centre = (lo + hi) / 2
    sx = (proj[:, 0] - centre[0]) * scale + width / 2
    sy = width / 2 - (proj[:, 1] - centre[1]) * scale
    pix, face, bary = _fragments(sx[scene.faces], sy[scene.faces], width)
    depth = np.einsum("ij,ij->i", bary, proj[scene.faces[face], 2])
    order = np.lexsort((-depth, pix))
    first = np.concatenate([[True], pix[order][1:] != pix[order][:-1]])
    keep = order[first]
    normal = _unit(
        np.einsum("ij,ijk->ik", bary[keep], scene.normals[scene.faces[face[keep]]]) @ cam.T
    )
    normal[normal[:, 2] < 0] *= -1.0
    light = _unit(np.array(numbers("render_light_camera")))
    shade = number("render_ambient") + number("render_diffuse") * np.clip(normal @ light, 0.0, None)
    palette = _palette()
    rgb = np.tile(np.array(palette["background"], dtype=float), (width * width, 1))
    rgb[pix[keep]] = np.clip(scene.colours[face[keep]] * shade[:, None], 0.0, 1.0)
    zbuf = np.zeros(width * width)
    ids = np.zeros(width * width, dtype=np.int64)
    zbuf[pix[keep]] = depth[keep]
    ids[pix[keep]] = scene.part_ids[face[keep]]
    threshold = number("render_outline_depth_frac") * float(np.max(hi - lo))
    edge = _outline(zbuf.reshape(width, width), ids.reshape(width, width), threshold, ss)
    img = rgb.reshape(width, width, 3)
    img[edge] = np.array(palette["outline"], dtype=float)
    small = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.round(small * 255).astype(np.uint8)


def render_png(meshes: list[PartMesh], view: str, px: int) -> bytes:
    """PNG bytes of render_rgb."""
    buffer = io.BytesIO()
    Image.fromarray(render_rgb(meshes, view, px)).save(buffer, format="PNG")
    return buffer.getvalue()
