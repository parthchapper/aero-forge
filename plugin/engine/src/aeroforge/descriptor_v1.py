"""SPEC §9 descriptor v1 (0.3): 16 lifting-surface numbers in [0, 1], independent of slot order and
encoding. Kept for the 0.3 baseline comparison; the search uses descriptor.desc (v2, U4)."""

import numpy as np

from aeroforge.constants import number, numbers
from aeroforge.models import Design
from aeroforge.style import panels

DESC_SIZE_V1 = 16


class DescriptorError(ValueError):
    """A design without lifting area has no descriptor."""


def _strips(d: Design) -> np.ndarray:
    """Spanwise strips of every panel and instance: rows (x, y, z, true area, |cos cant|),
    sorted so later sums do not depend on slot order [m, m2]."""
    k = int(number("desc_strips_per_panel"))
    t = (np.arange(k) + 0.5) / k
    rows = []
    for s in d.surfaces:
        for p in panels(s):
            le_a, te_a, le_b, te_b = p.pts_m
            le = le_a + t[:, None] * (le_b - le_a)
            te = te_a + t[:, None] * (te_b - te_a)
            chord = np.linalg.norm(te - le, axis=1)
            area = p.true_m2 / k * chord / chord.mean()
            cos = np.full(k, abs(np.cos(np.radians(p.cant_deg))))
            block = np.column_stack([(le + te) / 2, area, cos])
            rows.append(block)
            if s.mirrored:
                rows.append(block * (1, -1, 1, 1, 1))
    if not rows:
        raise DescriptorError("design has no surface panels")
    out = np.vstack(rows)
    return out[np.lexsort(out.T[::-1])]


def _soft_histogram(values: np.ndarray, weights: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Weight shares on bin centres, split linearly between the two nearest (ends clamp)."""
    pos = np.interp(values, centres, np.arange(len(centres)))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(centres) - 1)
    frac = pos - lo
    hist = np.zeros(len(centres))
    np.add.at(hist, lo, weights * (1.0 - frac))
    np.add.at(hist, hi, weights * frac)
    return hist / hist.sum()


def _centres(key: str) -> np.ndarray:
    """Bin centres from constants.yaml."""
    return np.array(numbers(key))


def lifting_terms(d: Design, keys: tuple[str, str, str]) -> np.ndarray:
    """Lifting-area shares along x, |y| and z (bin centres from the three constants keys) followed
    by the vertical-area fraction; every number in [0, 1]."""
    st = _strips(d)
    xyz, area, cos = st[:, :3], st[:, 3], st[:, 4]
    lift = area * cos
    if lift.sum() <= 0.0:
        raise DescriptorError("design has no lifting area")
    # Lengths scale with sqrt(S) and are taken about the lifting-area centroid, so a body shift,
    # the overall size and the slot order leave the descriptor unchanged.
    scale = np.sqrt(lift.sum())
    centroid = (lift @ xyz) / lift.sum()
    semi = float(np.abs(xyz[:, 1]).max())
    hx = _soft_histogram((xyz[:, 0] - centroid[0]) / scale, lift, _centres(keys[0]))
    y_frac = np.abs(xyz[:, 1]) / semi if semi > 0.0 else np.zeros(len(xyz))
    hy = _soft_histogram(y_frac, lift, _centres(keys[1]))
    hz = _soft_histogram((xyz[:, 2] - centroid[2]) / scale, lift, _centres(keys[2]))
    vertical = float(area @ np.sqrt(np.clip(1.0 - cos**2, 0.0, 1.0)) / area.sum())
    return np.concatenate([hx, hy, hz, [vertical]])


def desc_v1(d: Design) -> np.ndarray:
    """Lifting-area shares along x (5), |y| (4), z (3); vertical-area fraction; surface,
    joined-tip and body counts, each scaled to [0, 1]."""
    terms = lifting_terms(d, ("desc_x_centres", "desc_y_centres", "desc_z_centres"))
    bodies = sum(2 if b.mirrored else 1 for b in d.bodies)
    counts = [
        len(d.surfaces) / number("desc_surface_count_max"),
        sum(s.joined_to is not None for s in d.surfaces) / number("desc_joined_count_max"),
        bodies / number("desc_body_count_max"),
    ]
    out = np.concatenate([terms, np.clip(counts, 0.0, 1.0)])
    if out.shape != (DESC_SIZE_V1,):
        raise DescriptorError(f"descriptor has {out.shape[0]} numbers, expected {DESC_SIZE_V1}")
    return out
