"""SPEC §9 descriptor v2 (U4): 20 numbers in [0, 1] from geometry only, independent of slot order
and encoding: 11 lifting-surface numbers (descriptor_v1 terms on coarser bins) and the 9 body
numbers of body_terms. descriptor_v1 keeps the 0.3 descriptor for the baseline comparison."""

import numpy as np

from aeroforge.body_terms import BODY_NAMES, body_terms
from aeroforge.constants import number
from aeroforge.descriptor_v1 import DescriptorError, lifting_terms
from aeroforge.models import Design

NAMES = (
    "lift_x_1",
    "lift_x_2",
    "lift_x_3",
    "lift_x_4",
    "lift_y_1",
    "lift_y_2",
    "lift_y_3",
    "lift_z_1",
    "lift_z_2",
    "vertical_area",
    "surface_count",
    *BODY_NAMES,
)
DESC_SIZE = len(NAMES)
DESC_VERSION = 2  # history.jsonl rows carry it; 0.3 rows (16 numbers) have none
BODY_TOP_INDEX = NAMES.index("body_top_share")
TOP_CONTRIBUTORS = 3


def desc(d: Design) -> np.ndarray:
    """Lifting-area shares along x (4), |y| (3), z (2); vertical-area fraction; surface count; then
    the nine body numbers (areas.shares, body_terms)."""
    terms = lifting_terms(d, ("desc2_x_centres", "desc2_y_centres", "desc2_z_centres"))
    count = min(len(d.surfaces) / number("desc_surface_count_max"), 1.0)
    out = np.concatenate([terms, [count], body_terms(d)])
    if out.shape != (DESC_SIZE,):
        raise DescriptorError(f"descriptor has {out.shape[0]} numbers, expected {DESC_SIZE}")
    return out


def body_top_share(x: np.ndarray) -> float:
    """The body's share of the top-down shadow, read from a descriptor."""
    return float(x[BODY_TOP_INDEX])


def novelty(x: np.ndarray, others: np.ndarray) -> float:
    """Min Euclidean distance from x to the rows of others (inf when there are none)."""
    if len(others) == 0:
        return float("inf")
    return float(np.sqrt(((others - x) ** 2).sum(axis=1)).min())


def contributions(x: np.ndarray, nearest: np.ndarray) -> list[tuple[str, float]]:
    """The TOP_CONTRIBUTORS descriptor numbers that make up most of the squared distance between x
    and its nearest known design: (name, share of the squared distance), largest first."""
    sq = (np.asarray(x) - np.asarray(nearest)) ** 2
    total = float(sq.sum())
    order = np.argsort(-sq, kind="stable")[:TOP_CONTRIBUTORS]
    return [(NAMES[i], float(sq[i] / total) if total > 0.0 else 0.0) for i in order]


def median_nearest(points: np.ndarray) -> float:
    """Median over rows of the distance to the nearest other row (SPEC §9 d_min default)."""
    diff = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(diff, np.inf)
    return float(np.median(diff.min(axis=1)))
