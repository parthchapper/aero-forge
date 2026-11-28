"""The nine body numbers of descriptor v2 (SPEC §9, U4), each in [0, 1]: body share of the top, side
and front shadows, body-wing blend, fineness, sheet count, crease sharpness, face directions, pods.
Shape numbers read the centre body (the longest); a smooth or absent body has no sheets, creases
or directions (0)."""

import numpy as np

from aeroforge.areas import shares
from aeroforge.coarse_metrics import measure
from aeroforge.constants import number
from aeroforge.facet_metrics import largest_section_half_extents_m
from aeroforge.models import Body, Design

BODY_NAMES = (
    "body_top_share",
    "body_side_share",
    "body_front_share",
    "body_wing_blend",
    "body_fineness",
    "body_panels",
    "body_crease_sharpness",
    "body_face_directions",
    "pods",
)


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _fineness(b: Body) -> float:
    """Length over the diameter of the circle with the area of the widest section's bounding box."""
    if b.facets is not None:
        half_w, half_h = largest_section_half_extents_m(b.facets)
    else:
        half_w, half_h = b.half_width_m, b.half_height_m
    diameter = 2.0 * np.sqrt(half_w * half_h)
    return b.length_m / diameter if diameter > 0.0 else 0.0


def _facet_terms(b: Body) -> tuple[float, float, float]:
    """Sheets, mean crease angle and face directions of a flat-panel body, scaled (zeros for a
    smooth body or one whose sheets cannot be formed)."""
    m = measure(b.name, "body", b.facets) if b.facets is not None else None
    if m is None:
        return 0.0, 0.0, 0.0
    crease = float(m.creases_deg.mean()) if len(m.creases_deg) else 0.0
    return (
        _unit(m.sheets / number("desc2_panels_max")),
        _unit(crease / number("desc2_crease_ref_deg")),
        _unit(m.directions / number("desc2_directions_max")),
    )


def body_terms(d: Design) -> np.ndarray:
    """The nine body numbers in BODY_NAMES order."""
    s = shares(d)
    centre = max(d.bodies, key=lambda b: b.length_m) if d.bodies else None
    pods = sum(2 if b.mirrored else 1 for b in d.bodies if b is not centre)
    shape = (0.0, 0.0, 0.0, 0.0) if centre is None else (_fineness(centre), *_facet_terms(centre))
    fineness, panels, crease, directions = shape
    return np.array(
        [
            s.top,
            s.side,
            s.front,
            s.blend,
            _unit(fineness / number("desc2_fineness_max")),
            panels,
            crease,
            directions,
            _unit(pods / number("desc2_pods_max")),
        ]
    )
