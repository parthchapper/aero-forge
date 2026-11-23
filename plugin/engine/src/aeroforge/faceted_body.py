"""Body models for the layout: the tail-end size both body kinds share and the flat-panel body."""

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_metrics import inradius_on_axis_m
from aeroforge.facet_query import check_invariants
from aeroforge.facets import facet_body
from aeroforge.models import Body, Family
from aeroforge.shapes import axis_z_at_x


def tail_radius_frac(fam: Family) -> float:
    """Tail-end radius / max radius of a smooth body: the gene, floored off a degenerate ring."""
    return max(fam.shape.tail_end_size_frac, number("facet_tail_end_min_frac"))


def faceted_body(
    fam: Family, length_m: float, axes_m: tuple[float, float], name: str, upsweep_deg: float = 0.0
) -> Body:
    """Body of flat panels; its smooth-law fields give the same extent and envelope (widest
    station at nose_frac, boat-tail after it) to consumers that are not facet-aware yet."""
    m = fam.shape.max_section_x_frac
    poly = facet_body(fam.shape, length_m, axes_m, upsweep_deg)
    check_invariants(poly, length_m)
    return Body(
        name=name,
        mirrored=False,
        nose_m=(0.0, 0.0, 0.0),
        length_m=length_m,
        half_width_m=axes_m[0],
        half_height_m=axes_m[1],
        section_exp=fam.body_section_exp,
        nose_frac=m,
        tail_frac=1.0 - m,
        tail_radius_frac=tail_radius_frac(fam),
        tail_upsweep_deg=upsweep_deg,
        facets=poly,
    )


def firewall_attach(
    body: Body, end_attach: np.ndarray, diameter_m: float, pusher: bool
) -> np.ndarray:
    """Where a motor attaches to a flat-panel host (the tail end for a pusher, the nose for a
    tractor): the end itself when a firewall plate of at least mount_plate_min_frac of its full
    radius fits there, else the nearest place inside (within mount_slide_body_frac of the body
    length) where it does, else the place with the widest plate. The shaft then runs from there
    to the hub. The plate is the disc of hardpoints._mount: its centre mount_plate_inset_m + half
    its thickness inside the attach point."""
    if body.facets is None:
        return end_attach
    target = number("mount_plate_min_frac") * number("mount_plate_radius_frac") * diameter_m
    back = number("mount_plate_inset_m") + number("mount_plate_thickness_m") / 2.0
    length = body.length_m
    steps = int(number("mount_slide_steps"))
    reach = number("mount_slide_body_frac") * length
    sign = -1.0 if pusher else 1.0  # plate centre moves forward from a tail end, aft from a nose
    first = length - back if pusher else back
    best_x, best_r = first, -1.0
    for k in range(steps + 1):
        x = first + sign * reach * k / steps  # plate centre, body frame
        r = inradius_on_axis_m(body.facets, x)
        if r >= target:
            return end_attach if k == 0 else _attach_at(body, x + (back if pusher else -back))
        if r > best_r:
            best_x, best_r = x, r
    return _attach_at(body, best_x + (back if pusher else -back))


def _attach_at(body: Body, x_body_m: float) -> np.ndarray:
    """Point on the body's axis at x_body_m (body frame) in world coordinates."""
    x = body.nose_m[0] + x_body_m
    return np.array([x, body.nose_m[1], axis_z_at_x(body, x)])
