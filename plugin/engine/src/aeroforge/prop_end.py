"""The end and size of the centre body that hosts the motor (U2 follow-up): on a flat-panel body
they are made just big enough for a firewall plate, so a slim body leaves no mount outside."""

from aeroforge.constants import number
from aeroforge.faceted_body import faceted_body, plate_fits
from aeroforge.models import Body, Family, Requirements
from aeroforge.pod_fit import grow_to_fit, prop_diameter
from aeroforge.shift import Anchor, Growth, centre_body

PROP_COUNT = {"centre": 1, "pods": 2, "centre_pods": 3}


def centre_body_for_prop(
    req: Requirements,
    fam: Family,
    length_frac: float,
    min_length_m: float,
    g: Growth,
    holds_boxes: bool,
    mount: tuple[str, bool],
) -> tuple[Body, Anchor | None, Family]:
    """(centre body, root anchor, family) with the body's end (the family's shape) and, when the
    end is not enough, its section sized for the motor of the pattern (mount = (pattern, pusher));
    nothing changes for a smooth body or a pattern without a centre motor."""
    pattern, pusher = mount
    body, anchor = centre_body(req, fam, length_frac, min_length_m, g, holds_boxes)
    if fam.shape.style != "faceted" or pattern not in ("centre", "centre_pods"):
        return body, anchor, fam
    motor = (prop_diameter(req, PROP_COUNT[pattern]), pusher)
    if plate_fits(body, *motor):
        return body, anchor, fam
    up = fam.body_tail_upsweep_deg

    def build_at(k: float, f: Family) -> Body:
        base = centre_body(req, f, length_frac, min_length_m, g, holds_boxes)[0]
        axes = (base.half_width_m * k, base.half_height_m * k)
        return faceted_body(f, base.length_m, axes, "body0", up)

    found = grow_to_fit(fam, build_at, motor, number("prop_body_grow_max"))
    if found is None:
        return body, anchor, fam
    return build_at(found[1], found[0]), anchor, found[0]
