"""Body models for the layout: the tail-end size both body kinds share and the flat-panel body."""

from collections.abc import Callable

import numpy as np

from aeroforge.constants import number, numbers
from aeroforge.facet_metrics import inradius_on_axis_m
from aeroforge.facet_query import check_invariants
from aeroforge.facets import facet_body
from aeroforge.models import Body, Family
from aeroforge.shapes import axis_z_at_x


def tail_radius_frac(fam: Family) -> float:
    """Tail-end radius / max radius of a smooth body: the gene, floored off a degenerate ring."""
    return max(fam.shape.tail_end_size_frac, number("facet_tail_end_min_frac"))


def faceted_body(
    fam: Family,
    length_m: float,
    axes_m: tuple[float, float],
    name: str,
    upsweep_deg: float = 0.0,
    rise_x_frac: float | None = None,
) -> Body:
    """Body of flat panels; its smooth-law fields give the same extent and envelope (widest
    station at nose_frac, boat-tail after it) to consumers that are not facet-aware yet."""
    m = fam.shape.max_section_x_frac
    poly = facet_body(fam.shape, length_m, axes_m, upsweep_deg, rise_x_frac)
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
        rise_x_frac=rise_x_frac,
        facets=poly,
    )


def _plate_scan(body: Body, diameter_m: float, pusher: bool) -> tuple[float, float, float]:
    """(attach x [body frame], widest plate inradius [m], target inradius [m]) for a motor on a
    flat-panel host: the first place from the end (k = 0) where a plate of at least
    mount_plate_min_frac of its full radius fits, else the place with the widest plate (searched
    within mount_slide_body_frac of the length). The plate centre lies mount_plate_inset_m + half
    its thickness inside the attach point."""
    target = number("mount_plate_min_frac") * number("mount_plate_radius_frac") * diameter_m
    target *= 1.0 + number("mount_plate_fit_margin_frac")
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
            return x + (back if pusher else -back), r, target
        if r > best_r:
            best_x, best_r = x, r
    return best_x + (back if pusher else -back), best_r, target


def firewall_attach(
    body: Body, end_attach: np.ndarray, diameter_m: float, pusher: bool
) -> np.ndarray:
    """Where a motor attaches to a flat-panel host (the tail end for a pusher, the nose for a
    tractor): the end itself when a firewall plate of at least mount_plate_min_frac of its full
    radius fits there, else the nearest place inside where it does, else the place with the
    widest plate. The shaft then runs from there to the hub."""
    if body.facets is None:
        return end_attach
    x, _, _ = _plate_scan(body, diameter_m, pusher)
    end_x = body.length_m if pusher else 0.0
    if abs(x - end_x) <= number("facet_geom_tol_m"):
        return end_attach
    return _attach_at(body, x)


def plate_fits(body: Body, diameter_m: float, pusher: bool) -> bool:
    """A firewall plate of at least mount_plate_min_frac of its full radius fits the host's end
    region (within mount_slide_body_frac of the length)."""
    if body.facets is None:
        return True
    _, widest, target = _plate_scan(body, diameter_m, pusher)
    return widest >= target


def prop_end_family(
    fam: Family, build: Callable[[Family], Body], diameter_m: float, pusher: bool
) -> Family:
    """fam with the end of the body that hosts a motor made just big enough for its firewall plate
    (a bigger tail-end size for a pusher, a blunt nose for a tractor), tried up the ladder
    prop_end_ladder_frac; fam itself when the plate fits already or no rung helps. `build` makes
    the host body from a family."""
    if plate_fits(build(fam), diameter_m, pusher):
        return fam
    shape = fam.shape
    ladder = numbers("prop_end_ladder_frac")
    if pusher:  # a bigger end first, then a flat cap (a pointed tail ignores the end size)
        rungs = [
            shape.model_copy(update={"tail_end_size_frac": r})
            for r in ladder
            if r > shape.tail_end_size_frac
        ]
        rungs += [
            shape.model_copy(update={"tail": "flat", "tail_end_size_frac": r}) for r in ladder
        ]
    else:
        rungs = [shape.model_copy(update={"nose": "blunt"})] if shape.nose != "blunt" else []
    for rung in rungs:
        trial = fam.model_copy(update={"shape": rung})
        if plate_fits(build(trial), diameter_m, pusher):
            return trial
    return fam


def _attach_at(body: Body, x_body_m: float) -> np.ndarray:
    """Point on the body's axis at x_body_m (body frame) in world coordinates."""
    x = body.nose_m[0] + x_body_m
    return np.array([x, body.nose_m[1], axis_z_at_x(body, x)])
