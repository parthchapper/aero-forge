"""Strict edge alignment of a flat-panel body (SPEC §4, U3): its top-view nose and tail edges.

A body is made to have exactly two straight outline edges: the radius law is linear (BodyShape
.linear_edges), the nose a point or a blunt cap, the tail a flat cap. The nose edge then runs from
the nose (or the cap's edge) to the widest station and the tail edge from there to the tail cap,
so the widest-station position m and the tail end size set their angles in closed form. Because
the body's length and half-width come out of the box sizing, the decoder re-snaps after each
sizing pass until they stop changing (genome.py repair loop). Pure; metres, degrees."""

from dataclasses import dataclass, replace

import numpy as np

from aeroforge.coarse_body import coarsen
from aeroforge.constants import number
from aeroforge.edges import angle_gap_deg
from aeroforge.models import Body, BodyShape, Family


def coerced(shape: BodyShape) -> BodyShape:
    """The shape with straight edges only: linear radius law, chisel nose -> point, point or knife
    tail -> flat (their closing edges are not on the silhouette's straight lines)."""
    return shape.model_copy(
        update={
            "linear_edges": True,
            "nose": "point" if shape.nose == "chisel" else shape.nose,
            "tail": "flat",
        }
    )


def coercion_notes(shape: BodyShape) -> tuple[str, ...]:
    """Decoder notes for what coerced() changed."""
    out = []
    if shape.nose == "chisel":
        out.append("strict alignment: chisel nose made a point (its edges are not straight lines)")
    if shape.tail != "flat":
        out.append(f"strict alignment: {shape.tail} tail made a flat cap")
    return tuple(out)


def _ranked(angle_deg: float, allowed: tuple[float, ...], sign: float) -> list[float]:
    """Allowed angles of the wanted sign (nose edges sweep back, tail edges forward), strictly
    between 0 and 90 in size, nearest to angle_deg first."""
    ok = [a for a in allowed if sign * a > 1e-9 and abs(a) < 90.0 - 1e-9]
    return sorted(ok, key=lambda a: angle_gap_deg(angle_deg, a))


def snap_shape(
    shape: BodyShape, length_m: float, half_width_m: float, allowed: tuple[float, ...]
) -> BodyShape:
    """shape (already coerced) with max_section_x_frac and tail_end_size_frac moved so that the nose
    and tail edges of a body of this length and half-width lie on allowed angles: the allowed
    angle nearest the present one whose widest station and tail end stay in their ranges
    (tail_end_size_frac never drops below its present value: it may have been raised for a motor
    plate); the shape unchanged when no allowed angle does."""
    blunt = number("facet_blunt_nose_size_frac") if shape.nose == "blunt" else 0.0
    lo, hi = number("strict_widest_x_min_frac"), number("strict_widest_x_max_frac")
    run = length_m / (half_width_m * (1.0 - blunt))
    m0, end0 = shape.max_section_x_frac, shape.tail_end_size_frac
    nose_now = float(np.degrees(np.arctan(m0 * run)))
    if end0 >= 1.0 - 1e-6:  # a motor plate filled the tail end: its edge runs along x (90 deg)
        tail_now, tail_free = -90.0, False
    else:
        run_t = (1.0 - m0) * length_m / (half_width_m * (1.0 - end0))
        tail_now, tail_free = -float(np.degrees(np.arctan(run_t))), True
    for angle in _ranked(nose_now, allowed, 1.0):
        m = float(np.tan(np.radians(angle))) / run
        if not lo <= m <= hi:
            continue
        if not tail_free:
            return shape.model_copy(update={"max_section_x_frac": m})
        for tail in _ranked(tail_now, allowed, -1.0):
            end = 1.0 - (1.0 - m) * length_m / (half_width_m * float(np.tan(np.radians(-tail))))
            if end0 <= end <= number("strict_tail_end_max_frac"):
                return shape.model_copy(update={"max_section_x_frac": m, "tail_end_size_frac": end})
    return shape


@dataclass(frozen=True)
class BodySnap:
    """The body end shape of a strictly aligned design and how many re-snaps it has had."""

    shape: BodyShape
    count: int = 0
    coarse: bool = False  # the coarseness pass (coarse_body.coarsen) has been made
    notes: tuple[str, ...] = ()  # decoder notes of the change that made this snap


def initial_shape(fam: Family) -> BodySnap | None:
    """The coerced body shape a strictly aligned flat-panel design starts from (None: no change
    unless the coarseness has a pass to make on a flat-panel body)."""
    if fam.shape.style != "faceted":
        return None
    shape = fam.shape.model_copy(update={"coarseness": fam.facet_coarseness})
    if fam.edge_alignment != "strict":
        return BodySnap(shape) if fam.facet_coarseness > 0.0 else None
    return BodySnap(coerced(shape))


def next_shape(fam: Family, body: Body, snap: BodySnap) -> BodySnap | None:
    """The next re-snap for the sized centre body, None when its edges already sit on allowed
    angles, no allowed angle fits (the measured notes report it), or strict_body_snap_max re-snaps
    were spent (box sizing and snapping chase each other slowly on some bodies)."""
    if body.facets is None:
        return None
    if snap.count < int(number("strict_body_snap_max")) and fam.edge_body_angles_deg:
        new = snap_shape(fam.shape, body.length_m, body.half_width_m, fam.edge_body_angles_deg)
        same = (
            abs(new.max_section_x_frac - fam.shape.max_section_x_frac) < 1e-9
            and abs(new.tail_end_size_frac - fam.shape.tail_end_size_frac) < 1e-9
        )
        if not same:
            return replace(snap, shape=new, count=snap.count + 1, notes=())
    if fam.facet_coarseness <= 0.0 or snap.coarse:
        return None
    shape, notes = coarsen(fam.shape, body, fam.facet_coarseness)
    return replace(snap, shape=shape, coarse=True, notes=notes) if notes else None


def uncoarsened(fam: Family, snap: BodySnap, why: str) -> BodySnap:
    """The body shape of the genes (coerced when strict) with no coarseness, kept for good: the
    coarse body could not hold what the design needs (boxes, a plane) and the decoder gave the
    coarsening up for the body (wings keep theirs)."""
    shape = fam.shape.model_copy(update={"linear_edges": False, "dropped": (), "coarseness": 0.0})
    shape = coerced(shape) if fam.edge_alignment == "strict" else shape
    note = f"coarseness: body coarsening given up, the genes' body kept ({why})"
    return BodySnap(shape, snap.count, True, (note,))
