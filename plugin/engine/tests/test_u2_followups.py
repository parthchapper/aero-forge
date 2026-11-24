"""U2 follow-ups 1, 1b and 2 (written at the U2 audit, they had no tests): a flat-panel host of a
motor is big enough for its firewall plate, the hatch has headroom above the top boxes, and the tail
upsweep of a flat-panel body starts at a station behind which no body-hosted root is lifted out."""

import numpy as np
import pytest
from conftest import u2_random_genomes

from aeroforge.assemble import root_reach_m
from aeroforge.brief import brief_requirements
from aeroforge.constants import number
from aeroforge.facet_query import axis_half_height_m, slice_section
from aeroforge.faceted_body import plate_fits
from aeroforge.facets import axis_rise_between_m, facet_body
from aeroforge.fit_report import fit_rows
from aeroforge.genetable import pin_choice
from aeroforge.genome import decode
from aeroforge.hardpoints import containment_failures, hardpoints
from aeroforge.models import Body, Design
from aeroforge.pod_fit import prop_diameter
from aeroforge.prop_end import PROP_COUNT
from aeroforge.shapes import axis_z_at_x

SEED = 20261112
N = 100
MISFIT_BAR = 0.01


@pytest.fixture(scope="module")
def faceted_stub() -> list[Design]:
    req = brief_requirements("stub")
    genomes = pin_choice(u2_random_genomes(SEED, N), "body_style", "faceted")
    return [decode(g, req) for g in genomes]


def test_a_flat_panel_host_of_a_motor_holds_its_firewall_plate(faceted_stub: list[Design]) -> None:
    """Follow-up 1: the end (tail size, flat cap, blunt nose) and if needed the section of a body
    or pod that hosts a motor are sized for the plate: no mount of the sample is a misfit."""
    req = brief_requirements("stub")
    misfits = 0
    for d in faceted_stub:
        names = containment_failures(d, hardpoints(d, req.recovery))
        misfits += sum(n.startswith("mount_") for n in names)
    assert misfits <= MISFIT_BAR * N


def test_the_sized_end_is_reported_by_the_plate_rule_not_hidden(faceted_stub: list[Design]) -> None:
    """A host whose end still has no plate of half its nominal radius within the slide is a
    misfit (containment), never a smaller plate: plate_fits agrees with containment."""
    req = brief_requirements("stub")
    for d in faceted_stub:
        names = containment_failures(d, hardpoints(d, req.recovery))
        for p in d.propulsors:
            host = next((b for b in d.bodies if b.name == p.host), None)
            if host is None or host.facets is None or p.mirrored:
                continue
            fits = plate_fits(host, p.diameter_m, p.pusher)
            assert fits or f"mount_{p.name}" in names


def test_the_prop_diameter_follows_the_pattern_count() -> None:
    req = brief_requirements("stub")
    assert prop_diameter(req, PROP_COUNT["centre"]) > prop_diameter(req, PROP_COUNT["centre_pods"])


def test_boxes_and_hatch_have_headroom_in_a_flat_panel_body(faceted_stub: list[Design]) -> None:
    """Follow-up 1b: bays and hatches are contained (the hatch lies on the top boxes)."""
    req = brief_requirements("stub")
    bad = 0
    for d in faceted_stub:
        names = containment_failures(d, hardpoints(d, req.recovery))
        bad += any(n.startswith(("bay_", "hatch_")) for n in names)
    assert bad <= MISFIT_BAR * N


def _upswept(designs: list[Design]) -> list[Design]:
    return [
        d
        for d in designs
        if any(b.facets is not None and b.rise_x_frac is not None for b in d.bodies)
    ]


def test_a_moved_rise_tilts_the_cell_it_starts_in_and_nothing_before_it(
    faceted_stub: list[Design],
) -> None:
    """Follow-up 2: the axis of a flat-panel body is the straight line between stations
    (axis_rise_between_m): checked on the body's own slices, level up to the station before the
    rise start and rising at the station values after it."""
    moved = _upswept(faceted_stub)
    assert moved, "no sampled design moved its rise start"
    for d in moved:
        b = next(x for x in d.bodies if x.name == "body0")
        t = np.linspace(0.05, 0.95, 19)
        law = axis_rise_between_m(t, d.family.shape, b.length_m, b.tail_upsweep_deg, b.rise_x_frac)
        for ti, rise in zip(t, law, strict=True):
            z0 = axis_z_at_x(b, b.nose_m[0] + ti * b.length_m) - b.nose_m[2]
            assert rise == pytest.approx(z0 - _level_axis_m(d, b, ti), abs=1e-6 * b.length_m)


def _level_axis_m(d: Design, b: Body, t: float) -> float:
    """Axis height [m] of the same body with no upsweep at length fraction t."""
    level = facet_body(d.family.shape, b.length_m, (b.half_width_m, b.half_height_m))
    return slice_section(level, t * b.length_m)[3]


def test_no_body_hosted_root_is_lifted_out_by_a_kept_rise(faceted_stub: list[Design]) -> None:
    """The reach of every body-hosted root chord (assemble.root_reach_m) is at least what the same
    body has with no rise (or zero): the rise lifts no root chord out of the body."""
    checked = 0
    for d in faceted_stub:
        b = next((x for x in d.bodies if x.name == "body0" and x.facets is not None), None)
        if b is None or b.tail_upsweep_deg <= 0.0:
            continue
        level = facet_body(d.family.shape, b.length_m, (b.half_width_m, b.half_height_m))
        for s in d.surfaces:
            if s.host != "body0":
                continue
            le = s.sections[0].le_m[0]
            te = le + s.sections[0].chord_m
            flat = (
                number("root_z_usable_frac") * axis_half_height_m(level, le, te)
                - abs(slice_section(level, te)[3] - slice_section(level, le)[3]) / 2.0
            )
            assert root_reach_m(b, le, te) >= min(0.0, flat) - 1e-6
            checked += 1
    assert checked


def test_a_smooth_body_never_moves_its_rise_start() -> None:
    """The 0.3 baseline: a smooth body drops an upsweep that a root sits on, as before."""
    req = brief_requirements("stub")
    genomes = pin_choice(u2_random_genomes(SEED, 60), "body_style", "smooth")
    for g in genomes:
        for b in decode(g, req).bodies:
            assert b.rise_x_frac is None


def test_the_report_states_how_plates_hatches_and_servo_trays_sit(
    faceted_stub: list[Design],
) -> None:
    """U2 audit: a passing fit is not silent about a small plate, a shrunk hatch or a slid servo."""
    req = brief_requirements("stub")
    rows = [r for d in faceted_stub[:12] for r in fit_rows(d, req.recovery)]
    assert any("plate radius" in r and "nominal" in r for r in rows)
    assert any("of its bay footprint" in r for r in rows)
    assert all(r.startswith("- ") for r in rows)
