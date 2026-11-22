from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from aeroforge.aero import (
    buildup_samples,
    flight_state,
    lift_n,
    point_speeds,
    stall_load_factor,
)
from aeroforge.airplane import quarter_mac_x_m
from aeroforge.balance import ON_TARGET, balance, neutral_point_x_m
from aeroforge.brief import resolve
from aeroforge.checks import Inputs, check_power, check_trim
from aeroforge.constants import number, numbers
from aeroforge.fitcheck import fit_margins
from aeroforge.genome import decode
from aeroforge.mass import cg_range, mass_items
from aeroforge.models import Adjust, Brief, Design, Requirements
from aeroforge.prepare import INCIDENCE_SET, Prepared, has_vertical_area, prepare
from aeroforge.reference import BASELINE, reference_genome
from aeroforge.style import wing_surfaces
from aeroforge.tier2 import vlm_samples
from aeroforge.trim import tier1, trim_point

# user (step 3 gate r3): the balance tests go slow while the fast suite exceeds 60 s
pytestmark = pytest.mark.slow

TARGET_TOL = 1e-3  # fresh AeroBuildup run vs the balance's own samples
IDEMPOTENT_TOL_M = 1e-6
JOIN_SEARCH = 200
SHIFT_DESIGNS = 4
SHIFTS_M = (-0.1, 0.1)
FIN_SCALE = 1.5
ROUND_OFF_M = 1e-9


def _cruise(req: Requirements) -> float:
    return next(p.speed_ms for p in req.points if p.id == "P3")


def _body_roots(d: Design) -> tuple[tuple[float, float], list[float]]:
    body = next(b for b in d.bodies if b.name == "body0")
    ends = (body.nose_m[0], body.nose_m[0] + body.length_m)
    return ends, [s.sections[0].le_m[0] for s in d.surfaces if s.host == "body0"]


def test_prepared_boxes_stay_inside_their_host_and_apart(sample_prepared: list[Prepared]) -> None:
    gap = number("box_gap_m")
    for p in sample_prepared:
        boxes = sorted(p.design.boxes, key=lambda b: b.centre_m[0])
        for a, b in pairwise(boxes):
            if a.host == b.host:
                apart = b.centre_m[0] - a.centre_m[0] - (a.size_m[0] + b.size_m[0]) / 2
                assert apart >= gap - ROUND_OFF_M


def test_designs_balanced_on_target_have_the_target_static_margin(
    sample_prepared: list[Prepared], sample_req: Requirements, sample_brief: Brief
) -> None:
    on_target = [p for p in sample_prepared if p.balance_outcome == ON_TARGET]
    assert on_target
    for p in on_target:
        aft = cg_range(mass_items(p.design, sample_req, sample_brief.payload_mass_kg))[1]
        speeds = point_speeds(sample_req)
        fresh = buildup_samples(p.design, sample_req, speeds, _cruise(sample_req), aft)
        c = flight_state(p.design, sample_req).c_ref_m
        margin = (neutral_point_x_m(fresh, sample_req, p.design, aft) - aft[0]) / c
        assert margin == pytest.approx(number("static_margin_target", "checks"), abs=TARGET_TOL)


def test_balancing_a_balanced_design_moves_nothing(
    sample_prepared: list[Prepared], sample_req: Requirements, sample_brief: Brief
) -> None:
    p = next(p for p in sample_prepared if p.balance_outcome == ON_TARGET)
    again = balance(p.design, sample_req, sample_brief.payload_mass_kg, buildup_samples, p.samples)
    assert abs(again.battery_shift_m) + abs(again.stack_shift_m) < IDEMPOTENT_TOL_M


def test_trim_deflects_the_control_put_on_the_longest_pitch_arm(
    sample_prepared: list[Prepared], sample_req: Requirements, sample_brief: Brief
) -> None:
    for p in sample_prepared:
        aft = cg_range(mass_items(p.design, sample_req, sample_brief.payload_mass_kg))[1]
        longest = max(
            wing_surfaces(p.design),
            key=lambda s: (abs(quarter_mac_x_m(s) - aft[0]), s.projected_area_m2),
        )
        assert longest.control is not None
        assert longest.control.chord_frac >= number("pitch_control_chord_frac")
        lo, hi = numbers("pitch_control_span_frac")
        assert (longest.control.span_start_frac, longest.control.span_end_frac) == (lo, hi)
        assert p.samples.pitch_control == longest.name


def test_body_shift_moves_the_body_and_keeps_the_surface_spacing(
    legacy_genomes: np.ndarray, sample_req: Requirements
) -> None:
    for g in legacy_genomes[:SHIFT_DESIGNS]:
        base = decode(g, sample_req)
        if not any(b.name == "body0" for b in base.bodies):
            continue
        ends0, roots0 = _body_roots(base)
        if not roots0:
            continue
        for shift in SHIFTS_M:
            ends, roots = _body_roots(decode(g, sample_req, Adjust(body_shift_m=shift)))
            assert np.allclose(np.diff(roots), np.diff(roots0), atol=ROUND_OFF_M)
            # the end that did not have to grow moves by exactly the shift
            moved = [(e - roots[0]) - (e0 - roots0[0]) for e, e0 in zip(ends, ends0, strict=True)]
            assert min(abs(m - shift) for m in moved) < 1e-6


def test_adjusted_designs_keep_fit_and_roots_on_their_body(
    legacy_genomes: np.ndarray, sample_req: Requirements
) -> None:
    adjustments = [Adjust(body_shift_m=s) for s in SHIFTS_M] + [Adjust(fin_scale=FIN_SCALE)]
    for g in legacy_genomes[:SHIFT_DESIGNS]:
        for adj in adjustments:
            d = decode(g, sample_req, adj)
            assert min(fit_margins(d, sample_req).values()) >= 0.0
            for b in d.bodies:
                for s in (s for s in d.surfaces if s.host == b.name):
                    root = s.sections[0]
                    assert root.le_m[0] >= b.nose_m[0] - ROUND_OFF_M
                    assert root.le_m[0] + root.chord_m <= b.nose_m[0] + b.length_m + ROUND_OFF_M


def test_gust_point_is_structure_only_and_capped_at_stall(
    conventional: Design, sample_brief: Brief, sample_req: Requirements
) -> None:
    items = mass_items(conventional, sample_req, sample_brief.payload_mass_kg)
    aero = tier1(conventional, sample_req, cg_range(items))
    gust = next(p for p in sample_req.points if p.gust_ms > 0.0)
    cap = stall_load_factor(aero.clmax, gust.speed_ms, aero.flight)
    assert all(t.load_factor <= cap * (1 + 1e-12) for t in aero.trims if t.point == gust.id)
    wild = tuple(
        replace(t, cl=10 * aero.clmax, cd=10.0, deflection_deg=90.0) if t.point == gust.id else t
        for t in aero.trims
    )
    x = Inputs(conventional, resolve(sample_brief), sample_req, items, aero)
    x_wild = replace(x, aero=replace(aero, trims=wild))
    assert check_trim(x_wild).margin == check_trim(x).margin
    assert check_power(x_wild).margin == check_power(x).margin


def test_tip_joined_design_has_positive_tier2_lift_slopes(
    legacy_genomes: np.ndarray, sample_req: Requirements, sample_brief: Brief
) -> None:
    joined = next(
        d
        for d in (decode(g, sample_req) for g in legacy_genomes[:JOIN_SEARCH])
        if any(s.joined_to for s in d.surfaces)
    )
    aft = cg_range(mass_items(joined, sample_req, sample_brief.payload_mass_kg))[1]
    speeds = np.array([_cruise(sample_req)])
    sm, _ = vlm_samples(joined, sample_req, speeds, float(speeds[0]), aft)
    a0, a1 = sm.alpha_deg
    rise = lift_n(sm.surface_force_n[0, 1], a1) - lift_n(sm.surface_force_n[0, 0], a0)
    slope = rise / (a1 - a0)
    names = [s.name for s in joined.surfaces]
    assert all(slope[names.index(s.name)] > 0.0 for s in wing_surfaces(joined))


def test_incidence_zeroes_the_cruise_deflection_at_the_mid_cg(
    sample_req: Requirements, sample_brief: Brief
) -> None:
    p = prepare(reference_genome(BASELINE), sample_req, sample_brief.payload_mass_kg)
    assert p.incidence_rule == INCIDENCE_SET
    items = mass_items(p.design, sample_req, sample_brief.payload_mass_kg)
    fore, aft = cg_range(items)
    cruise = next(pt for pt in sample_req.points if pt.id == "P3")
    fresh = buildup_samples(p.design, sample_req, point_speeds(sample_req), cruise.speed_ms, aft)
    flight = flight_state(p.design, sample_req)
    trim = trim_point(fresh, cruise, 1.0, "mid", (fore + aft) / 2, flight)
    assert abs(trim.deflection_deg) <= number("incidence_tol_deg")


def test_a_design_without_vertical_area_gets_a_centreline_fin(
    sample_prepared: list[Prepared], legacy_genomes: np.ndarray, sample_req: Requirements
) -> None:
    for g, p in zip(legacy_genomes, sample_prepared, strict=False):
        bare = decode(g, sample_req)
        if has_vertical_area(bare):
            continue
        assert p.adjust.add_fin
        fins = [s for s in p.design.surfaces if not s.mirrored and abs(s.cant_deg) == 90.0]
        assert len(fins) == 1 and fins[0].host == bare.surfaces[0].host
        return
    pytest.skip("no design without vertical area in the prepared sample")
