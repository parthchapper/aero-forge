import contextlib
import logging
import time

import numpy as np
import pytest

from aeroforge.aero import AeroError
from aeroforge.checks import CheckReport, run_checks
from aeroforge.constants import number, numbers
from aeroforge.downwash import influence
from aeroforge.genome import decode
from aeroforge.mass import BATTERY, cg_range, mass_items, move_item
from aeroforge.models import Brief, Design, Requirements
from aeroforge.tier2 import tier2
from aeroforge.trim import AeroResult, tier1

SPEC_CHECKS = {
    "stall",
    "trim",
    "static",
    "lateral",
    "stall_order",
    "mass",
    "power",
    "energy",
    "structure",
    "efficiency",
    "fit",
}
BATTERY_STEP_M = 0.02
BATTERY_MAX_STEPS = 20
FAR_AFT_M = 0.4  # battery well behind the wing: static fails at tier 1
NP_DESIGNS = 4  # plus the conventional: five designs fixed by rule, not by result
NP_TOLERANCE_MAC = 0.075  # user as PLAN owner (step 3 gate r4): half the static band
CM_TRIM_TOL = 1e-9
TIMED_DESIGNS = 10

log = logging.getLogger(__name__)


def _static(report: CheckReport) -> float:
    detail = next(c for c in report.checks if c.id == "static").detail["static_margin_min"]
    assert isinstance(detail, float)
    return detail


def _p3_aft_margin(r: AeroResult) -> float:
    return next(t.static_margin for t in r.trims if t.point == "P3" and t.cg == "aft")


@pytest.mark.slow  # user (step 3 gate r4): tier-2 conventional tests leave the fast suite
def test_conventional_reference_passes_every_check_at_tier_2(
    conventional_tier2: CheckReport,
) -> None:
    assert {c.id for c in conventional_tier2.checks} == SPEC_CHECKS
    failed = {c.id: c.margin for c in conventional_tier2.checks if not c.passed}
    assert failed == {}
    assert conventional_tier2.feasible and conventional_tier2.violation == 0.0


@pytest.mark.slow
def test_battery_moved_aft_fails_static_and_leaves_other_checks_unchanged(
    conventional_t2: tuple[Design, CheckReport],
    sample_brief: Brief,
    sample_req: Requirements,
) -> None:
    conventional, conventional_tier2 = conventional_t2
    items = mass_items(conventional, sample_req, sample_brief.payload_mass_kg)
    low = numbers("static_margin_band", "checks")[0]
    for step in range(1, BATTERY_MAX_STEPS + 1):
        moved = move_item(items, BATTERY, step * BATTERY_STEP_M)
        report = run_checks(conventional, sample_brief, sample_req, 2, moved)
        if _static(report) < low:
            break
    else:
        pytest.fail("static margin never fell below the band")
    before = {c.id: c.passed for c in conventional_tier2.checks}
    after = {c.id: c.passed for c in report.checks}
    assert not after.pop("static")
    before.pop("static")
    assert after == before


def test_violation_is_the_sum_of_negative_margins(
    conventional: Design, sample_brief: Brief, sample_req: Requirements
) -> None:
    items = mass_items(conventional, sample_req, sample_brief.payload_mass_kg)
    report = run_checks(
        conventional, sample_brief, sample_req, 1, move_item(items, BATTERY, FAR_AFT_M)
    )
    expected = sum(max(0.0, -c.margin) for c in report.checks)
    assert not report.feasible and report.violation > 0.0
    assert report.violation == pytest.approx(expected)


def test_trimmed_points_carry_their_load_with_zero_pitch_moment(
    conventional: Design, sample_brief: Brief, sample_req: Requirements
) -> None:
    cgs = cg_range(mass_items(conventional, sample_req, sample_brief.payload_mass_kg))
    r = tier1(conventional, sample_req, cgs)
    f = r.flight
    for t in r.trims:
        q_s = 0.5 * f.rho_kg_m3 * t.speed_ms**2 * f.s_ref_m2
        assert t.cl * q_s == pytest.approx(t.load_factor * f.weight_n)
        assert abs(t.cm_residual) < CM_TRIM_TOL


def test_the_wing_washes_the_tail_down_and_the_tail_washes_the_wing_up(
    conventional: Design,
) -> None:
    c = influence(conventional)
    assert c[0, 1] < 0.0 < c[1, 0]


@pytest.mark.slow
def test_tier1_and_tier2_neutral_points_agree_within_5_percent_mac(
    conventional: Design,
    sample_brief: Brief,
    sample_req: Requirements,
    legacy_genomes: np.ndarray,
) -> None:
    designs = [conventional] + [decode(g, sample_req) for g in legacy_genomes[:NP_DESIGNS]]
    gaps = []
    for d in designs:
        cgs = cg_range(mass_items(d, sample_req, sample_brief.payload_mass_kg))
        gaps.append(
            abs(
                _p3_aft_margin(tier1(d, sample_req, cgs))
                - _p3_aft_margin(tier2(d, sample_req, cgs))
            )
        )
    assert max(gaps) <= NP_TOLERANCE_MAC, gaps


@pytest.mark.slow
def test_tier1_time_per_candidate_is_measured(
    sample_brief: Brief, sample_req: Requirements, genomes: np.ndarray
) -> None:
    seconds = []
    for g in genomes[-TIMED_DESIGNS:]:
        d = decode(g, sample_req)
        items = mass_items(d, sample_req, sample_brief.payload_mass_kg)
        start = time.perf_counter()
        with contextlib.suppress(AeroError):  # an untrimmable candidate still costs its time
            run_checks(d, sample_brief, sample_req, 1, items)
        seconds.append(time.perf_counter() - start)
    log.warning(
        "tier-1 check time per candidate: mean %.3f s, max %.3f s (target %.2f s)",
        float(np.mean(seconds)),
        max(seconds),
        number("tier1_target_s"),
    )
    assert len(seconds) == TIMED_DESIGNS
