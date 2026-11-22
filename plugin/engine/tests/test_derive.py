import pytest

from aeroforge.constants import number
from aeroforge.derive import InfeasibleError, derive
from aeroforge.mass import mass_items, total_mass_kg
from aeroforge.models import Brief, Design, Requirements

TEN_KM_IN_ONE_HOUR_MS = 10_000 / 3600
IMPOSSIBLE_ENDURANCE_MIN = 1000.0
WIDE_SPAN_M = 5000.0


def test_mass_closure_converges_on_the_sample_brief(
    sample_brief: Brief, sample_req: Requirements, conventional: Design
) -> None:
    items = mass_items(conventional, sample_req, sample_brief.payload_mass_kg)
    margins = (1 + number("mass_growth_frac")) * (1 + number("mass_reserve_frac"))
    closed = margins * total_mass_kg(items)
    battery_kg = sample_req.battery_wh / number("battery_specific_energy_wh_per_kg")
    # converged: MTOW and battery each moved by at most derive_rel_tol on the last plain step,
    # and MTOW = margins x (sum + battery swap)
    slack = number("derive_rel_tol") * (sample_req.mtow_kg + margins * battery_kg)
    assert abs(closed - sample_req.mtow_kg) <= slack


def test_requirements_carry_each_flight_point_once(sample_req: Requirements) -> None:
    assert [p.id for p in sample_req.points] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert sample_req.baseline_cruise_ld_tier1 is not None
    assert sample_req.baseline_cruise_ld_tier2 is not None


def test_ten_km_in_one_hour_is_infeasible_and_names_stall(sample_brief: Brief) -> None:
    slow = sample_brief.model_copy(
        update={"cruise_speed_ms": TEN_KM_IN_ONE_HOUR_MS, "range_km": 10.0, "endurance_min": None}
    )
    with pytest.raises(InfeasibleError) as info:
        derive(slow)
    assert info.value.check == "stall"


def test_impossible_endurance_raises_the_typed_mass_error(sample_brief: Brief) -> None:
    # a wide span limit keeps the growing wing feasible, so the mass closure is what fails
    endless = sample_brief.model_copy(
        update={"endurance_min": IMPOSSIBLE_ENDURANCE_MIN, "span_max_m": WIDE_SPAN_M}
    )
    with pytest.raises(InfeasibleError) as info:
        derive(endless)
    assert info.value.check == "mass"
