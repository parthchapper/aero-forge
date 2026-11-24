"""SPEC §8 checks as normalised margins (m >= 0 passes) from mass, aero and fit."""

from dataclasses import dataclass

import numpy as np

from aeroforge.aero import Trim
from aeroforge.brief import launch_speed_ms, launch_value, resolve
from aeroforge.constants import constant, number, numbers
from aeroforge.fitcheck import fit_margins, shaft_margin
from aeroforge.mass import (
    MassItem,
    cg_range,
    installed_power_w,
    spar_length_m,
    total_mass_kg,
)
from aeroforge.models import Brief, Design, Frozen, Requirements
from aeroforge.prepare import prepare, prepare_tier2, prepared_aero
from aeroforge.style import wing_surfaces
from aeroforge.tier2 import tier2
from aeroforge.trim import AeroResult, cruise_ld, tier1

STRUCTURE_POINTS = ("P5", "P6")
UNTRIMMABLE_MARGIN = -1.0  # pitch control present but the trim system is singular


class CheckError(RuntimeError):
    """Requirements lack what a check needs (run derive first)."""


class CheckResult(Frozen):
    """One check: worst normalised margin and the quantities behind it."""

    id: str
    margin: float
    passed: bool
    detail: dict[str, float | str]


class CheckReport(Frozen):
    """All checks of one design at one tier."""

    tier: int
    feasible: bool
    violation: float
    checks: tuple[CheckResult, ...]
    passable: bool
    body_shift_m: float = 0.0
    fin_scale: float = 1.0
    battery_shift_m: float = 0.0
    stack_shift_m: float = 0.0
    shift_outcome: str = "not_prepared"
    balance: str = "not_balanced"
    tier1_body_shift_m: float | None = None
    fin_scale_capped: bool = False
    added_fin: bool = False
    incidence_deg: float = 0.0
    incidence_rule: str = "not_prepared"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Inputs:
    """Everything a check reads."""

    design: Design
    brief: Brief
    req: Requirements
    items: tuple[MassItem, ...]
    aero: AeroResult
    fin_capped: bool = False


def _result(cid: str, margins: dict[str, float], detail: dict[str, float | str]) -> CheckResult:
    """CheckResult with the worst named margin; the margins join the detail."""
    worst = min(margins.values())
    return CheckResult(
        id=cid, margin=float(worst), passed=bool(worst >= 0.0), detail={**margins, **detail}
    )


def _power_w(x: Inputs, cd: float, speed_ms: float) -> float:
    """Drag power D V [W]."""
    f = x.aero.flight
    return cd * 0.5 * f.rho_kg_m3 * speed_ms**3 * f.s_ref_m2


def check_stall(x: Inputs) -> CheckResult:
    """Vs <= stall_max and launch speed >= 1.2 Vs."""
    vs, launch = x.aero.stall_speed_ms, launch_speed_ms(x.brief)
    ratio = number("launch_speed_stall_ratio", "checks")
    stall_max = float(x.brief.stall_max_ms or 0.0)
    margins = {
        "stall_speed": (stall_max - vs) / stall_max,
        "launch_speed": (launch - ratio * vs) / launch,
    }
    return _result("stall", margins, {"stall_speed_ms": vs, "clmax": x.aero.clmax})


def _flown(x: Inputs) -> list[Trim]:
    """Trimmed points that are flown, not structure-only (gust points are structure-only)."""
    gusts = {p.id for p in x.req.points if p.gust_ms > 0.0}
    return [t for t in x.aero.trims if t.point not in gusts]


def check_trim(x: Inputs) -> CheckResult:
    """Every non-gust point trims with |deflection| <= 15 deg and CL <= 0.85 CLmax (P6 is
    structure-only: user Deviation, step 3 gate r3)."""
    d_max = number("trim_deflection_max_deg", "checks")
    cl_lim = number("trim_cl_max_frac", "checks") * x.aero.clmax
    defl, cl, worst_cm = [], [], 0.0
    for t in _flown(x):
        if x.aero.pitch_control is None:
            worst_cm = max(worst_cm, abs(t.cm_residual))
            defl.append(d_max)
        else:
            defl.append(np.inf if np.isnan(t.deflection_deg) else abs(t.deflection_deg))
        cl.append(t.cl)
    margins = {"deflection": (d_max - max(defl)) / d_max, "cl": (cl_lim - max(cl)) / cl_lim}
    if x.aero.pitch_control is None:
        margins["untrimmed_cm"] = -worst_cm / number("trim_cm_scale", "checks")
    if not np.isfinite(margins["deflection"]):
        margins["deflection"] = UNTRIMMABLE_MARGIN
    return _result("trim", margins, {"pitch_control": x.aero.pitch_control or "none"})


def check_static(x: Inputs) -> CheckResult:
    """Static margin within the band at the aft CG at every point."""
    lo, hi = numbers("static_margin_band", "checks")
    sm = [t.static_margin for t in _flown(x) if t.cg == "aft"]
    margins = {"low": (min(sm) - lo) / (hi - lo), "high": (hi - max(sm)) / (hi - lo)}
    return _result("static", margins, {"static_margin_min": min(sm), "static_margin_max": max(sm)})


def check_lateral(x: Inputs) -> CheckResult:
    """Cn_beta > 0 and Cl_beta < 0 at cruise; with vertical-area scaling at its cap, Cn_beta must
    also reach cn_beta_target (user Deviation, step 3 gate r4)."""
    scale = number("lateral_scale_per_rad", "checks")
    margins = {"cn_beta": x.aero.cnb_per_rad / scale, "cl_beta": -x.aero.clb_per_rad / scale}
    if x.fin_capped:
        target = number("cn_beta_target", "checks")
        margins["cn_beta_at_scale_cap"] = (x.aero.cnb_per_rad - target) / scale
    return _result(
        "lateral",
        margins,
        {"cn_beta_per_rad": x.aero.cnb_per_rad, "cl_beta_per_rad": x.aero.clb_per_rad},
    )


def check_stall_order(x: Inputs) -> CheckResult:
    """Tier 2: the foremost lifting surface stalls first, peak cl/clmax inboard of 70 % span."""
    if not x.aero.stall_order:
        raise CheckError("stall order needs the tier-2 evaluation")
    span_max = number("stall_order_span_frac_max", "checks")
    first = min(
        min(-1.0, (so.fore_ratio - so.other_ratio) / max(abs(so.other_ratio), 1.0))
        if so.fore_ratio <= 0.0
        else (so.fore_ratio - so.other_ratio) / max(so.fore_ratio, abs(so.other_ratio))
        for so in x.aero.stall_order
    )
    peak = max(so.peak_span_frac for so in x.aero.stall_order)
    margins = {"foremost_first": first, "peak_inboard": (span_max - peak) / span_max}
    return _result("stall_order", margins, {"peak_span_frac": peak})


def check_mass(x: Inputs) -> CheckResult:
    """Sum of component masses <= MTOW <= mass_max."""
    total, mtow = total_mass_kg(x.items), x.req.mtow_kg
    margins = {"closure": (mtow - total) / mtow}
    if x.brief.mass_max_kg is not None:
        margins["mass_max"] = (x.brief.mass_max_kg - mtow) / x.brief.mass_max_kg
    return _result("mass", margins, {"component_sum_kg": total, "mtow_kg": mtow})


def _static_thrust_n(x: Inputs) -> float:
    """Momentum-theory static thrust of all propellers sharing the installed power [N]."""
    discs = [p.diameter_m for p in x.design.propulsors for _ in range(2 if p.mirrored else 1)]
    per = installed_power_w(x.req) / len(discs)
    rho, fm = x.aero.flight.rho_kg_m3, number("prop_figure_of_merit")
    return float(sum((fm * per * np.sqrt(2 * rho * np.pi * dd**2 / 4)) ** (2 / 3) for dd in discs))


def check_power(x: Inputs) -> CheckResult:
    """P_required <= 0.8 P_available at every point; static T/W >= the launch minimum."""
    available = (
        number("power_available_frac", "checks")
        * installed_power_w(x.req)
        * number("prop_efficiency")
    )
    required = max(_power_w(x, t.cd, t.speed_ms) for t in _flown(x))
    tw = _static_thrust_n(x) / x.aero.flight.weight_n
    tw_min = launch_value("launch_thrust_to_weight_min", x.brief.launch)
    margins = {
        "power": (available - required) / available,
        "thrust_to_weight": (tw - tw_min) / tw_min,
    }
    return _result("power", margins, {"power_required_w": required, "thrust_to_weight": tw})


def check_energy(x: Inputs) -> CheckResult:
    """Endurance at P2 or range at P3 meets the brief."""
    eta = number("prop_efficiency") * number("motor_efficiency") * number("esc_efficiency")
    usable_j = x.req.battery_wh * number("battery_usable_frac") * 3600.0
    pid = "P2" if x.brief.endurance_min is not None else "P3"
    trims = [t for t in x.aero.trims if t.point == pid]
    p_elec = max(_power_w(x, t.cd, t.speed_ms) for t in trims) / eta
    if x.brief.endurance_min is not None:
        need_j = p_elec * x.brief.endurance_min * 60.0
    else:
        need_j = p_elec * float(x.brief.range_km or 0.0) * 1000.0 / trims[0].speed_ms
    return _result(
        "energy",
        {"energy": (usable_j - need_j) / need_j},
        {"electric_power_w": p_elec, "point": pid},
    )


def check_structure(x: Inputs) -> CheckResult:
    """At P5, P6: spar root stress x 1.5 <= allowable, tip deflection <= 10 % of spar length."""
    allow, e_mod = number("spar_allowable_pa"), number("spar_modulus_pa")
    sf, defl_frac = (
        number("structure_safety_factor", "checks"),
        number("tip_deflection_max_frac", "checks"),
    )
    index = {s.name: i for i, s in enumerate(x.design.surfaces)}
    stress_m, defl_m = np.inf, np.inf
    for t in (t for t in x.aero.trims if t.point in STRUCTURE_POINTS):
        for s in wing_surfaces(x.design):
            side = abs(t.surface_lift_n[index[s.name]]) / (2 if s.mirrored else 1)
            span = spar_length_m(s)
            inertia = np.pi * (s.spar.od_m**4 - s.spar.id_m**4) / 64
            stress = side * 4 * span / (3 * np.pi) * (s.spar.od_m / 2) / inertia
            tip = side * span**3 / (8 * e_mod * inertia)
            stress_m = min(stress_m, (allow - sf * stress) / allow)
            defl_m = min(defl_m, (defl_frac * span - tip) / (defl_frac * span))
    return _result("structure", {"stress": float(stress_m), "deflection": float(defl_m)}, {})


def check_efficiency(x: Inputs) -> CheckResult:
    """Cruise L/D >= efficiency_floor x baseline L/D at the same tier."""
    base = x.req.baseline_cruise_ld_tier1 if x.aero.tier == 1 else x.req.baseline_cruise_ld_tier2
    if base is None:
        raise CheckError(f"requirements carry no tier-{x.aero.tier} baseline L/D (run derive)")
    ld = cruise_ld(x.aero)
    need = float(x.brief.efficiency_floor or 0.0) * base
    return _result(
        "efficiency", {"efficiency": (ld - need) / need}, {"cruise_ld": ld, "baseline_ld": base}
    )


def check_fit(x: Inputs) -> CheckResult:
    """Invariants 5-7 re-measured, and propeller shafts clear of surfaces (user, step 4)."""
    return _result("fit", {**fit_margins(x.design, x.req), "shaft": shaft_margin(x.design)}, {})


CHECKS = (
    check_stall,
    check_trim,
    check_static,
    check_lateral,
    check_mass,
    check_power,
    check_energy,
    check_structure,
    check_efficiency,
    check_fit,
)

LATERAL_INDEX = CHECKS.index(check_lateral)


def _report(
    d: Design,
    brief: Brief,
    req: Requirements,
    items: tuple[MassItem, ...],
    aero: AeroResult,
    fin_capped: bool = False,
) -> CheckReport:
    """Every check of d from its masses and one aero evaluation."""
    tier = aero.tier
    x = Inputs(d, resolve(brief), req, items, aero, fin_capped)
    results = [c(x) for c in CHECKS]
    if tier == 2:
        results.insert(LATERAL_INDEX + 1, check_stall_order(x))
    violation = float(sum(max(0.0, -r.margin) for r in results))
    bar = constant("checks", "passable_checks")
    if not isinstance(bar, list):
        raise CheckError("checks.yaml passable_checks must be a list")
    return CheckReport(
        tier=tier,
        feasible=all(r.passed for r in results),
        violation=violation,
        checks=tuple(results),
        passable=all(r.passed for r in results if r.id in bar),
        notes=fidelity_notes(d, tier),
    )


def fidelity_notes(d: Design, tier: int) -> tuple[str, ...]:
    """Model limits that apply to this design at this tier."""
    if not any(s.joined_to for s in d.surfaces):
        return ()
    how = "as separate wings" if tier == 1 else "with a 5 % gap at the joined tip (VLM)"
    return (f"tip-joined surfaces analysed {how}: joint load sharing not modelled",)


def run_checks(
    d: Design, brief: Brief, req: Requirements, tier: int, items: tuple[MassItem, ...]
) -> CheckReport:
    """Every check of d at tier 1 (AeroBuildup) or 2 (VLM + stall order) with given masses,
    without balancing."""
    cgs = cg_range(items)
    aero = tier1(d, req, cgs) if tier == 1 else tier2(d, req, cgs)
    return _report(d, brief, req, items, aero)


def evaluate(
    genome: np.ndarray, brief: Brief, req: Requirements, tier: int
) -> tuple[Design, CheckReport]:
    """Prepare the genome at tier 1 (and re-balance on the tier-2 neutral point for tier 2), then
    check the delivered design; the report carries the adjustments that rebuild it."""
    payload = brief.payload_mass_kg
    p = prepare(genome, req, payload)
    first = p.adjust.body_shift_m
    if tier == 2:
        p = prepare_tier2(genome, req, payload)
    items, aero = prepared_aero(p, req, payload, tier)
    report = _report(p.design, brief, req, items, aero, p.fin_capped)
    return p.design, report.model_copy(
        update={
            "body_shift_m": p.adjust.body_shift_m,
            "fin_scale": p.adjust.fin_scale,
            "fin_scale_capped": p.fin_capped,
            "added_fin": p.adjust.add_fin,
            "incidence_deg": p.adjust.incidence_deg,
            "incidence_rule": p.incidence_rule,
            "battery_shift_m": p.battery_shift_m,
            "stack_shift_m": p.stack_shift_m,
            "shift_outcome": p.shift_outcome,
            "balance": p.balance_outcome,
            "tier1_body_shift_m": first if tier == 2 else None,
        }
    )
