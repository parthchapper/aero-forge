"""Brief → Requirements: flight points and MTOW / S / battery closure on the baseline (SPEC §8)."""

from dataclasses import dataclass, replace

import aerosandbox as asb
import numpy as np

from aeroforge.brief import launch_speed_ms, resolve
from aeroforge.constants import number
from aeroforge.genome import DecodeError, decode
from aeroforge.mass import installed_power_w, total_mass_kg
from aeroforge.models import Brief, Design, FlightPoint, Requirements
from aeroforge.prepare import Prepared, prepare, prepare_tier2, prepared_aero
from aeroforge.reference import BASELINE, reference_genome
from aeroforge.trim import AeroResult, cruise_ld

LOITER_POINT, DASH_POINT = "P2", "P4"


class InfeasibleError(RuntimeError):
    """The brief cannot be met; `check` names the SPEC §8 check that rules it out."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


class ClosureError(InfeasibleError):
    """The mass closure did not converge within derive_max_iter iterations."""


@dataclass(frozen=True)
class Guess:
    """One closure iterate: MTOW [kg], battery [Wh], aircraft CLmax, loiter speed [m/s]."""

    mtow_kg: float
    battery_wh: float
    clmax: float
    loiter_ms: float


def stall_target_ms(b: Brief) -> float:
    """Stall speed the baseline is sized for [m/s]: the launch, stall_max and cruise-CL limits."""
    cl_frac = number("trim_cl_max_frac", "checks")
    limits = (
        float(b.stall_max_ms or 0.0),
        launch_speed_ms(b) / number("launch_speed_stall_ratio", "checks"),
        b.cruise_speed_ms * np.sqrt(cl_frac),
    )
    return min(limits) / number("stall_speed_margin")


def _points(b: Brief, vs_ms: float, loiter_ms: tuple[float, ...]) -> tuple[FlightPoint, ...]:
    """P1-P6; several P2 entries while the loiter speed is being searched."""
    n_limit = float(b.n_limit or 0.0)
    pull_up = (
        vs_ms
        * np.sqrt(n_limit / number("trim_cl_max_frac", "checks"))
        * number("pullup_speed_margin")
    )
    dash = float(b.dash_speed_ms or 0.0)
    return (
        FlightPoint(id="P1", speed_ms=launch_speed_ms(b), load_factor=1.0),
        *(FlightPoint(id=LOITER_POINT, speed_ms=v, load_factor=1.0) for v in loiter_ms),
        FlightPoint(id="P3", speed_ms=b.cruise_speed_ms, load_factor=1.0),
        FlightPoint(id="P4", speed_ms=dash, load_factor=1.0),
        FlightPoint(id="P5", speed_ms=max(dash, float(pull_up)), load_factor=n_limit),
        FlightPoint(
            id="P6", speed_ms=b.cruise_speed_ms, load_factor=1.0, gust_ms=float(b.gust_ms or 0.0)
        ),
    )


def _loiter_candidates(b: Brief, vs_ms: float) -> tuple[float, ...]:
    """Speeds searched for the minimum-power loiter point [m/s]."""
    lo = min(number("loiter_stall_ratio") * vs_ms, b.cruise_speed_ms)
    return tuple(
        float(v) for v in np.linspace(lo, b.cruise_speed_ms, int(number("loiter_speed_samples")))
    )


def _requirements(b: Brief, g: Guess, vs_ms: float, loiter: tuple[float, ...]) -> Requirements:
    """Requirements for one iterate; the stall speed fixes S, the span limit bounds it."""
    rho = float(asb.Atmosphere(altitude=float(b.altitude_m or 0.0)).density())
    area = 2 * g.mtow_kg * number("gravity_m_s2") / (rho * vs_ms**2 * g.clmax)
    area_max = b.span_max_m**2 / number("derive_min_aspect_ratio")
    if area > area_max:
        raise InfeasibleError(
            "stall",
            f"stall speed {vs_ms:.2f} m/s needs {area:.2f} m2 of wing;"
            f" span_max allows {area_max:.2f} m2",
        )
    return Requirements(
        mtow_kg=g.mtow_kg,
        lifting_area_min_m2=area,
        span_max_m=b.span_max_m,
        ref_length_m=number("ref_length_per_sqrt_area") * float(np.sqrt(area)),
        battery_wh=g.battery_wh,
        payload_box_m=b.payload_box_m,
        recovery=b.recovery,
        altitude_m=float(b.altitude_m or 0.0),
        points=_points(b, vs_ms, loiter),
    )


def _electric_w(aero: AeroResult, point: str, speed_ms: float) -> float:
    """Worst-CG electrical power at one point and speed [W]."""
    f = aero.flight
    eta = number("prop_efficiency") * number("motor_efficiency") * number("esc_efficiency")
    cds = [t.cd for t in aero.trims if t.point == point and t.speed_ms == speed_ms]
    return max(cds) * 0.5 * f.rho_kg_m3 * speed_ms**3 * f.s_ref_m2 / eta


def _battery_wh(b: Brief, aero: AeroResult, loiter_ms: float) -> float:
    """Pack energy for the mission at the loiter (endurance) or cruise (range) point [Wh]."""
    if b.endurance_min is not None:
        need_j = _electric_w(aero, LOITER_POINT, loiter_ms) * b.endurance_min * 60.0
    else:
        need_j = (
            _electric_w(aero, "P3", b.cruise_speed_ms)
            * float(b.range_km or 0.0)
            * 1000.0
            / b.cruise_speed_ms
        )
    return number("battery_sizing_margin") * need_j / (number("battery_usable_frac") * 3600.0)


def baseline_design(req: Requirements) -> Design:
    """The `conventional` reference decoded under req."""
    try:
        return decode(reference_genome(BASELINE), req)
    except DecodeError as exc:
        raise InfeasibleError("fit", f"baseline does not decode: {exc}") from exc


def _iterate(b: Brief, g: Guess, vs_ms: float) -> Guess:
    """Next closure iterate from the baseline decoded under the current one.

    The new battery's mass replaces the current one's in the MTOW sum (Gauss-Seidel): MTOW lagging
    the battery by a step made the closure oscillate and take 9 of its 10 iterations.
    """
    candidates = _loiter_candidates(b, vs_ms)
    req = _requirements(b, g, vs_ms, candidates)
    items, aero = prepared_aero(
        baseline_prepared(req, b.payload_mass_kg), req, b.payload_mass_kg, 1
    )
    loiter = min(candidates, key=lambda v: _electric_w(aero, LOITER_POINT, v))
    battery = _battery_wh(b, aero, loiter)
    swap_kg = (battery - g.battery_wh) / number("battery_specific_energy_wh_per_kg")
    return Guess(
        mtow_kg=_margins() * (total_mass_kg(items) + swap_kg),
        battery_wh=battery,
        clmax=aero.clmax,
        loiter_ms=loiter,
    )


def _converged(a: Guess, b: Guess) -> bool:
    """Relative change of MTOW, battery and CLmax within derive_rel_tol."""
    tol = number("derive_rel_tol")
    pairs = ((a.mtow_kg, b.mtow_kg), (a.battery_wh, b.battery_wh), (a.clmax, b.clmax))
    return all(abs(y - x) <= tol * abs(x) for x, y in pairs)


def _aitken(x0: float, x1: float, x2: float) -> float:
    """Aitken extrapolation of a monotone, linearly converging sequence; x2 when it is not."""
    d1, d2 = x1 - x0, x2 - x1
    if d1 == 0.0 or not 0.0 < d2 / d1 < 1.0:
        return x2
    return x2 - d2 * d2 / (d2 - d1)


def _accelerate(x0: Guess, x1: Guess, x2: Guess) -> Guess:
    """Steffensen step on MTOW and battery (the closure contracts by ~0.6 per plain step)."""
    return replace(
        x2,
        mtow_kg=_aitken(x0.mtow_kg, x1.mtow_kg, x2.mtow_kg),
        battery_wh=_aitken(x0.battery_wh, x1.battery_wh, x2.battery_wh),
    )


def derive(brief: Brief) -> Requirements:
    """Requirements fixed once per run by fixed-point closure on the baseline.

    Convergence is judged on a plain step (input vs its image); every third iterate is an
    Aitken extrapolation of the two plain steps before it.
    """
    b = resolve(brief)
    vs = stall_target_ms(b)
    mtow0 = number("derive_mtow_payload_ratio") * b.payload_mass_kg
    g = Guess(
        mtow0, number("derive_battery_wh_per_kg") * mtow0, number("derive_clmax_initial"), 0.0
    )
    ceiling = min(number("derive_mtow_ceiling_kg"), b.mass_max_kg or np.inf)
    chain = [g]
    for _ in range(int(number("derive_max_iter"))):
        nxt = _iterate(b, g, vs)
        if nxt.mtow_kg > ceiling:
            raise InfeasibleError(
                "mass", f"closure MTOW {nxt.mtow_kg:.2f} kg exceeds {ceiling:.2f} kg"
            )
        if _converged(g, nxt):
            final = _requirements(b, g, vs, (nxt.loiter_ms,))
            first = baseline_prepared(final, b.payload_mass_kg)
            aero = prepared_aero(first, final, b.payload_mass_kg, 1)[1]
            _require_dash_power(b, final, aero)
            return _with_baselines(final, b.payload_mass_kg, aero)
        chain.append(nxt)
        g = _accelerate(*chain) if len(chain) == 3 else nxt
        chain = [g] if len(chain) == 3 else chain
    raise ClosureError(
        "mass", f"MTOW closure did not converge in {int(number('derive_max_iter'))} iterations"
    )


def _require_dash_power(b: Brief, req: Requirements, aero: AeroResult) -> None:
    """Stop a brief whose baseline cannot come near the dash speed: electrical power at the dash
    point over installed_power_w (mass.py) above derive_power_infeasible_ratio. Search would only
    rank designs by how little drag they have (user, novelty plan step 8)."""
    dash = float(b.dash_speed_ms or 0.0)
    need_w = _electric_w(aero, DASH_POINT, dash)
    installed, limit = installed_power_w(req), number("derive_power_infeasible_ratio")
    if need_w > limit * installed:
        raise InfeasibleError(
            "power",
            f"the baseline needs {need_w:.0f} W electrical at the {dash:.0f} m/s dash,"
            f" {need_w / installed:.1f} x the {installed:.0f} W installed (limit {limit:g} x):"
            " lower the dash speed or raise the mass limit",
        )


def _with_baselines(req: Requirements, payload_mass_kg: float, aero: AeroResult) -> Requirements:
    """req with the balanced baseline's cruise L/D at both tiers (efficiency compares within a
    tier); aero is the tier-1 baseline's aerodynamics under req."""
    second = prepare_tier2(reference_genome(BASELINE), req, payload_mass_kg)
    return req.model_copy(
        update={
            "baseline_cruise_ld_tier1": cruise_ld(aero),
            "baseline_cruise_ld_tier2": cruise_ld(
                prepared_aero(second, req, payload_mass_kg, 2)[1]
            ),
        }
    )


def _margins() -> float:
    """MTOW / sum of baseline masses: growth and reserve (user Deviation, step 3 gate r3)."""
    return (1 + number("mass_growth_frac")) * (1 + number("mass_reserve_frac"))


def baseline_prepared(req: Requirements, payload_mass_kg: float) -> Prepared:
    """The `conventional` reference through the same preparation as every candidate."""
    try:
        return prepare(reference_genome(BASELINE), req, payload_mass_kg)
    except DecodeError as exc:
        raise InfeasibleError("fit", f"baseline does not decode: {exc}") from exc
