"""Prepare a genome for its checks (user Deviations, step 3 gate r3-r4): body shift on the CG,
pitch control on the longest arm, vertical-area scaling to the Cn_beta target (an added fin when
there is none), box slide, then the pitch surface's incidence for zero deflection at cruise."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from aeroforge.aero import (
    CRUISE_POINT,
    AeroError,
    Samples,
    buildup_samples,
    flight_state,
    lift_n,
    point_speeds,
)
from aeroforge.airplane import quarter_mac_x_m
from aeroforge.balance import Balance, balance, neutral_point_x_m
from aeroforge.constants import number, numbers
from aeroforge.genome import decode
from aeroforge.mass import MassItem, cg_range, mass_items
from aeroforge.models import Adjust, ControlSurface, Design, Requirements, Surface
from aeroforge.style import panels, wing_surfaces
from aeroforge.tier2 import tier2, vlm_samples
from aeroforge.trim import AeroResult, SampleFn, evaluate, lateral, slopes, trim_point

SHIFT_ON_TARGET, SHIFT_CAPPED, NO_BODY = "on_target", "capped", "no_body"
CAPPED_NOTE = "body shift capped"
INCIDENCE_SET, SINGLE_SURFACE, JOINED, BOX_HOST, NO_CONTROL = (
    "set",
    "skipped: single lifting surface",
    "skipped: joined or join target",
    "skipped: hosts boxes",
    "skipped: no pitch control",
)
MEMO = 64


@dataclass(frozen=True)
class Prepared:
    """Checked design, the adjustments that rebuild it, outcomes and its samples."""

    design: Design
    adjust: Adjust
    battery_shift_m: float
    stack_shift_m: float
    shift_outcome: str
    balance_outcome: str
    fin_capped: bool
    incidence_rule: str
    samples: Samples


def with_pitch_control(d: Design, aft_x_m: float, incidence_deg: float) -> Design:
    """d with a control (chord >= pitch_control_chord_frac, full structural span) and the given
    incidence on the lifting surface with the longest pitch arm to aft_x_m."""
    lifting = wing_surfaces(d)
    if not lifting:
        return d
    best = max(lifting, key=lambda s: (abs(quarter_mac_x_m(s) - aft_x_m), s.projected_area_m2))
    lo, hi = numbers("pitch_control_span_frac")
    gene = best.control.chord_frac if best.control is not None else 0.0
    frac = max(number("pitch_control_chord_frac"), gene)
    control = ControlSurface(chord_frac=frac, span_start_frac=lo, span_end_frac=hi)
    update = {"control": control, "incidence_deg": incidence_deg}
    fixed: tuple[Surface, ...] = tuple(
        s.model_copy(update=update) if s.name == best.name else s for s in d.surfaces
    )
    return d.model_copy(update={"surfaces": fixed})


def build(genome: np.ndarray, req: Requirements, adjust: Adjust, payload_mass_kg: float) -> Design:
    """Decode with the adjustments, then add the pitch control at the decoded aft CG."""
    d = decode(genome, req, adjust)
    return with_pitch_control(d, float(_aft(d, req, payload_mass_kg)[0]), adjust.incidence_deg)


def _aft(d: Design, req: Requirements, payload_mass_kg: float) -> np.ndarray:
    """Aft CG [m]."""
    return cg_range(mass_items(d, req, payload_mass_kg))[1]


def _ref_x(d: Design) -> float:
    """Surface-fixed reference: surface0's root leading-edge x [m]."""
    return float(d.surfaces[0].sections[0].le_m[0])


def _speeds(req: Requirements) -> tuple[np.ndarray, float]:
    """Flight-point speeds and the cruise speed [m/s]."""
    return point_speeds(req), next(p.speed_ms for p in req.points if p.id == CRUISE_POINT)


def _samples(d: Design, req: Requirements, payload_mass_kg: float, sampler: SampleFn) -> Samples:
    """Sampler output for d at its aft CG."""
    speeds, cruise = _speeds(req)
    return sampler(d, req, speeds, cruise, _aft(d, req, payload_mass_kg))


def _vlm_sampler(
    d: Design, req: Requirements, speeds_ms: np.ndarray, cruise_ms: float, ref_m: np.ndarray
) -> Samples:
    """Tier-2 samples (VLM wings + buildup fuselages)."""
    return vlm_samples(d, req, speeds_ms, cruise_ms, ref_m)[0]


def _moved_fraction(d: Design, req: Requirements, payload_mass_kg: float) -> float:
    """Mass share that moves with body0: shell, boxes and body-hosted propulsion."""
    movers = {"body0", *(b.name for b in d.boxes)}
    hosted = [p.name for p in d.propulsors if p.host == "body0"]
    movers |= {f"{name}_{part}" for name in hosted for part in ("drive", "prop")}
    items = mass_items(d, req, payload_mass_kg)
    return sum(it.mass_kg for it in items if it.name in movers) / sum(it.mass_kg for it in items)


def _miss_m(d: Design, req: Requirements, payload_mass_kg: float, target_m: float) -> float:
    """Aft-CG x behind surface0's root minus the target [m]."""
    return float(_aft(d, req, payload_mass_kg)[0]) - _ref_x(d) - target_m


def _solve_shift(
    genome: np.ndarray, req: Requirements, payload_mass_kg: float, adjust: Adjust, target_m: float
) -> tuple[float, str]:
    """Body shift putting the aft CG target_m behind surface0's root (secant on CG alone)."""
    d = decode(genome, req, adjust)
    if not any(b.name == "body0" for b in d.bodies):
        return 0.0, NO_BODY
    tol = number("balance_target_tol") * flight_state(d, req).c_ref_m
    s, miss = adjust.body_shift_m, _miss_m(d, req, payload_mass_kg, target_m)
    slope = _moved_fraction(d, req, payload_mass_kg)
    for _ in range(int(number("shift_iter"))):
        if abs(miss) <= tol:
            return s, SHIFT_ON_TARGET
        s_new = s - miss / slope
        d = decode(genome, req, adjust.model_copy(update={"body_shift_m": s_new}))
        if any(n.startswith(CAPPED_NOTE) for n in d.notes):
            return s_new, SHIFT_CAPPED
        miss_new = _miss_m(d, req, payload_mass_kg, target_m)
        if s_new != s and miss_new != miss:
            slope = (miss_new - miss) / (s_new - s)
        s, miss = s_new, miss_new
    return s, SHIFT_ON_TARGET if abs(miss) <= tol else SHIFT_CAPPED


def _target_m(d: Design, req: Requirements, payload_mass_kg: float, sm: Samples) -> float:
    """Aft-CG x behind surface0's root that gives static_margin_target [m]."""
    np_x = neutral_point_x_m(sm, req, d, _aft(d, req, payload_mass_kg))
    c = flight_state(d, req).c_ref_m
    return np_x - _ref_x(d) - number("static_margin_target", "checks") * c


def _shifted(
    genome: np.ndarray, req: Requirements, payload_mass_kg: float, adjust: Adjust, sampler: SampleFn
) -> tuple[Adjust, str]:
    """adjust with the body shift re-solved on the sampler's NP of the design it builds."""
    d = build(genome, req, adjust, payload_mass_kg)
    target = _target_m(d, req, payload_mass_kg, _samples(d, req, payload_mass_kg, sampler))
    shift, outcome = _solve_shift(genome, req, payload_mass_kg, adjust, target)
    return adjust.model_copy(update={"body_shift_m": shift}), outcome


def _cn_beta(d: Design, req: Requirements, payload_mass_kg: float) -> float:
    """Fresh tier-1 Cn_beta [1/rad] about the aft CG at cruise."""
    sm = _samples(d, req, payload_mass_kg, buildup_samples)
    speeds, cruise = _speeds(req)
    s = int(np.flatnonzero(speeds == cruise)[0])
    return lateral(sm, s, _aft(d, req, payload_mass_kg), flight_state(d, req), cruise)[0]


def has_vertical_area(d: Design) -> bool:
    """Any panel canted over vertical_cant_deg (fins, V-tails, steep outer panels)."""
    steep = number("vertical_cant_deg")
    return any(abs(p.cant_deg) > steep for s in d.surfaces for p in panels(s))


def _vertical_scale(
    genome: np.ndarray, req: Requirements, payload_mass_kg: float, adjust: Adjust
) -> tuple[Adjust, bool]:
    """adjust with an added fin when there is no vertical area and the vertical-area factor that
    meets cn_beta_target (linear in the factor, at most fin_scale_max); whether the cap binds."""
    d = build(genome, req, adjust, payload_mass_kg)
    if not has_vertical_area(d):
        adjust = adjust.model_copy(update={"add_fin": True})
        d = build(genome, req, adjust, payload_mass_kg)
    target, cap = number("cn_beta_target", "checks"), number("fin_scale_max")
    base = _cn_beta(d, req, payload_mass_kg)
    if base >= target:
        return adjust, False
    k0 = number("fin_scale_trial")
    trial = build(genome, req, adjust.model_copy(update={"fin_scale": k0}), payload_mass_kg)
    slope = (_cn_beta(trial, req, payload_mass_kg) - base) / (k0 - 1)
    needed = 1.0 + (target - base) / slope if slope > 0.0 else np.inf
    return adjust.model_copy(update={"fin_scale": float(min(needed, cap))}), bool(needed > cap)


def _incidence_rule(d: Design, pitch: str | None) -> str:
    """Why the pitch surface may not take an incidence, else INCIDENCE_SET."""
    if pitch is None:
        return NO_CONTROL
    if len(wing_surfaces(d)) < 2:
        return SINGLE_SURFACE
    s = next(s for s in d.surfaces if s.name == pitch)
    if s.joined_to is not None or any(o.joined_to == pitch for o in d.surfaces):
        return JOINED
    if any(b.host == pitch for b in d.boxes):
        return BOX_HOST
    return INCIDENCE_SET


def _incidence_deg(d: Design, req: Requirements, items: tuple[MassItem, ...], sm: Samples) -> float:
    """Pitch-surface incidence [deg] trimming cruise at the mid CG with zero deflection: the
    incidence acts as that surface's own angle of attack (force slope at its quarter-MAC point)."""
    mid = np.mean(cg_range(items), axis=0)
    speeds, cruise = _speeds(req)
    si = int(np.flatnonzero(speeds == cruise)[0])
    k = slopes(sm, si, mid)
    idx = [s.name for s in d.surfaces].index(str(sm.pitch_control))
    a0, a1 = sm.alpha_deg
    d_force = (sm.surface_force_n[si, 1, idx] - sm.surface_force_n[si, 0, idx]) / (a1 - a0)
    s = d.surfaces[idx]
    ac = np.array([quarter_mac_x_m(s), 0.0, np.mean([c.le_m[2] for c in s.sections])])
    arm = ac - mid
    li, mi = float(lift_n(d_force, a0)), float(arm[2] * d_force[0] - arm[0] * d_force[2])
    system = np.array([[k.la, li], [k.ma, mi]])
    rhs = np.array([flight_state(d, req).weight_n - k.l0, -k.m0])
    return float(np.linalg.solve(system, rhs)[1])


def _tilted(d: Design, name: str, incidence_deg: float) -> Design:
    """d with the named surface at incidence_deg."""
    surfaces = tuple(
        s.model_copy(update={"incidence_deg": incidence_deg}) if s.name == name else s
        for s in d.surfaces
    )
    return d.model_copy(update={"surfaces": surfaces})


def _cruise_deflection_deg(
    d: Design, req: Requirements, payload_mass_kg: float, sm: Samples
) -> float:
    """Pitch deflection [deg] trimming cruise P3 at the mid CG, from sm."""
    mid = np.mean(cg_range(mass_items(d, req, payload_mass_kg)), axis=0)
    cruise = next(p for p in req.points if p.id == CRUISE_POINT)
    return trim_point(sm, cruise, 1.0, "mid", mid, flight_state(d, req)).deflection_deg


def _measured_deg(bal: Balance, req: Requirements, payload_mass_kg: float) -> float:
    """|cruise deflection| [deg] of a balanced design; inf when it does not trim."""
    try:
        deflection = _cruise_deflection_deg(bal.design, req, payload_mass_kg, bal.samples)
    except AeroError:
        return float("inf")
    return abs(deflection) if np.isfinite(deflection) else float("inf")


def _next_incidence(
    tried: list[tuple[float, float, float]], bal: Balance, req: Requirements, payload_mass_kg: float
) -> float:
    """Linear estimate first, then a secant on the last two (incidence, signed deflection)."""
    if len(tried) == 1:
        items = mass_items(bal.design, req, payload_mass_kg)
        return tried[0][0] + _incidence_deg(bal.design, req, items, bal.samples)
    (i0, d0, _), (i1, d1, _) = tried[-2:]
    return i1 - d1 * (i1 - i0) / (d1 - d0) if d1 != d0 else i1


def _finish(
    genome: np.ndarray,
    req: Requirements,
    payload_mass_kg: float,
    adjust: Adjust,
    outcomes: tuple[str, bool],
    sampler: SampleFn,
) -> Prepared:
    """Build and slide the boxes on the sampler's NP; then drive the measured cruise deflection
    at the mid CG to zero with the pitch surface's incidence (linear first step, then secant),
    re-sliding the boxes each round because the NP moves with the incidence. The round with the
    smallest measured deflection is kept (zero incidence included), so a diverging or
    untrimmable step never replaces a better one."""
    d = build(genome, req, adjust.model_copy(update={"incidence_deg": 0.0}), payload_mass_kg)
    bal = balance(d, req, payload_mass_kg, sampler, _samples(d, req, payload_mass_kg, sampler))
    rule = _incidence_rule(bal.design, bal.samples.pitch_control)
    rounds: list[tuple[float, float, Balance, np.ndarray]] = []
    slides, incidence = np.array([bal.battery_shift_m, bal.stack_shift_m]), 0.0
    tried: list[tuple[float, float, float]] = []
    for _ in range(int(number("incidence_rounds")) + 1 if rule == INCIDENCE_SET else 1):
        size = _measured_deg(bal, req, payload_mass_kg)
        rounds.append((size, incidence, bal, slides.copy()))
        if size <= number("incidence_tol_deg") or not np.isfinite(size) or rule != INCIDENCE_SET:
            break
        signed = _cruise_deflection_deg(bal.design, req, payload_mass_kg, bal.samples)
        tried.append((incidence, signed, size))
        incidence = _next_incidence(tried, bal, req, payload_mass_kg)
        if not np.isfinite(incidence):
            break
        tilted = _tilted(bal.design, str(bal.samples.pitch_control), incidence)
        bal = balance(
            tilted, req, payload_mass_kg, sampler, _samples(tilted, req, payload_mass_kg, sampler)
        )
        slides = np.add(slides, (bal.battery_shift_m, bal.stack_shift_m))
    _, incidence, bal, slides = min(rounds, key=lambda r: r[0])
    final = adjust.model_copy(update={"incidence_deg": incidence})
    shift_outcome, capped = outcomes
    return Prepared(
        bal.design,
        final,
        float(slides[0]),
        float(slides[1]),
        shift_outcome,
        bal.outcome,
        capped,
        rule,
        bal.samples,
    )


@lru_cache(maxsize=MEMO)
def _prepare_memo(genome_bytes: bytes, req_json: str, payload_mass_kg: float) -> Prepared:
    """prepare() keyed on hashable inputs (derive's final baselines and later checks share it)."""
    genome = np.frombuffer(genome_bytes)
    req = Requirements.model_validate_json(req_json)
    adjust, outcome = _shifted(genome, req, payload_mass_kg, Adjust(), buildup_samples)
    adjust, capped = _vertical_scale(genome, req, payload_mass_kg, adjust)
    if adjust.fin_scale != 1.0:
        adjust, outcome = _shifted(genome, req, payload_mass_kg, adjust, buildup_samples)
    return _finish(genome, req, payload_mass_kg, adjust, (outcome, capped), buildup_samples)


def prepare(genome: np.ndarray, req: Requirements, payload_mass_kg: float) -> Prepared:
    """Tier-1 preparation: shift, pitch control, vertical-area scaling (re-shift if scaled), box
    slide, incidence."""
    return _prepare_memo(
        np.asarray(genome, float).tobytes(), req.model_dump_json(), payload_mass_kg
    )


@lru_cache(maxsize=MEMO)
def _tier2_memo(genome_bytes: bytes, req_json: str, payload_mass_kg: float) -> Prepared:
    """prepare_tier2() keyed on hashable inputs."""
    genome = np.frombuffer(genome_bytes)
    req = Requirements.model_validate_json(req_json)
    first = _prepare_memo(genome_bytes, req_json, payload_mass_kg)
    start = first.adjust.model_copy(update={"incidence_deg": 0.0})
    adjust, outcome = _shifted(genome, req, payload_mass_kg, start, _vlm_sampler)
    return _finish(genome, req, payload_mass_kg, adjust, (outcome, first.fin_capped), _vlm_sampler)


def prepare_tier2(genome: np.ndarray, req: Requirements, payload_mass_kg: float) -> Prepared:
    """Re-balance on the tier-2 NP (shift, box slide, incidence; tier-1 vertical scaling kept)."""
    return _tier2_memo(np.asarray(genome, float).tobytes(), req.model_dump_json(), payload_mass_kg)


def prepared_aero(
    p: Prepared, req: Requirements, payload_mass_kg: float, tier: int
) -> tuple[tuple[MassItem, ...], AeroResult]:
    """Masses and aero of a prepared design (tier 1 re-uses its samples)."""
    items = mass_items(p.design, req, payload_mass_kg)
    cgs = cg_range(items)
    if tier == 1:
        return items, evaluate(p.design, req, cgs, lambda *_: p.samples, 1)
    return items, tier2(p.design, req, cgs)
