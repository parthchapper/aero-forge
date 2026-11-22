"""Balance a design (user Deviation, step 3 gate): slide the battery, then the whole box stack,
along x inside its host so the aft-CG static margin meets static_margin_target."""

from dataclasses import dataclass

import numpy as np

from aeroforge.aero import CRUISE_POINT, Samples, flight_state, point_speeds
from aeroforge.airplane import pitch_surface
from aeroforge.constants import number, numbers
from aeroforge.fitcheck import box_points_inside
from aeroforge.mass import BATTERY, PAYLOAD, MassItem, cg_range, mass_items, move_item
from aeroforge.models import Design, Requirements
from aeroforge.trim import SampleFn, slopes

ON_TARGET, IN_BAND, OUT_OF_BAND = "on_target", "clamped_in_band", "out_of_band"
GAP_TOL_M = 1e-9  # round-off allowed on the box gap of the decoded stack


@dataclass(frozen=True)
class Balance:
    """Balanced design, applied slides [m], outcome and its tier-1 samples."""

    design: Design
    battery_shift_m: float
    stack_shift_m: float
    outcome: str
    samples: Samples


def _moved(d: Design, names: tuple[str, ...], dx_m: float) -> Design:
    """d with the named boxes slid by dx_m along x."""
    boxes = tuple(
        b.model_copy(update={"centre_m": (b.centre_m[0] + dx_m, b.centre_m[1], b.centre_m[2])})
        if b.name in names
        else b
        for b in d.boxes
    )
    return d.model_copy(update={"boxes": boxes})


def _fits(d: Design) -> bool:
    """Every box inside its host and boxes on one host apart by at least box_gap_m along x."""
    if not all(box_points_inside(d, b).all() for b in d.boxes):
        return False
    gap = number("box_gap_m")
    for i, a in enumerate(d.boxes):
        for b in d.boxes[i + 1 :]:
            apart = abs(a.centre_m[0] - b.centre_m[0]) - (a.size_m[0] + b.size_m[0]) / 2
            if a.host == b.host and apart < gap - GAP_TOL_M:
                return False
    return True


def _reach(d: Design, names: tuple[str, ...], dx_m: float) -> float:
    """Signed slide toward dx_m, as far as it stays fitting (bisection ends on the fitting side)."""
    if dx_m == 0.0 or _fits(_moved(d, names, dx_m)):
        return dx_m
    ok, bad = 0.0, dx_m
    for _ in range(int(number("balance_bisect_iter"))):
        mid = (ok + bad) / 2
        ok, bad = (mid, bad) if _fits(_moved(d, names, mid)) else (ok, mid)
    return ok


def _configs(items: tuple[MassItem, ...]) -> list[tuple[MassItem, ...]]:
    """The cg_range loadings: payload on/off x battery at -/+ its position tolerance."""
    tol = number("battery_x_tol_m")
    out = []
    for payload_on in (True, False):
        kept = items if payload_on else tuple(it for it in items if it.name != PAYLOAD)
        out += [move_item(kept, BATTERY, dx) for dx in (-tol, tol)]
    return out


def _slide_to(items: tuple[MassItem, ...], names: tuple[str, ...], target_x_m: float) -> float:
    """Slide of the named items putting the aft CG (max over loadings) exactly at target_x_m."""
    best = np.inf
    for cfg in _configs(items):
        m = np.array([it.mass_kg for it in cfg])
        x = np.array([it.xyz_m[0] for it in cfg])
        moved = sum(it.mass_kg for it in cfg if it.name in names)
        best = min(best, (target_x_m - float((m * x).sum() / m.sum())) * m.sum() / moved)
    return float(best)


def neutral_point_x_m(sm: Samples, req: Requirements, d: Design, cg_m: np.ndarray) -> float:
    """Mean over the flight-point speeds of the tier-1 neutral point x [m]."""
    c = flight_state(d, req).c_ref_m
    margins = [
        -(k.ma / (k.la * c)) for k in (slopes(sm, s, cg_m) for s in range(len(sm.speeds_ms)))
    ]
    return float(cg_m[0] + c * np.mean(margins))


def _outcome(target_x_m: float, aft_x_m: float, np_x_m: float, c_m: float) -> str:
    """Target hit, clamped but inside the static band, or out of it."""
    lo, hi = numbers("static_margin_band", "checks")
    sm = (np_x_m - aft_x_m) / c_m
    if abs(aft_x_m - target_x_m) / c_m <= number("balance_target_tol"):
        return ON_TARGET
    return IN_BAND if lo <= sm <= hi else OUT_OF_BAND


def _slide(
    d: Design, req: Requirements, payload_mass_kg: float, target_x_m: float
) -> tuple[Design, float, float]:
    """One battery slide then one stack slide toward target_x_m: (design, battery dx, stack dx)."""
    to_battery = _slide_to(mass_items(d, req, payload_mass_kg), (BATTERY,), target_x_m)
    battery_dx = _reach(d, (BATTERY,), to_battery)
    d1 = _moved(d, (BATTERY,), battery_dx)
    stack = tuple(b.name for b in d.boxes)
    stack_dx = _reach(d1, stack, _slide_to(mass_items(d1, req, payload_mass_kg), stack, target_x_m))
    return _moved(d1, stack, stack_dx), battery_dx, stack_dx


def balance(
    d: Design, req: Requirements, payload_mass_kg: float, sampler: SampleFn, sm: Samples
) -> Balance:
    """Box slide (battery, then stack, cumulative) toward the mid-band static margin; sm are the
    sampler's samples of d at its aft CG (resampled only if the pitch surface changes).

    The neutral point x_cg - M'/L' drifts slightly with the CG (moment transfer uses the normal
    force, the slope uses lift), so the slide repeats from the same samples until on target.
    """
    speeds = point_speeds(req)
    cruise = next(p.speed_ms for p in req.points if p.id == CRUISE_POINT)
    aft = cg_range(mass_items(d, req, payload_mass_kg))[1]
    c = flight_state(d, req).c_ref_m
    shifts = np.zeros(2)
    for _ in range(int(number("balance_np_iter"))):
        target = neutral_point_x_m(sm, req, d, aft) - number("static_margin_target", "checks") * c
        if abs(aft[0] - target) / c <= number("balance_target_tol"):
            break
        d, battery_dx, stack_dx = _slide(d, req, payload_mass_kg, target)
        shifts += (battery_dx, stack_dx)
        moved_aft = cg_range(mass_items(d, req, payload_mass_kg))[1]
        if moved_aft[0] == aft[0]:
            break
        aft = moved_aft
    np_x = neutral_point_x_m(sm, req, d, aft)
    target = np_x - number("static_margin_target", "checks") * c
    pitch = pitch_surface(d, float(aft[0]))
    if (pitch.name if pitch else None) != sm.pitch_control:
        sm = sampler(d, req, speeds, cruise, aft)
    outcome = _outcome(target, float(aft[0]), np_x, c)
    return Balance(d, float(shifts[0]), float(shifts[1]), outcome, sm)
