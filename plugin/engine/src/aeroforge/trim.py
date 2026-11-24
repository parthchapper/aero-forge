"""Trim every flight point at fore and aft CG from linearised samples (SPEC §8)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aeroforge.aero import (
    CRUISE_POINT,
    STALL_POINT,
    AeroError,
    Flight,
    Samples,
    Trim,
    aircraft_clmax,
    buildup_samples,
    flight_points,
    flight_state,
    gust_load_factor,
    lift_n,
    pitch_nm,
    point_speeds,
    stall_load_factor,
)
from aeroforge.constants import number
from aeroforge.models import Design, FlightPoint, Requirements

CG_LABELS = ("fore", "aft")
SINGULAR_TOL = 1e-12  # relative determinant below which the trim system has no solution

SampleFn = Callable[[Design, Requirements, np.ndarray, float, np.ndarray], Samples]


@dataclass(frozen=True)
class StallOrder:
    """Tier-2 stall order at P1: cl/clmax peaks and the peak's semi-span fraction."""

    fore_ratio: float
    other_ratio: float
    peak_span_frac: float


@dataclass(frozen=True)
class AeroResult:
    """Trimmed points and stability/stall quantities of one design at one tier."""

    tier: int
    trims: tuple[Trim, ...]
    clmax: float
    stall_speed_ms: float
    cl_alpha_per_rad: float
    cnb_per_rad: float
    clb_per_rad: float
    flight: Flight
    pitch_control: str | None
    stall_order: tuple[StallOrder, ...]


@dataclass(frozen=True)
class Slopes:
    """Lift [N] and pitch moment about a CG [N m] at alpha lo, per degree alpha and deflection."""

    l0: float
    la: float
    m0: float
    ma: float
    ld: float
    md: float


def slopes(sm: Samples, s: int, cg_m: np.ndarray) -> Slopes:
    """Linear lift and pitch-moment model at speed index s about cg_m."""
    a0, a1 = sm.alpha_deg
    f0, f1 = sm.force_n[s]
    m0 = pitch_nm(sm.moment_nm[s, 0], f0, sm.xyz_ref_m, cg_m)
    m1 = pitch_nm(sm.moment_nm[s, 1], f1, sm.xyz_ref_m, cg_m)
    l0, l1 = float(lift_n(f0, a0)), float(lift_n(f1, a1))
    return Slopes(
        l0=l0,
        la=(l1 - l0) / (a1 - a0),
        m0=m0,
        ma=(m1 - m0) / (a1 - a0),
        ld=float(lift_n(sm.d_force_n[s], a0)),
        md=pitch_nm(sm.d_moment_nm[s], sm.d_force_n[s], sm.xyz_ref_m, cg_m),
    )


def _solve(k: Slopes, target_n: float, has_control: bool) -> tuple[float, float]:
    """(alpha increment over alpha lo, deflection) [deg] meeting lift and zero pitch moment."""
    if not has_control:
        return (target_n - k.l0) / k.la, 0.0
    det = k.la * k.md - k.ld * k.ma
    if abs(det) <= SINGULAR_TOL * abs(k.la * k.md) or det == 0.0:
        return (target_n - k.l0) / k.la, float("nan")
    da = ((target_n - k.l0) * k.md + k.ld * k.m0) / det
    dd = (-k.m0 * k.la - k.ma * (target_n - k.l0)) / det
    return da, dd


def trim_point(
    sm: Samples, p: FlightPoint, n_load: float, label: str, cg_m: np.ndarray, f: Flight
) -> Trim:
    """Alpha and pitch deflection giving lift n W and zero pitch moment about cg_m."""
    s = int(np.flatnonzero(sm.speeds_ms == p.speed_ms)[0])
    k = slopes(sm, s, cg_m)
    if k.la <= 0.0:
        raise AeroError(f"no positive lift slope at {p.id}")
    q_s = 0.5 * f.rho_kg_m3 * p.speed_ms**2 * f.s_ref_m2
    da, dd = _solve(k, n_load * f.weight_n, sm.pitch_control is not None)
    d_eff = 0.0 if np.isnan(dd) else dd
    alpha = float(sm.alpha_deg[0] + da)
    a_span = sm.alpha_deg[1] - sm.alpha_deg[0]
    profile = sm.profile_n[s, 0] + da * (sm.profile_n[s, 1] - sm.profile_n[s, 0]) / a_span
    profile = max(profile, 0.0) + abs(d_eff) * max(sm.d_profile_n[s], 0.0)
    cl = n_load * f.weight_n / q_s
    surf = (
        sm.surface_force_n[s, 0]
        + da * (sm.surface_force_n[s, 1] - sm.surface_force_n[s, 0]) / a_span
    )
    surf = surf + d_eff * sm.d_surface_force_n[s]
    return Trim(
        point=p.id,
        cg=label,
        speed_ms=p.speed_ms,
        load_factor=n_load,
        alpha_deg=alpha,
        deflection_deg=float(dd),
        cl=cl,
        cd=float(profile / q_s + sm.induced_k * cl**2),
        cm_residual=float((k.m0 + da * k.ma + d_eff * k.md) / (q_s * f.c_ref_m)),
        static_margin=-k.ma / (k.la * f.c_ref_m),
        surface_lift_n=tuple(float(x) for x in lift_n(surf, alpha)),
    )


def lateral(
    sm: Samples, s: int, cg_m: np.ndarray, f: Flight, speed_ms: float
) -> tuple[float, float]:
    """(Cn_beta, Cl_beta) [1/rad] about cg_m from the sideslip sample at speed index s."""
    q_sb = 0.5 * f.rho_kg_m3 * speed_ms**2 * f.s_ref_m2 * f.b_ref_m
    arm = sm.xyz_ref_m - cg_m

    def body(m: np.ndarray, force: np.ndarray) -> np.ndarray:
        total = m + np.cross(arm, force)
        return np.array([-total[0], -total[2]]) / q_sb

    beta_rad = np.radians(number("aero_beta_deg"))
    roll_yaw = (
        body(sm.beta_moment_nm, sm.beta_force_n) - body(sm.moment_nm[s, 0], sm.force_n[s, 0])
    ) / beta_rad
    return float(roll_yaw[1]), float(roll_yaw[0])


def _point(points: tuple[FlightPoint, ...], pid: str) -> FlightPoint:
    """Flight point by id."""
    for p in points:
        if p.id == pid:
            return p
    raise AeroError(f"flight point {pid} missing")


def evaluate(
    d: Design, req: Requirements, cgs_m: tuple[np.ndarray, np.ndarray], sampler: SampleFn, tier: int
) -> AeroResult:
    """Trim every flight point at both CGs with the given sampler."""
    points = flight_points(req)
    f = flight_state(d, req)
    speeds = point_speeds(req)
    cruise = _point(points, CRUISE_POINT)
    sm = sampler(d, req, speeds, cruise.speed_ms, cgs_m[1])
    s3 = int(np.flatnonzero(speeds == cruise.speed_ms)[0])
    k3 = slopes(sm, s3, cgs_m[1])
    q3_s = 0.5 * f.rho_kg_m3 * cruise.speed_ms**2 * f.s_ref_m2
    cl_alpha = k3.la / q3_s * 180.0 / np.pi
    clmax = aircraft_clmax(d, _point(points, STALL_POINT).speed_ms, f)
    trims = []
    for p in points:
        n_load = p.load_factor
        if p.gust_ms > 0.0:
            # user Deviation (step 3 gate): a gust cannot load the wing past its stall
            n_load = min(
                gust_load_factor(cl_alpha, p.speed_ms, p.gust_ms, f),
                stall_load_factor(clmax, p.speed_ms, f),
            )
        trims += [
            trim_point(sm, p, n_load, lab, cg_m, f)
            for lab, cg_m in zip(CG_LABELS, cgs_m, strict=True)
        ]
    cnb, clb = lateral(sm, s3, cgs_m[1], f, cruise.speed_ms)
    return AeroResult(
        tier=tier,
        trims=tuple(trims),
        clmax=clmax,
        stall_speed_ms=float(np.sqrt(2 * f.weight_n / (f.rho_kg_m3 * f.s_ref_m2 * clmax))),
        cl_alpha_per_rad=float(cl_alpha),
        cnb_per_rad=cnb,
        clb_per_rad=clb,
        flight=f,
        pitch_control=sm.pitch_control,
        stall_order=(),
    )


def tier1(d: Design, req: Requirements, cgs_m: tuple[np.ndarray, np.ndarray]) -> AeroResult:
    """Tier-1 (AeroBuildup) evaluation."""
    return evaluate(d, req, cgs_m, buildup_samples, 1)


def cruise_ld(aero: AeroResult) -> float:
    """Worst-CG lift-to-drag ratio at cruise."""
    return min(t.cl / t.cd for t in aero.trims if t.point == CRUISE_POINT)
