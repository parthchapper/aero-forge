"""Linearised aero samples and trimmed flight points; tier 1 = asb.AeroBuildup (SPEC §8)."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import aerosandbox as asb
import numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup import softmax_scalefree

from aeroforge.airplane import (
    airplane,
    mac_m,
    pitch_surface,
    section_kulfan,
    span_m,
    wing,
)
from aeroforge.constants import number, numbers
from aeroforge.downwash import bound_span_m, influence
from aeroforge.models import Design, FlightPoint, Requirements, Surface
from aeroforge.style import wing_surfaces

STALL_POINT, CRUISE_POINT = "P1", "P3"


class AeroError(RuntimeError):
    """A design cannot be evaluated (no flight points, no lift slope)."""


@dataclass(frozen=True)
class Samples:
    """Forces [N] and moments [N m] about xyz_ref in geometry axes at the linearisation angles.

    Arrays are per speed s and per alpha sample (lo, hi); d_* are per degree of pitch-control
    deflection at alpha lo; body_* are the fuselage share of force/moment (tier 2 keeps them).
    """

    speeds_ms: np.ndarray
    alpha_deg: np.ndarray
    force_n: np.ndarray
    moment_nm: np.ndarray
    surface_force_n: np.ndarray
    body_force_n: np.ndarray
    body_moment_nm: np.ndarray
    profile_n: np.ndarray
    d_force_n: np.ndarray
    d_moment_nm: np.ndarray
    d_surface_force_n: np.ndarray
    d_profile_n: np.ndarray
    beta_force_n: np.ndarray
    beta_moment_nm: np.ndarray
    beta_body_force_n: np.ndarray
    beta_body_moment_nm: np.ndarray
    induced_k: float
    xyz_ref_m: np.ndarray
    pitch_control: str | None


@dataclass(frozen=True)
class Trim:
    """One flight point trimmed at one CG."""

    point: str
    cg: str
    speed_ms: float
    load_factor: float
    alpha_deg: float
    deflection_deg: float
    cl: float
    cd: float
    cm_residual: float
    static_margin: float
    surface_lift_n: tuple[float, ...]


@dataclass(frozen=True)
class Flight:
    """Shared flight state: weight [N], density [kg/m³], viscosity [Pa s], reference sizes [m]."""

    weight_n: float
    rho_kg_m3: float
    mu_pa_s: float
    s_ref_m2: float
    c_ref_m: float
    b_ref_m: float


def flight_state(d: Design, req: Requirements) -> Flight:
    """Weight, air properties at the brief altitude and reference sizes."""
    atmo = asb.Atmosphere(altitude=req.altitude_m)
    return Flight(
        weight_n=req.mtow_kg * number("gravity_m_s2"),
        rho_kg_m3=float(atmo.density()),
        mu_pa_s=float(atmo.dynamic_viscosity()),
        s_ref_m2=d.lifting_area_m2,
        c_ref_m=mac_m(d.surfaces[0]),
        b_ref_m=span_m(d),
    )


def _vec3(parts: list[object], n: int) -> np.ndarray:
    """asb [x, y, z] list of scalars/arrays → (n, 3) array."""
    return np.stack(
        [np.broadcast_to(np.ravel(np.asarray(p, dtype=float)), (n,)) for p in parts], axis=1
    )


def _by_alpha(a: np.ndarray, n_speeds: int) -> np.ndarray:
    """Rows ordered (speed, alpha lo/hi) → shape (n_speeds, 2, ...)."""
    return a[: 2 * n_speeds].reshape(n_speeds, 2, *a.shape[1:])


def _operating_point(
    req: Requirements, speeds_ms: np.ndarray, cruise_ms: float
) -> asb.OperatingPoint:
    """Each speed at both alpha samples, then the cruise speed sideslipped at alpha lo."""
    a_lo, a_hi = numbers("aero_alpha_deg")
    beta = np.zeros(2 * len(speeds_ms) + 1)
    beta[-1] = number("aero_beta_deg")
    return asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=req.altitude_m),
        velocity=np.append(np.repeat(speeds_ms, 2), cruise_ms),
        alpha=np.append(np.tile([a_lo, a_hi], len(speeds_ms)), a_lo),
        beta=beta,
    )


def _deflection_deltas(
    d: Design, req: Requirements, ref_m: np.ndarray, pitch: Surface | None, out: dict, n_speeds: int
) -> tuple[np.ndarray, ...]:
    """Per-degree change of the pitch surface's force, moment and drag at alpha lo (0 if none)."""
    if pitch is None:
        zero = np.zeros((n_speeds, 3))
        return zero, zero, np.zeros((n_speeds, len(d.surfaces), 3)), np.zeros(n_speeds)
    k = [s.name for s in d.surfaces].index(pitch.name)
    a_lo = numbers("aero_alpha_deg")[0]
    speeds_ms = np.asarray(out["op_point"].velocity)[: 2 * n_speeds : 2]
    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=req.altitude_m), velocity=speeds_ms, alpha=a_lo
    )
    ap = airplane(d, req, ref_m, None, 0.0, vlm=False)
    defl = number("aero_deflection_deg")
    comp = asb.AeroBuildup(ap, op, xyz_ref=list(ref_m)).wing_aerodynamics(
        wing(pitch, defl, vlm=False), include_induced_drag=False
    )
    base = out["wing_aero_components"][k]
    lo = slice(0, 2 * n_speeds, 2)
    d_f = (_vec3(comp.F_g, n_speeds) - _vec3(base.F_g, 2 * n_speeds + 1)[lo]) / defl
    d_m = (_vec3(comp.M_g, n_speeds) - _vec3(base.M_g, 2 * n_speeds + 1)[lo]) / defl
    d_drag = (np.ravel(comp.D) - np.ravel(base.D)[lo]) / defl
    d_surf = np.zeros((n_speeds, len(d.surfaces), 3))
    d_surf[:, k] = d_f
    return d_f, d_m, d_surf, d_drag


def _sum_components(
    op: asb.OperatingPoint, wings: list[object], bodies: list[object]
) -> tuple[np.ndarray, ...]:
    """Total geometry force/moment, per-wing force/moment, lift, induced drag and profile drag."""
    comps = [*wings, *bodies]
    n = len(np.ravel(op.velocity))
    f_raw = [sum(c.F_g[i] for c in comps) for i in range(3)]
    m_raw = [sum(c.M_g[i] for c in comps) for i in range(3)]
    span_sq = softmax_scalefree([c.span_effective**2 * c.oswalds_efficiency for c in comps])
    _, side, lift = op.convert_axes(*f_raw, from_axes="geometry", to_axes="wind")
    induced = (lift**2 + side**2) / (op.dynamic_pressure() * np.pi * span_sq)
    ind_g = op.convert_axes(-induced, 0, 0, from_axes="wind", to_axes="geometry")
    force = _vec3([f_raw[i] + ind_g[i] for i in range(3)], n)
    moment = _vec3(m_raw, n)
    surf = np.stack([_vec3(c.F_g, n) for c in wings], axis=1)
    surf_m = np.stack([_vec3(c.M_g, n) for c in wings], axis=1)
    profile = np.broadcast_to(np.ravel(sum(c.D for c in comps)), (n,))
    return force, moment, surf, surf_m, np.ravel(lift), np.ravel(induced), profile


def _assemble_samples(
    d: Design,
    req: Requirements,
    speeds_ms: np.ndarray,
    cruise_ms: float,
    ref_m: np.ndarray,
    op: asb.OperatingPoint,
    wings: list[object],
    bodies: list[object],
    pitch: Surface | None,
) -> Samples:
    """Assemble tier-1 Samples from per-component AeroBuildup outputs (exact match to .run())."""
    n, ns = 2 * len(speeds_ms) + 1, len(speeds_ms)
    force, moment, surf, surf_m, lift, induced, profile = _sum_components(op, wings, bodies)
    wash_f, wash_m = downwash(d, surf, surf_m, op, speeds_ms)
    force, moment, surf = force + wash_f.sum(axis=1), moment + wash_m.sum(axis=1), surf + wash_f
    body_f = sum((_vec3(c.F_g, n) for c in bodies), np.zeros((n, 3)))
    body_m = sum((_vec3(c.M_g, n) for c in bodies), np.zeros((n, 3)))
    q_s = 0.5 * float(op.atmosphere.density()) * cruise_ms**2 * d.lifting_area_m2
    hi = 2 * int(np.flatnonzero(speeds_ms == cruise_ms)[0]) + 1
    if abs(float(lift[hi])) < 1e-12:
        raise AeroError("zero lift at alpha_hi: cannot compute induced_k")
    deltas = _deflection_deltas(
        d, req, ref_m, pitch, {"op_point": op, "wing_aero_components": wings}, ns
    )
    return Samples(
        speeds_ms=speeds_ms,
        alpha_deg=np.array(numbers("aero_alpha_deg")),
        force_n=_by_alpha(force, ns),
        moment_nm=_by_alpha(moment, ns),
        surface_force_n=_by_alpha(surf, ns),
        body_force_n=_by_alpha(body_f, ns),
        body_moment_nm=_by_alpha(body_m, ns),
        profile_n=_by_alpha(profile, ns),
        d_force_n=deltas[0],
        d_moment_nm=deltas[1],
        d_surface_force_n=deltas[2],
        d_profile_n=deltas[3],
        beta_force_n=force[-1],
        beta_moment_nm=moment[-1],
        beta_body_force_n=body_f[-1],
        beta_body_moment_nm=body_m[-1],
        induced_k=float(induced[hi] * q_s / lift[hi] ** 2),
        xyz_ref_m=ref_m,
        pitch_control=None if pitch is None else pitch.name,
    )


def buildup_samples(
    d: Design, req: Requirements, speeds_ms: np.ndarray, cruise_ms: float, ref_m: np.ndarray
) -> Samples:
    """Tier-1 samples from one vectorised AeroBuildup run plus the deflected pitch surface."""
    ap = airplane(d, req, ref_m, None, 0.0, vlm=False)
    op = _operating_point(req, speeds_ms, cruise_ms)
    ab = asb.AeroBuildup(ap, op, xyz_ref=list(ref_m))
    wings = [ab.wing_aerodynamics(w, include_induced_drag=False) for w in ap.wings]
    bodies = [ab.fuselage_aerodynamics(f, include_induced_drag=False) for f in ap.fuselages]
    return _assemble_samples(
        d, req, speeds_ms, cruise_ms, ref_m, op, wings, bodies, pitch_surface(d, float(ref_m[0]))
    )


def buildup_pitch_updater(
    d: Design, req: Requirements, speeds_ms: np.ndarray, cruise_ms: float, ref_m: np.ndarray
) -> tuple[Samples, Callable[[Design, np.ndarray], Samples]]:
    """Initial Samples plus a fast updater when only the pitch surface's incidence_deg changes."""
    ap = airplane(d, req, ref_m, None, 0.0, vlm=False)
    op = _operating_point(req, speeds_ms, cruise_ms)
    ab = asb.AeroBuildup(ap, op, xyz_ref=list(ref_m))
    wings = [ab.wing_aerodynamics(w, include_induced_drag=False) for w in ap.wings]
    bodies = [ab.fuselage_aerodynamics(f, include_induced_drag=False) for f in ap.fuselages]
    pitch = pitch_surface(d, float(ref_m[0]))
    first = _assemble_samples(d, req, speeds_ms, cruise_ms, ref_m, op, wings, bodies, pitch)

    def update(tilted: Design, cg_m: np.ndarray) -> Samples:
        new_pitch = pitch_surface(tilted, float(cg_m[0]))
        if pitch is None or new_pitch is None or new_pitch.name != pitch.name:
            return buildup_samples(tilted, req, speeds_ms, cruise_ms, cg_m)
        k = [s.name for s in tilted.surfaces].index(pitch.name)
        swapped = list(wings)
        swapped[k] = ab.wing_aerodynamics(
            wing(tilted.surfaces[k], 0.0, vlm=False), include_induced_drag=False
        )
        return _assemble_samples(
            tilted, req, speeds_ms, cruise_ms, ref_m, op, swapped, bodies, new_pitch
        )

    return first, update


def cn_beta_buildup(d: Design, req: Requirements, cruise_ms: float, cg_m: np.ndarray) -> float:
    """Fast tier-1 Cn_beta [1/rad] about cg_m at cruise (3-point run, no deflection)."""
    a_lo, a_hi = numbers("aero_alpha_deg")
    beta_deg = number("aero_beta_deg")
    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=req.altitude_m),
        velocity=np.full(3, cruise_ms),
        alpha=np.array([a_lo, a_hi, a_lo]),
        beta=np.array([0.0, 0.0, beta_deg]),
    )
    ap = airplane(d, req, cg_m, None, 0.0, vlm=False)
    out = asb.AeroBuildup(ap, op, xyz_ref=list(cg_m)).run()
    moment = _vec3(out["M_g"], 3)
    surf = np.stack([_vec3(c.F_g, 3) for c in out["wing_aero_components"]], axis=1)
    surf_m = np.stack([_vec3(c.M_g, 3) for c in out["wing_aero_components"]], axis=1)
    _, wash_m = downwash(d, surf, surf_m, op, np.array([cruise_ms]))
    moment = moment + wash_m.sum(axis=1)
    f = flight_state(d, req)
    q_sb = 0.5 * f.rho_kg_m3 * cruise_ms**2 * f.s_ref_m2 * f.b_ref_m
    return float(-(moment[2, 2] - moment[0, 2]) / (q_sb * np.radians(beta_deg)))


def downwash(
    d: Design,
    surf_force_n: np.ndarray,
    surf_moment_nm: np.ndarray,
    op: asb.OperatingPoint,
    speeds_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-surface force and moment change [N, N m] per run row from the other lifting surfaces'
    horseshoe vortices (downwash.py): each surface's own lo/hi samples at the row's speed give its
    force at the induced change of angle of attack."""
    a_lo, a_hi = numbers("aero_alpha_deg")
    v = np.ravel(op.velocity)
    rows = np.array([int(np.flatnonzero(speeds_ms == vr)[0]) for vr in v])
    lift = lift_n(surf_force_n, np.ravel(op.alpha)[:, None])
    span = np.array([bound_span_m(s) if s.projected_area_m2 > 0 else 1.0 for s in d.surfaces])
    gamma = lift / (float(op.atmosphere.density()) * v[:, None] * span[None, :])
    d_alpha_deg = np.degrees(gamma @ influence(d) / v[:, None])
    d_f, d_m = np.zeros_like(surf_force_n), np.zeros_like(surf_moment_nm)
    for src, dst in ((surf_force_n, d_f), (surf_moment_nm, d_m)):
        slope = (src[2 * rows + 1] - src[2 * rows]) / (a_hi - a_lo)
        dst[:] = d_alpha_deg[:, :, None] * slope
    return d_f, d_m


def lift_n(force_n: np.ndarray, alpha_deg: float | np.ndarray) -> np.ndarray:
    """Wind-axis lift of geometry-axis force(s) at alpha (x aft, z up)."""
    a = np.radians(alpha_deg)
    return -force_n[..., 0] * np.sin(a) + force_n[..., 2] * np.cos(a)


def pitch_nm(
    moment_nm: np.ndarray, force_n: np.ndarray, ref_m: np.ndarray, cg_m: np.ndarray
) -> float:
    """Pitch-up moment about cg_m from a moment about ref_m: M + (ref_m - cg_m) x F, y component."""
    dv = ref_m - cg_m
    return float(moment_nm[1] + dv[2] * force_n[0] - dv[0] * force_n[2])


@lru_cache(maxsize=4096)
def _section_clmax(camber: float, loc: float, tc: float, re: float) -> float:
    """Max section lift coefficient over the alpha sweep (NeuralFoil, step-0 Decision b)."""
    lo, hi, step = numbers("section_clmax_alpha_deg")
    alpha = np.arange(lo, hi + step / 2, step)
    aero = section_kulfan(camber, loc, tc).get_aero_from_neuralfoil(alpha=alpha, Re=re)
    return float(np.max(aero["CL"]))


def section_clmax(s: Surface, speed_ms: float, f: Flight) -> float:
    """Section clmax of s at its MAC Reynolds number and speed."""
    re = f.rho_kg_m3 * speed_ms * mac_m(s) / f.mu_pa_s
    return _section_clmax(s.camber_frac, s.camber_loc_frac, s.tc, float(re))


def aircraft_clmax(d: Design, speed_ms: float, f: Flight) -> float:
    """3-D factor x projected-area-weighted section clmax of the lifting surfaces."""
    wings = wing_surfaces(d)
    area = np.array([s.projected_area_m2 for s in wings])
    cl = np.array([section_clmax(s, speed_ms, f) for s in wings])
    return number("clmax_3d_factor", "checks") * float((area * cl).sum() / area.sum())


def gust_load_factor(cl_alpha_per_rad: float, speed_ms: float, gust_ms: float, f: Flight) -> float:
    """1 + Pratt-alleviated sharp-edged gust increment."""
    wing_loading = f.weight_n / f.s_ref_m2
    a_k, b_k = numbers("gust_alleviation")
    mu = 2 * wing_loading / (f.rho_kg_m3 * f.c_ref_m * cl_alpha_per_rad * number("gravity_m_s2"))
    k = a_k * mu / (b_k + mu)
    return 1.0 + k * f.rho_kg_m3 * speed_ms * gust_ms * cl_alpha_per_rad / (2 * wing_loading)


def flight_points(req: Requirements) -> tuple[FlightPoint, ...]:
    """Requirements' flight points; derive must have filled them."""
    if not req.points:
        raise AeroError("requirements carry no flight points (run derive)")
    return req.points


def point_speeds(req: Requirements) -> np.ndarray:
    """Distinct flight-point speeds, ascending [m/s]."""
    return np.array(sorted({p.speed_ms for p in flight_points(req)}))


def stall_load_factor(clmax: float, speed_ms: float, f: Flight) -> float:
    """Largest load factor the wing can reach at speed_ms: CLmax q S / W."""
    return clmax * 0.5 * f.rho_kg_m3 * speed_ms**2 * f.s_ref_m2 / f.weight_n
