"""Tier 2: VLM wing lift, moments, induced drag + buildup fuselages, profile drag (SPEC §8)."""

from dataclasses import dataclass, replace
from itertools import pairwise

import aerosandbox as asb
import numpy as np

from aeroforge.aero import (
    STALL_POINT,
    AeroError,
    Flight,
    Samples,
    Trim,
    buildup_samples,
    lift_n,
    section_clmax,
)
from aeroforge.airplane import aero_sections, airplane, quarter_mac_x_m
from aeroforge.constants import number, numbers
from aeroforge.models import Design, Requirements, Surface
from aeroforge.style import wing_surfaces
from aeroforge.trim import AeroResult, StallOrder, evaluate


@dataclass(frozen=True)
class VlmRun:
    """One VLM solution: totals about xyz_ref, per-wing forces, per-panel forces [N, N m]."""

    force_n: np.ndarray
    moment_nm: np.ndarray
    wing_force_n: np.ndarray
    panel_force_n: np.ndarray
    cl: float
    cd: float


@dataclass(frozen=True)
class Panels:
    """Panel layout and the runs needed to superpose a trimmed panel loading."""

    centres_m: np.ndarray
    areas_m2: np.ndarray
    wing_of_panel: np.ndarray
    lo: VlmRun
    hi: VlmRun
    deflected: VlmRun | None
    q_ref_pa: float


def _panel_counts(ap: asb.Airplane) -> list[int]:
    """Panels per wing in VLM mesh order."""
    span_res, chord_res = (
        int(number("vlm_spanwise_resolution")),
        int(number("vlm_chordwise_resolution")),
    )
    return [(len(w.xsecs) - 1) * span_res * chord_res * (2 if w.symmetric else 1) for w in ap.wings]


def _vlm(
    ap: asb.Airplane, op: asb.OperatingPoint, ref_m: np.ndarray
) -> tuple[VlmRun, asb.VortexLatticeMethod]:
    """Solve one VLM and split its panel forces by wing."""
    vlm = asb.VortexLatticeMethod(
        ap,
        op,
        xyz_ref=list(map(float, ref_m)),
        spanwise_resolution=int(number("vlm_spanwise_resolution")),
        chordwise_resolution=int(number("vlm_chordwise_resolution")),
    )
    out = vlm.run()
    counts = _panel_counts(ap)
    if sum(counts) != len(vlm.areas):
        raise AeroError(f"VLM mesh has {len(vlm.areas)} panels, expected {sum(counts)}")
    bounds = np.cumsum([0, *counts])
    forces = np.asarray(vlm.forces_geometry)
    wings = np.array([forces[a:b].sum(axis=0) for a, b in pairwise(bounds)])
    run = VlmRun(
        force_n=np.asarray(out["F_g"], dtype=float),
        moment_nm=np.asarray(out["M_g"], dtype=float),
        wing_force_n=wings,
        panel_force_n=forces,
        cl=float(out["CL"]),
        cd=float(out["CD"]),
    )
    return run, vlm


def _runs(
    d: Design, req: Requirements, ref_m: np.ndarray, cruise_ms: float, pitch: str | None
) -> tuple[list[VlmRun], asb.VortexLatticeMethod]:
    """VLM at alpha lo, alpha hi, sideslip (alpha lo), deflected pitch control; the lo mesh."""
    a_lo, a_hi = numbers("aero_alpha_deg")
    atmo = asb.Atmosphere(altitude=req.altitude_m)
    plain = airplane(d, req, ref_m, None, 0.0, vlm=True)
    lo, vlm = _vlm(
        plain, asb.OperatingPoint(atmosphere=atmo, velocity=cruise_ms, alpha=a_lo), ref_m
    )
    hi, _ = _vlm(plain, asb.OperatingPoint(atmosphere=atmo, velocity=cruise_ms, alpha=a_hi), ref_m)
    beta = asb.OperatingPoint(
        atmosphere=atmo, velocity=cruise_ms, alpha=a_lo, beta=number("aero_beta_deg")
    )
    side, _ = _vlm(plain, beta, ref_m)
    runs = [lo, hi, side]
    if pitch is not None:
        bent = airplane(d, req, ref_m, pitch, number("aero_deflection_deg"), vlm=True)
        runs.append(
            _vlm(bent, asb.OperatingPoint(atmosphere=atmo, velocity=cruise_ms, alpha=a_lo), ref_m)[
                0
            ]
        )
    return runs, vlm


def vlm_samples(
    d: Design, req: Requirements, speeds_ms: np.ndarray, cruise_ms: float, ref_m: np.ndarray
) -> tuple[Samples, Panels]:
    """Buildup samples with wing forces and moments replaced by VLM (scaled by dynamic pressure)."""
    base = buildup_samples(d, req, speeds_ms, cruise_ms, ref_m)
    runs, vlm = _runs(d, req, ref_m, cruise_ms, base.pitch_control)
    lo, hi, side = runs[:3]
    bent = runs[3] if len(runs) > 3 else None
    scale = (speeds_ms / cruise_ms) ** 2
    per_deg = 1.0 / number("aero_deflection_deg")
    pair_f = np.stack([lo.force_n, hi.force_n])
    pair_m = np.stack([lo.moment_nm, hi.moment_nm])
    pair_w = np.stack([lo.wing_force_n, hi.wing_force_n])
    zero = np.zeros(3)
    d_f = (bent.force_n - lo.force_n) * per_deg if bent else zero
    d_m = (bent.moment_nm - lo.moment_nm) * per_deg if bent else zero
    d_w = (
        (bent.wing_force_n - lo.wing_force_n) * per_deg if bent else np.zeros_like(lo.wing_force_n)
    )
    samples = replace(
        base,
        force_n=scale[:, None, None] * pair_f + base.body_force_n,
        moment_nm=scale[:, None, None] * pair_m + base.body_moment_nm,
        surface_force_n=scale[:, None, None, None] * pair_w,
        d_force_n=scale[:, None] * d_f,
        d_moment_nm=scale[:, None] * d_m,
        d_surface_force_n=scale[:, None, None] * d_w,
        beta_force_n=side.force_n + base.beta_body_force_n,
        beta_moment_nm=side.moment_nm + base.beta_body_moment_nm,
        induced_k=hi.cd / hi.cl**2,
    )
    q_ref = 0.5 * float(asb.Atmosphere(altitude=req.altitude_m).density()) * cruise_ms**2
    wing_of = np.repeat(
        np.arange(len(d.surfaces)), _panel_counts(airplane(d, req, ref_m, None, 0.0, vlm=True))
    )
    panels = Panels(
        np.asarray(vlm.vortex_centers), np.asarray(vlm.areas), wing_of, lo, hi, bent, q_ref
    )
    return samples, panels


def _span_frac(s: Surface, centres_m: np.ndarray) -> np.ndarray:
    """Spanwise distance of strip centres_m from the root / total span length of one side."""
    secs = aero_sections(s)
    root = np.asarray(secs[0].le_m)
    les = np.array([c.le_m for c in secs])
    total = float(np.hypot(np.diff(les[:, 1]), np.diff(les[:, 2])).sum())
    y = np.abs(centres_m[:, 1]) if s.mirrored else centres_m[:, 1]
    return np.hypot(y - root[1], centres_m[:, 2] - root[2]) / total


def stall_order(d: Design, p: Panels, t: Trim, f: Flight, alpha_deg: np.ndarray) -> StallOrder:
    """Peak strip cl/clmax of the foremost lifting surface vs the others; peak span fraction."""
    frac_a = (t.alpha_deg - alpha_deg[0]) / (alpha_deg[1] - alpha_deg[0])
    forces = p.lo.panel_force_n + frac_a * (p.hi.panel_force_n - p.lo.panel_force_n)
    if p.deflected is not None and np.isfinite(t.deflection_deg):
        forces = forces + t.deflection_deg / number("aero_deflection_deg") * (
            p.deflected.panel_force_n - p.lo.panel_force_n
        )
    chord_res = int(number("vlm_chordwise_resolution"))
    strip_f = forces.reshape(-1, chord_res, 3).sum(axis=1)
    strip_a = p.areas_m2.reshape(-1, chord_res).sum(axis=1)
    strip_c = p.centres_m.reshape(-1, chord_res, 3).mean(axis=1)
    strip_w = p.wing_of_panel.reshape(-1, chord_res)[:, 0]
    cl = lift_n(strip_f, t.alpha_deg) / (p.q_ref_pa * strip_a)
    lifting = wing_surfaces(d)
    fore = min(lifting, key=quarter_mac_x_m)
    peaks: dict[str, tuple[float, float]] = {}
    for s in lifting:
        k = [x.name for x in d.surfaces].index(s.name)
        rows = strip_w == k
        ratio = cl[rows] / section_clmax(s, t.speed_ms, f)
        i = int(np.argmax(ratio))
        peaks[s.name] = (float(ratio[i]), float(_span_frac(s, strip_c[rows][i : i + 1])[0]))
    others = [r for name, (r, _) in peaks.items() if name != fore.name]
    peak_name = max(peaks, key=lambda n: peaks[n][0])
    return StallOrder(
        fore_ratio=peaks[fore.name][0],
        other_ratio=max(others, default=0.0),
        peak_span_frac=peaks[peak_name][1],
    )


def tier2(d: Design, req: Requirements, cgs_m: tuple[np.ndarray, np.ndarray]) -> AeroResult:
    """Tier-2 (VLM) evaluation with the stall order at P1 for both CGs."""
    kept: list[Panels] = []

    def sampler(
        dd: Design, rq: Requirements, speeds_ms: np.ndarray, cruise: float, ref_m: np.ndarray
    ) -> Samples:
        samples, panels = vlm_samples(dd, rq, speeds_ms, cruise, ref_m)
        kept.append(panels)
        return samples

    result = evaluate(d, req, cgs_m, sampler, 2)
    alpha = np.array(numbers("aero_alpha_deg"))
    orders = tuple(
        stall_order(d, kept[0], t, result.flight, alpha)
        for t in result.trims
        if t.point == STALL_POINT
    )
    return replace(result, stall_order=orders)
