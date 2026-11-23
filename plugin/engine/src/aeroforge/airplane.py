"""Design → aerosandbox Airplane (wings with pitch control, fuselages) for both aero tiers."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from math import hypot

import aerosandbox as asb
import numpy as np

from aeroforge.airfoil import naca_outline
from aeroforge.constants import number
from aeroforge.facet_query import equivalent_exponent, slice_section, surface_area_m2
from aeroforge.models import Body, Design, Polyhedron, Requirements, Section, Surface
from aeroforge.shapes import body_axis_z, body_radius_frac, body_stations
from aeroforge.style import wing_surfaces

FRAC_TOL = 1e-9  # control span fractions are compared to station fractions with this slack


@dataclass(frozen=True)
class Station:
    """Aero cross-section: leading edge [m], chord [m], structural span fraction."""

    le_m: np.ndarray
    chord_m: float
    frac: float
    twist_deg: float


@lru_cache(maxsize=1024)
def section_airfoil(camber_frac: float, camber_loc_frac: float, tc: float) -> asb.Airfoil:
    """NACA 4-digit airfoil shared by every section of a surface."""
    xy = np.array(
        naca_outline(camber_frac, camber_loc_frac, tc, int(number("airfoil_points_per_side")))
    )
    return asb.Airfoil(name="naca", coordinates=xy)


def surface_airfoil(s: Surface) -> asb.Airfoil:
    """Airfoil of surface s."""
    return section_airfoil(s.camber_frac, s.camber_loc_frac, s.tc)


def _span_m(a: Section, b: Section) -> float:
    """Spanwise (y-z) distance between two section leading edges [m]."""
    return hypot(b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2])


def aero_sections(s: Surface) -> tuple[Section, ...]:
    """Structural sections plus one end station of equal area replacing a multi-station tip."""
    n = s.n_structural
    if len(s.sections) <= n + 1:
        return s.sections
    run = s.sections[n - 1 :]
    lengths = [_span_m(a, b) for a, b in pairwise(run)]
    area = sum(
        ln * (a.chord_m + b.chord_m) / 2 for ln, (a, b) in zip(lengths, pairwise(run), strict=True)
    )
    end_chord = 2 * area / sum(lengths) - run[0].chord_m
    return (*s.sections[:n], s.sections[-1].model_copy(update={"chord_m": end_chord}))


def _tip_gapped(secs: tuple[Section, ...]) -> tuple[Section, ...]:
    """Sections with the tip pulled inboard by vlm_join_gap_frac of the last non-zero panel.

    Both surfaces of a tip join end on one chord line, so the VLM's narrow tip panels sit on the
    other surface's vortex legs and the solve is near-singular (user Deviation, step 3 gate).
    """
    tip = secs[-1]
    inner = next((a for a in reversed(secs[:-1]) if _span_m(a, tip) > 0.0), None)
    if inner is None:
        return secs
    k = number("vlm_join_gap_frac")
    le = np.asarray(tip.le_m) + k * (np.asarray(inner.le_m) - np.asarray(tip.le_m))
    chord = tip.chord_m + k * (inner.chord_m - tip.chord_m)
    gapped = tip.model_copy(
        update={"le_m": (float(le[0]), float(le[1]), float(le[2])), "chord_m": chord}
    )
    return (*secs[:-1], gapped)


def _stations(s: Surface, vlm: bool) -> list[Station]:
    """Aero stations with structural span fractions, plus stations at the control span ends;
    a joined surface's tip is gapped for the VLM."""
    secs = aero_sections(s)
    if vlm and s.joined_to is not None:
        secs = _tip_gapped(secs)
    cum = np.concatenate([[0.0], np.cumsum([_span_m(a, b) for a, b in pairwise(secs)])])
    structural = cum[s.n_structural - 1]
    fracs = cum / structural
    les = np.array([c.le_m for c in secs])
    chords = np.array([c.chord_m for c in secs])
    out = list(fracs)
    if s.control is not None:
        out += [s.control.span_start_frac, s.control.span_end_frac]
    stations = []
    for f in sorted(set(np.round(out, 12))):
        le = np.array([np.interp(f, fracs, les[:, k]) for k in range(3)])
        twist = float(np.interp(f, fracs, [c.twist_deg for c in secs]))
        stations.append(Station(le, float(np.interp(f, fracs, chords)), float(f), twist))
    return stations


def wing(s: Surface, deflection_deg: float, vlm: bool) -> asb.Wing:
    """asb.Wing of surface s; a non-zero deflection deflects its control symmetrically.

    AeroBuildup takes the deflection as a ControlSurface on the inboard xsec of each controlled
    section; the VLM ignores ControlSurface, so it gets deflected airfoils on the control stations.
    """
    af, c = surface_airfoil(s), s.control
    deflect = c is not None and deflection_deg != 0.0
    xsecs = []
    for st in _stations(s, vlm):
        lo, hi = (c.span_start_frac, c.span_end_frac) if c is not None else (0.0, 0.0)
        hinge = 1.0 - c.chord_frac if c is not None else 1.0
        xsec_af, controls = af, []
        if deflect and vlm and lo - FRAC_TOL <= st.frac <= hi + FRAC_TOL:
            xsec_af = af.add_control_surface(deflection=deflection_deg, hinge_point_x=hinge)
        if deflect and not vlm and lo - FRAC_TOL <= st.frac < hi - FRAC_TOL:
            controls = [asb.ControlSurface(deflection=deflection_deg, hinge_point=hinge)]
        xsecs.append(
            asb.WingXSec(
                xyz_le=st.le_m,
                chord=st.chord_m,
                twist=s.incidence_deg + st.twist_deg,
                airfoil=xsec_af,
                control_surfaces=controls,
            )
        )
    return asb.Wing(name=s.name, xsecs=xsecs, symmetric=s.mirrored)


class FacetedFuselage(asb.Fuselage):
    """Equivalent fuselage of a flat-panel body whose skin-friction area is the exact area of its
    faces: the equivalent sections keep width, height and area but their perimeters undercount the
    polygon's by 2 to 20 %, and AeroBuildup takes friction from `area_wetted`."""

    def __init__(self, *args: object, wetted_m2: float, **kwargs: object) -> None:
        # AeroBuildup keys its fuselage defaults on the exact class, so a subclass gets them here
        defaults = asb.AeroBuildup.default_analysis_specific_options[asb.Fuselage]
        kwargs.setdefault("analysis_specific_options", {asb.AeroBuildup: dict(defaults)})
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.wetted_m2 = wetted_m2

    def area_wetted(self) -> float:
        """Exact wetted area [m^2] of the flat faces."""
        return self.wetted_m2


def _faceted_fuselage(b: Body, y_sign: float, poly: Polyhedron) -> asb.Fuselage:
    """Equivalent fuselage of a flat-panel body (SPEC §8): each station keeps the polygon's width,
    height and area (super-ellipse exponent from asb's own area formula) and the friction area is
    the faces' own. The extra drag of sharp chines is not modelled."""
    xsecs = []
    for ti in body_stations(int(number("aero_body_stations"))):
        width, height, area, zc = slice_section(poly, ti * b.length_m)
        xsecs.append(
            asb.FuselageXSec(
                xyz_c=[b.nose_m[0] + ti * b.length_m, y_sign * b.nose_m[1], b.nose_m[2] + zc],
                width=width,
                height=height,
                shape=equivalent_exponent(width, height, area),
            )
        )
    return FacetedFuselage(name=b.name, xsecs=xsecs, wetted_m2=surface_area_m2(poly))


def _fuselage(b: Body, y_sign: float) -> asb.Fuselage:
    """asb.Fuselage of one body instance (y mirrored by y_sign)."""
    if b.facets is not None:
        return _faceted_fuselage(b, y_sign, b.facets)
    t = body_stations(int(number("aero_body_stations")))
    r = body_radius_frac(b, t)
    xsecs = [
        asb.FuselageXSec(
            xyz_c=[b.nose_m[0] + ti * b.length_m, y_sign * b.nose_m[1], float(zi)],
            width=2 * b.half_width_m * ri,
            height=2 * b.half_height_m * ri,
            shape=b.section_exp,
        )
        for ti, ri, zi in zip(t, r, body_axis_z(b, t), strict=True)
    ]
    return asb.Fuselage(name=b.name, xsecs=xsecs)


def fuselages(d: Design) -> list[asb.Fuselage]:
    """One asb.Fuselage per body instance."""
    return [
        _fuselage(b, sign) for b in d.bodies for sign in ((1.0, -1.0) if b.mirrored else (1.0,))
    ]


def mac_m(s: Surface) -> float:
    """Mean aerodynamic chord of a surface [m]."""
    secs = aero_sections(s)
    num = den = 0.0
    for a, b in pairwise(secs):
        ln = _span_m(a, b)
        num += ln * (a.chord_m**2 + a.chord_m * b.chord_m + b.chord_m**2) / 3
        den += ln * (a.chord_m + b.chord_m) / 2
    return num / den


def span_m(d: Design) -> float:
    """Tip-to-tip span of all surfaces [m]."""
    return 2 * max(abs(c.le_m[1]) for s in d.surfaces for c in s.sections)


def quarter_mac_x_m(s: Surface) -> float:
    """x of the area-weighted quarter-chord point [m]."""
    secs = aero_sections(s)
    num = den = 0.0
    for a, b in pairwise(secs):
        area = _span_m(a, b) * (a.chord_m + b.chord_m) / 2
        num += area * ((a.le_m[0] + b.le_m[0]) / 2 + 0.25 * (a.chord_m + b.chord_m) / 2)
        den += area
    return num / den


def pitch_surface(d: Design, cg_x_m: float) -> Surface | None:
    """Controlled lifting surface with the longest pitch arm to cg (larger area breaks a tie).

    Area x arm picks main-wing ailerons over a tail elevator on a conventional layout (0.057 vs
    0.054), but flaperons near the CG mostly change lift, not moment.
    """
    ranked = sorted(
        (s for s in wing_surfaces(d) if s.control is not None),
        key=lambda s: (abs(quarter_mac_x_m(s) - cg_x_m), s.projected_area_m2),
    )
    return ranked[-1] if ranked else None


def airplane(
    d: Design,
    req: Requirements,
    xyz_ref_m: np.ndarray,
    deflected: str | None,
    deflection_deg: float,
    vlm: bool,
) -> asb.Airplane:
    """asb.Airplane of d with the named surface's control deflected."""
    wings = [wing(s, deflection_deg if s.name == deflected else 0.0, vlm) for s in d.surfaces]
    return asb.Airplane(
        wings=wings,
        fuselages=fuselages(d),
        xyz_ref=list(map(float, xyz_ref_m)),
        s_ref=d.lifting_area_m2,
        c_ref=mac_m(d.surfaces[0]),
        b_ref=span_m(d),
    )
