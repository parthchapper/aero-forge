"""Surface planforms: sections from area, aspect ratio, sweep, cant, break and tip treatment."""

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from aeroforge.airfoil import local_thickness_frac
from aeroforge.align import snap_sections
from aeroforge.coarse import stepped_levels
from aeroforge.constants import constant, number
from aeroforge.models import Family, Section, Spar, TipStyle, Vec3


@dataclass(frozen=True)
class SurfaceGenes:
    """Decoded per-surface parameters before sizing (family already applied)."""

    index: int
    host: str
    mirrored: bool
    root_x_frac: float
    root_z_frac: float
    weight: float
    aspect_ratio: float
    sweep_deg: float
    cant_deg: float
    break_frac: float
    outer_sweep_deg: float
    outer_cant_deg: float
    taper: float
    tip_style: TipStyle
    camber_frac: float
    camber_loc_frac: float
    tc: float
    control_frac: float
    join_target: int | None
    area_share: float | None = None  # true area / S, overriding the weight (added fin)
    root_chord_scale: float = 1.0  # strake: inner-panel root chord x this, area kept
    twist_deg: float = 0.0  # linear washout from root to tip
    airfoil_facets: int = 0  # flat segments per side of the section polygon (0 = smooth)
    coarseness: float = 0.0  # U3b facet coarseness of a faceted surface (stepped washout)
    align_angles_deg: tuple[float, ...] = ()  # strict alignment: allowed edge angles (() = free)


@dataclass(frozen=True)
class Planform:
    """Built sections of one surface instance; first n_structural carry the spar."""

    sections: tuple[Section, ...]
    n_structural: int
    aspect_ratio: float
    tip_style: TipStyle
    root_chord_m: float
    note: str = ""  # strict alignment: what could not be snapped ('' when everything was)


@dataclass(frozen=True)
class TipShape:
    """Tip extension: length / semi-span, (ext frac, chord / tip chord) stations, held chord."""

    span_frac: float
    stations: tuple[tuple[float, float], ...]
    align_chord_frac: float


def tip_shape(style: TipStyle) -> TipShape | None:
    """Tip extension parameters of a tip style (None for square)."""
    if style == "square":
        return None
    raw = constant("constants", f"tip_{style}")
    if not isinstance(raw, dict):
        raise TypeError(f"tip_{style} must be a mapping")
    stations = tuple((float(e), float(k)) for e, k in raw["stations"])
    return TipShape(float(raw["span_frac"]), stations, float(raw["align_chord_frac"]))


def _ext_chord_integral(tip: TipShape | None) -> float:
    """Extension area / (extension length x tip chord)."""
    if tip is None:
        return 0.0
    total, prev_e, prev_k = 0.0, 0.0, 1.0
    for e, k in tip.stations:
        total += (e - prev_e) * (prev_k + k) / 2
        prev_e, prev_k = e, k
    return total


def _direction(sweep_deg: float, cant_deg: float) -> np.ndarray:
    """Quarter-chord line step per unit yz length."""
    c = np.radians(cant_deg)
    return np.array([np.tan(np.radians(sweep_deg)), np.cos(c), np.sin(c)])


def _twisted(
    sections: list[Section], n_structural: int, twist_deg: float, steps: int = 0
) -> list[Section]:
    """Sections with a linear washout over the structural span (arc length root to tip); the tip
    extension holds the tip twist. steps > 0: the washout in that many whole steps (coarseness),
    so a strip between two sections of one step is not twisted. Zero twist returns the sections."""
    if twist_deg == 0.0:
        return sections
    pts = np.array([s.le_m[1:] for s in sections[:n_structural]])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    frac = np.concatenate([arc / arc[-1], np.ones(len(sections) - n_structural)])
    if steps:
        frac = np.round(frac * steps) / steps
    return [
        s.model_copy(update={"twist_deg": float(twist_deg * f)})
        for s, f in zip(sections, frac, strict=True)
    ]


def _section(qc_m: np.ndarray, chord_m: float, cant_deg: float) -> Section:
    """Section from its quarter-chord point."""
    le = qc_m - np.array([0.25 * chord_m, 0.0, 0.0])
    return Section(
        le_m=(float(le[0]), float(le[1]), float(le[2])), chord_m=chord_m, cant_deg=cant_deg
    )


def _semi_span(g: SurfaceGenes, half_area_m2: float, root_y_m: float, y_limit_m: float) -> float:
    """Structural semi-span [m] from aspect ratio, clipped so the tip stays within y_limit_m."""
    semi = float(np.sqrt(2.0 * g.aspect_ratio * half_area_m2) / 2.0)
    tip = tip_shape(g.tip_style)
    kappa = tip.span_frac if tip else 0.0
    c1, c2 = np.radians(g.cant_deg), np.radians(g.outer_cant_deg)
    reach = g.break_frac * np.cos(c1) + (1 - g.break_frac + kappa) * np.cos(c2)
    if g.mirrored and reach > 1e-9 and root_y_m + semi * reach > y_limit_m:
        semi = float((y_limit_m - root_y_m) / reach)
    return semi


def _capped_break_frac(g: SurfaceGenes, semi_m: float, kappa: float, height_max_m: float) -> float:
    """Break fraction pushed outboard until the outer panel, tip extension included, is at most
    height_max_m tall; unchanged when it already is or when the surface has no outer panel."""
    rise_m = semi_m * abs(np.sin(np.radians(g.outer_cant_deg)))
    if g.break_frac >= 1.0 or rise_m * (1.0 - g.break_frac + kappa) <= height_max_m:
        return g.break_frac
    return float(min(1.0, 1.0 + kappa - height_max_m / rise_m))


def build_planform(
    g: SurfaceGenes,
    area_m2: float,
    root_mid_m: Vec3,
    y_limit_m: float,
    mitre_root: bool,
    outer_height_max_m: float = np.inf,
) -> Planform:
    """Sections of the right/centre instance for true area area_m2 (all instances).

    mitre_root puts the root section in the y = 0 plane so mirrored halves meet flush there.
    outer_height_max_m caps the z extent of the outer panel by moving the break outboard (area
    and aspect ratio keep their values); it is chosen from the unclipped semi-span, so the
    span limit can only shorten the panel further.
    """
    half_area = area_m2 / 2.0 if g.mirrored else area_m2
    tip = tip_shape(g.tip_style)
    kappa = tip.span_frac if tip else 0.0
    unclipped_m = _semi_span(g, half_area, root_mid_m[1], np.inf)
    g = replace(g, break_frac=_capped_break_frac(g, unclipped_m, kappa, outer_height_max_m))
    semi = _semi_span(g, half_area, root_mid_m[1], y_limit_m)
    lam = g.taper
    c_root = half_area / (semi * ((1 + lam) / 2 + kappa * lam * _ext_chord_integral(tip)))
    strake = g.root_chord_scale
    if (
        strake != 1.0
    ):  # the stretched root adds area to the inner panel: shrink every chord to keep it
        inner_m = g.break_frac * semi
        c_root *= half_area / (half_area + inner_m * (strake - 1.0) * c_root / 2.0)
    c_root_section = strake * c_root
    qc0 = np.asarray(root_mid_m) - np.array([0.25 * c_root_section, 0.0, 0.0])
    d1 = _direction(g.sweep_deg, g.cant_deg)
    d2 = _direction(g.outer_sweep_deg, g.outer_cant_deg)
    sections = [_section(qc0, c_root_section, 0.0 if mitre_root else g.cant_deg)]
    qc_break = qc0 + g.break_frac * semi * d1
    if g.break_frac < 1.0:
        c_break = c_root * (1 - (1 - lam) * g.break_frac)
        sections.append(_section(qc_break, c_break, (g.cant_deg + g.outer_cant_deg) / 2))
    tip_section = _section(
        qc_break + (1 - g.break_frac) * semi * d2, lam * c_root, g.outer_cant_deg
    )
    sections.append(tip_section)
    n_structural = len(sections)
    if tip is not None:
        sections += _tip_sections(tip_section, tip, kappa * semi, d2)
    aspect = (2 * semi) ** 2 / (2 * half_area)
    note = ""
    if g.align_angles_deg and g.mirrored:
        snap = snap_sections(tuple(sections), g.align_angles_deg, root_mid_m[0])
        sections, note = list(snap.sections), snap.note
        c_root_section = sections[0].chord_m
    steps = stepped_levels(g.twist_deg, g.coarseness, n_structural - 1) if g.airfoil_facets else 0
    sections = _twisted(sections, n_structural, g.twist_deg, steps)
    if steps:
        step_note = f"coarseness: washout {g.twist_deg:.1f} deg in {steps} step(s), not linear"
        note = f"{note}; {step_note}" if note else step_note
    return Planform(tuple(sections), n_structural, aspect, g.tip_style, c_root_section, note)


def _tip_sections(tip: Section, shape: TipShape, ext_m: float, d2: np.ndarray) -> list[Section]:
    """Tip-extension sections holding the chord point at align_chord_frac."""
    held = np.asarray(tip.le_m) + np.array([shape.align_chord_frac * tip.chord_m, 0.0, 0.0])
    out = []
    for e, k in shape.stations:
        chord = k * tip.chord_m
        le = held + e * ext_m * d2 - np.array([shape.align_chord_frac * chord, 0.0, 0.0])
        out.append(
            Section(
                le_m=(float(le[0]), float(le[1]), float(le[2])),
                chord_m=chord,
                cant_deg=tip.cant_deg,
            )
        )
    return out


def joined_root_chord(g: SurfaceGenes, area_m2: float) -> float:
    """Root chord [m] of a joined panel of true area area_m2 (both instances)."""
    half_area = area_m2 / 2.0
    semi = float(np.sqrt(2.0 * g.aspect_ratio * half_area) / 2.0)
    return half_area / (semi * (1 + g.taper) / 2)


def joined_sweep_deg(pf: Planform) -> float:
    """Quarter-chord sweep [deg] of a two-section (joined) panel."""
    root, tip = pf.sections[0], pf.sections[-1]
    dx = (tip.le_m[0] + 0.25 * tip.chord_m) - (root.le_m[0] + 0.25 * root.chord_m)
    span = float(np.hypot(tip.le_m[1] - root.le_m[1], tip.le_m[2] - root.le_m[2]))
    return float(np.degrees(np.arctan2(dx, span)))


def join_too_swept(pf: Planform) -> bool:
    """A joined panel steeper than join_max_sweep_deg reads as tangled."""
    return abs(joined_sweep_deg(pf)) > number("join_max_sweep_deg")


def build_joined(
    g: SurfaceGenes, area_m2: float, root_mid_m: Vec3, target_tip: Section
) -> Planform | None:
    """Single panel from the own root to a shared target tip; None if root and tip coincide."""
    half_area = area_m2 / 2.0
    c_root = joined_root_chord(g, area_m2)
    root_le = np.asarray(root_mid_m) - np.array([0.5 * c_root, 0.0, 0.0])
    tip_le = np.asarray(target_tip.le_m)
    dy, dz = tip_le[1] - root_le[1], tip_le[2] - root_le[2]
    length = float(np.hypot(dy, dz))
    if length < number("join_min_length_m"):
        return None
    cant = float(np.degrees(np.arctan2(dz, dy)))
    root = Section(
        le_m=(float(root_le[0]), float(root_le[1]), float(root_le[2])),
        chord_m=c_root,
        cant_deg=cant,
    )
    aspect = (2 * length) ** 2 / (2 * half_area)
    return Planform((root, target_tip), 2, aspect, "square", c_root)


def section_areas(sections: tuple[Section, ...]) -> tuple[float, float]:
    """(true, xy-projected) area of one instance [m^2] from trapezoids between sections."""
    true, projected = 0.0, 0.0
    for a, b in pairwise(sections):
        chord = (a.chord_m + b.chord_m) / 2
        dy, dz = b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]
        true += chord * float(np.hypot(dy, dz))
        projected += chord * abs(dy)
    return true, projected


def choose_spar(sections: tuple[Section, ...], g: SurfaceGenes, tc: float) -> Spar | None:
    """Largest stock tube fitting local thickness minus two skins on every structural section."""
    frac = number("spar_chord_frac")
    n_af = int(number("airfoil_points_per_side"))
    t_frac = local_thickness_frac(
        g.camber_frac, g.camber_loc_frac, tc, frac, n_af, g.airfoil_facets
    )
    room = t_frac * min(s.chord_m for s in sections) - 2 * number("spar_skin_thickness_m")
    tubes = constant("constants", "spar_tubes_m")
    if not isinstance(tubes, list):
        raise TypeError("spar_tubes_m must be a list")
    fitting = [(float(od), float(i_d)) for od, i_d in tubes if float(od) <= room]
    if not fitting:
        return None
    od, i_d = max(fitting)
    return Spar(od_m=od, id_m=i_d, chord_frac=frac)


def fit_tc(g: SurfaceGenes, pf: Planform) -> tuple[float, Spar] | None:
    """Thinnest tc from g.tc up to tc_max (in tc_step) at which a stock spar fits."""
    structural = pf.sections[: pf.n_structural]
    tc_max, step = number("tc_max"), number("tc_step")
    k = 0
    while (tc := min(g.tc + k * step, tc_max)) <= tc_max:
        spar = choose_spar(structural, g, tc)
        if spar is not None:
            return tc, spar
        if tc >= tc_max:
            return None
        k += 1
    return None


def added_fin_spec(fam: Family, tc: float, index: int) -> SurfaceGenes:
    """Centreline fin at the aft end of the centre host (user Deviation, step 3 gate r4), added
    when a design has no panel canted over vertical_cant_deg; family sweep, taper and tip."""
    return SurfaceGenes(
        index=index,
        host="centre",
        mirrored=False,
        root_x_frac=1.0,
        root_z_frac=number("added_fin_root_z_frac"),
        weight=0.0,
        aspect_ratio=number("added_fin_aspect_ratio"),
        sweep_deg=fam.sweep_deg,
        cant_deg=90.0,
        break_frac=1.0,
        outer_sweep_deg=fam.sweep_deg,
        outer_cant_deg=90.0,
        taper=fam.taper,
        tip_style=fam.tip_style,
        camber_frac=0.0,
        camber_loc_frac=fam.camber_loc_frac,
        tc=tc,
        control_frac=0.0,
        join_target=None,
        area_share=number("added_fin_area_frac"),
        airfoil_facets=fam.airfoil_facets,
    )
