"""SPEC §5 decoder invariants on 1,000 seeded random genomes (stub brief) + 400 (small boxes).

Prop clearance and wing-box fit use independent samplers in test_clearance.py.
"""

import itertools

import numpy as np

from aeroforge.airfoil import local_thickness_frac
from aeroforge.constants import constant, number
from aeroforge.faceted_body import firewall_attach
from aeroforge.genetable import family_of, gene_table, unpack
from aeroforge.genome import decode, fin_capped
from aeroforge.layout import host_end
from aeroforge.models import Body, Design, Requirements, Surface
from aeroforge.shapes import MIRROR_Y, component_points, inside_body

TOL = 1e-9
CHORD_POINTS = 11
OUTER_PANEL_MIN_STRUCTURAL = 3  # root, break and tip sections: a surface with an outer panel
STEEP_CANT_DEG = 60.0  # the plan's trigger; panels below it exceeded the cap too (probe, step 7)


def test_brief_exercises_its_box_host(
    decoded: list[Design], brief: Requirements, req: Requirements
) -> None:
    in_wing = any(b.host == "surface0" for d in decoded for b in d.boxes)
    # User decision A1 (upgrade U0): small_boxes must use the wing host; stub may (a large wing
    # area scale lets a wing_root layout hold the boxes).
    if brief != req:
        assert in_wing


def _bodies(d: Design) -> dict[str, Body]:
    return {b.name: b for b in d.bodies}


def _surface(d: Design, name: str) -> Surface:
    return next(s for s in d.surfaces if s.name == name)


def _root_chord_points(s: Surface) -> np.ndarray:
    root = s.sections[0]
    t = np.linspace(0, 1, CHORD_POINTS)[:, None]
    return np.asarray(root.le_m) + t * np.array([root.chord_m, 0, 0])


def _index(s: Surface) -> int:
    return int(s.name.removeprefix("surface"))


def _sorted_rows(p: np.ndarray) -> np.ndarray:
    q = np.round(p, 7)
    return q[np.lexsort(q.T)]


def test_part_points_are_mirror_symmetric_about_y0(decoded: list[Design]) -> None:
    for d in decoded:
        pts = component_points(d, include_props=False)
        for name, p in pts.items():
            partner = name[:-1] + "R" if name.endswith("_L") else name
            if name.endswith("_R"):
                continue
            assert np.array_equal(_sorted_rows(p), _sorted_rows(pts[partner] * MIRROR_Y)), name
        assert all(p.hub_m[1] == 0.0 for p in d.propulsors if not p.mirrored)


def test_whole_root_chord_lies_on_its_host(decoded: list[Design]) -> None:
    for d in decoded:
        bodies = _bodies(d)
        for s in d.surfaces:
            pts = _root_chord_points(s)
            if s.host in bodies:
                assert inside_body(bodies[s.host], pts).all(), s.name
            elif s.host == "surface0":
                host = _surface(d, "surface0").sections[0]
                assert pts[0, 0] >= host.le_m[0] - TOL, s.name
                assert pts[-1, 0] <= host.le_m[0] + host.chord_m + TOL, s.name
                assert pts[0, 1] == 0.0 and abs(pts[0, 2] - host.le_m[2]) < TOL
            else:
                assert s.name == "surface0" and s.host == "centre"


def _surface0_chord_at(d: Design, y: float) -> tuple[float, float]:
    """(le x, te x) of surface0's inner panel at span station y."""
    a, b = _surface(d, "surface0").sections[:2]
    t = (y - a.le_m[1]) / (b.le_m[1] - a.le_m[1])
    le = a.le_m[0] + t * (b.le_m[0] - a.le_m[0])
    return le, le + a.chord_m + t * (b.chord_m - a.chord_m)


def _roots_by_host(d: Design) -> dict[str, list[tuple[float, float, float]]]:
    """(le x, te x, cant) of every root per host; a pod also carries surface0's chord."""
    out: dict[str, list[tuple[float, float, float]]] = {}
    for s in d.surfaces:
        r = s.sections[0]
        out.setdefault(s.host, []).append((r.le_m[0], r.le_m[0] + r.chord_m, s.cant_deg))
    for b in d.bodies:
        if b.name == "pod":
            le, te = _surface0_chord_at(d, b.nose_m[1])
            out.setdefault("pod", []).append((le, te, d.surfaces[0].cant_deg))
    return out


def test_similar_cant_roots_on_one_host_keep_the_separation_gap(decoded: list[Design]) -> None:
    similar, frac = number("surface_separation_cant_deg"), number("surface_gap_chord_frac")
    checked = 0
    for d in decoded:
        for host, roots in _roots_by_host(d).items():
            for a, b in itertools.combinations(roots, 2):
                if abs(a[2] - b[2]) >= similar:
                    continue
                checked += 1
                gap = frac * max(a[1] - a[0], b[1] - b[0])
                assert b[0] >= a[1] + gap - TOL or a[0] >= b[1] + gap - TOL, host
    assert checked > 0


def _chord_line_at_y(s: Surface, y: float) -> np.ndarray:
    les = np.array([sec.le_m for sec in s.sections[: s.n_structural]])
    chords = np.array([sec.chord_m for sec in s.sections[: s.n_structural]])
    le = np.array([np.interp(y, les[:, 1], les[:, k]) for k in range(3)])
    chord = float(np.interp(y, les[:, 1], chords))
    return le + np.linspace(0, 1, CHORD_POINTS)[:, None] * np.array([chord, 0, 0])


def test_pods_sit_on_surface0(decoded: list[Design]) -> None:
    for d in decoded:
        for b in d.bodies:
            if b.name == "pod":
                s0 = _surface(d, "surface0")
                assert inside_body(b, _chord_line_at_y(s0, b.nose_m[1])).any()


def test_joins_and_pods_coexist(full_designs: list[Design]) -> None:
    both = [
        d
        for d in full_designs
        if any(s.joined_to for s in d.surfaces) and any(b.name == "pod" for b in d.bodies)
    ]
    assert both


def test_propellers_attach_to_their_host_end(decoded: list[Design]) -> None:
    for d in decoded:
        bodies = _bodies(d)
        for p in d.propulsors:
            if p.host in bodies:
                expected = host_end(bodies[p.host], p.pusher)[0]
                # a flat-panel end too slim for a motor plate: the firewall sits inside it
                expected = firewall_attach(bodies[p.host], expected, p.diameter_m, p.pusher)
            else:
                root = _surface(d, "surface0").sections[0]
                expected = np.asarray(root.le_m) + np.array([root.chord_m if p.pusher else 0, 0, 0])
            assert np.allclose(p.attach_m, expected, atol=TOL)


def test_projected_area_equals_required_area(decoded: list[Design], brief: Requirements) -> None:
    for d in decoded:
        total = sum(s.projected_area_m2 for s in d.surfaces)
        assert abs(total - d.lifting_area_m2) <= 1e-7 * d.lifting_area_m2


def test_span_stays_within_limit(decoded: list[Design], brief: Requirements) -> None:
    limit = brief.span_max_m / 2 + TOL
    for d in decoded:
        for s in d.surfaces:
            for sec in s.sections:
                assert abs(sec.le_m[1]) <= limit
        for b in d.bodies:
            assert abs(b.nose_m[1]) + b.half_width_m <= limit
        for p in d.propulsors:
            assert abs(p.hub_m[1]) + p.diameter_m / 2 <= limit


def _fins(d: Design) -> list[Surface]:
    return [
        s for s in d.surfaces if s.joined_to is None and abs(s.cant_deg) >= number("fin_cant_deg")
    ]


def _fin_height_m(s: Surface) -> float:
    return float(np.ptp([sec.le_m[2] for sec in s.sections]))


def test_fins_stay_within_the_area_and_height_caps(
    decoded: list[Design], brief: Requirements
) -> None:
    height_max = number("fin_height_max_ref_frac") * brief.ref_length_m
    fins = [(d, s) for d in decoded for s in _fins(d)]
    assert fins, "seeded genomes should include fins"
    assert all(
        s.area_m2 <= number("fin_area_max_frac") * d.lifting_area_m2 * (1 + TOL) for d, s in fins
    )
    fins = [s for _, s in fins]
    assert all(_fin_height_m(s) <= height_max * (1 + TOL) for s in fins)


def _outer_panel_height_m(s: Surface) -> float | None:
    """z extent [m] from the break section outward, tip extension included (None: one panel)."""
    if s.joined_to is not None or s.n_structural < OUTER_PANEL_MIN_STRUCTURAL:
        return None
    return float(np.ptp([sec.le_m[2] for sec in s.sections[1:]]))


def test_no_outer_panel_is_taller_than_the_cap(decoded: list[Design], brief: Requirements) -> None:
    height_max = number("outer_panel_height_max_ref_frac") * brief.ref_length_m
    heights = [
        h for d in decoded for s in d.surfaces if (h := _outer_panel_height_m(s)) is not None
    ]
    assert heights, "seeded genomes should include broken (two-panel) surfaces"
    assert max(heights) <= height_max * (1 + TOL)


def test_the_outer_panel_cap_binds_on_steep_and_shallow_panels(
    decoded: list[Design], brief: Requirements
) -> None:
    height_max = number("outer_panel_height_max_ref_frac") * brief.ref_length_m
    at_cap = [
        s
        for d in decoded
        for s in d.surfaces
        if (h := _outer_panel_height_m(s)) is not None and np.isclose(h, height_max)
    ]
    cants = [abs(s.sections[-1].cant_deg) for s in at_cap]
    assert any(c >= STEEP_CANT_DEG for c in cants), "no steep panel reached the cap"
    assert any(c < STEEP_CANT_DEG for c in cants), "no shallow panel reached the cap"


def test_a_capped_design_has_a_fin_at_a_cap(decoded: list[Design], brief: Requirements) -> None:
    height_max = number("fin_height_max_ref_frac") * brief.ref_length_m
    capped = [d for d in decoded if fin_capped(d)]
    assert capped, "seeded genomes should hit the fin cap"
    tip_y_max = brief.span_max_m / 2
    for d in capped:  # the span limit may shorten a capped fin below the height cap
        area_max = number("fin_area_max_frac") * d.lifting_area_m2
        assert any(
            np.isclose(s.area_m2, area_max)
            or np.isclose(_fin_height_m(s), height_max)
            or np.isclose(max(abs(sec.le_m[1]) for sec in s.sections), tip_y_max)
            for s in _fins(d)
        )


def test_surface0_is_always_present(decoded: list[Design]) -> None:
    assert all(d.surfaces[0].name == "surface0" for d in decoded)


def test_no_surface_is_below_min_area_share(decoded: list[Design]) -> None:
    assert all(
        s.area_m2 >= number("min_area_share") * d.lifting_area_m2
        for d in decoded
        for s in d.surfaces[1:]
    )


def test_every_design_carries_the_three_boxes(decoded: list[Design]) -> None:
    assert all(
        sorted(b.name for b in d.boxes) == ["avionics", "battery", "payload"] for d in decoded
    )


def _box_corners(
    centre: tuple[float, float, float], size: tuple[float, float, float]
) -> np.ndarray:
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    return np.asarray(centre) + signs * np.asarray(size) / 2


def test_body_hosted_boxes_fit_inside_their_body(decoded: list[Design]) -> None:
    for d in decoded:
        bodies = _bodies(d)
        for box in d.boxes:
            if box.host in bodies:
                assert inside_body(bodies[box.host], _box_corners(box.centre_m, box.size_m)).all()


def _tubes() -> list[tuple[float, float]]:
    raw = constant("constants", "spar_tubes_m")
    assert isinstance(raw, list)
    return [(float(od), float(i_d)) for od, i_d in raw]


def test_spar_is_largest_tube_fitting_the_whole_spar_line(decoded: list[Design]) -> None:
    skin = number("spar_skin_thickness_m")
    for d in decoded:
        for s in d.surfaces:
            t = local_thickness_frac(s.camber_frac, s.camber_loc_frac, s.tc, s.spar.chord_frac, 40)
            room = min(t * sec.chord_m for sec in s.sections[: s.n_structural]) - 2 * skin
            assert s.spar.od_m <= room
            assert all(od <= s.spar.od_m or od > room for od, _ in _tubes())


def test_tip_joins_share_tip_point_and_chord(decoded: list[Design]) -> None:
    joined = 0
    for d in decoded:
        for s in d.surfaces:
            if s.joined_to is None:
                continue
            joined += 1
            target = _surface(d, s.joined_to)
            mine, theirs = s.sections[s.n_structural - 1], target.sections[target.n_structural - 1]
            assert mine.le_m == theirs.le_m and mine.chord_m == theirs.chord_m
            assert target.tip_style == "square"
    assert joined > 0


def _delta_bounds(name: str) -> tuple[float, float]:
    gene = next(g for g in gene_table() if g.name == name)
    return gene.lo, gene.hi


def _is_fin(v_num: dict[str, float], dihedral_deg: float, i: int) -> bool:
    cant = float(np.clip(v_num[f"s{i}_cant_deg"] + dihedral_deg, -90.0, 90.0))
    return abs(cant) >= number("fin_snap_cant_deg")


def test_style_family_is_respected(decoded: list[Design], genomes: np.ndarray) -> None:
    tc_max = number("tc_max")
    for d, g in zip(decoded, genomes[: len(decoded)], strict=True):
        v = unpack(g)
        fam = family_of(v)
        # the decoder owns two things of a flat-panel family: the end of the shape (U2 follow-up 1:
        # tail size, flat cap, blunt nose, for a motor plate) and where the upsweep starts
        # (follow-up 2); everything else is the genes'
        assert d.family.model_copy(update={"shape": fam.shape, "body_rise_x_frac": None}) == fam
        assert d.family.body_rise_x_frac is None or d.family.shape.style == "faceted"
        changed = {k for k, x in d.family.shape if getattr(fam.shape, k) != x}
        assert changed <= {"nose", "tail", "tail_end_size_frac"}
        for b in d.bodies:
            fineness = b.length_m / (2 * max(b.half_width_m, b.half_height_m))
            packed = b.name == "body0" and fam.box_layout in ("side_by_side", "stacked")
            if packed and any(x.host == "body0" for x in d.boxes):
                # B1 (user, upgrade U0): the fineness gene is measured on the inline diameter, so
                # a wider packed section reads at most the gene; equality holds everywhere else.
                assert fineness <= fam.body_fineness + 1e-9
            elif b.facets is not None and any(p.host == b.name for p in d.propulsors):
                # U2 audit: a flat-panel host of a motor may be fatter (end sized for the plate)
                assert fineness <= fam.body_fineness + 1e-9
                assert fineness >= 1.0 / (2.0 * number("pod_radius_max_length_frac")) - 1e-9
            else:
                assert abs(fineness - fam.body_fineness) < 1e-9
            assert b.section_exp == fam.body_section_exp
        for s in d.surfaces:
            i = _index(s)
            assert fam.tc - TOL <= s.tc <= tc_max + TOL
            assert s.camber_loc_frac == fam.camber_loc_frac
            assert s.camber_frac == (
                0.0 if _is_fin(v.num, fam.dihedral_deg, i) else fam.camber_frac
            )
            if s.joined_to is not None:
                continue
            assert s.tip_style in (fam.tip_style, "square")
            sign = 1.0 if v.pick[f"s{i}_sweep_sign"] == "aft" else -1.0
            lo, hi = _delta_bounds(f"s{i}_sweep_delta_deg")
            assert lo - TOL <= s.sweep_deg - sign * fam.sweep_deg <= hi + TOL
            assert s.taper == fam.taper if i else s.taper >= fam.taper


def test_joined_surfaces_keep_the_join_style(decoded: list[Design]) -> None:
    limit = number("join_max_sweep_deg")
    joined = [s for d in decoded for s in d.surfaces if s.joined_to is not None]
    assert joined
    for s in joined:
        assert s.tip_style == "square" and s.n_structural == len(s.sections) == 2
        assert abs(s.sweep_deg) <= limit + TOL
        assert s.mirrored


def test_decode_is_deterministic(
    genomes: np.ndarray, decoded: list[Design], brief: Requirements
) -> None:
    for g, d in zip(genomes[:100], decoded[:100], strict=True):
        again = decode(g.copy(), brief)
        assert again == d and again.model_dump_json() == d.model_dump_json()
