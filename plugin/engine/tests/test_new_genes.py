"""Novelty plan step 10: the new shape genes (layout, area scale, strake, twist, body genes, box
layouts), each exact at its default, and the geometry each one is supposed to give."""

from itertools import pairwise

import numpy as np
import pytest

from aeroforge.airplane import wing
from aeroforge.constants import load_table, number
from aeroforge.fitcheck import fit_margins
from aeroforge.genetable import family_of, gene_table, unpack
from aeroforge.genome import decode
from aeroforge.layout import arrange_boxes, box_sizes, size_centre_body
from aeroforge.models import Body, Design, Requirements, Section, Surface
from aeroforge.reference import BASELINE, encode, reference_genome, reference_names
from aeroforge.shapes import body_axis_z, body_stations, section_incidence_deg, section_outline

LAYOUTS = ("free", "conventional", "canard", "tandem", "three_surface", "flying_wing", "joined")
LIFTING_CANT_DEG = 30.0  # a panel flatter than this reads as a lifting surface, steeper as a fin
TAIL_AREA_RATIO = 0.35  # an aft lifting surface this big against surface0 reads as a tandem wing
TOL = 1e-9


def _with(genome: np.ndarray, **values: float | str) -> np.ndarray:
    """genome with named genes set from physical values or choice labels."""
    out = np.array(genome, dtype=float)
    index = {g.name: i for i, g in enumerate(gene_table())}
    for name, value in values.items():
        g = gene_table()[index[name]]
        if g.kind == "choice":
            out[index[name]] = (g.choices.index(str(value)) + 0.5) / len(g.choices)
        else:
            out[index[name]] = (float(value) - g.lo) / (g.hi - g.lo)
    return out


def _base() -> np.ndarray:
    return reference_genome(BASELINE)


def _body0(d: Design) -> Body:
    return next(b for b in d.bodies if b.name == "body0")


# --- ref_default and the layout table ---------------------------------------------------------


def test_unlisted_genes_take_their_ref_default_else_the_range_middle() -> None:
    v = unpack(encode({}))
    for g in gene_table():
        if g.kind == "float":
            want = g.ref_default if g.ref_default is not None else (g.lo + g.hi) / 2
            assert v.num[g.name] == pytest.approx(want), g.name


def test_every_reference_sits_at_the_defaults_of_the_new_genes() -> None:
    for name in reference_names():
        v = unpack(reference_genome(name))
        assert v.pick["layout"] == "free" and v.pick["box_layout"] == "inline", name
        assert v.num["wing_area_scale"] == pytest.approx(1.0), name
        assert v.num["body_width_ratio"] == pytest.approx(1.0), name
        assert v.num["body_tail_upsweep_deg"] == pytest.approx(0.0, abs=TOL), name
        for i in range(4):
            assert v.num[f"s{i}_root_chord_scale"] == pytest.approx(1.0), name
            assert v.num[f"s{i}_twist_deg"] == pytest.approx(0.0, abs=TOL), name


def test_the_layout_table_lists_every_named_layout_with_valid_genes() -> None:
    table = load_table("layouts")["layouts"]
    assert set(table) == set(LAYOUTS) - {"free"}
    rows = {g.name: g for g in gene_table()}
    for name, entries in table.items():
        for key, value in entries.items():
            g = rows[key]
            if g.kind == "choice":
                assert value in g.choices, (name, key)
            else:
                assert g.lo <= value <= g.hi, (name, key)


def test_a_named_layout_sets_the_slot_roles_and_the_slot_gene_fine_tunes_it() -> None:
    base = _with(_base(), layout="canard")
    mid = unpack(_with(base, s1_root_x_frac=0.5)).num["s1_root_x_frac"]
    low = unpack(_with(base, s1_root_x_frac=0.05)).num["s1_root_x_frac"]
    high = unpack(_with(base, s1_root_x_frac=0.95)).num["s1_root_x_frac"]
    canard_x = load_table("layouts")["layouts"]["canard"]["s1_root_x_frac"]
    assert mid == pytest.approx(canard_x)
    assert 0.05 <= low <= mid <= high <= 0.95
    assert high - low <= number("layout_tune_frac") * 0.9 + TOL  # a share of the gene's range


def test_the_free_layout_leaves_the_slot_genes_alone() -> None:
    free = unpack(_with(_base(), layout="free", s1_root_x_frac=0.77))
    assert free.num["s1_root_x_frac"] == pytest.approx(0.77)


def test_the_decoded_design_records_its_layout_label(
    genomes: np.ndarray, designs: list[Design]
) -> None:
    labels = {unpack(g).pick["layout"] for g in genomes}
    assert labels == set(LAYOUTS) and all(
        d.family.layout == unpack(g).pick["layout"] for g, d in zip(genomes, designs, strict=True)
    )


def _lifting(d: Design) -> list[Surface]:
    return [s for s in d.surfaces if s.mirrored and abs(s.cant_deg) < LIFTING_CANT_DEG]


def _root_x(s: Surface) -> float:
    return s.sections[0].le_m[0] + s.sections[0].chord_m / 2


def _forward(d: Design) -> bool:
    s0 = d.surfaces[0]
    return any(_root_x(s) < _root_x(s0) for s in _lifting(d) if s.name != s0.name)


def _aft(d: Design) -> list[Surface]:
    s0 = d.surfaces[0]
    return [s for s in _lifting(d) if s.name != s0.name and _root_x(s) > _root_x(s0)]


PREDICATES = {
    "free": lambda d: True,
    "conventional": lambda d: bool(_aft(d)) and not _forward(d),
    "canard": _forward,
    "tandem": lambda d: any(s.area_m2 >= TAIL_AREA_RATIO * d.surfaces[0].area_m2 for s in _aft(d)),
    "three_surface": lambda d: _forward(d) and bool(_aft(d)),
    "flying_wing": lambda d: len(_lifting(d)) == 1,
    "joined": lambda d: any(s.joined_to is not None for s in d.surfaces),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_some_random_genome_decodes_to_each_layout_by_its_geometry(
    layout: str, genomes: np.ndarray, designs: list[Design]
) -> None:
    """Read the geometry back (not the gene label): forward surface for canard, tip join for
    joined, no other lifting surface for flying_wing, and so on."""
    labelled = [
        d for g, d in zip(genomes, designs, strict=True) if unpack(g).pick["layout"] == layout
    ]
    assert labelled, f"no random genome carries layout {layout}"
    assert any(PREDICATES[layout](d) for d in labelled), layout


# --- area scale ---------------------------------------------------------------------------------


def test_wing_area_scale_one_keeps_the_stall_area_and_two_doubles_it(req: Requirements) -> None:
    one = decode(_with(_base(), wing_area_scale=1.0), req)
    two = decode(_with(_base(), wing_area_scale=2.0), req)
    assert one.lifting_area_m2 == pytest.approx(req.lifting_area_min_m2, rel=TOL)
    assert two.lifting_area_m2 == pytest.approx(2.0 * req.lifting_area_min_m2, rel=TOL)
    assert sum(s.projected_area_m2 for s in two.surfaces) == pytest.approx(
        two.lifting_area_m2, rel=1e-7
    )


def test_the_area_stops_growing_where_the_span_limit_would_force_the_minimum_aspect_ratio(
    req: Requirements,
) -> None:
    cap_m2 = req.span_max_m**2 / number("derive_min_aspect_ratio")
    tight = req.model_copy(update={"span_max_m": req.span_max_m * 0.7})
    d = decode(_with(_base(), wing_area_scale=3.0), tight)
    assert (
        d.lifting_area_m2
        <= max(tight.span_max_m**2 / number("derive_min_aspect_ratio"), tight.lifting_area_min_m2)
        + TOL
    )
    assert d.lifting_area_m2 < 3.0 * tight.lifting_area_min_m2 and cap_m2 > 0
    span = 2 * max(abs(sec.le_m[1]) for s in d.surfaces if s.mirrored for sec in s.sections)
    assert span <= tight.span_max_m * (1 + 1e-7)


def test_no_design_has_less_than_the_stall_area(designs: list[Design], req: Requirements) -> None:
    assert all(d.lifting_area_m2 >= req.lifting_area_min_m2 * (1 - TOL) for d in designs)


# --- strake and twist ---------------------------------------------------------------------------


def test_a_strake_stretches_the_root_chord_and_keeps_the_area_and_aspect_ratio(
    req: Requirements,
) -> None:
    plain = decode(_with(_base(), s0_root_chord_scale=1.0), req).surfaces[0]
    strake = decode(_with(_base(), s0_root_chord_scale=1.8), req).surfaces[0]
    assert strake.sections[0].chord_m > 1.2 * plain.sections[0].chord_m
    assert strake.sections[-1].chord_m < plain.sections[-1].chord_m  # the rest shrinks to keep S
    assert strake.projected_area_m2 == pytest.approx(plain.projected_area_m2, rel=1e-6)


def test_twist_is_a_linear_washout_from_zero_at_the_root_to_the_gene_at_the_tip(
    req: Requirements,
) -> None:
    s = decode(_with(_base(), s0_twist_deg=-4.0, s0_break_frac=0.6), req).surfaces[0]
    twists = [sec.twist_deg for sec in s.sections[: s.n_structural]]
    assert twists[0] == 0.0 and twists[-1] == pytest.approx(-4.0)
    assert all(a >= b for a, b in pairwise(twists))
    assert all(sec.twist_deg == pytest.approx(-4.0) for sec in s.sections[s.n_structural :])


def test_zero_twist_leaves_every_section_untwisted(req: Requirements) -> None:
    d = decode(_base(), req)
    assert all(sec.twist_deg == 0.0 for s in d.surfaces for sec in s.sections)


def test_a_centreline_fin_ignores_twist(req: Requirements) -> None:
    d = decode(_with(_base(), s2_twist_deg=-5.0), req)
    fins = [s for s in d.surfaces if not s.mirrored]
    assert fins and all(sec.twist_deg == 0.0 for s in fins for sec in s.sections)


def test_a_washed_out_tip_turns_nose_down_in_the_outline_and_in_the_aero_stations(
    req: Requirements,
) -> None:
    s = decode(_with(_base(), s0_twist_deg=-4.0), req).surfaces[0]
    root, tip = s.sections[0], s.sections[s.n_structural - 1]
    assert section_incidence_deg(s, root) == pytest.approx(s.incidence_deg)
    assert section_incidence_deg(s, tip) == pytest.approx(s.incidence_deg - 4.0)

    def rise(sec: Section) -> float:
        """Trailing-edge height above the leading edge, per chord."""
        return (section_outline(sec, s, 16)[:, 2].mean() - sec.le_m[2]) / sec.chord_m

    assert rise(tip) > rise(root)
    twists = [x.twist for x in wing(s, 0.0, False).xsecs]
    assert twists[0] == pytest.approx(s.incidence_deg)
    assert twists[-1] == pytest.approx(s.incidence_deg - 4.0)


# --- body genes ---------------------------------------------------------------------------------


def test_the_body_nose_and_tail_cones_follow_their_genes(req: Requirements) -> None:
    b = _body0(decode(_with(_base(), body_nose_frac=0.3, body_tail_frac=0.2), req))
    assert b.nose_frac == pytest.approx(0.3) and b.tail_frac == pytest.approx(0.2)


def test_a_tail_upsweep_bends_the_axis_up_over_the_tail_cone_only(req: Requirements) -> None:
    wing_only = {"s1_active": "off", "s2_active": "off", "s3_active": "off"}
    flat = _body0(decode(_with(_base(), **wing_only), req))
    bent = _body0(decode(_with(_base(), body_tail_upsweep_deg=10.0, **wing_only), req))
    t = body_stations(40)
    assert np.array_equal(body_axis_z(flat, t), np.full_like(t, flat.nose_m[2]))
    rise = body_axis_z(bent, t) - bent.nose_m[2]
    assert rise[-1] == pytest.approx(bent.length_m * bent.tail_frac * np.tan(np.radians(10.0)))
    assert np.all(rise[t <= 1.0 - bent.tail_frac] == 0.0) and np.all(np.diff(rise) >= 0.0)


def test_a_tail_upsweep_is_dropped_when_a_root_chord_sits_on_the_rising_tail_cone(
    req: Requirements,
) -> None:
    d = decode(_with(_base(), body_tail_upsweep_deg=10.0), req)  # the conventional tail root is aft
    assert _body0(d).tail_upsweep_deg == 0.0
    assert any("tail upsweep dropped" in n for n in d.notes)


def test_the_width_ratio_multiplies_the_box_fit_aspect_and_the_boxes_still_fit(
    req: Requirements,
) -> None:
    base = _body0(decode(_base(), req))
    aspect = base.half_width_m / base.half_height_m
    for ratio in (0.6, 1.5, 2.0):
        d = decode(_with(_base(), body_width_ratio=ratio), req)
        b = _body0(d)
        assert b.half_width_m / b.half_height_m == pytest.approx(aspect * ratio, rel=1e-6)
        assert min(fit_margins(d, req).values()) >= 0.0


def test_a_widened_body_keeps_its_fineness(req: Requirements) -> None:
    fineness = unpack(_with(_base(), body_width_ratio=1.8)).num["fam_body_fineness"]
    b = _body0(decode(_with(_base(), body_width_ratio=1.8), req))
    assert b.length_m / (2 * max(b.half_width_m, b.half_height_m)) == pytest.approx(fineness)


# --- box layouts --------------------------------------------------------------------------------


def _aabbs_apart(sizes: dict, offsets: dict) -> bool:
    names = list(sizes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            gap = [
                abs(offsets[a][k] - offsets[b][k]) - (sizes[a][k] + sizes[b][k]) / 2
                for k in range(3)
            ]
            if max(gap) < 0.0:
                return False
    return True


@pytest.mark.parametrize("layout", ["inline", "side_by_side", "stacked", "wing_root"])
def test_boxes_do_not_overlap_in_any_layout_and_sit_inside_their_envelope(
    layout: str, req: Requirements
) -> None:
    sizes = box_sizes(req)
    env, offsets = arrange_boxes(sizes, layout)
    assert _aabbs_apart(sizes, offsets)
    for name, size in sizes.items():
        for k in range(3):
            assert abs(offsets[name][k]) + size[k] / 2 <= env[k] / 2 + TOL


def test_side_by_side_and_stacked_boxes_shorten_the_stack_floor_of_the_body(
    req: Requirements,
) -> None:
    sizes = box_sizes(req)
    inline = arrange_boxes(sizes, "inline")[0][0]
    assert arrange_boxes(sizes, "side_by_side")[0][0] < inline
    assert arrange_boxes(sizes, "stacked")[0][0] < inline


@pytest.mark.parametrize("layout", ["side_by_side", "stacked"])
def test_a_packed_body_is_never_longer_than_the_inline_body_and_never_below_the_box_fit(
    layout: str, req: Requirements
) -> None:
    """B1 (user, upgrade U0): fineness is measured on the inline diameter."""
    fam = family_of(unpack(_with(_base(), box_layout=layout)))
    inline = size_centre_body(req, fam.model_copy(update={"box_layout": "inline"}), 0.5, 0.0)
    packed = size_centre_body(req, fam, 0.5, 0.0)
    assert packed.length_m <= inline.length_m + TOL
    env, _ = arrange_boxes(box_sizes(req), layout)
    assert 2 * packed.half_width_m >= env[1] - TOL and 2 * packed.half_height_m >= env[2] - TOL


@pytest.mark.parametrize("layout", ["side_by_side", "stacked"])
def test_a_packed_body_holds_every_box_inside_its_skin(layout: str, req: Requirements) -> None:
    d = decode(_with(_base(), box_layout=layout), req)
    assert min(fit_margins(d, req).values()) >= 0.0
    ys = {round(b.centre_m[1], 9) for b in d.boxes}
    zs = {round(b.centre_m[2], 9) for b in d.boxes}
    assert len(ys if layout == "side_by_side" else zs) == len(d.boxes)


def _check_the_wing_root_layout(
    genomes: np.ndarray, decoded: list[Design], brief: Requirements, req: Requirements
) -> None:
    rows = [
        d
        for g, d in zip(genomes, decoded, strict=False)
        if unpack(g).pick["box_layout"] == "wing_root"
    ]
    in_wing = [d for d in rows if d.boxes[0].host == "surface0"]
    in_body = [d for d in rows if d.boxes[0].host == "body0"]
    assert in_body, "a wing too thin for the boxes must fall back to the body"
    assert bool(in_wing) == (brief != req)  # only the small-boxes brief fits a wing
    for d in in_wing:  # the boxes sit inside surface0's root chord
        root = d.surfaces[0].sections[0]
        assert all(root.le_m[0] <= b.centre_m[0] <= root.le_m[0] + root.chord_m for b in d.boxes)
        assert all(b.host == "surface0" for b in d.boxes)


def test_the_wing_root_layout_holds_the_boxes_in_the_wing_when_they_fit_and_in_the_body_else(
    genomes_0_3: np.ndarray, decoded_0_3: list[Design], brief: Requirements, req: Requirements
) -> None:
    _check_the_wing_root_layout(genomes_0_3, decoded_0_3, brief, req)


def test_the_wing_root_layout_on_the_99_gene_population(
    genomes: np.ndarray, decoded: list[Design], brief: Requirements, req: Requirements
) -> None:
    """The same assertions on today's random draw (the one above keeps the 0.3 draw)."""
    _check_the_wing_root_layout(genomes, decoded, brief, req)


FACETED_TWIN_GENOMES = 400


def test_the_wing_root_layout_holds_the_boxes_in_the_wing_or_the_body_with_faceted_bodies(
    genomes: np.ndarray, brief: Requirements, req: Requirements
) -> None:
    """The same rule on the current random population with every body flat-panel."""
    style = [g.name for g in gene_table()].index("body_style")
    rows = []
    for g in genomes[:FACETED_TWIN_GENOMES]:
        if unpack(g).pick["box_layout"] == "wing_root":
            faceted = g.copy()
            faceted[style] = 0.9
            rows.append(decode(faceted, brief))
    in_wing = [d for d in rows if d.boxes[0].host == "surface0"]
    in_body = [d for d in rows if d.boxes[0].host == "body0"]
    assert in_body, "a wing too thin for the boxes must fall back to the body"
    assert all(b.facets is not None for d in in_body for b in d.bodies if b.name == "body0")
    assert bool(in_wing) == (brief != req)
    for d in in_wing:
        root = d.surfaces[0].sections[0]
        assert all(root.le_m[0] <= b.centre_m[0] <= root.le_m[0] + root.chord_m for b in d.boxes)
