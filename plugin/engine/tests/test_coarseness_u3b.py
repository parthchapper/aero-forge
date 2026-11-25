"""U3b facet coarseness: few, large, sharp, flat panels on faceted parts; every override a note."""

import itertools

import numpy as np
import pytest

from aeroforge.airfoil import facet_outline, facet_sides
from aeroforge.brief import brief_requirements
from aeroforge.cli import _pinned
from aeroforge.coarse import level, stepped_levels
from aeroforge.coarse_body import min_turn_deg, polygon_move_m
from aeroforge.coarse_metrics import coarse_notes, coarse_rows, design_measures, measure
from aeroforge.coarse_prior import weighted
from aeroforge.constants import number
from aeroforge.facet_decimate import decimate, min_crease_deg, pack, unpack
from aeroforge.facet_planar import planarize
from aeroforge.facet_query import is_closed, planarity_m
from aeroforge.facet_wing import _part_length, wing_geometry
from aeroforge.facets import facet_body_info
from aeroforge.genetable import gene_table, genome_length, pin_choice, random_genomes
from aeroforge.genome import decode
from aeroforge.models import Family, Polyhedron

REQ = brief_requirements("stub")


def _genomes(n: int, label: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    g = _pinned(random_genomes(rng, n), {"airfoil_facets": "faceted", "edge_alignment": "strict"})
    return pin_choice(g, "facet_coarseness", label)


@pytest.fixture(scope="module")
def pairs() -> list[tuple]:
    """The same 6 genomes decoded with the coarseness off and high."""
    off, high = _genomes(6, "off"), _genomes(6, "high")
    return [(decode(a, REQ), decode(b, REQ)) for a, b in zip(off, high, strict=True)]


def test_gene_is_last_and_off_is_the_first_label() -> None:
    genes = gene_table()
    assert genes[-1].name == "facet_coarseness" and genes[-1].choices[0] == "off"
    assert genome_length() == 99 + 3
    assert level("off") == 0.0 and level("high") == 1.0 and 0.0 < level("mid") < 1.0


def test_prior_weights_faceted_rows_toward_high_and_leaves_smooth_rows_alone() -> None:
    genes = gene_table()
    names = [g.name for g in genes]
    ia, ib, ic = (names.index(n) for n in ("airfoil_facets", "body_style", "facet_coarseness"))
    u = np.random.default_rng(0).random((4000, len(genes)))
    out = weighted(u, genes)
    smooth = (u[:, ia] < 1 / 7) & (u[:, ib] < 0.5)  # airfoil smooth and body smooth
    assert (out[smooth] == u[smooth]).all()
    faceted = ~smooth
    labels = np.minimum((out[faceted, ic] * 4).astype(int), 3)
    shares = np.bincount(labels, minlength=4) / labels.size
    for share, key in zip(
        shares,
        ("coarse_prior_off", "coarse_prior_low", "coarse_prior_mid", "coarse_prior_high"),
        strict=True,
    ):
        assert share == pytest.approx(number(key), abs=0.03)
    padded = np.full(len(genes), 0.5)
    padded[ic] = 0.125  # the value fixtures are padded with: still the first label, "off"
    assert gene_table()[ic].choices[int(0.125 * 4)] == "off" and weighted(padded[None], genes).shape


def test_facet_code_round_trips() -> None:
    assert pack(5, 0.0) == 5 and unpack(5) == (5, 0.0)
    assert unpack(pack(7, 1.0)) == (7, 1.0) and unpack(pack(4, 0.35)) == (4, 0.35)


def test_decimation_drops_the_flattest_corner_first_within_the_cap() -> None:
    x = np.linspace(0.0, 1.0, 6)
    y = np.array([0.0, 0.05, 0.06, 0.0, -0.05, 0.0]) * 0.1  # nearly flat between 1 and 2
    pts = np.column_stack([x, y])
    got = decimate(pts, 15.0, 10, 1.0)
    assert 2 not in got.keep or 1 not in got.keep  # one of the flat pair went, cap is loose
    tight = decimate(pts, 15.0, 10, 1e-9)
    assert tight.blocked and tight.keep == tuple(range(6))
    assert decimate(pts, 0.0, 2, 1.0, min_points=3).keep.__len__() >= 3


def test_coarse_wing_section_keeps_its_thickness_and_symmetry_and_moves_little() -> None:
    n = int(number("airfoil_points_per_side"))
    for k0 in range(3, 9):
        code = pack(k0, 1.0)
        up, lo, info = facet_sides(0.0, 0.4, 0.1, n, code)
        assert info is not None and np.allclose(lo[:, 1], -up[:, 1]) and len(up) >= 3
        assert info[0].move <= number("coarse_move_frac") + 1e-12
        assert up[:, 1].max() > 0.02  # thickness kept
        cam = facet_sides(0.04, 0.4, 0.1, n, code)
        assert cam[2] is not None and len(cam[0]) >= 3
    plain = facet_outline(0.04, 0.4, 0.1, n, 5)
    assert len(plain) == 2 * 5 + 1


def test_high_coarseness_is_noted_and_cuts_the_wing_sheets(pairs: list[tuple]) -> None:
    for off, high in pairs:
        assert high.family.facet_coarseness == 1.0 and high.family.airfoil_facets >= 10
        sheets_off = sum(m.sheets for m in design_measures(off) if m.kind == "wing")
        sheets_high = sum(m.sheets for m in design_measures(high) if m.kind == "wing")
        assert sheets_high <= sheets_off


def test_wing_polyhedra_are_closed_flat_and_moved_within_the_cap(pairs: list[tuple]) -> None:
    quads = 0
    for _, high in pairs:
        for s in high.surfaces:
            if not s.airfoil_facets:
                continue
            poly, planar = wing_geometry(s)
            assert is_closed(poly) and planarity_m(poly) <= 1e-9
            cap = number("coarse_move_frac") * _part_length(s)
            assert planar is not None and planar.move_m <= cap + 1e-12
            quads += sum(len(f) == 4 for f in poly.faces)
    assert quads > 0


def test_planarize_flattens_a_twisted_strip_and_gives_up_over_the_cap() -> None:
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    twist = np.radians(6.0)
    rot = np.array(
        [[np.cos(twist), 0, np.sin(twist)], [0, 1, 0], [-np.sin(twist), 0, np.cos(twist)]]
    )
    v = np.vstack([sq, (sq - 0.5) @ rot.T + 0.5 + np.array([0, 1.0, 0])])
    quads = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
    caps = [(0, 1, 2, 3), (4, 5, 6, 7)]
    free = planarize(v, quads, caps, np.zeros(v.size, bool), 0.5)
    assert free.blocked == 0 and len(free.merged) == 4 and free.move_m > 0.0
    tight = planarize(v, quads, caps, np.zeros(v.size, bool), 1e-6)
    assert tight.blocked == 4 and tight.move_m < 1e-6


def test_stepped_levels_bounds() -> None:
    assert stepped_levels(0.0, 1.0, 3) == 0
    assert stepped_levels(-6.0, 0.0, 3) == 0
    assert stepped_levels(-0.2, 1.0, 3) == 1  # stepped, not none
    assert stepped_levels(-6.0, 1.0, 3) == 2  # a strip stays flat
    assert stepped_levels(-6.0, 1.0, 1) == 1


def test_washout_is_stepped_and_keeps_the_tip_twist(pairs: list[tuple]) -> None:
    seen = 0
    for off, high in pairs:
        for s0, s1 in zip(off.surfaces, high.surfaces, strict=False):
            if not (s0.mirrored and s0.name == s1.name and s1.airfoil_facets):
                continue
            t0 = [s.twist_deg for s in s0.sections]
            t1 = [s.twist_deg for s in s1.sections]
            if abs(t0[-1]) < 1e-9:
                continue
            seen += 1
            assert t1[-1] == pytest.approx(t0[-1])
            equal = sum(a == b for a, b in itertools.pairwise(t1))
            assert equal >= 1 or len(t1) == 2  # at least one strip is not twisted
    assert seen >= 3


def test_coarse_bodies_are_linear_merged_and_noted(pairs: list[tuple]) -> None:
    bodies = 0
    for off, high in pairs:
        b0 = next((b for b in off.bodies if b.facets is not None), None)
        b1 = next((b for b in high.bodies if b.facets is not None), None)
        if b0 is None or b1 is None:
            continue
        bodies += 1
        assert is_closed(b1.facets) and planarity_m(b1.facets) <= 1e-9 * b1.length_m
        shape = high.family.shape
        if shape.coarseness > 0.0:  # else the decoder gave the body's coarsening up, noted
            assert shape.linear_edges
            axes = (b1.half_width_m, b1.half_height_m)
            info = facet_body_info(shape, b1.length_m, axes, b1.tail_upsweep_deg, b1.rise_x_frac)[1]
            assert info is not None and info.move_m <= number("coarse_move_frac") * b1.length_m
        else:
            assert any("coarsening given up" in n for n in high.notes)
    assert bodies >= 2


def test_coarse_is_flatter_in_total(pairs: list[tuple]) -> None:
    def totals(designs: list) -> tuple[int, int]:
        ms = [m for d in designs for m in design_measures(d)]
        return sum(m.sheets for m in ms), sum(int((m.creases_deg < 15.0).sum()) for m in ms)

    sheets_off, low_off = totals([a for a, _ in pairs])
    sheets_high, low_high = totals([b for _, b in pairs])
    assert sheets_high < sheets_off
    assert low_high < low_off


def test_off_leaves_no_coarseness_note_and_smooth_parts_are_untouched(pairs: list[tuple]) -> None:
    for off, _ in pairs:
        assert off.family.facet_coarseness == 0.0
        assert not any("coarseness" in n for n in off.notes)
        assert coarse_notes(off) == ()
    rng = np.random.default_rng(3)
    g = pin_choice(
        _pinned(random_genomes(rng, 3), {"airfoil_facets": "smooth"}), "body_style", "smooth"
    )
    for row in g:
        a = decode(pin_choice(row[None, :], "facet_coarseness", "off")[0], REQ)
        b = decode(pin_choice(row[None, :], "facet_coarseness", "high")[0], REQ)
        assert a.surfaces == b.surfaces and a.bodies == b.bodies
        assert not any("coarseness" in n for n in b.notes)


def _cube() -> Polyhedron:
    v = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return Polyhedron(vertices_m=tuple(v), faces=faces)


def test_measure_of_a_cube() -> None:
    m = measure("cube", "body", _cube())
    assert m is not None
    assert m.sheets == 6 and m.directions == 6 and len(m.creases_deg) == 12
    assert np.allclose(m.creases_deg, 90.0)


def test_turn_and_move_helpers() -> None:
    sq = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    assert min_turn_deg(sq) == pytest.approx(90.0)
    assert polygon_move_m(sq, sq) == pytest.approx(0.0)
    assert polygon_move_m(sq, sq * 2.0) > 0.5
    assert min_crease_deg(1.0) == number("coarse_min_crease_deg")


def test_report_has_the_coarseness_table(pairs: list[tuple]) -> None:
    text = "\n".join(coarse_rows(pairs[0][1]))
    assert "facet_coarseness 1.00" in text and "aero cost" in text and "clmax" in text


def test_family_model_accepts_the_new_fields() -> None:
    assert Family.model_fields["facet_coarseness"].default == 0.0


def test_a_coarse_body_can_be_given_up_and_the_genes_body_returns() -> None:
    from aeroforge.body_align import BodySnap, uncoarsened
    from aeroforge.models import BodyShape

    fam = Family.model_construct(
        shape=BodyShape(style="faceted", sides=8, linear_edges=True, dropped=(1,), coarseness=1.0),
        edge_alignment="off",
    )
    snap = uncoarsened(fam, BodySnap(fam.shape), "boxes do not fit")
    assert snap.shape.coarseness == 0.0 and snap.shape.dropped == () and not snap.shape.linear_edges
    assert snap.coarse and "given up" in snap.notes[0] and "boxes do not fit" in snap.notes[0]
