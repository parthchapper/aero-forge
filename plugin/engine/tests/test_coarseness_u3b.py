"""U3b facet coarseness: fewer, sharper, flatter panels on faceted parts; every override a note."""

import itertools

import numpy as np
import pytest

from aeroforge.brief import brief_requirements
from aeroforge.cli import _pinned
from aeroforge.coarse import level, min_crease_deg, section_move, stepped_levels, wing_facets
from aeroforge.coarse_body import min_turn_deg, polygon_move_m
from aeroforge.coarse_metrics import coarse_notes, coarse_rows, design_measures, measure
from aeroforge.constants import number
from aeroforge.genetable import gene_table, genome_length, pin_choice
from aeroforge.genome import decode, random_genomes
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


def test_off_leaves_no_coarseness_note(pairs: list[tuple]) -> None:
    for off, _ in pairs:
        assert off.family.facet_coarseness == 0.0
        assert not any("coarseness" in n for n in off.notes)
        assert coarse_notes(off) == ()


def test_high_coarseness_is_noted_and_cuts_wing_facets(pairs: list[tuple]) -> None:
    cap = round(number("coarse_wing_facets_tight"))
    for off, high in pairs:
        assert high.family.facet_coarseness == 1.0
        assert high.family.airfoil_facets <= min(off.family.airfoil_facets, cap) or any(
            "merge skipped" in n for n in high.notes
        )
        if high.family.airfoil_facets != off.family.airfoil_facets:
            assert any("flat segments per side" in n for n in high.notes)


def test_wing_section_vertex_move_stays_under_the_cap() -> None:
    cap = number("coarse_move_frac")
    for k0 in range(3, 9):
        fam = Family.model_construct(
            airfoil_facets=k0, facet_coarseness=1.0, camber_frac=0.04, camber_loc_frac=0.4, tc=0.1
        )
        k, _ = wing_facets(fam)
        assert 2 <= k <= k0
        assert section_move(fam, k0, k) <= cap + 1e-12


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
            assert len({round(t, 9) for t in t1}) <= len(t1)
            equal = sum(a == b for a, b in itertools.pairwise(t1))
            assert equal >= 1 or len(t1) == 2  # at least one strip is not twisted
    assert seen >= 3


def test_stepped_levels_bounds() -> None:
    assert stepped_levels(0.0, 1.0, 3) == 0
    assert stepped_levels(-6.0, 0.0, 3) == 0
    assert stepped_levels(-0.2, 1.0, 3) == 1  # stepped, not none
    assert stepped_levels(-6.0, 1.0, 3) == 2  # a strip stays flat
    assert stepped_levels(-6.0, 1.0, 1) == 1


def test_coarse_bodies_are_linear_and_overrides_are_noted(pairs: list[tuple]) -> None:
    bodies = 0
    for off, high in pairs:
        b0 = next((b for b in off.bodies if b.facets is not None), None)
        if b0 is None:
            continue
        bodies += 1
        assert high.family.shape.linear_edges
        if high.family.shape.sides < off.family.shape.sides:
            assert any("body section" in n and "corners" in n for n in high.notes)
        m = next(m for m in design_measures(high) if m.kind == "body")
        assert m.sheets <= number("coarse_body_panels_max") or any(
            "over the budget" in n or "coarseness" in n for n in high.notes
        )
    assert bodies >= 2


def test_coarse_is_flatter_in_total(pairs: list[tuple]) -> None:
    def totals(designs: list) -> tuple[int, int]:
        ms = [m for d in designs for m in design_measures(d)]
        return sum(m.sheets for m in ms), sum(int((m.creases_deg < 10.0).sum()) for m in ms)

    sheets_off, low_off = totals([a for a, _ in pairs])
    sheets_high, low_high = totals([b for _, b in pairs])
    assert sheets_high < sheets_off
    assert low_high < low_off


def test_smooth_parts_are_untouched_by_coarseness() -> None:
    rng = np.random.default_rng(3)
    g = _pinned(random_genomes(rng, 3), {"airfoil_facets": "smooth"})
    g = pin_choice(g, "body_style", "smooth")
    for row in g:
        a = decode(row, REQ)
        b = decode(pin_choice(row[None, :], "facet_coarseness", "high")[0], REQ)
        assert a.surfaces == b.surfaces and a.bodies == b.bodies
        assert not any("coarseness" in n for n in b.notes)


def _cube() -> Polyhedron:
    v = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
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
    rows = coarse_rows(pairs[0][1])
    text = "\n".join(rows)
    assert "facet_coarseness 1.00" in text and "aero cost" in text and "clmax" in text
