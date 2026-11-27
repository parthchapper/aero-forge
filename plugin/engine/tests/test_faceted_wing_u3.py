"""U3: flat-panel wings (airfoil_facets): flat faces, closed solids, exact volume, cut list, aero
increment, one polygon for every consumer."""

import time

import numpy as np
import pytest
from conftest import SAMPLE_BRIEF

from aeroforge.airfoil import corner_turn_sq_sum, facet_outline, naca_outline, wing_outline
from aeroforge.brief import brief_requirements, load_brief
from aeroforge.checks import evaluate
from aeroforge.constants import number
from aeroforge.derive import derive
from aeroforge.facet_aero import section_cd_increment
from aeroforge.facet_query import is_closed, planarity_m, surface_area_m2, volume_m3
from aeroforge.facet_wing import wing_polyhedron
from aeroforge.fitcheck import fit_margins
from aeroforge.genetable import gene_table, pin_choice
from aeroforge.genome import decode, random_genomes
from aeroforge.geom import build_parts, exact_volumes_m3
from aeroforge.mesh import solid_volume_m3
from aeroforge.models import Design, Requirements, instance_names
from aeroforge.panels import design_panels, surface_panels

SEED = 20261125
COUNTS = ("3", "4", "5", "6", "7", "8")


@pytest.fixture(scope="module")
def req() -> Requirements:
    return brief_requirements("stub")


@pytest.fixture(scope="module")
def faceted(req: Requirements) -> list[Design]:
    """24 seeded designs whose wings have 3..8 facets per side (counts cycle)."""
    gs = random_genomes(np.random.default_rng(SEED), 24)
    out = []
    for i, g in enumerate(gs):
        out.append(decode(_u3(g, COUNTS[i % 6]), req))
    return out


def _u3(g: np.ndarray, label: str) -> np.ndarray:
    """g with the wing facet count pinned and the U3b coarseness off (these tests are U3's)."""
    return pin_choice(pin_choice(g[None, :], "airfoil_facets", label), "facet_coarseness", "off")[0]


def test_the_u3_genes_precede_the_u3b_gene_and_their_first_labels_are_no_ops() -> None:
    rows = gene_table()[-3:-1]
    assert [g.name for g in rows] == ["airfoil_facets", "edge_alignment"]
    assert rows[0].choices[0] == "smooth" and rows[1].choices[0] == "off"


def test_every_surface_carries_its_facet_count(faceted: list[Design]) -> None:
    for i, d in enumerate(faceted):
        assert d.family.airfoil_facets == int(COUNTS[i % 6])
        assert {s.airfoil_facets for s in d.surfaces} == {d.family.airfoil_facets}


def test_the_polygon_has_two_facets_corners_on_the_parent_and_is_thinner_than_it() -> None:
    parent = naca_outline(0.04, 0.4, 0.12, 40)
    for k in (3, 5, 8):
        poly = facet_outline(0.04, 0.4, 0.12, 40, k)
        assert len(poly) == 2 * k + 1 and np.array_equal(poly[0], poly[-1])
        assert wing_outline(0.04, 0.4, 0.12, 40, k) is poly
        split = len(parent) // 2
        up = np.interp(poly[: k + 1, 0], parent[: split + 1][::-1, 0], parent[: split + 1][::-1, 1])
        assert np.all(np.abs(poly[: k + 1, 1][::-1] - up[::-1]) < 3e-3)
        assert (poly[:, 1].max() - poly[:, 1].min()) <= (parent[:, 1].max() - parent[:, 1].min())


def test_smooth_wings_read_the_unchanged_naca_outline() -> None:
    assert wing_outline(0.02, 0.4, 0.1, 40, 0) is naca_outline(0.02, 0.4, 0.1, 40)


def test_every_face_of_a_faceted_wing_is_flat_and_the_solid_is_closed(
    faceted: list[Design],
) -> None:
    triangles = quads = 0
    for d in faceted:
        for s in d.surfaces:
            poly = wing_polyhedron(s)
            span = max(c.le_m[1] for c in s.sections) - min(c.le_m[1] for c in s.sections)
            assert planarity_m(poly) <= number("facet_geom_tol_m") * max(span, 1.0)
            assert is_closed(poly) and volume_m3(poly) > 0.0
            triangles += sum(len(f) == 3 for f in poly.faces)
            quads += sum(len(f) == 4 for f in poly.faces)
    assert triangles > 0 and quads > 0, "twisted strips split into triangles, straight ones not"


def test_area_and_span_invariants_hold_for_faceted_wings(
    faceted: list[Design], req: Requirements
) -> None:
    for d in faceted:
        total = sum(s.projected_area_m2 for s in d.surfaces)
        assert total == pytest.approx(d.lifting_area_m2, rel=number("area_check_rel_tol"))
        span = max(
            (abs(c.le_m[1]) for s in d.surfaces if s.mirrored for c in s.sections), default=0.0
        )
        assert 2 * span <= req.span_max_m * (1 + 1e-9)


def test_cad_volume_of_a_faceted_wing_is_its_polyhedron_volume(faceted: list[Design]) -> None:
    seen = 0
    for d in faceted[:6]:
        parts, exact = build_parts(d), exact_volumes_m3(d)
        for s in d.surfaces:
            name = instance_names(s.name, s.mirrored)[-1]
            assert solid_volume_m3(parts[name]) == pytest.approx(exact[name], rel=1e-6)
            seen += 1
    assert seen >= 6


def test_boxes_spar_and_props_still_fit_with_polygon_sections(
    faceted: list[Design], req: Requirements
) -> None:
    for d in faceted:
        margins = fit_margins(d, req)
        assert margins["boxes"] == 0.0 and margins["spar"] >= 0.0


def test_wing_sheets_feed_the_cut_list(faceted: list[Design]) -> None:
    for d in faceted[:8]:
        rows = design_panels(d)
        for s in d.surfaces:
            mine = [p for p in rows if p.part == s.name]
            assert [p.id for p in mine] == [p.id for p in surface_panels(s)]
            poly = wing_polyhedron(s)
            assert 0 < len(mine) <= len(poly.faces)
            assert all(p.count == (2 if s.mirrored else 1) for p in mine)
            assert sum(p.area_m2 for p in mine) == pytest.approx(surface_area_m2(poly), rel=1e-6)


def test_the_drag_increment_is_zero_when_smooth_and_falls_with_more_facets(
    faceted: list[Design],
) -> None:
    by_k = {}
    for d in faceted:
        s = d.surfaces[0]
        by_k.setdefault(s.airfoil_facets, []).append(section_cd_increment(s))
    assert 0.0 < np.mean(by_k[8]) < np.mean(by_k[3])
    smooth = decode(random_genomes(np.random.default_rng(1), 1)[0], brief_requirements("stub"))
    assert all(section_cd_increment(s) == 0.0 for s in smooth.surfaces if not s.airfoil_facets)


def test_corner_turns_of_the_polygon_are_positive_and_shrink_with_facets() -> None:
    turns = [corner_turn_sq_sum(facet_outline(0.02, 0.4, 0.12, 40, k)) for k in range(3, 9)]
    assert all(t > 0 for t in turns) and turns == sorted(turns, reverse=True)


def test_checks_take_less_than_twenty_percent_longer_with_faceted_wings() -> None:
    brief = load_brief(SAMPLE_BRIEF)
    r = derive(brief)
    gs = random_genomes(np.random.default_rng(SEED), 4)

    def run(label: str) -> float:
        t = time.perf_counter()
        for g in gs:
            evaluate(_u3(g, label), brief, r, 1)
        return time.perf_counter() - t

    run("smooth"), run("5")  # warm the section and import caches of both before timing
    times = {label: min(run(label) for _ in range(3)) for label in ("smooth", "5")}
    assert times["5"] < 1.2 * times["smooth"]
