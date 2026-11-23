"""U2: flat-panel bodies in CAD, aero equivalent section, mass, internals and the panel cut list."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import aeroforge.facet_hatch as fh
import aeroforge.facet_query as q
from aeroforge.airplane import fuselages
from aeroforge.checks import evaluate
from aeroforge.constants import number
from aeroforge.facet_pieces import pieces
from aeroforge.genetable import pin_choice
from aeroforge.genome import decode, random_genomes
from aeroforge.geom import GeometryError, build_parts, exact_volumes_m3
from aeroforge.hardpoints import containment_failures, hardpoints
from aeroforge.internals import missing
from aeroforge.mass import _body_item
from aeroforge.mesh import solid_volume_m3, tessellate
from aeroforge.models import Brief, Design, Polyhedron, Requirements
from aeroforge.panels import design_panels, panels_csv, panels_svg, write_panels
from aeroforge.reference import BASELINE, reference_genome
from aeroforge.report import _faceted_fidelity

SEED = 20261123
FAST_DESIGNS = 40
CAD_SAMPLE = 100
CAD_SUCCESS_FLOOR = 0.98
VOLUME_DESIGNS = 8
VOLUME_REL_TOL = 1e-6  # plan U2 bar is 1 %; planar faces are exact, the real error is ~1e-9
MISFIT_BAR = 0.01  # F6: internals containment, target 1 % (SPEC §5 bar is 2 %)
AREA_REL_TOL = 3e-3
WINNERS = Path(__file__).parent / "fixtures" / "winners"


def _faceted(n: int, req: Requirements) -> list[Design]:
    genomes = pin_choice(random_genomes(np.random.default_rng(SEED), n), "body_style", "faceted")
    return [decode(g, req) for g in genomes]


@pytest.fixture(scope="module")
def faceted(req: Requirements) -> list[Design]:
    return _faceted(FAST_DESIGNS, req)


def _cube() -> Polyhedron:
    v = [(x, y, z) for x in (0.0, 1.0) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    faces = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    return Polyhedron(vertices_m=tuple(v), faces=tuple(f[::-1] for f in faces))  # outward


def test_polyhedron_volume_area_and_slice_of_a_cube_are_exact() -> None:
    cube = _cube()
    assert q.volume_m3(cube) == pytest.approx(1.0)
    assert q.surface_area_m2(cube) == pytest.approx(6.0)
    assert q.area_centroid_m(cube) == pytest.approx([0.5, 0.0, 0.0])
    assert q.slice_section(cube, 0.4) == pytest.approx((1.0, 1.0, 1.0, 0.0))


def test_the_equivalent_exponent_runs_from_the_diamond_to_the_rectangle() -> None:
    assert q.equivalent_exponent(1.0, 1.0, 0.5) == pytest.approx(1.0)
    assert q.equivalent_exponent(1.0, 1.0, 1.0) == number("equiv_section_exp_max")
    assert q.equivalent_exponent(1.0, 1.0, np.pi / 4) == pytest.approx(2.0, rel=1e-6)


def test_the_equivalent_aero_section_has_the_polygons_area(faceted: list[Design]) -> None:
    checked = 0
    for d in faceted:
        for body, fus in zip(d.bodies, fuselages(d)[: len(d.bodies)], strict=False):
            if body.facets is None or body.mirrored:
                continue
            for xs in fus.xsecs:
                w, h, area, _ = q.slice_section(body.facets, xs.xyz_c[0] - body.nose_m[0])
                assert xs.width == pytest.approx(w)
                assert xs.height == pytest.approx(h)
                if xs.shape < number("equiv_section_exp_max") and area > 0.0:
                    assert xs.xsec_area() == pytest.approx(area, rel=AREA_REL_TOL)
                    checked += 1
    assert checked > FAST_DESIGNS


def test_skin_mass_comes_from_the_exact_wetted_area(faceted: list[Design]) -> None:
    for d in faceted:
        for b in d.bodies:
            density = number("faceted_panel_thickness_m") * number("faceted_panel_density_kg_m3")
            instances = 2 if b.mirrored else 1
            assert _body_item(b).mass_kg == pytest.approx(
                instances * q.surface_area_m2(b.facets) * density
            )


def test_panels_csv_has_one_row_per_sheet_and_the_areas_sum_to_the_wetted_area(
    faceted: list[Design],
) -> None:
    for d in faceted:
        panels = design_panels(d)
        rows = panels_csv(panels).splitlines()[1:]
        assert len(rows) == sum(len(pieces(b.facets)) for b in d.bodies)
        assert len(rows) <= sum(len(b.facets.faces) for b in d.bodies)
        for b in d.bodies:
            mine = [r.split(",") for r in rows if r.split(",")[1] == b.name]
            assert len(mine) == len(pieces(b.facets))
            total = sum(float(r[3]) for r in mine)
            assert total == pytest.approx(q.surface_area_m2(b.facets), rel=1e-5)


def test_count_times_area_over_the_rows_is_the_wetted_area_of_every_faceted_body(
    faceted: list[Design],
) -> None:
    pods = 0
    for d in faceted:
        rows = design_panels(d)
        wetted = sum((2 if b.mirrored else 1) * q.surface_area_m2(b.facets) for b in d.bodies)
        assert sum(p.count * p.area_m2 for p in rows) == pytest.approx(wetted, rel=1e-9)
        pods += any(b.mirrored for b in d.bodies)
    assert pods > 0, "the sample must include a mirrored pod, where the plain sum falls short"


def test_each_panel_edge_has_a_neighbour_and_a_bevel_angle(faceted: list[Design]) -> None:
    for p in design_panels(faceted[0]):
        n = len(p.corners_m)
        assert len(p.edge_lengths_m) == len(p.neighbour_ids) == len(p.bevel_deg) == n
        assert all(0.0 <= a <= 180.0 for a in p.bevel_deg)


def test_panels_svg_has_every_face_numbered(faceted: list[Design]) -> None:
    panels = design_panels(faceted[0])
    svg = panels_svg(panels)
    assert svg.count("<polygon") == len(panels)
    assert all(f'id="{p.id}"' in svg for p in panels)


def test_the_panel_writer_writes_faceted_designs_and_nothing_for_smooth_ones(
    faceted: list[Design], tmp_path: Path, req: Requirements
) -> None:
    assert write_panels(tmp_path, faceted[0]) == len(design_panels(faceted[0]))
    assert {p.name for p in (tmp_path / "panels").iterdir()} == {"panels.csv", "panels.svg"}
    smooth = decode(
        pin_choice(random_genomes(np.random.default_rng(SEED), 1), "body_style", "smooth")[0], req
    )
    assert write_panels(tmp_path, smooth) == 0
    assert not list((tmp_path / "panels").iterdir())


def test_the_fidelity_statement_names_the_unmodelled_chine_drag(
    faceted: list[Design], req: Requirements
) -> None:
    assert "not modelled" in _faceted_fidelity(faceted[0])[0]
    smooth = decode(
        pin_choice(random_genomes(np.random.default_rng(SEED), 1), "body_style", "smooth")[0], req
    )
    assert _faceted_fidelity(smooth) == []


def test_internals_are_complete_and_mostly_contained_on_random_faceted_bodies(
    faceted: list[Design], req: Requirements
) -> None:
    misfits = 0
    for d in faceted:
        items = hardpoints(d, req.recovery)
        assert not missing(d, req.recovery, items)
        misfits += bool(containment_failures(d, items))
    assert misfits <= MISFIT_BAR * len(faceted)


@pytest.mark.parametrize("i", range(5))
def test_internals_are_contained_on_the_winner_genomes_decoded_as_faceted(i: int) -> None:
    w = json.loads((WINNERS / f"winner_{i}.json").read_text(encoding="utf-8"))
    req = Requirements.model_validate_json((WINNERS / "requirements.json").read_text("utf-8"))
    d = decode(pin_choice(np.array(w["genome"]), "body_style", "faceted"), req)
    assert any(b.facets is not None for b in d.bodies)
    items = hardpoints(d, req.recovery)
    assert not missing(d, req.recovery, items)
    assert containment_failures(d, items) == []


def test_a_hatch_on_a_flat_panel_body_lies_inside_one_roof_panel(faceted: list[Design]) -> None:
    checked = 0
    for d in faceted:
        body = next(b for b in d.bodies if b.name == d.boxes[0].host) if d.boxes else None
        if body is None or body.facets is None:
            continue
        hatch = next(i for i in hardpoints(d, "belly") if i.item == "hatch")
        c, size = np.asarray(hatch.centre_m) - np.asarray(body.nose_m), hatch.size_m
        corners = np.array(
            [[c[0] + a * size[0] / 2, c[1] + b * size[1] / 2] for a in (-1, 1) for b in (-1, 1)]
        )
        panels = fh.roof_panels_xy(body.facets)
        assert any(fh._in_convex(p, corners).all() for p in panels)
        checked += 1
    assert checked > 0


@pytest.mark.slow
def test_cad_builds_for_at_least_98_percent_of_random_faceted_genomes(req: Requirements) -> None:
    built = 0
    for d in _faceted(CAD_SAMPLE, req):
        try:
            build_parts(d)
            built += 1
        except GeometryError:
            continue
    assert built >= CAD_SUCCESS_FLOOR * CAD_SAMPLE


@pytest.mark.slow
def test_cad_and_mesh_volumes_match_the_polyhedron_within_one_percent(req: Requirements) -> None:
    for d in _faceted(VOLUME_DESIGNS, req):
        parts = build_parts(d)
        meshes = {m.name: m for m in tessellate(parts, exact_volumes_m3(d))}
        for b in d.bodies:
            name = f"{b.name}_R" if b.mirrored else b.name
            exact = q.volume_m3(b.facets)
            assert solid_volume_m3(parts[name]) == pytest.approx(exact, rel=VOLUME_REL_TOL)
            m = meshes[name]
            tm = trimesh.Trimesh(m.vertices, m.faces, process=False)
            assert tm.volume == pytest.approx(exact, rel=VOLUME_REL_TOL)
            assert tm.area == pytest.approx(q.surface_area_m2(b.facets), rel=VOLUME_REL_TOL)


@pytest.mark.parametrize("tier", [1, 2])
def test_a_faceted_conventional_design_is_checked_with_finite_margins(
    tier: int, sample_brief: Brief, sample_req: Requirements
) -> None:
    genome = pin_choice(reference_genome(BASELINE), "body_style", "faceted")
    design, report = evaluate(genome, sample_brief, sample_req, tier)
    assert design.bodies[0].facets is not None
    margins = [c.margin for c in report.checks]
    assert margins and np.isfinite(margins).all()
