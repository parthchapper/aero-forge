"""Faceted bodies (upgrade U1): section polygons, flat-face solids, panel plan, containment, fit,
decoded bodies and their shadow areas. Properties over seeded random shapes."""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from aeroforge import facet_query as q
from aeroforge import facets as f
from aeroforge.areas import areas
from aeroforge.brief import brief_requirements
from aeroforge.genetable import gene_table, genome_length
from aeroforge.genome import decode
from aeroforge.models import BodyShape
from aeroforge.reference import encode
from aeroforge.shapes import inside_body

SEED = 20261122
N_SHAPES = 120
LENGTH_M = 1.3
AXES_M = (0.09, 0.07)


def random_shape(rng: np.random.Generator) -> BodyShape:
    """A faceted shape with every style gene drawn from its genes.yaml range."""
    pick = {g.name: g for g in gene_table()}
    num = {k: float(rng.uniform(pick[k].lo, pick[k].hi)) for k in pick if pick[k].kind == "float"}
    return BodyShape(
        style="faceted",
        sides=int(rng.integers(3, 13)),
        stations=int(rng.integers(1, 9)),
        chine_z_frac=num["chine_z_frac"],
        belly_flat_frac=num["belly_flat_frac"],
        roof_pitch_deg=num["roof_pitch_deg"],
        side_tumblehome_deg=num["side_tumblehome_deg"],
        lower_flare_deg=num["lower_flare_deg"],
        nose=str(rng.choice(f.NOSES)),  # type: ignore[arg-type]
        tail=str(rng.choice(f.TAILS)),  # type: ignore[arg-type]
        tail_flatten=num["tail_flatten"],
        tail_end_size_frac=num["tail_end_size_frac"],
        max_section_x_frac=num["max_section_x_frac"],
    )


@pytest.fixture(scope="module")
def shapes() -> list[BodyShape]:
    rng = np.random.default_rng(SEED)
    return [random_shape(rng) for _ in range(N_SHAPES)]


def _turns(poly: np.ndarray) -> np.ndarray:
    edge = np.roll(poly, -1, axis=0) - poly
    nxt = np.roll(edge, -1, axis=0)
    return edge[:, 0] * nxt[:, 1] - edge[:, 1] * nxt[:, 0]


def test_section_polygon_is_strictly_convex_symmetric_and_holds_the_axis(
    shapes: list[BodyShape],
) -> None:
    for sh in shapes:
        poly = f.section_polygon(sh, sh.sides)
        assert len(poly) == sh.sides
        turns = _turns(poly)
        assert np.all(turns < 0) or np.all(turns > 0)  # one turning direction, no collinear corner
        mirrored = poly * np.array([-1.0, 1.0])
        assert all(np.isclose(mirrored, p).all(axis=1).any() for p in poly)
        assert poly[:, 1].min() == pytest.approx(-1.0) and poly[:, 1].max() == pytest.approx(1.0)
        assert np.abs(poly[:, 0]).max() == pytest.approx(1.0)


def test_every_face_is_flat_and_every_solid_closed(shapes: list[BodyShape]) -> None:
    for sh in shapes:
        poly = f.facet_body(sh, LENGTH_M, AXES_M)
        assert q.planarity_m(poly) < 1e-9 * LENGTH_M
        assert q.is_closed(poly)


def test_panel_count_formula_equals_the_faces_built(shapes: list[BodyShape]) -> None:
    for sh in shapes:
        plan = f.effective_plan(sh)
        poly = f.facet_body(sh, LENGTH_M, AXES_M)
        assert len(poly.faces) == f.panel_count(plan, f.section_polygon(sh, sh.sides))


def test_the_plan_example_gives_25_panels() -> None:
    sh = BodyShape(
        style="faceted", sides=6, stations=3, nose="point", tail="flat", tail_flatten=0.5
    )
    plan = f.effective_plan(sh)
    assert (plan.sides, plan.stations, plan.breaks) == (6, 3, 1)
    assert f.panel_count(plan, f.section_polygon(sh, 6)) == 12 + 6 + 6 + 1
    assert len(f.facet_body(sh, LENGTH_M, AXES_M).faces) == 25


def test_a_flat_tail_with_a_blunt_nose_forces_three_stations() -> None:
    plan = f.effective_plan(BodyShape(style="faceted", stations=1, nose="blunt", tail="flat"))
    assert plan.stations == 3


def test_panel_plan_hits_a_reachable_count_exactly_and_reports_an_unreachable_one() -> None:
    sh = BodyShape(style="faceted", sides=6, stations=3, nose="point", tail="flat")
    wish = f.effective_plan(sh)
    plan, exact = f.panel_plan(25, wish, sh)
    assert exact
    assert len(f.body_polyhedron(plan, sh, LENGTH_M, AXES_M).faces) == 25
    _, exact = f.panel_plan(1000, wish, sh)
    assert not exact


def test_stations_of_one_shape_make_trapezoids_and_a_break_makes_triangles() -> None:
    plain = BodyShape(style="faceted", sides=6, stations=3, nose="point", tail="point")
    broken = plain.model_copy(update={"tail_flatten": 0.5})
    quads = [fc for fc in f.facet_body(plain, LENGTH_M, AXES_M).faces if len(fc) == 4]
    assert len(quads) == 12
    tris = [fc for fc in f.facet_body(broken, LENGTH_M, AXES_M).faces if len(fc) == 3]
    assert len(tris) == 12 + 12  # two end pyramids of 6, plus the break's 12


def test_inside_matches_an_independent_hull_test_on_convex_cells(shapes: list[BodyShape]) -> None:
    rng = np.random.default_rng(SEED)
    for sh in shapes[:30]:
        poly = f.facet_body(sh, LENGTH_M, AXES_M)
        pts = rng.uniform([-0.05, -0.12, -0.1], [LENGTH_M + 0.05, 0.12, 0.1], size=(400, 3))
        got = q.inside_faceted(poly, pts)
        cells = q._cells(poly)
        want = np.zeros(len(pts), dtype=bool)
        v = np.asarray(poly.vertices_m)
        for lo, hi, _, _ in cells:
            cell_v = v[(v[:, 0] >= lo - 1e-9) & (v[:, 0] <= hi + 1e-9)]
            if len(cell_v) >= 4:
                want |= Delaunay(cell_v).find_simplex(pts, tol=1e-9) >= 0
        assert (got == want).mean() > 0.995


def test_the_solid_holds_its_vertices_and_refuses_points_beyond_its_ends() -> None:
    sh = BodyShape(style="faceted", sides=8, stations=4, nose="chisel", tail="knife")
    poly = f.facet_body(sh, LENGTH_M, AXES_M)
    assert q.inside_faceted(poly, np.asarray(poly.vertices_m)).all()
    outside = np.array([[-0.01, 0.0, 0.0], [LENGTH_M + 0.01, 0.0, 0.0], [0.6, 0.5, 0.0]])
    assert not q.inside_faceted(poly, outside).any()


def test_fit_scale_is_the_smallest_even_scale_holding_the_corners() -> None:
    sh = BodyShape(style="faceted", sides=6, stations=3, belly_flat_frac=0.4, chine_z_frac=0.1)
    poly = f.facet_body(sh, LENGTH_M, AXES_M)
    x_mid = sh.max_section_x_frac * LENGTH_M
    corners = np.array(
        [[x_mid + dx, y, z] for dx in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (-0.04, 0.04)]
    )
    k = q.fit_scale(poly, corners)
    assert np.isfinite(k)
    at = lambda s: f.facet_body(sh, LENGTH_M, (AXES_M[0] * s, AXES_M[1] * s))  # noqa: E731
    assert q.inside_faceted(at(k * (1 + 1e-6)), corners).all()
    assert not q.inside_faceted(at(k * (1 - 1e-3)), corners).all()


def test_fit_scale_is_infinite_outside_the_body_length() -> None:
    poly = f.facet_body(BodyShape(style="faceted"), LENGTH_M, AXES_M)
    assert q.fit_scale(poly, np.array([[LENGTH_M + 0.5, 0.0, 0.0]])) == np.inf


def _faceted_genome(**genes: object) -> np.ndarray:
    return encode({"body_style": "faceted", **genes})


@pytest.mark.parametrize("brief_name", ["stub", "sample"])
def test_decoded_faceted_bodies_hold_their_boxes_and_pods_follow_the_style(
    brief_name: str, sample_req: object
) -> None:
    req = brief_requirements("stub") if brief_name == "stub" else sample_req
    rng = np.random.default_rng(SEED)
    seen_pod = False
    for _ in range(25):
        genome = rng.random(genome_length())
        genome[[g.name for g in gene_table()].index("body_style")] = 0.9
        design = decode(genome, req)  # type: ignore[arg-type]
        for body in design.bodies:
            assert body.facets is not None
            seen_pod |= body.name == "pod"
        for box in design.boxes:
            body = next((b for b in design.bodies if b.name == box.host), None)
            if body is None:
                continue
            half = np.array(box.size_m) / 2
            corners = np.array(box.centre_m) + half * np.array(
                [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
            assert inside_body(body, corners).all()
    assert seen_pod


def test_faceted_shadow_areas_match_a_grid_count_of_the_solid() -> None:
    req = brief_requirements("stub")
    genome = _faceted_genome(
        body_section_sides=8,
        body_stations=4,
        nose_style="chisel",
        tail_style="flat",
        body_on="on",
    )
    design = decode(genome, req)
    body = design.bodies[0]
    assert body.facets is not None
    pts = (
        np.mgrid[
            0 : body.length_m : 60j,
            -body.half_width_m : body.half_width_m : 41j,
            -body.half_height_m : body.half_height_m : 41j,
        ]
        .reshape(3, -1)
        .T
    )
    inside = q.inside_faceted(body.facets, pts)
    cell = (body.length_m / 59) * (2 * body.half_width_m / 40)
    top = inside.reshape(60, 41, 41).any(axis=2).sum() * cell
    body_only = design.model_copy(update={"surfaces": ()})
    assert areas(body_only).top_m2 == pytest.approx(top, rel=0.1)


def test_an_upswept_faceted_body_keeps_flat_faces_and_follows_the_axis_law(
    shapes: list[BodyShape],
) -> None:
    for sh in shapes[:40]:
        poly = f.facet_body(sh, LENGTH_M, AXES_M, upsweep_deg=12.0)
        level = f.facet_body(sh, LENGTH_M, AXES_M)
        assert q.planarity_m(poly) < 1e-9 * LENGTH_M
        assert q.is_closed(poly)
        assert len(poly.faces) == len(level.faces)
        v, v0 = np.asarray(poly.vertices_m), np.asarray(level.vertices_m)
        t = v0[:, 0] / LENGTH_M
        rise = f.axis_rise_m(t, sh, LENGTH_M, 12.0)
        assert np.allclose((v - v0)[:, 2], rise)
        x = np.linspace(0.05, 0.95, 12) * LENGTH_M
        axis = np.column_stack([x, np.zeros(12), f.axis_rise_m(x / LENGTH_M, sh, LENGTH_M, 12.0)])
        assert q.inside_faceted(poly, axis).all()  # the axis still runs through the solid
