"""Faceted bodies, the plan's remaining U1 tests: volume, symmetry, panel plans for every count,
upsweep at several angles and sizes, and shadow areas in both views."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from test_facets import SEED, random_shape

from aeroforge import areas as ar
from aeroforge import facet_query as q
from aeroforge import facets as f
from aeroforge.areas import areas
from aeroforge.brief import brief_requirements
from aeroforge.facet_ends import end_panel_count
from aeroforge.genetable import gene_table, genome_length
from aeroforge.genome import decode
from aeroforge.models import BodyShape, Design, Polyhedron

SIZES = [(0.05, (0.006, 0.005)), (0.4, (0.04, 0.03)), (1.3, (0.09, 0.07)), (60.0, (3.0, 2.0))]
UPSWEEPS_DEG = [0.0, 5.0, 15.0]
MIN_COUNT, MAX_COUNT = 4, 200
N_PLAN_SHAPES = 2
SCAN_LINES = 3000


@pytest.fixture(scope="module")
def shapes() -> list[BodyShape]:
    rng = np.random.default_rng(SEED + 1)
    return [random_shape(rng) for _ in range(24)]


def _divergence_volume(poly: Polyhedron) -> float:
    v = np.asarray(poly.vertices_m)
    total = 0.0
    for face in poly.faces:
        p = v[list(face)]
        total += sum(np.dot(p[0], np.cross(p[k], p[k + 1])) for k in range(1, len(face) - 1)) / 6
    return float(total)


def _cell_hull_volume(poly: Polyhedron) -> float:
    v = np.asarray(poly.vertices_m)
    layers = np.unique(v[:, 0])
    return float(
        sum(ConvexHull(v[(v[:, 0] >= lo) & (v[:, 0] <= hi)]).volume for lo, hi in pairwise(layers))
    )


@pytest.mark.parametrize("upsweep_deg", UPSWEEPS_DEG)
def test_the_volume_by_the_divergence_theorem_equals_the_sum_of_cell_volumes(
    shapes: list[BodyShape], upsweep_deg: float
) -> None:
    for sh in shapes:
        poly = f.facet_body(sh, 1.3, (0.09, 0.07), upsweep_deg)
        volume = _divergence_volume(poly)
        assert volume > 0.0
        assert _cell_hull_volume(poly) == pytest.approx(volume, rel=1e-9)


@pytest.mark.parametrize(("length_m", "axes_m"), SIZES)
def test_the_polyhedron_is_mirror_symmetric_about_the_y0_plane(
    shapes: list[BodyShape], length_m: float, axes_m: tuple[float, float]
) -> None:
    for sh in shapes:
        v = np.asarray(f.facet_body(sh, length_m, axes_m).vertices_m)
        mirrored = v * np.array([1.0, -1.0, 1.0])
        gap = [np.abs(v - m).sum(axis=1).min() for m in mirrored]
        assert max(gap) < 1e-9 * length_m


@pytest.mark.parametrize("upsweep_deg", UPSWEEPS_DEG)
@pytest.mark.parametrize(("length_m", "axes_m"), SIZES)
def test_upswept_bodies_stay_flat_closed_counted_and_on_the_axis_law(
    shapes: list[BodyShape], upsweep_deg: float, length_m: float, axes_m: tuple[float, float]
) -> None:
    for sh in shapes[:12]:
        plan = f.effective_plan(sh)
        poly = f.facet_body(sh, length_m, axes_m, upsweep_deg)
        level = f.facet_body(sh, length_m, axes_m)
        assert q.planarity_m(poly) < 1e-9 * length_m
        assert q.is_closed(poly)
        assert len(poly.faces) == f.panel_count(plan, f.section_polygon(sh, sh.sides))
        t = np.asarray(level.vertices_m)[:, 0] / length_m
        rise = f.axis_rise_m(t, sh, length_m, upsweep_deg)
        shift = np.asarray(poly.vertices_m)[:, 2] - np.asarray(level.vertices_m)[:, 2]
        assert np.allclose(shift, rise, atol=1e-9 * length_m)
        x = np.linspace(0.05, 0.95, 15) * length_m
        rise_x = f.axis_rise_m(x / length_m, sh, length_m, upsweep_deg)
        assert q.inside_faceted(poly, np.column_stack([x, np.zeros(15), rise_x])).all()


def _reachable(shape: BodyShape) -> set[int]:
    """Every panel count the plan space can make, by the formula N = K (S - 1 + B) + ends."""
    reach: set[int] = set()
    for k in range(3, 13):
        poly = f.section_polygon(shape, k)
        for s in range(1, 9):
            for nose in f.NOSES:
                for tail in f.TAILS:
                    if s < 1 + (nose == "blunt") + (tail == "flat"):
                        continue
                    base = f.PanelPlan(k, s, 0, nose, tail)
                    breaks = len(f.break_stations(base, shape.max_section_x_frac))
                    ends = end_panel_count(poly, nose) + end_panel_count(poly, tail)
                    reach |= {k * (s - 1 + b) + ends for b in range(breaks + 1)}
    return reach


def test_panel_plan_is_exact_for_every_reachable_count_and_nearest_for_the_rest(
    shapes: list[BodyShape],
) -> None:
    for sh in shapes[:N_PLAN_SHAPES]:
        wish = f.effective_plan(sh)
        reach = _reachable(sh)
        for count in range(MIN_COUNT, MAX_COUNT + 1):
            plan, exact = f.panel_plan(count, wish, sh)
            got = f.panel_count(plan, f.section_polygon(sh, plan.sides))
            assert exact == (count in reach), count
            if exact:
                assert got == count
                assert len(f.body_polyhedron(plan, sh, 1.0, (0.1, 0.08)).faces) == count
            else:
                assert abs(got - count) == min(abs(n - count) for n in reach), count


def _faceted_body_only_designs(n: int) -> list[Design]:
    names = [g.name for g in gene_table()]
    req = brief_requirements("stub")
    rng = np.random.default_rng(SEED + 2)
    out: list[Design] = []
    while len(out) < n:
        genome = rng.random(genome_length())
        genome[names.index("body_style")] = 0.9
        design = decode(genome, req)
        body = next((b for b in design.bodies if b.name == "body0"), None)
        if body is not None and body.facets is not None:
            out.append(
                design.model_copy(update={"surfaces": (), "bodies": (body,), "propulsors": ()})
            )
    return out


def _grid_shadows(design: Design) -> tuple[float, float]:
    """(top, frontal) shadow area of the single body by counting solid grid points."""
    body = design.bodies[0]
    assert body.facets is not None
    nx, ny, nz = 160, 81, 81
    z = np.asarray(body.facets.vertices_m)[:, 2]
    gx = np.linspace(0.0, body.length_m, nx)
    gy = np.linspace(-body.half_width_m, body.half_width_m, ny)
    gz = np.linspace(float(z.min()), float(z.max()), nz)
    pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = q.inside_faceted(body.facets, pts).reshape(nx, ny, nz)
    top = inside.any(axis=2).sum() * (gx[1] - gx[0]) * (gy[1] - gy[0])
    front = inside.any(axis=0).sum() * (gy[1] - gy[0]) * (gz[1] - gz[0])
    return float(top), float(front)


def test_shadow_areas_in_both_views_match_a_grid_count_of_the_solid() -> None:
    for design in _faceted_body_only_designs(4):
        top, front = _grid_shadows(design)
        measured = areas(design)
        assert measured.top_m2 == pytest.approx(top, rel=0.04)
        assert measured.frontal_m2 == pytest.approx(front, rel=0.04)


def _union_area(polys: np.ndarray) -> float:
    lo, hi = polys[..., 0].min(), polys[..., 0].max()
    strip = (hi - lo) / SCAN_LINES
    s = lo + (np.arange(SCAN_LINES) + 0.5) * strip
    return float(ar._union_length(*ar._polygon_intervals(polys, s)).sum() * strip)


def _face_polygons(poly: Polyhedron, columns: tuple[int, int]) -> np.ndarray:
    v = np.asarray(poly.vertices_m)[:, list(columns)]
    width = max(len(fc) for fc in poly.faces)
    return np.array(
        [
            np.vstack([v[list(fc)], np.repeat(v[[fc[-1]]], width - len(fc), axis=0)])
            for fc in poly.faces
        ]
    )


@pytest.mark.parametrize("columns", [(1, 0), (1, 2)])
def test_the_cell_hull_shadow_equals_the_union_of_the_face_shadows(
    columns: tuple[int, int],
) -> None:
    for design in _faceted_body_only_designs(4):
        facets = design.bodies[0].facets
        assert facets is not None
        cells = _union_area(q.cell_projection(facets, columns))
        assert cells == pytest.approx(_union_area(_face_polygons(facets, columns)), rel=1e-6)
