import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import trimesh

from aeroforge.constants import number
from aeroforge.facet_query import volume_m3
from aeroforge.facet_wing import wing_polyhedron
from aeroforge.genome import decode
from aeroforge.geom import GeometryError, build_parts
from aeroforge.mesh import PartMesh, export_mesh, glb_bytes, tessellate, write_stls
from aeroforge.models import Body, Design, Requirements, Surface, instance_names
from aeroforge.render import render_png
from aeroforge.shapes import body_radius_frac, section_outline

CAD_SAMPLE = 100
CAD_SUCCESS_FLOOR = 0.98
VOLUME_DESIGNS = 8
# Spline-vs-polygon outlines, the smooth body loft and tessellation chords stay well inside this.
VOLUME_REL_TOL = 0.02
RADIUS_SAMPLES = 20001


@pytest.fixture(scope="module")
def meshes(designs: list[Design]) -> list[PartMesh]:
    return tessellate(build_parts(designs[1]))


def _expected_names(d: Design) -> set[str]:
    parts = [*d.surfaces, *d.bodies, *d.propulsors]
    return {n for p in parts for n in instance_names(p.name, p.mirrored)}


@pytest.mark.slow
def test_cad_builds_for_at_least_98_percent_of_random_genomes(designs: list[Design]) -> None:
    built = 0
    for d in designs[:CAD_SAMPLE]:
        try:
            build_parts(d)
            built += 1
        except GeometryError:
            pass
    assert built / CAD_SAMPLE >= CAD_SUCCESS_FLOOR


@pytest.mark.slow
def test_part_names_are_the_stable_instance_names(designs: list[Design]) -> None:
    for d in designs[:3]:
        assert set(build_parts(d)) == _expected_names(d)


def test_broken_section_raises_typed_geometry_error(designs: list[Design]) -> None:
    d = designs[0]
    s0 = d.surfaces[0]
    flat = s0.model_copy(
        update={"sections": tuple(c.model_copy(update={"chord_m": 0.0}) for c in s0.sections)}
    )
    broken = d.model_copy(update={"surfaces": (flat, *d.surfaces[1:])})
    with pytest.raises(GeometryError):
        build_parts(broken)


@pytest.mark.slow
def test_glb_node_names_match_part_names(meshes: list[PartMesh]) -> None:
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes(meshes)), file_type="glb")
    names = {m.name for m in meshes}
    assert set(scene.graph.nodes_geometry) == names
    assert set(scene.geometry) == names


@pytest.mark.slow
def test_same_genome_renders_identical_png_bytes(genomes: np.ndarray, req: Requirements) -> None:
    def pipeline() -> bytes:
        return render_png(tessellate(build_parts(decode(genomes[2].copy(), req))), "iso", 320)

    assert pipeline() == pipeline()


@pytest.mark.slow
def test_one_stl_is_written_per_part(meshes: list[PartMesh], tmp_path: Path) -> None:
    write_stls(tmp_path, meshes)
    assert sorted(p.stem for p in tmp_path.glob("*.stl")) == sorted(m.name for m in meshes)


@pytest.mark.slow
def test_written_stls_are_watertight(meshes: list[PartMesh], tmp_path: Path) -> None:
    write_stls(tmp_path, meshes)
    assert all(trimesh.load(p).is_watertight for p in tmp_path.glob("*.stl"))


GAUSS_NODES = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))  # exact for the integrand


def _ruled_side_volume(a: np.ndarray, b: np.ndarray) -> float:
    """Divergence-theorem volume term (1/3 integral of P . (P_f x P_u)) of the ruled strip between
    closed outlines a and b: every point pair (a_i, b_i) is joined by a straight line, so each
    polygon edge sweeps a bilinear patch, and the integrand is biquadratic (2-point Gauss exact)."""
    a1, b1 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    total = 0.0
    for f in GAUSS_NODES:
        for u in GAUSS_NODES:
            p = (1 - f) * ((1 - u) * a + u * a1) + f * ((1 - u) * b + u * b1)
            p_f = (1 - u) * (b - a) + u * (b1 - a1)
            p_u = (1 - f) * (a1 - a) + f * (b1 - b)
            total += float((p * np.cross(p_f, p_u)).sum()) / 12.0
    return total


def _cap_volume(outline: np.ndarray) -> float:
    """Volume term of a planar end cap: signed tetrahedra of a fan from its first point."""
    p0 = outline[0]
    return (
        sum(
            float(np.dot(outline[i], np.cross(outline[i + 1], p0)))
            for i in range(1, len(outline) - 1)
        )
        / 6.0
    )


def _surface_volume(s: Surface) -> float:
    """Exact volume of the ruled loft through the sections' own outlines (twist, cant and chord
    included, as the decoder specifies them): ruled strips plus the two planar end caps. The
    outline of a smooth surface is the NACA polygon the CAD spline passes through. A flat-panel
    wing's volume is that of its polyhedron (the planarised strips move its vertices off the
    section outlines, up to a few per cent on coarse wings; the audit checks CAD = polyhedron)."""
    if s.airfoil_facets:
        return volume_m3(wing_polyhedron(s))
    n_af = int(number("airfoil_points_per_side"))
    rings = [section_outline(c, s, n_af) for c in s.sections]
    side = sum(_ruled_side_volume(a, b) for a, b in pairwise(rings))
    return abs(side + _cap_volume(rings[0]) - _cap_volume(rings[-1]))


def _body_volume(b: Body) -> float:
    """Super-ellipse section area 4ab G(1+1/n)^2 / G(1+2/n) integrated along the radius law; a
    flat-panel body's volume is that of its polyhedron (divergence theorem over its faces)."""
    if b.facets is not None:
        return volume_m3(b.facets)
    t = np.linspace(0.0, 1.0, RADIUS_SAMPLES)
    shape = 4 * math.gamma(1 + 1 / b.section_exp) ** 2 / math.gamma(1 + 2 / b.section_exp)
    radius_sq = float(np.trapezoid(body_radius_frac(b, t) ** 2, t))
    return b.length_m * b.half_width_m * b.half_height_m * shape * radius_sq


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["surface", "body"])
def test_tessellated_volumes_match_the_analytic_shapes(decoded: list[Design], kind: str) -> None:
    for d in decoded[:VOLUME_DESIGNS]:
        meshes = {m.name: m for m in tessellate(build_parts(d))}
        parts = (
            [(s, _surface_volume(s)) for s in d.surfaces]
            if kind == "surface"
            else [(b, _body_volume(b)) for b in d.bodies]
        )
        for part, analytic in parts:
            for name in instance_names(part.name, part.mirrored):
                volume = export_mesh(meshes[name], merge=True).volume
                assert abs(volume - analytic) <= VOLUME_REL_TOL * analytic, name
