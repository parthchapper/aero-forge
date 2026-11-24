import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import trimesh

from aeroforge.airfoil import naca_outline
from aeroforge.constants import number
from aeroforge.genome import decode
from aeroforge.geom import GeometryError, build_parts
from aeroforge.mesh import PartMesh, export_mesh, glb_bytes, tessellate, write_stls
from aeroforge.models import Body, Design, Requirements, Surface, instance_names
from aeroforge.render import render_png
from aeroforge.shapes import X_HAT, body_radius_frac, section_incidence_deg, section_normal

CAD_SAMPLE = 100
CAD_SUCCESS_FLOOR = 0.98
VOLUME_DESIGNS = 8
# Spline-vs-polygon outlines, the smooth body loft and tessellation chords stay well inside this.
VOLUME_REL_TOL = 0.02
GAUSS_POINTS = 3  # the loft volume integrand is cubic in the spanwise fraction
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


def _cross_yz(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[1] * v[2] - u[2] * v[1])


def _outline_moments(s: Surface) -> tuple[float, float, float]:
    """(area, x-moment, y-moment) of the unit-chord outline polygon about the leading edge."""
    xy = naca_outline(
        s.camber_frac, s.camber_loc_frac, s.tc, int(number("airfoil_points_per_side"))
    )
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    return (
        float(cross.sum() / 2),
        float(((x + x1) * cross).sum() / 6),
        float(((y + y1) * cross).sum() / 6),
    )


def _section_uv(s: Surface, sec: object) -> tuple[np.ndarray, np.ndarray]:
    """(chord vector, thickness vector) [m] of one section, incidence and twist included."""
    normal = section_normal(sec.cant_deg)  # type: ignore[attr-defined]
    ang = np.radians(section_incidence_deg(s, sec))  # type: ignore[arg-type]
    cd = np.cos(ang) * X_HAT - np.sin(ang) * normal
    td = np.sin(ang) * X_HAT + np.cos(ang) * normal
    return sec.chord_m * cd, sec.chord_m * td  # type: ignore[attr-defined]


def _surface_volume(s: Surface) -> float:
    """Exact volume of ruled loft L(f) + xc U(f) + yc V(f) via Gauss quadrature on |J|."""
    area, mx, my = _outline_moments(s)
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_POINTS)
    total = 0.0
    for a, b in pairwise(s.sections):
        dl = np.subtract(b.le_m, a.le_m)
        u0, v0 = _section_uv(s, a)
        u1, v1 = _section_uv(s, b)
        du, dv = u1 - u0, v1 - v0
        for node, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
            w = np.cross(u0 + node * du, v0 + node * dv)
            total += weight * float(np.dot(dl, w) * area + np.dot(du, w) * mx + np.dot(dv, w) * my)
    return abs(total)


def _body_volume(b: Body) -> float:
    """Super-ellipse section area 4ab G(1+1/n)^2 / G(1+2/n) integrated along the radius law."""
    t = np.linspace(0.0, 1.0, RADIUS_SAMPLES)
    shape = 4 * math.gamma(1 + 1 / b.section_exp) ** 2 / math.gamma(1 + 2 / b.section_exp)
    radius_sq = float(np.trapezoid(body_radius_frac(b, t) ** 2, t))
    return b.length_m * b.half_width_m * b.half_height_m * shape * radius_sq


@pytest.mark.slow
def test_tessellated_volumes_match_the_analytic_shapes(decoded: list[Design]) -> None:
    for d in decoded[:VOLUME_DESIGNS]:
        meshes = {m.name: m for m in tessellate(build_parts(d))}
        parts = [(s, _surface_volume(s)) for s in d.surfaces]
        parts += [(b, _body_volume(b)) for b in d.bodies]
        for part, analytic in parts:
            for name in instance_names(part.name, part.mirrored):
                volume = export_mesh(meshes[name], merge=True).volume
                assert abs(volume - analytic) <= VOLUME_REL_TOL * analytic, name
