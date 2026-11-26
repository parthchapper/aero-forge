"""U3 audit: a planarised wing stays buildable (flat caps, simple quads), and face areas are exact
on concave faces. The designs are seeds the audit found failing (cap 13 um off its plane; a quad
flipped into a figure of eight; wing caps whose area was over-counted by up to 19 %)."""

import numpy as np
import pytest

from aeroforge.brief import brief_requirements
from aeroforge.constants import number
from aeroforge.facet_planar import Planar, planarisation_holds
from aeroforge.facet_query import (
    area_centroid_m,
    face_areas_m2,
    is_closed,
    planarity_m,
    surface_area_m2,
    volume_m3,
)
from aeroforge.facet_wing import wing_polyhedron
from aeroforge.genetable import pin_choice
from aeroforge.genome import decode, random_genomes
from aeroforge.geom import surface_solid
from aeroforge.models import Design, Polyhedron
from aeroforge.panels import design_panels, flat_outline_mm

FACETS = ("smooth", "3", "4", "5", "6", "7", "8")
COARSE = ("off", "low", "mid", "high")
BRIEFS = ("stub", "small_boxes")
CAP_OFF_PLANE, FLIPPED_QUAD, CONCAVE_CAP = 55, 1168, 6  # audit seeds (U3 audit, 2026-11-26)


def _design(i: int) -> Design:
    """The audit's design i: faceted wings and coarseness chosen by the index, strict alignment."""
    g = random_genomes(np.random.default_rng(900000 + i), 1)
    g = pin_choice(g, "airfoil_facets", FACETS[i % 7])
    g = pin_choice(g, "facet_coarseness", COARSE[(i // 7) % 4])
    g = pin_choice(g, "edge_alignment", "strict" if i % 5 else "soft")
    return decode(g[0], brief_requirements(BRIEFS[i % 2]))


def _shoelace_m2(corners_xy_m: np.ndarray) -> float:
    x, y = corners_xy_m[:, 0], corners_xy_m[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _l_prism() -> Polyhedron:
    """A prism of height 1 over the concave L polygon of area 3 (perimeter 8)."""
    ring = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    v = [(x, y, 0.0) for x, y in ring] + [(x, y, 1.0) for x, y in ring]
    n = len(ring)
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return Polyhedron(vertices_m=tuple(v), faces=tuple(faces))


def test_a_concave_face_has_its_true_area() -> None:
    poly = _l_prism()
    assert volume_m3(poly) == pytest.approx(3.0)
    assert face_areas_m2(poly)[0] == pytest.approx(3.0) and face_areas_m2(poly)[1] == 3.0
    assert surface_area_m2(poly) == pytest.approx(2 * 3.0 + 8.0)


def test_the_skin_centroid_of_a_concave_prism_is_on_its_mirror_plane_and_mid_height() -> None:
    c = area_centroid_m(_l_prism())
    assert c[2] == pytest.approx(0.5) and c[0] == pytest.approx(c[1])


@pytest.mark.parametrize("i", [CONCAVE_CAP, CAP_OFF_PLANE, FLIPPED_QUAD])
def test_every_cut_list_area_equals_the_area_of_its_flat_outline(i: int) -> None:
    for p in design_panels(_design(i)):
        assert _shoelace_m2(flat_outline_mm(p) / 1000.0) == pytest.approx(p.area_m2, rel=1e-9)


@pytest.mark.parametrize("i", [CAP_OFF_PLANE, FLIPPED_QUAD])
def test_wings_that_once_failed_to_build_are_flat_closed_and_build(i: int) -> None:
    d = _design(i)
    faceted = [s for s in d.surfaces if s.airfoil_facets]
    assert faceted
    for s in faceted:
        poly = wing_polyhedron(s)
        assert planarity_m(poly) <= number("facet_geom_tol_m")
        assert is_closed(poly)
        assert surface_solid(s).isValid()


def test_a_rejected_strip_merge_is_a_decoder_note() -> None:
    assert any("no strip merge was accepted" in n for n in _design(FLIPPED_QUAD).notes)


def test_coarse_wings_of_random_genomes_are_flat_closed_and_build() -> None:
    req = brief_requirements("stub")
    gs = random_genomes(np.random.default_rng(20261126), 10)
    for g in gs:
        g = pin_choice(pin_choice(g[None, :], "facet_coarseness", "high"), "airfoil_facets", "8")[0]
        for s in (s for s in decode(g, req).surfaces if s.airfoil_facets):
            poly = wing_polyhedron(s)
            assert planarity_m(poly) <= number("facet_geom_tol_m") and is_closed(poly)
            assert surface_solid(s).isValid()


def _planar(vertices: np.ndarray, merged: tuple[tuple[int, ...], ...]) -> Planar:
    return Planar(vertices, frozenset(merged), 0, 0.0)


def test_the_planarisation_gate_rejects_a_figure_of_eight_quad() -> None:
    flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    bow = flat[[0, 1, 3, 2]]
    assert planarisation_holds(flat, _planar(flat, ((0, 1, 2, 3),)), [])
    assert not planarisation_holds(flat, _planar(bow, ((0, 1, 2, 3),)), [])


def test_the_planarisation_gate_rejects_a_cap_off_its_plane() -> None:
    ring = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1e-4]], dtype=float)
    none: tuple[tuple[int, ...], ...] = ()
    assert planarisation_holds(ring[:4], _planar(ring[:4], none), [(0, 1, 2, 3)])
    assert not planarisation_holds(ring, _planar(ring, none), [(0, 1, 2, 3, 4)])
