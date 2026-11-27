"""U3: top-view edge angles are measured on every design; strict alignment snaps them, soft
rewards them in the style score, off changes nothing; whatever is not snapped is a decoder note."""

import numpy as np
import pytest

from aeroforge.brief import brief_requirements
from aeroforge.constants import number
from aeroforge.edge_report import edge_rows, shared_angles
from aeroforge.edges import (
    Edge,
    alignment_misses,
    allowed_angles_deg,
    angle_gap_deg,
    body_angles_deg,
    body_edges,
    soft_style_bonus,
    surface_edges,
)
from aeroforge.genetable import pin_choice
from aeroforge.genome import decode, random_genomes
from aeroforge.models import Design, Requirements

SEED = 20261126
N = 30
TOL = number("edge_align_tol_deg")


@pytest.fixture(scope="module")
def req() -> Requirements:
    return brief_requirements("stub")


def _pinned(mode: str, facets: str = "5") -> np.ndarray:
    gs = random_genomes(np.random.default_rng(SEED), N)
    return pin_choice(pin_choice(gs, "edge_alignment", mode), "airfoil_facets", facets)


@pytest.fixture(scope="module")
def strict(req: Requirements) -> list[Design]:
    return [decode(g, req) for g in _pinned("strict")]


def test_the_angle_set_holds_the_fixed_angles_the_sweep_and_its_complement() -> None:
    got = allowed_angles_deg(30.0)
    assert got == (-60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
    assert allowed_angles_deg(-30.0) == got and 90.0 in allowed_angles_deg(0.0)
    steep = body_angles_deg(30.0)
    assert set(got) < set(steep) and any(a > 80.0 for a in steep)


def test_angle_gap_wraps_at_a_half_turn() -> None:
    assert angle_gap_deg(89.0, -89.0) == pytest.approx(2.0)
    assert angle_gap_deg(10.0, 30.0) == pytest.approx(20.0)


def test_every_wing_and_tail_edge_of_a_strict_design_is_on_the_angle_set(
    strict: list[Design],
) -> None:
    seen = 0
    for d in strict:
        assert d.family.edge_alignment == "strict" and d.family.edge_angles_deg
        for s in d.surfaces:
            for e in surface_edges(s):
                gap = min(angle_gap_deg(e.sweep_deg, a) for a in d.family.edge_angles_deg)
                assert gap <= TOL, (s.name, e.kind, e.sweep_deg, d.family.edge_angles_deg)
                seen += 1
    assert seen > 100


def test_strict_keeps_area_and_span(strict: list[Design], req: Requirements) -> None:
    for d in strict:
        total = sum(s.projected_area_m2 for s in d.surfaces)
        assert total == pytest.approx(d.lifting_area_m2, rel=number("area_check_rel_tol"))
        reach = max(
            (abs(c.le_m[1]) for s in d.surfaces if s.mirrored for c in s.sections), default=0
        )
        assert 2 * reach <= req.span_max_m * (1 + 1e-9)


def test_a_flat_panel_body_of_a_strict_design_is_aligned_or_reported(req: Requirements) -> None:
    gs = pin_choice(_pinned("strict"), "body_style", "faceted")
    aligned = total = 0
    for g in gs:
        d = decode(g, req)
        for b in d.bodies:
            for e in body_edges(b):
                gap = min(angle_gap_deg(e.sweep_deg, a) for a in d.family.edge_body_angles_deg)
                total += 1
                aligned += gap <= TOL
                if gap > TOL:
                    assert any(n.startswith(f"strict alignment: {b.name} has") for n in d.notes)
    assert total > 40 and aligned / total > 0.4


def test_every_override_of_strict_is_a_decoder_note(req: Requirements) -> None:
    for g in _pinned("strict"):
        d = decode(g, req)
        assert any(n.startswith("strict alignment: edges snapped to") for n in d.notes)
        if d.family.tip_style != "square":
            assert any("tips made square" in n for n in d.notes)
        assert all(s.tip_style == "square" for s in d.surfaces)
        assert all(s.joined_to is None for s in d.surfaces)


def test_misses_are_measured_on_the_finished_design(strict: list[Design]) -> None:
    for d in strict:
        listed = {e.part for e, _ in alignment_misses(d)}
        noted = {n.split()[2] for n in d.notes if "outline edges up to" in n}
        assert listed == noted


def test_off_and_soft_decode_the_same_geometry_and_only_soft_pays_a_style_bonus(
    req: Requirements,
) -> None:
    off = [decode(g, req) for g in _pinned("off")[:8]]
    soft = [decode(g, req) for g in _pinned("soft")[:8]]
    for a, b in zip(off, soft, strict=True):
        assert a.surfaces == b.surfaces and a.bodies == b.bodies
        assert soft_style_bonus(a) == 0.0
        assert 0.0 < soft_style_bonus(b) <= number("edge_soft_weight")


def test_the_report_lists_measured_edges_for_every_alignment_mode(req: Requirements) -> None:
    for mode in ("off", "soft", "strict"):
        d = decode(_pinned(mode)[0], req)
        text = "\n".join(edge_rows(d))
        assert "| part | edge |" in text and f"edge_alignment: {mode}" in text
        assert "measured edges are within" in text


def test_shared_angles_group_edges_of_different_parts_only() -> None:
    edges = [
        Edge("surface0", "leading", 30.0, 1.0),
        Edge("surface0", "trailing", 30.3, 1.0),
        Edge("body0", "outline", 30.4, 1.0),
        Edge("surface1", "tip", 90.0, 0.2),
    ]
    got = shared_angles(edges)
    assert len(got) == 1 and len(got[0][1]) == 3
    assert shared_angles([edges[0], edges[1]]) == []
