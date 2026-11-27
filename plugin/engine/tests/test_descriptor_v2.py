"""U4: descriptor v2 (body-aware), the bomb cap, history versions and the report's contributors."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from aeroforge import areas
from aeroforge.areas import body_top_share, shares
from aeroforge.candidate import Candidate
from aeroforge.checks import CheckReport, CheckResult
from aeroforge.descriptor import (
    BODY_TOP_INDEX,
    DESC_SIZE,
    NAMES,
    contributions,
    desc,
)
from aeroforge.descriptor_v1 import DESC_SIZE_V1, desc_v1
from aeroforge.genetable import gene_table, pin_choice
from aeroforge.genome import DecodeError, decode, random_genomes
from aeroforge.models import Design, Requirements
from aeroforge.report import Novelty, _novelty_rows
from aeroforge.search import Scored, over_bomb_cap, select
from aeroforge.search_cli import append_history, read_history

SPEC = Path(__file__).parents[3] / "SPEC.md"
LIFT_NUMBERS = 11  # the first 11 numbers read lifting surfaces only
BAND_R = 0.5
D_MIN = 0.2


def test_every_number_is_named_and_documented_in_the_spec() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert len(NAMES) == DESC_SIZE == 20
    for name in NAMES[LIFT_NUMBERS:]:
        assert f"`{name}`" in text, name


def test_v1_descriptor_is_kept_with_its_sixteen_numbers(designs: list[Design]) -> None:
    x = np.array([desc_v1(d) for d in designs[:20]])
    assert x.shape == (20, DESC_SIZE_V1) and x.min() >= 0.0 and x.max() <= 1.0


@pytest.mark.slow
def test_v2_is_in_unit_range_on_one_thousand_genomes(full_designs: list[Design]) -> None:
    x = np.array([desc(d) for d in full_designs])
    assert x.shape == (len(full_designs), DESC_SIZE) and len(full_designs) == 1000
    assert np.isfinite(x).all() and x.min() >= 0.0 and x.max() <= 1.0


def _scaled_body(d: Design, factor: float) -> Design:
    bodies = tuple(b.model_copy(update={"length_m": b.length_m * factor}) for b in d.bodies)
    return d.model_copy(update={"bodies": bodies})


def test_designs_that_differ_only_in_body_length_get_different_descriptors(
    conventional: Design,
) -> None:
    a, b = desc(conventional), desc(_scaled_body(conventional, 1.4))
    assert np.array_equal(a[:LIFT_NUMBERS], b[:LIFT_NUMBERS])
    assert not np.allclose(a[LIFT_NUMBERS:], b[LIFT_NUMBERS:])
    assert a[NAMES.index("body_fineness")] < b[NAMES.index("body_fineness")]


def _decoded_pair(req: Requirements, gene: str, labels: tuple[str, str]) -> list[Design]:
    """The first seeded genome that decodes with each label of a choice gene, wings the same."""
    for seed in range(40):
        g = random_genomes(np.random.default_rng(seed), 1)
        try:
            pair = [decode(pin_choice(g, gene, label)[0], req) for label in labels]
        except DecodeError:
            continue
        if all(d.bodies for d in pair):
            return pair
    pytest.fail("no genome decodes with both labels")


def test_a_flat_panel_body_and_a_smooth_one_under_the_same_wings_differ(req: Requirements) -> None:
    smooth, faceted = _decoded_pair(req, "body_style", ("smooth", "faceted"))
    a, b = desc(smooth), desc(faceted)
    sheets = [NAMES.index(n) for n in ("body_panels", "body_crease_sharpness")]
    assert np.all(a[sheets] == 0.0) and np.all(b[sheets] > 0.0)
    assert not np.array_equal(a, b)


def test_the_body_changes_the_descriptor_only_through_the_body_numbers(req: Requirements) -> None:
    names = [g.name for g in gene_table()]
    assert "fam_body_fineness" in names
    g = random_genomes(np.random.default_rng(5), 1)
    short, long_ = g.copy(), g.copy()
    short[0, names.index("fam_body_fineness")], long_[0, names.index("fam_body_fineness")] = (
        0.0,
        1.0,
    )
    a, b = desc(decode(short[0], req)), desc(decode(long_[0], req))
    assert a[NAMES.index("body_fineness")] < b[NAMES.index("body_fineness")]


def test_a_body_that_leaves_the_wing_has_no_blend(conventional: Design) -> None:
    assert shares(conventional).blend > 0.05
    moved = [
        b.model_copy(update={"nose_m": (b.nose_m[0] + 50.0, *b.nose_m[1:])})
        for b in conventional.bodies
    ]
    apart = conventional.model_copy(update={"bodies": tuple(moved)})
    assert shares(apart).blend == pytest.approx(0.0, abs=1e-9)


def test_top_and_front_shares_match_the_area_module(designs: list[Design]) -> None:
    for d in designs[:25]:
        if not d.bodies:
            continue
        s = shares(d)
        assert s.top == pytest.approx(body_top_share(d), abs=0.01)
        assert 0.0 <= s.side <= 1.0 and 0.0 <= s.front <= 1.0 and 0.0 <= s.blend <= 1.0


def _triangles(d: Design) -> np.ndarray:
    """Every face of every flat-panel body, fanned into triangles in the world frame (x, z)."""
    out = []
    for b in d.bodies:
        if b.facets is None:
            continue
        v = np.asarray(b.facets.vertices_m) + np.asarray(b.nose_m)
        for face in b.facets.faces:
            out += [v[[face[0], face[k], face[k + 1]]][:, [0, 2]] for k in range(1, len(face) - 1)]
    return np.array(out)


def _raster_area(tri: np.ndarray, n: int = 400) -> float:
    """Area of the union of the triangles by testing a grid of points against each (barycentric)."""
    lo, hi = tri.reshape(-1, 2).min(axis=0), tri.reshape(-1, 2).max(axis=0)
    gx = lo[0] + (np.arange(n) + 0.5) * (hi[0] - lo[0]) / n
    gz = lo[1] + (np.arange(n) + 0.5) * (hi[1] - lo[1]) / n
    p = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
    inside = np.zeros(len(p), dtype=bool)
    for a, b, c in tri:
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-18:
            continue
        u = ((p[:, 0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[:, 1] - a[1])) / det
        v = ((b[0] - a[0]) * (p[:, 1] - a[1]) - (p[:, 0] - a[0]) * (b[1] - a[1])) / det
        inside |= (u >= 0) & (v >= 0) & (u + v <= 1)
    return float(inside.sum() * (hi[0] - lo[0]) * (hi[1] - lo[1]) / n**2)


def test_the_side_shadow_of_a_flat_panel_body_matches_a_triangle_raster(req: Requirements) -> None:
    checked = 0
    for seed in range(60):
        g = pin_choice(random_genomes(np.random.default_rng(seed), 1), "body_style", "faceted")
        try:
            d = decode(g[0], req)
        except DecodeError:
            continue
        tri = _triangles(d)
        if len(tri) == 0:
            continue
        body_only = areas._side_shadow(d.model_copy(update={"surfaces": ()}), 600)[0]
        assert body_only == pytest.approx(_raster_area(tri), rel=0.02), seed
        checked += 1
        if checked == 6:
            return
    pytest.fail(f"only {checked} flat-panel bodies checked")


def test_the_side_shadow_of_a_smooth_body_matches_its_section_integral(
    conventional: Design,
) -> None:
    from aeroforge.shapes import body_radius_frac

    body = conventional.bodies[0]
    assert body.facets is None
    t = np.linspace(0.0, 1.0, 20001)
    by_hand = np.trapezoid(2.0 * body.half_height_m * body_radius_frac(body, t), t) * body.length_m
    bare = conventional.model_copy(update={"surfaces": ()})
    assert areas._side_shadow(bare, 600)[0] == pytest.approx(by_hand, rel=0.01)


def _scored(index: int, share: float) -> Scored:
    x = [0.0] * DESC_SIZE
    x[BODY_TOP_INDEX] = share
    report = CheckReport(
        tier=1,
        feasible=True,
        violation=0.0,
        checks=(CheckResult(id="trim", margin=0.1, passed=True, detail={}),),
        passable=True,
    )
    c = Candidate((float(index),), tuple(x), 0.9, report, None, None)
    return Scored(index, c, 0.5 * D_MIN * (1.0 + index / 100.0), "random")


def test_a_design_over_the_bomb_cap_is_never_a_finalist() -> None:
    from aeroforge.constants import number
    from aeroforge.search import novelty_band

    cap = number("bomb_cap")
    lo, hi = novelty_band(BAND_R)
    mid = (lo + hi) / 2 * D_MIN
    evaluated = [
        Scored(k, _scored(k, share).candidate, mid * (1 + k / 50), "random")
        for k, share in enumerate([cap + 0.01, cap - 0.2, 0.9, cap, 0.0])
    ]
    picked = {s.index for s in select(evaluated, D_MIN, 0.6, BAND_R)}
    assert picked <= {1, 3, 4} and picked  # over the cap: 0 and 2
    assert over_bomb_cap(evaluated[0].candidate) and not over_bomb_cap(evaluated[3].candidate)
    assert select([evaluated[0], evaluated[2]], D_MIN, 0.6, BAND_R) == []


def test_contributions_name_the_three_numbers_that_set_the_distance() -> None:
    x, near = np.full(DESC_SIZE, 0.5), np.full(DESC_SIZE, 0.5)
    near[NAMES.index("body_fineness")] -= 0.4
    near[NAMES.index("pods")] -= 0.3
    near[NAMES.index("lift_x_1")] -= 0.1
    top = contributions(x, near)
    assert [n for n, _ in top] == ["body_fineness", "pods", "lift_x_1"]
    assert sum(s for _, s in top) == pytest.approx(1.0) and top[0][1] == pytest.approx(16 / 26)
    assert contributions(x, x.copy())[0][1] == 0.0


def test_the_report_lists_the_contributors() -> None:
    n = Novelty(0.2, "conventional", 0.3, None, None, (("pods", 0.5), ("body_fineness", 0.25)))
    text = "\n".join(_novelty_rows(n))
    assert "pods (50%)" in text and "body_fineness (25%)" in text


def test_history_keeps_only_rows_of_the_current_descriptor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.jsonl"
    old = {"id": "old", "desc": [0.1] * 16}
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    append_history(path, {"id": "new", "desc": [0.2] * DESC_SIZE})
    with caplog.at_level(logging.WARNING):
        rows = read_history(path)
    assert rows.shape == (1, DESC_SIZE) and rows[0, 0] == 0.2
    assert "older descriptor" in caplog.text
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == old
