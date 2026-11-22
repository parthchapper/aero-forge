import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import SAMPLE_BRIEF

from aeroforge.areas import areas
from aeroforge.candidate import Candidate, evaluate_candidate, passable_bar
from aeroforge.checks import CheckReport, CheckResult
from aeroforge.constants import number
from aeroforge.demo_cli import WINNER_FILE, _one_run, pick_winner
from aeroforge.descriptor import desc, median_nearest
from aeroforge.genome import decode, random_genomes
from aeroforge.geom import GeometryError
from aeroforge.models import Adjust, Brief, Design, Requirements
from aeroforge.physics_cli import BRIEF_FILE, REQUIREMENTS_FILE
from aeroforge.reference import reference_genome, reference_names
from aeroforge.search import Run, Scored, _rank01, novelty_band, objective, search, select
from aeroforge.search_cli import (
    FINALISTS_FILE,
    HISTORY_FILE,
    VERIFIED_FILE,
    accept_other_failures_max,
    accepted,
    finalist_entry,
    read_history,
    run_search,
    worker_pool,
)

SEED_NAMES = ("conventional", "canard", "flying_wing")
CENTROID_DESIGNS = 400
POOL_WORKERS = 8
POOL_GENOMES = 6
SMALL_BUDGET = 100
CHAIN_RUNS = 5
CHAIN_BUDGET = 300
SAFE_R = 0.0  # the step-4 band [1, 2] x d_min
DEFAULT_R = 0.5
FAR = 10.0  # descriptor spacing of unrelated test candidates (d_min is 0.2)


def _report(margins: dict[str, float]) -> CheckReport:
    checks = tuple(
        CheckResult(id=k, margin=m, passed=m >= 0.0, detail={}) for k, m in margins.items()
    )
    return CheckReport(
        tier=1,
        feasible=all(c.passed for c in checks),
        violation=sum(max(0.0, -c.margin) for c in checks),
        checks=checks,
        passable=all(c.passed for c in checks if c.id in passable_bar()),
    )


def test_objective_is_style_when_passable_else_minus_bar_violation() -> None:
    passing = Candidate((0.5,), None, 0.7, _report({"trim": 0.1, "energy": -3.0}), None, None)
    failing = Candidate((0.5,), None, 0.9, _report({"trim": -0.2, "energy": -3.0}), None, None)
    broken = Candidate((0.5,), None, 0.9, None, None, "injected")
    assert objective(passing) == pytest.approx(0.7)
    assert objective(failing) == pytest.approx(-0.2)
    assert objective(broken) == number("search_fail_objective")


def _run(req: Requirements, designs: list[Design], budget: int) -> Run:
    seeds = np.array([reference_genome(n) for n in SEED_NAMES])
    refs = np.array([desc(decode(g, req)) for g in seeds])
    samples = np.array([desc(d) for d in designs[:CENTROID_DESIGNS]])
    return Run(
        seeds, SEED_NAMES, refs, samples, median_nearest(refs), budget, seed=3, randomness=DEFAULT_R
    )


def _injected(req: Requirements):  # noqa: ANN202
    """Every other candidate fails to decode; the rest are infeasible or carry NaN margins."""

    def evaluate(genomes: np.ndarray) -> list[Candidate]:
        out = []
        for k, g in enumerate(genomes):
            x = tuple(map(float, desc(decode(g, req)))) if k % 2 else None
            report = _report({"trim": float("nan") if k % 3 else -1.0}) if x else None
            out.append(Candidate(tuple(map(float, g)), x, 0.5, report, None, "injected"))
        return out

    return evaluate


@pytest.mark.slow
def test_injected_failures_never_crash_and_end_in_a_flagged_fallback(
    req: Requirements, designs: list[Design]
) -> None:
    result = search(_run(req, designs, budget=1), _injected(req))
    assert result.doubled and result.fallback
    assert result.budget_used >= 2
    assert all(not s.candidate.passable for s in result.finalists)


def test_cad_failure_moves_to_the_next_finalist(conventional: Design, tmp_path: Path) -> None:
    def build(design: Design, folder: Path, tile_id: str) -> tuple[str, str]:
        if tile_id == "bad" or design.notes == ("cad fails",):
            raise GeometryError("surface0", "injected CAD failure")
        return ("iso.png", "top.png")

    broken = conventional.model_copy(update={"notes": ("cad fails",)})
    c = Candidate((0.5,), None, 0.5, None, None, None)
    checked = {"f00": (c, broken), "f01": (c, None), "f02": (c, conventional)}
    fid, views, errors = pick_winner(["f00", "f01", "f02"], checked, build, tmp_path, "t")
    assert fid == "f02" and views == ("iso.png", "top.png") and len(errors) == 2
    assert pick_winner(["f00"], checked, build, tmp_path, "t")[0] is None


def _scored(
    gene: float,
    index: int,
    nov: float,
    power: float = 0.3,
    style: float = 0.9,
    desc: tuple[float, ...] | None = None,
) -> Scored:
    """A passable candidate; its descriptor sits far from the others unless `desc` is given."""
    report = _report({"trim": 0.1, "power": power})
    x = desc if desc is not None else (gene * FAR,)
    return Scored(index, Candidate((gene,), x, style, report, None, None), nov, "random")


def test_select_keeps_the_novelty_band_and_drops_repeats() -> None:
    d_min = 0.2
    lo, hi = novelty_band(SAFE_R)
    mid = (lo + hi) / 2 * d_min
    evaluated = [
        _scored(0.1, 0, lo * d_min),
        _scored(0.2, 1, hi * d_min),
        _scored(0.3, 2, lo * d_min * 0.99),
        _scored(0.4, 3, hi * d_min * 1.01),
        _scored(0.1, 4, mid),
        _scored(0.5, 5, mid, style=0.7),
        _scored(0.6, 6, mid, style=0.5),
    ]
    picked = select(evaluated, d_min, 0.6, SAFE_R)
    assert sorted(s.index for s in picked) == [0, 1, 5]


def _in_band(r: float, d_min: float = 0.2) -> float:
    """Novelty at the middle of the band for r, so candidates qualify at every r."""
    lo, hi = novelty_band(r)
    return (lo + hi) / 2 * d_min


def test_a_qualifier_with_a_huge_power_failure_can_still_rank_first() -> None:
    d_min = 0.2
    for r in (0.0, DEFAULT_R, 1.0):
        mid = _in_band(r)
        evaluated = [
            _scored(0.1, 0, mid, power=0.3, style=0.8),
            _scored(0.2, 1, mid * 1.1, power=-1.0e6, style=0.99),
            _scored(0.3, 2, mid * 0.9, power=0.3, style=0.7),
        ]
        assert select(evaluated, d_min, 0.6, r)[0].index == 1


def test_at_r1_the_most_novel_qualifier_ranks_first_and_at_r0_style_decides() -> None:
    d_min = 0.2
    for r, first in ((1.0, 1), (0.0, 0)):
        mid = _in_band(r)
        evaluated = [
            _scored(0.1, 0, mid, style=0.99),
            _scored(0.2, 1, mid * 1.2, style=0.7),
            _scored(0.3, 2, mid * 0.8, style=0.8),
        ]
        assert select(evaluated, d_min, 0.6, r)[0].index == first
    assert [s.index for s in select(evaluated, d_min, 0.6, 0.0)] != [1, 0, 2]


def test_the_ranking_ignores_total_violation() -> None:
    d_min = 0.2
    mid = _in_band(1.0)
    evaluated = [_scored(0.1, 0, mid, power=-50.0), _scored(0.2, 1, mid, power=0.3)]
    # equal novelty and style tie on index; a least-violation ranking would put index 1 first
    assert [s.index for s in select(evaluated, d_min, 0.6, 1.0)] == [0, 1]


def test_all_finalists_are_at_least_d_min_apart_and_the_best_scores_first() -> None:
    d_min = 0.2
    line = [
        _scored(0.01 * (k + 1), k, _in_band(1.0) + 0.001 * k, desc=(0.06 * k, 0.0))
        for k in range(30)
    ]
    picked = select(line, d_min, 0.6, 1.0)
    assert picked[0].index == 29  # highest novelty at r = 1
    xs = np.array([s.candidate.desc for s in picked])
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(xs) for b in xs[:i]]
    assert gaps and min(gaps) >= d_min - 1e-12
    assert [s.index for s in picked] == [
        29,
        25,
        21,
        17,
        13,
        9,
        5,
        1,
    ]  # four 0.06 steps clear d_min, three do not


def test_rank01_spreads_values_over_zero_one_with_mean_ranks_for_ties() -> None:
    assert _rank01(np.array([5.0])) == pytest.approx([0.5])
    assert _rank01(np.array([3.0, 1.0, 2.0])) == pytest.approx([1.0, 0.0, 0.5])
    assert _rank01(np.array([1.0, 1.0, 2.0])) == pytest.approx([0.25, 0.25, 1.0])


def _folder(tmp: Path, req: Requirements) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE_BRIEF, tmp / BRIEF_FILE)
    (tmp / REQUIREMENTS_FILE).write_text(req.model_dump_json(), encoding="utf-8")
    return tmp


@pytest.mark.slow
def test_pool_and_inline_evaluation_agree(sample_brief: Brief, sample_req: Requirements) -> None:
    genomes = random_genomes(np.random.default_rng(11), POOL_GENOMES)
    inline = [evaluate_candidate(g, sample_brief, sample_req, 1) for g in genomes]
    with worker_pool(POOL_WORKERS) as pool:
        fn = functools.partial(evaluate_candidate, brief=sample_brief, req=sample_req, tier=1)
        pooled = list(pool.map(fn, genomes))
    assert pooled == inline


@pytest.mark.slow
def test_same_seed_history_and_brief_give_identical_finalists(
    sample_req: Requirements, tmp_path: Path
) -> None:
    history = tmp_path / HISTORY_FILE
    texts = []
    with worker_pool(POOL_WORKERS) as pool:
        for k in range(2):
            folder = _folder(tmp_path / f"run{k}", sample_req)
            run_search(pool, folder, 5, SMALL_BUDGET, history, DEFAULT_R)
            texts.append((folder / FINALISTS_FILE).read_bytes())
    assert texts[0] == texts[1]


def _no_cad(design: Design, folder: Path, tile_id: str) -> tuple[str, str]:
    return ("", "")


@pytest.mark.slow
def test_chained_runs_give_tier2_passing_winners_beyond_d_min(
    sample_req: Requirements, tmp_path: Path
) -> None:
    out = _folder(tmp_path, sample_req)
    with worker_pool(int(number("search_workers"))) as pool:
        for i in range(CHAIN_RUNS):
            _one_run(pool, out, i, 100 + i, CHAIN_BUDGET, _no_cad, DEFAULT_R)
    refs = np.array([desc(decode(reference_genome(n), sample_req)) for n in reference_names()])
    d_min = median_nearest(refs)
    winners = read_history(out / HISTORY_FILE)
    assert len(winners) == CHAIN_RUNS
    for i in range(CHAIN_RUNS):
        run = out / f"run{i:02d}"
        verified = json.loads((run / VERIFIED_FILE).read_text(encoding="utf-8"))
        winner = json.loads((run / WINNER_FILE).read_text(encoding="utf-8"))
        assert winner["finalist"] in verified["passing"]
        others = np.vstack([refs, winners[:i]])
        assert np.sqrt(((others - winners[i]) ** 2).sum(axis=1)).min() >= d_min


def test_finalist_entries_carry_the_shadow_areas(sample_req: Requirements) -> None:
    genome = tuple(float(x) for x in reference_genome("conventional"))
    c = Candidate(genome, (0.0,), 0.9, _report({"trim": 0.1}), Adjust(), None)
    entry = finalist_entry(0, Scored(0, c, 0.3, "random"), sample_req)
    shadow = areas(decode(np.array(genome), sample_req, Adjust()))
    assert entry["areas"] == {
        "top_area_m2": shadow.top_m2,
        "frontal_area_m2": shadow.frontal_m2,
        "body_top_share": shadow.body_top_share,
    }
    failed = Candidate(genome, None, 0.0, None, None, "injected")
    assert finalist_entry(1, Scored(1, failed, 0.0, "random"), sample_req)["areas"] is None


def _row(fid: str, passable: bool, failed: list[str]) -> dict:
    return {"id": fid, "passable": passable, "failed": failed}


def test_accepted_keeps_passable_rows_with_at_most_max_other_failures_in_score_order() -> None:
    rows = [
        _row("f00", False, ["lateral"]),
        _row("f01", True, ["stall_order", "energy", "efficiency"]),
        _row("f02", True, ["stall_order"]),
        _row("f03", True, []),
        _row("f04", True, ["energy", "power"]),
        _row("f05", True, []),
    ]
    assert accepted(rows, 2, 3) == ["f02", "f03", "f04"]


def test_accept_other_failures_max_is_2_at_r0_and_covers_every_check_at_r1(
    conventional_tier2: CheckReport,
) -> None:
    assert accept_other_failures_max(0.0) == 2
    assert accept_other_failures_max(1.0) >= len(conventional_tier2.checks)
    assert 2 < accept_other_failures_max(DEFAULT_R) < accept_other_failures_max(1.0)


def test_at_r1_a_finalist_failing_power_energy_and_efficiency_is_still_delivered() -> None:
    rows = [_row("f00", True, ["power", "energy", "efficiency"]), _row("f01", False, ["trim"])]
    assert accepted(rows, accept_other_failures_max(1.0), 3) == ["f00"]
    assert accepted(rows, accept_other_failures_max(0.0), 3) == []
