"""Step 7 (novelty plan): randomness scale r, the six knobs, first-batch composition, the seed."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import SAMPLE_BRIEF
from pydantic import ValidationError

from aeroforge.brief import BriefError, effective_randomness, load_brief
from aeroforge.cli import main
from aeroforge.constants import knob, number
from aeroforge.models import Brief, Requirements
from aeroforge.physics_cli import BRIEF_FILE, REQUIREMENTS_FILE
from aeroforge.reference import reference_genome, reference_names
from aeroforge.search import RANDOM_SEED, Run, first_batch
from aeroforge.search_cli import FINALISTS_FILE, resolve_seed

# the plan's table, written out independently of constants.yaml: (r = 0, r = 0.5, r = 1)
PLAN_TABLE = {
    "knob_novelty_band_top": (2.0, 6.0, 10.0),
    "knob_novelty_band_bottom": (1.0, 1.5, 2.0),
    "knob_explore_sigma": (0.1, 0.25, 0.4),
    "knob_first_batch_reference_share": (1.0, 0.5, 0.0),
    "knob_rank_novelty_weight": (0.2, 0.6, 1.0),
    "knob_critic_strangeness_weight": (1.0, 2.0, 3.0),
}
R_POINTS = (0.0, 0.5, 1.0)
BATCH = int(number("search_batch_size"))
SMALL_BUDGET = 100
SEED_WORKERS = 8
LONG_SEED = 2**100 + 12345  # beyond 64 bits: any int64 cast would break


def _brief(**fields: object) -> Brief:
    return Brief.model_validate({**load_brief(SAMPLE_BRIEF).model_dump(), **fields})


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, float("nan")])
def test_randomness_outside_zero_one_fails_validation(bad: float) -> None:
    with pytest.raises(ValidationError):
        _brief(randomness=bad)


@pytest.mark.parametrize("ok", R_POINTS)
def test_randomness_inside_zero_one_is_accepted(ok: float) -> None:
    assert _brief(randomness=ok).randomness == ok


def test_unset_randomness_takes_the_constants_default() -> None:
    assert _brief().randomness is None
    assert effective_randomness(_brief()) == number("randomness_default") == 0.5


def test_a_flag_override_beats_the_brief_and_is_validated() -> None:
    assert effective_randomness(_brief(randomness=0.2), 0.9) == 0.9
    assert effective_randomness(_brief(randomness=0.2)) == 0.2
    for bad in (-0.1, 1.5):
        with pytest.raises(BriefError):
            effective_randomness(_brief(), bad)


@pytest.mark.parametrize("name", sorted(PLAN_TABLE))
def test_knob_values_at_r_0_half_and_1_match_the_plan_table(name: str) -> None:
    got = tuple(knob(name, r) for r in R_POINTS)
    assert got == pytest.approx(PLAN_TABLE[name])


def test_knob_refuses_r_outside_zero_one() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="randomness"):
            knob("knob_explore_sigma", bad)


def test_an_explicit_seed_beats_the_brief_seed_which_beats_a_fresh_one() -> None:
    assert resolve_seed(7, _brief(seed=3)) == 7
    assert resolve_seed(None, _brief(seed=3)) == 3
    assert resolve_seed(0, _brief()) == 0


def test_two_runs_without_a_seed_draw_different_seeds() -> None:
    draws = {resolve_seed(None, _brief()) for _ in range(5)}
    assert len(draws) == 5 and all(isinstance(s, int) and s >= 0 for s in draws)


def _run(r: float, seed: int = 3) -> Run:
    names = reference_names()
    seeds = np.array([reference_genome(n) for n in names])
    return Run(seeds, names, np.zeros((1, 1)), np.zeros((1, 1)), 0.1, 1, seed, r)


def _batch(r: float, seed: int = 3) -> tuple[np.ndarray, list[str]]:
    return first_batch(_run(r, seed), np.random.default_rng(seed))


def test_first_batch_is_all_reference_derived_at_r0_and_all_random_at_r1() -> None:
    size = len(reference_names()) + 2 * BATCH
    genomes, labels = _batch(0.0)
    assert genomes.shape[0] == size and RANDOM_SEED not in labels
    genomes, labels = _batch(1.0)
    assert genomes.shape[0] == size and set(labels) == {RANDOM_SEED}


def test_first_batch_reference_share_follows_the_knob_with_half_up_rounding() -> None:
    size = len(reference_names()) + 2 * BATCH
    for r in (0.25, 0.5, 0.75):
        _, labels = _batch(r)
        slots = int(np.floor(knob("knob_first_batch_reference_share", r) * size + 0.5))
        assert sum(label != RANDOM_SEED for label in labels) == slots


def test_first_batch_uses_each_reference_once_before_any_mutated_copy() -> None:
    refs = np.array([reference_genome(n) for n in reference_names()])
    genomes, labels = _batch(0.0)
    plain = [k for k, label in enumerate(labels) if label.startswith("reference:")]
    assert len(plain) == len(refs)
    assert {labels[k].split(":", 1)[1] for k in plain} == set(reference_names())
    for k in plain:
        name = labels[k].split(":", 1)[1]
        assert np.array_equal(genomes[k], refs[reference_names().index(name)])
    assert all(
        label.startswith("reference-mutated:")
        for label in labels
        if label not in {labels[k] for k in plain}
    )
    assert genomes.min() >= 0.0 and genomes.max() <= 1.0


def test_first_batch_is_deterministic_and_seed_dependent() -> None:
    a, b, c = _batch(0.5, 3), _batch(0.5, 3), _batch(0.5, 4)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert not np.array_equal(a[0], c[0])


@pytest.fixture()
def design_dir(tmp_path: Path, sample_req: Requirements) -> Path:
    shutil.copyfile(SAMPLE_BRIEF, tmp_path / BRIEF_FILE)
    (tmp_path / REQUIREMENTS_FILE).write_text(sample_req.model_dump_json(), encoding="utf-8")
    return tmp_path


def _search(folder: Path, history: Path, *extra: str) -> dict:
    args = ["search", str(folder), "--budget", str(SMALL_BUDGET), "--history", str(history)]
    assert main([*args, "--workers", str(SEED_WORKERS), *extra]) == 0
    return json.loads((folder / FINALISTS_FILE).read_text(encoding="utf-8"))


@pytest.mark.slow
def test_search_without_a_seed_records_it_and_replays_byte_for_byte(
    design_dir: Path, tmp_path: Path
) -> None:
    history = tmp_path / "history.jsonl"
    first = _search(design_dir, history)
    first_bytes = (design_dir / FINALISTS_FILE).read_bytes()
    assert first["randomness"] == 0.5 and isinstance(first["seed"], int)
    replay = _search(design_dir, history, "--seed", str(first["seed"]), "--randomness", "0.5")
    assert replay["seed"] == first["seed"]
    assert (design_dir / FINALISTS_FILE).read_bytes() == first_bytes


@pytest.mark.slow
def test_a_seed_beyond_64_bits_survives_the_run_and_the_json(
    design_dir: Path, tmp_path: Path
) -> None:
    out = _search(design_dir, tmp_path / "history.jsonl", "--seed", str(LONG_SEED))
    assert out["seed"] == LONG_SEED


def test_the_randomness_flag_out_of_range_exits_2(design_dir: Path, tmp_path: Path) -> None:
    code = main(
        ["search", str(design_dir), "--randomness", "1.5", "--history", str(tmp_path / "h")]
    )
    assert code == 2
