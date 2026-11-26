import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from aeroforge.brief import brief_requirements
from aeroforge.cli import main
from aeroforge.finish import fin_capped
from aeroforge.genome import decode, random_genomes
from aeroforge.models import Design
from aeroforge.style import FEATURES, StyleModel, features, model_from_table, scores
from aeroforge.tune import (
    Marks,
    TuneError,
    fit_logistic,
    gate_good_in_top,
    kfold_accuracy,
    top_and_random,
)

SYNTH_SEED = 4242
CLI_GENOMES = 400


def _true_model(x: np.ndarray, rng: np.random.Generator) -> StyleModel:
    """Known weights on standardised features (a constant column gets weight 0)."""
    std = x.std(axis=0)
    weights = np.where(std > 0.0, rng.uniform(-1.5, 1.5, x.shape[1]), 0.0)
    return StyleModel(x.mean(axis=0), np.where(std > 0.0, std, 1.0), weights, 0.3)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fit_recovers_known_weights_from_synthetic_marks(full_designs: list[Design]) -> None:
    rng = np.random.default_rng(SYNTH_SEED)
    x = np.array([features(d) for d in full_designs])
    truth = _true_model(x, rng)
    draws = 8  # repeated marks of the same designs shrink label noise
    xs = np.repeat(x, draws, axis=0)
    y = (rng.random(len(xs)) < scores(truth, xs)).astype(float)
    fitted = fit_logistic(xs, y, l2=1e-3)
    recovered = fitted.weights * truth.std / fitted.std  # back onto the truth's scale
    assert _cosine(recovered, truth.weights) > 0.97
    assert np.max(np.abs(recovered - truth.weights)) < 0.3


def test_fit_needs_both_classes() -> None:
    with pytest.raises(TuneError):
        fit_logistic(np.ones((4, 2)), np.ones(4), l2=1.0)


def test_kfold_keeps_a_genomes_duplicate_rows_in_one_fold() -> None:
    # 12 genomes, 2 identical rows each, more columns than rows: a twin in training is memorised
    x = np.repeat(np.random.default_rng(1).normal(size=(12, len(FEATURES) + 2)), 2, axis=0)
    y = np.repeat(np.arange(12) % 2, 2).astype(float)
    split = kfold_accuracy(x, y, np.arange(24), 5, 1e-3, np.random.default_rng(2))
    grouped = kfold_accuracy(x, y, np.arange(24) // 2, 5, 1e-3, np.random.default_rng(2))
    assert split > 0.8 and grouped < 0.65


def test_top_and_random_keeps_the_best_half_and_no_repeats() -> None:
    s = np.arange(100, dtype=float)
    rows = top_and_random(s, 20, np.random.default_rng(3))
    assert len(set(rows.tolist())) == 20 and set(range(90, 100)) <= set(rows.tolist())


def test_top_and_random_rejects_a_small_pool() -> None:
    with pytest.raises(TuneError):
        top_and_random(np.zeros(5), 6, np.random.default_rng(0))


def test_gate_counts_good_marks_among_top_scores() -> None:
    items = [
        {"id": f"g{i:03d}", "genome": [0.5], "mark": "good" if i % 2 else "bad", "score": i / 10}
        for i in range(6)
    ]
    marks = Marks.model_validate({"seed": 1, "items": items})
    assert gate_good_in_top(marks, 3) == 2  # scores 0.5 good, 0.4 bad, 0.3 good


def _synthetic_marks(path: Path, designs: list[Design], genomes: np.ndarray) -> StyleModel:
    """marks.json labelled by a known model; returns that model."""
    rng = np.random.default_rng(SYNTH_SEED)
    x = np.array([features(d) for d in designs])
    truth = _true_model(x, rng)
    good = rng.random(len(x)) < scores(truth, x)
    items = [
        {"id": f"g{i:04d}", "genome": g.tolist(), "mark": "good" if ok else "bad"}
        for i, (g, ok) in enumerate(zip(genomes, good, strict=True))
    ]
    path.write_text(
        json.dumps({"seed": 0, "brief": "stub", "fin_cap": True, "items": items}), encoding="utf-8"
    )
    return truth


@pytest.mark.slow
def test_cli_tune_style_recovers_weights_and_is_deterministic(
    tmp_path: Path, designs: list[Design], genomes: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    n = CLI_GENOMES  # the fast fit test covers recovery at 1,000; this one covers the CLI path
    truth = _synthetic_marks(tmp_path / "marks.json", designs[:n], genomes[:n])
    outs = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
    assert main(["tune-style", str(tmp_path / "marks.json"), "--out", str(outs[0])]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert main(["tune-style", str(tmp_path / "marks.json"), "--out", str(outs[1])]) == 0
    assert summary["kfold_accuracy"] > summary["majority_baseline"]
    assert outs[0].read_bytes() == outs[1].read_bytes()
    fitted = model_from_table(yaml.safe_load(outs[0].read_text(encoding="utf-8")))
    assert _cosine(fitted.weights * truth.std / fitted.std, truth.weights) > 0.9


def test_cli_tune_style_needs_a_brief(tmp_path: Path) -> None:
    g = random_genomes(np.random.default_rng(0), 1)[0].tolist()
    items = [{"id": "g000", "genome": g, "mark": "good"}]
    (tmp_path / "m.json").write_text(json.dumps({"seed": 0, "items": items}), encoding="utf-8")
    assert main(["tune-style", str(tmp_path / "m.json"), "--out", str(tmp_path / "s.yaml")]) == 2


def test_cli_tune_style_uses_a_files_own_brief_over_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gs = random_genomes(
        np.random.default_rng(0), 10
    ).tolist()  # both classes in every training fold
    items = [
        {"id": f"g{i:03d}", "genome": g, "mark": m}
        for i, (g, m) in enumerate(zip(gs, ["good", "bad"] * 5, strict=True))
    ]
    (tmp_path / "own.json").write_text(
        json.dumps({"seed": 0, "brief": "stub", "fin_cap": True, "items": items}), encoding="utf-8"
    )
    (tmp_path / "old.json").write_text(
        json.dumps({"seed": 0, "fin_cap": True, "items": items}), encoding="utf-8"
    )
    files = [str(tmp_path / "own.json"), str(tmp_path / "old.json")]
    args = ["tune-style", *files, "--brief", "stub", "--brief", "small_boxes"]
    assert main([*args, "--out", str(tmp_path / "s.yaml")]) == 0
    assert (
        json.loads(capsys.readouterr().out)["rows"] == 10 + 2 * 10
    )  # own brief once, old under both


def test_cli_tune_style_drops_pre_cap_marks_on_capped_fins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    req = brief_requirements("stub")
    pool = random_genomes(np.random.default_rng(5), 60)
    capped = [g for g in pool if fin_capped(decode(g, req))][:2]
    plain = [g for g in pool if not fin_capped(decode(g, req))][:10]
    marks = ["good", "bad"] * 6
    items = [
        {"id": f"g{i:03d}", "genome": g.tolist(), "mark": m}
        for i, (g, m) in enumerate(zip(plain + capped, marks, strict=True))
    ]
    for name, fin_cap in (("old", False), ("new", True)):
        doc = {"seed": 0, "brief": "stub", "fin_cap": fin_cap, "items": items}
        (tmp_path / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    out = str(tmp_path / "s.yaml")
    assert main(["tune-style", str(tmp_path / "old.json"), "--out", out]) == 0
    old = json.loads(capsys.readouterr().out)
    assert main(["tune-style", str(tmp_path / "new.json"), "--out", out]) == 0
    new = json.loads(capsys.readouterr().out)
    assert (old["rows"], old["stale_marks_dropped"], new["rows"]) == (10, 2, 12)
