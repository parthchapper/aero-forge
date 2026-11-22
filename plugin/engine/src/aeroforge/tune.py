"""tune-style core: marks parsing, L2 logistic fit, grouped k-fold accuracy (SPEC §7)."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from aeroforge.constants import number
from aeroforge.style import FEATURES, StyleModel, scores, sigmoid

Mark = Literal["good", "bad", "none"]


class MarkItem(BaseModel):
    """One exported tile mark; score is the style score the tile was rendered with."""

    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)

    id: str
    genome: tuple[float, ...]
    mark: Mark
    score: float | None = None


class Marks(BaseModel):
    """A gallery's marks.json; older exports carry no brief."""

    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)

    seed: int
    brief: str | None = None
    fin_cap: bool = False  # marked on fin-capped decodes; older marks saw uncapped fins
    items: tuple[MarkItem, ...]


class TuneError(ValueError):
    """Marks cannot be fitted (no labels, or one class only)."""


def labelled(marks: Marks) -> list[MarkItem]:
    """Items marked good or bad (unmarked tiles carry no label)."""
    return [it for it in marks.items if it.mark != "none"]


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and std; a constant column gets std 1 so its z is 0."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.where(std > 0.0, std, 1.0)


def _loss(z: np.ndarray, y: np.ndarray, penalty: np.ndarray, theta: np.ndarray) -> float:
    """Penalised negative log-likelihood, overflow-free."""
    t = z @ theta
    return float(np.sum(np.logaddexp(0.0, t) - y * t) + 0.5 * theta @ penalty @ theta)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> StyleModel:
    """Ridge logistic regression; bias unpenalised. x (n, F), y in {0, 1}.

    Newton steps with step halving: separable marks (few designs, many features) make plain
    Newton overshoot and leave a singular Hessian, hence the line search and least squares.
    """
    if len(np.unique(y)) < 2:
        raise TuneError("marks need at least one good and one bad design")
    mean, std = _standardise(x)
    z = np.column_stack([(x - mean) / std, np.ones(len(x))])
    penalty = np.diag([l2] * x.shape[1] + [0.0])
    theta = np.zeros(z.shape[1])
    tol = number("style_newton_tol")
    for _ in range(int(number("style_newton_iter"))):
        p = sigmoid(z @ theta)
        grad = z.T @ (p - y) + penalty @ theta
        hess = (z * (p * (1.0 - p))[:, None]).T @ z + penalty
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        base, t = _loss(z, y, penalty, theta), 1.0
        while _loss(z, y, penalty, theta - t * step) > base and t > tol:
            t /= 2.0
        theta -= t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return StyleModel(mean, std, theta[:-1].copy(), float(theta[-1]))


def deployed(model: StyleModel, kept: int) -> StyleModel:
    """The scoring model: the first `kept` columns; trailing nuisance columns sit at their mean."""
    return StyleModel(model.mean[:kept], model.std[:kept], model.weights[:kept], model.bias)


def kfold_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
    l2: float,
    rng: np.random.Generator,
) -> float:
    """Held-out accuracy of the deployed model (len(FEATURES) columns) at threshold 0.5.

    Rows sharing a group stay in one fold; x may carry trailing nuisance columns.
    """
    kept = len(FEATURES)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq.tolist(), (rng.permutation(len(uniq)) % k).tolist(), strict=True))
    folds = np.array([fold_of[g] for g in groups.tolist()])
    hits = 0
    for f in range(k):
        test = folds == f
        if not test.any():
            continue
        model = fit_logistic(x[~test], y[~test], l2)
        predicted = scores(deployed(model, kept), x[test][:, :kept]) >= 0.5
        hits += int(np.sum(predicted == (y[test] == 1)))
    return hits / len(y)


def model_table(model: StyleModel, source: str) -> dict[str, object]:
    """style.yaml mapping for a fitted model (plain floats, FEATURES order)."""
    rows = [
        {"name": n, "mean": float(m), "std": float(s), "weight": float(w)}
        for n, m, s, w in zip(FEATURES, model.mean, model.std, model.weights, strict=True)
    ]
    return {"source": source, "bias": model.bias, "features": rows}


def gate_good_in_top(marks: Marks, top_n: int) -> int | None:
    """Good marks among the top_n tiles by rendered score; None if tiles carry no score."""
    scored = [it for it in marks.items if it.score is not None]
    if len(scored) < top_n:
        return None
    top = sorted(scored, key=lambda it: (-float(it.score or 0.0), it.id))[:top_n]
    return sum(it.mark == "good" for it in top)


def top_and_random(score_by_row: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices: the n // 2 best scores plus random others, shuffled so tiles hide the split."""
    if n > len(score_by_row):
        raise TuneError(f"pool of {len(score_by_row)} scored designs is smaller than n = {n}")
    order = np.argsort(-score_by_row, kind="stable")
    top, rest = order[: n // 2], np.sort(order[n // 2 :])
    chosen = np.concatenate([top, rng.choice(rest, n - len(top), replace=False)])
    return chosen[rng.permutation(n)]
