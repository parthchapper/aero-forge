"""Quality-diversity search (SPEC §9): CVT archive over the descriptor, emitters seeded with the
reference genomes and random genomes; candidates are evaluated by an injected batch function."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from ribs.archives import CVTArchive
from ribs.emitters import EmitterBase, GaussianEmitter, IsoLineEmitter
from ribs.schedulers import Scheduler

from aeroforge.candidate import Candidate
from aeroforge.constants import knob, number
from aeroforge.descriptor import DESC_SIZE, body_top_share, novelty
from aeroforge.genetable import genome_length
from aeroforge.genome import random_genomes

BatchFn = Callable[[np.ndarray], Sequence[Candidate]]
RANDOM_SEED, REFINE, EXPLORE, CROSS = "random", "refine", "explore", "isoline"
KMEANS_SEED_LIMIT = 2**32  # exclusive upper bound of scikit-learn random_state


class SearchError(ValueError):
    """Unusable search inputs (empty seeds, non-positive budget)."""


@dataclass(frozen=True)
class Scored:
    """An evaluated candidate with its evaluation index, novelty and lineage."""

    index: int
    candidate: Candidate
    novelty: float
    lineage: str


@dataclass(frozen=True)
class Run:
    """Inputs of one search: seed genomes (named), known descriptors (references and history),
    descriptor samples for the CVT centroids, the novelty bar, budget and RNG seed."""

    seeds: np.ndarray
    seed_names: tuple[str, ...]
    known: np.ndarray
    centroid_samples: np.ndarray
    d_min: float
    budget: int
    seed: int
    randomness: float


@dataclass(frozen=True)
class SearchResult:
    """Every evaluation in order, the finalists and how the budget went."""

    evaluated: tuple[Scored, ...]
    finalists: tuple[Scored, ...]
    budget_used: int
    doubled: bool
    fallback: bool
    d_min: float
    s_min: float
    randomness: float


def objective(c: Candidate) -> float:
    """Style score when passable, else minus the bar violation; a failure gets a finite floor."""
    if c.report is None or not np.isfinite(c.bar_violation):
        return number("search_fail_objective")
    return c.style if c.passable else -c.bar_violation


def _measures(c: Candidate) -> np.ndarray:
    """Descriptor of a candidate; zeros when it failed to decode (its objective is the floor)."""
    return np.array(c.desc) if c.desc is not None else np.zeros(DESC_SIZE)


def first_batch(run: Run, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Genomes and lineage labels of the first ask (user, novelty plan step 7).

    The reference share of the batch (knob at r, rounded half up) is filled with reference
    genomes in a seeded order; slots beyond the reference count are refine-sigma mutated copies
    that cycle through them. Random genomes take the other slots. Every draw has a fixed size,
    so the random stream does not depend on r.
    """
    n_refs = len(run.seeds)
    size = n_refs + 2 * int(number("search_batch_size"))
    share = knob("knob_first_batch_reference_share", run.randomness)
    slots = int(np.floor(share * size + 0.5))
    randoms = random_genomes(rng, size)
    order = rng.permutation(n_refs)
    noise = rng.normal(0.0, number("search_refine_sigma"), size=(size, genome_length()))
    genomes, labels = [], []
    for k in range(size):
        if k >= slots:
            genomes.append(randoms[k])
            labels.append(RANDOM_SEED)
            continue
        j = int(order[k % n_refs])
        mutated = k >= n_refs
        genomes.append(np.clip(run.seeds[j] + noise[k], 0.0, 1.0) if mutated else run.seeds[j])
        labels.append(f"{'reference-mutated' if mutated else 'reference'}:{run.seed_names[j]}")
    return np.array(genomes), labels


def _emitters(
    archive: CVTArchive, run: Run, rng: np.random.Generator, first: np.ndarray
) -> list[EmitterBase]:
    """Refine, explore (sigma from r) and iso+line emitters; the first batch is split among them."""
    n, batch = genome_length(), int(number("search_batch_size"))
    bounds = [(0.0, 1.0)] * n
    cut = len(run.seeds)
    ids = rng.integers(0, 2**31 - 1, size=3)
    return [
        GaussianEmitter(
            archive,
            sigma=number("search_refine_sigma"),
            initial_solutions=first[:cut],
            bounds=bounds,
            batch_size=batch,
            seed=int(ids[0]),
        ),
        GaussianEmitter(
            archive,
            sigma=knob("knob_explore_sigma", run.randomness),
            initial_solutions=first[cut : cut + batch],
            bounds=bounds,
            batch_size=batch,
            seed=int(ids[1]),
        ),
        IsoLineEmitter(
            archive,
            iso_sigma=number("search_iso_sigma"),
            line_sigma=number("search_line_sigma"),
            initial_solutions=first[cut + batch :],
            bounds=bounds,
            batch_size=batch,
            seed=int(ids[2]),
        ),
    ]


def _spend(
    scheduler: Scheduler,
    run: Run,
    evaluate: BatchFn,
    out: list[Scored],
    until: int,
    first_lineages: list[str],
) -> None:
    """Ask, evaluate and tell whole batches until at least `until` evaluations are done."""
    batch = int(number("search_batch_size"))
    later = [*[REFINE] * batch, *[EXPLORE] * batch, *[CROSS] * batch]
    while len(out) < until:
        lineages = later if out else first_lineages
        solutions = scheduler.ask()
        found = list(evaluate(solutions))
        if len(found) != len(solutions) or len(lineages) != len(solutions):
            raise SearchError(f"batch of {len(solutions)} came back with {len(found)} results")
        for c, lineage in zip(found, lineages, strict=True):
            nov = novelty(np.array(c.desc), run.known) if c.desc is not None else 0.0
            out.append(Scored(len(out), c, nov, lineage))
        scheduler.tell(
            np.array([objective(c) for c in found]), np.array([_measures(c) for c in found])
        )


def novelty_band(r: float) -> tuple[float, float]:
    """Novelty band [bottom, top] in units of d_min at randomness r (user, novelty plan)."""
    return knob("knob_novelty_band_bottom", r), knob("knob_novelty_band_top", r)


def band_gap(novelty_value: float, d_min: float, r: float) -> float:
    """Distance of a novelty from the band [lo, hi] x d_min, 0 inside (user Deviation §9)."""
    lo, hi = novelty_band(r)
    return max(lo * d_min - novelty_value, novelty_value - hi * d_min, 0.0)


def total_violation(c: Candidate) -> float:
    """Sum of negative margins over every check (inf when evaluation failed)."""
    return c.report.violation if c.report is not None else float("inf")


def over_bomb_cap(c: Candidate) -> bool:
    """The body takes more of the top-down shadow than bomb_cap allows (U4); a candidate without a
    descriptor counts as over, so it never qualifies."""
    return c.desc is None or body_top_share(np.array(c.desc)) > number("bomb_cap")


def qualifies(s: Scored, d_min: float, s_min: float, r: float) -> bool:
    """SPEC §9 filter with the passable bar (§14.3), the novelty band (user Deviation §9) and the
    bomb cap (U4)."""
    c = s.candidate
    return (
        c.passable
        and c.style >= s_min
        and band_gap(s.novelty, d_min, r) == 0.0
        and not over_bomb_cap(c)
    )


def _unique(items: Sequence[Scored]) -> list[Scored]:
    """First occurrence of each genome, order kept."""
    seen: set[tuple[float, ...]] = set()
    out = []
    for s in items:
        if s.candidate.genome not in seen:
            seen.add(s.candidate.genome)
            out.append(s)
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    """Rank-normalised values in [0, 1]: ties share their mean rank; one value gives 0.5."""
    n = len(values)
    if n == 1:
        return np.full(1, 0.5)
    ranks = np.empty(n)
    ranks[np.argsort(values, kind="stable")] = np.arange(n)
    for v in np.unique(values):
        tied = values == v
        ranks[tied] = ranks[tied].mean()
    return ranks / (n - 1)


def _spread(ranked: Sequence[Scored], d_min: float) -> list[Scored]:
    """Greedy finalists in rank order, each at least d_min (descriptor distance) from every
    finalist already chosen, so the critic sees designs that differ from each other."""
    chosen: list[Scored] = []
    for s in ranked:
        x = np.array(s.candidate.desc)
        if all(np.linalg.norm(x - np.array(c.candidate.desc)) >= d_min for c in chosen):
            chosen.append(s)
        if len(chosen) == int(number("search_finalists")):
            break
    return chosen


def select(evaluated: Sequence[Scored], d_min: float, s_min: float, r: float) -> list[Scored]:
    """Finalists (user, novelty plan step 8): qualifiers ranked by
    w(r) z(novelty) + (1 - w(r)) z(style), z = rank-normalised among the qualifiers, then spread
    at least d_min apart. Drag, power and energy margins are not in the ranking."""
    good = _unique([s for s in evaluated if qualifies(s, d_min, s_min, r)])
    if not good:
        return []
    w = knob("knob_rank_novelty_weight", r)
    score = w * _rank01(np.array([s.novelty for s in good]))
    score += (1.0 - w) * _rank01(np.array([s.candidate.style for s in good]))
    order = sorted(range(len(good)), key=lambda i: (-score[i], good[i].index))
    return _spread([good[i] for i in order], d_min)


def fallback(evaluated: Sequence[Scored], d_min: float, s_min: float, r: float) -> list[Scored]:
    """Nothing qualified: closest candidates (under the bomb cap, passable, styled, near the band,
    least violation)."""
    checked = [s for s in evaluated if s.candidate.report is not None]

    def key(s: Scored) -> tuple[float, ...]:
        c = s.candidate
        return (
            float(over_bomb_cap(c)),
            -float(c.passable),
            -float(c.style >= s_min),
            band_gap(s.novelty, d_min, r),
            c.bar_violation,
            total_violation(c),
            s.index,
        )

    return sorted(_unique(checked), key=key)[: int(number("search_finalists"))]


def search(run: Run, evaluate: BatchFn) -> SearchResult:
    """Run the budget; nothing qualifies → one more budget; still nothing → flagged fallback."""
    if run.budget < 1 or len(run.seeds) == 0:
        raise SearchError("search needs a positive budget and at least one seed genome")
    rng = np.random.default_rng(run.seed)
    archive = CVTArchive(
        solution_dim=genome_length(),
        centroids=int(number("search_cells")),
        ranges=[(0.0, 1.0)] * DESC_SIZE,
        samples=run.centroid_samples,
        # scikit-learn's k-means takes seeds in [0, 2^32): a fresh 128-bit seed is reduced here
        seed=int(rng.integers(0, KMEANS_SEED_LIMIT)),
    )
    first, first_lineages = first_batch(run, rng)
    scheduler = Scheduler(archive, _emitters(archive, run, rng, first))
    evaluated: list[Scored] = []
    s_min, r = number("s_min"), run.randomness
    _spend(scheduler, run, evaluate, evaluated, run.budget, first_lineages)
    finalists, doubled = select(evaluated, run.d_min, s_min, r), False
    if not finalists:
        doubled = True
        _spend(scheduler, run, evaluate, evaluated, 2 * run.budget, first_lineages)
        finalists = select(evaluated, run.d_min, s_min, r)
    picked = finalists or fallback(evaluated, run.d_min, s_min, r)
    return SearchResult(
        tuple(evaluated), tuple(picked), len(evaluated), doubled, not finalists, run.d_min, s_min, r
    )
