"""CLI handlers for search and verify (SPEC §9, §12): worker pool, history file, finalists.json.

Candidates are evaluated in spawned worker processes; `map` keeps the input order, so results
do not depend on scheduling (RULES determinism)."""

import functools
import hashlib
import json
import logging
import multiprocessing
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from aeroforge.areas import areas
from aeroforge.brief import effective_randomness, load_brief
from aeroforge.candidate import (
    Candidate,
    decoded_desc,
    evaluate_candidate,
    pool_desc,
    verify_candidate,
)
from aeroforge.constants import knob, number
from aeroforge.descriptor import DESC_SIZE, DESC_VERSION, median_nearest, novelty
from aeroforge.genome import decode, random_genomes
from aeroforge.models import Brief, Design, Requirements
from aeroforge.physics_cli import (
    BRIEF_FILE,
    CHECK_FILE,
    DESIGN_FILE,
    REQUIREMENTS_FILE,
    SpecError,
    write_design,
)
from aeroforge.reference import reference_genome, reference_names
from aeroforge.search import BatchFn, Run, Scored, SearchResult, over_bomb_cap, qualifies, search

log = logging.getLogger(__name__)
FINALISTS_FILE, VERIFIED_FILE, HISTORY_FILE = "finalists.json", "verified.json", "history.jsonl"
DEFAULT_HISTORY = Path.home() / ".aero-forge" / HISTORY_FILE
_LIMITS: list[threadpool_limits] = []

Summary = dict[str, object]


def _init_worker() -> None:
    """Worker start: one BLAS thread (numpy is loaded first, else the limit does nothing)."""
    import numpy  # noqa: F401  (load the BLAS libraries the limit applies to)

    _LIMITS.append(threadpool_limits(limits=int(number("blas_threads")), user_api="blas"))


@contextmanager
def worker_pool(workers: int) -> Iterator[ProcessPoolExecutor]:
    """Spawned worker processes, each limited to one BLAS thread."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as pool:
        yield pool


def history_rows(path: Path) -> list[dict[str, object]]:
    """The rows of a history.jsonl that carry the current descriptor (U4: rows of the 0.3
    descriptor are skipped, with a warning; the file is never rewritten). Empty without the file."""
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    kept = [r for r in rows if r.get("desc_version") == DESC_VERSION]
    if len(kept) < len(rows):
        log.warning("%s: %d rows of an older descriptor skipped", path, len(rows) - len(kept))
    return kept


def read_history(path: Path) -> np.ndarray:
    """Descriptors of the current version in a history.jsonl (empty when there are none)."""
    rows = history_rows(path)
    return np.array([r["desc"] for r in rows]).reshape(-1, DESC_SIZE)


def append_history(path: Path, entry: dict[str, object]) -> None:
    """Append one delivered design (id, seed, brief hash, desc) to history.jsonl, tagged with the
    descriptor version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({**entry, "desc_version": DESC_VERSION}, sort_keys=True) + "\n")


def brief_hash(b: Brief) -> str:
    """SHA-256 of the brief's canonical JSON."""
    return hashlib.sha256(b.model_dump_json().encode()).hexdigest()


def load_folder(folder: Path) -> tuple[Brief, Requirements]:
    """brief.yaml + requirements.json of a design dir."""
    b = load_brief(folder / BRIEF_FILE)
    try:
        req = Requirements.model_validate_json(
            (folder / REQUIREMENTS_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise SpecError(f"{folder / REQUIREMENTS_FILE}: {exc} (run derive first)") from exc
    return b, req


def _descs(
    pool: ProcessPoolExecutor, genomes: np.ndarray, req: Requirements
) -> list[tuple[float, ...] | None]:
    """Decoded descriptors in input order (None for genomes that do not decode)."""
    return list(pool.map(functools.partial(pool_desc, req=req), genomes, chunksize=8))


def resolve_seed(seed: int | None, b: Brief) -> int:
    """Search seed: the explicit one, else the brief's, else fresh OS entropy (user, novelty plan:
    the same brief gives a different design every run; the seed is recorded for replay)."""
    if seed is not None:
        return seed
    if b.seed is not None:
        return b.seed
    return int(np.random.SeedSequence().entropy)


def make_run(
    pool: ProcessPoolExecutor,
    req: Requirements,
    history: np.ndarray,
    budget: int,
    seed: int,
    randomness: float,
) -> Run:
    """Seeds = reference genomes; known = references + history; d_min from the references."""
    names = reference_names()
    seeds = np.array([reference_genome(n) for n in names])
    refs = _descs(pool, seeds, req)
    if any(r is None for r in refs):
        raise SpecError("a reference configuration does not decode under these requirements")
    ref_arr = np.array(refs)
    rng = np.random.default_rng(seed)
    pool_genomes = random_genomes(rng, int(number("search_centroid_samples")))
    samples = [x for x in _descs(pool, pool_genomes, req) if x is not None]
    return Run(
        seeds=seeds,
        seed_names=names,
        known=np.vstack([ref_arr, history]),
        centroid_samples=np.vstack([np.array(samples), ref_arr]),
        d_min=median_nearest(ref_arr),
        budget=budget,
        seed=seed,
        randomness=randomness,
    )


def batch_fn(pool: ProcessPoolExecutor, b: Brief, req: Requirements) -> BatchFn:
    """Ordered tier-1 evaluation of a batch in the pool."""
    fn = functools.partial(evaluate_candidate, brief=b, req=req, tier=1)
    return lambda genomes: list(pool.map(fn, genomes, chunksize=1))


def finalist_areas(c: Candidate, req: Requirements) -> dict[str, float] | None:
    """Shadow areas of a finalist's decoded design (None when its evaluation failed)."""
    if c.report is None or c.adjust is None:
        return None
    shadow = areas(decode(np.array(c.genome), req, c.adjust))
    return {
        "top_area_m2": shadow.top_m2,
        "frontal_area_m2": shadow.frontal_m2,
        "body_top_share": shadow.body_top_share,
    }


def finalist_entry(k: int, s: Scored, req: Requirements) -> dict[str, object]:
    """finalists.json row: genome, descriptor, novelty, style, lineage and every check row."""
    c = s.candidate
    return {
        "id": f"f{k:02d}",
        "evaluation": s.index,
        "lineage": s.lineage,
        "novelty": s.novelty,
        "style": c.style,
        "passable": c.passable,
        "areas": finalist_areas(c, req),
        "bar_violation": c.bar_violation if c.report is not None else None,
        "violation": c.report.violation if c.report is not None else None,
        "genome": list(c.genome),
        "desc": list(c.desc) if c.desc is not None else None,
        "adjust": c.adjust.model_dump() if c.adjust is not None else None,
        "report": c.report.model_dump() if c.report is not None else None,
        "reason": c.reason,
    }


def funnel(result: SearchResult) -> dict[str, object]:
    """Counts along the selection filter and lineage of the finalists."""
    ev = result.evaluated
    passable = [s for s in ev if s.candidate.passable]
    styled = [s for s in passable if s.candidate.style >= result.s_min]
    return {
        "evaluated": len(ev),
        "failed": sum(s.candidate.report is None for s in ev),
        "passable": len(passable),
        "passable_styled": len(styled),
        "over_bomb_cap": sum(over_bomb_cap(s.candidate) for s in styled),
        "novel": sum(s.novelty >= result.d_min for s in styled),
        "qualifying": sum(
            qualifies(s, result.d_min, result.s_min, result.randomness) for s in styled
        ),
        "finalist_lineage": dict(sorted(Counter(s.lineage for s in result.finalists).items())),
    }


def search_payload(
    result: SearchResult, b: Brief, req: Requirements, seed: int, budget: int
) -> dict[str, object]:
    """finalists.json content (no wall-clock: same inputs give the same bytes)."""
    return {
        "seed": seed,
        "randomness": result.randomness,
        "budget": budget,
        "budget_used": result.budget_used,
        "doubled": result.doubled,
        "fallback": result.fallback,
        "d_min": result.d_min,
        "s_min": result.s_min,
        "brief_hash": brief_hash(b),
        "funnel": funnel(result),
        "finalists": [finalist_entry(k, s, req) for k, s in enumerate(result.finalists)],
    }


def run_search(
    pool: ProcessPoolExecutor,
    folder: Path,
    seed: int,
    budget: int,
    history: Path,
    randomness: float,
) -> tuple[dict[str, object], float]:
    """Search one design dir; writes finalists.json; returns its content and wall time [s]."""
    b, req = load_folder(folder)
    start = time.perf_counter()
    run = make_run(pool, req, read_history(history), budget, seed, randomness)
    result = search(run, batch_fn(pool, b, req))
    payload = search_payload(result, b, req, seed, budget)
    write_json(folder / FINALISTS_FILE, payload)
    return payload, time.perf_counter() - start


def write_json(path: Path, payload: object) -> None:
    """Indented JSON, LF line ends."""
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")


def search_cmd(
    folder: Path,
    seed: int | None,
    budget: int,
    history: Path,
    workers: int,
    randomness: float | None,
) -> Summary:
    """`search`: finalists.json in the design dir; stdout carries the wall time.

    seed and randomness are resolved here, once, from the flags, the brief and the defaults.
    """
    b, _ = load_folder(folder)
    resolved_seed, r = resolve_seed(seed, b), effective_randomness(b, randomness)
    with worker_pool(workers) as pool:
        payload, wall = run_search(pool, folder, resolved_seed, budget, history, r)
    return {
        "cmd": "search",
        **{
            k: payload[k]
            for k in ("seed", "randomness", "budget_used", "doubled", "fallback", "d_min")
        },
        "funnel": payload["funnel"],
        "finalists": [(f["id"], round(f["novelty"], 4)) for f in payload["finalists"]],
        "wall_time_s": round(wall, 1),
        "workers": workers,
        "finalists_file": str(folder / FINALISTS_FILE),
    }


def _verified_row(entry: dict, c: Candidate, known: np.ndarray) -> dict[str, object]:
    """verified.json row of one finalist at tier 2."""
    return {
        "id": entry["id"],
        "passable": c.passable,
        "feasible": c.report is not None and c.report.feasible,
        "novelty": novelty(np.array(c.desc), known) if c.desc is not None else 0.0,
        "style": c.style,
        "failed": [] if c.report is None else [k.id for k in c.report.checks if not k.passed],
        "reason": c.reason,
    }


def accept_other_failures_max(r: float) -> int:
    """Most failed checks outside the bar a finalist may carry at tier 2: 2 at r = 0, every check
    at r = 1 (user, novelty plan step 8; was a fixed 2 sorted by failure count)."""
    return int(knob("knob_accept_other_failures_max", r))


def payload_randomness(payload: dict, source: Path) -> float:
    """Search randomness recorded in finalists.json (searches before step 7 lack it)."""
    if "randomness" not in payload:
        raise SpecError(f"{source}: no randomness recorded (written before step 7; search again)")
    return float(payload["randomness"])


def accepted(rows: list[dict], max_other: int, limit: int) -> list[str]:
    """Passable rows with at most max_other failed checks, in finalist (score) order."""
    ok = [r for r in rows if r["passable"] and len(r["failed"]) <= max_other]
    return [r["id"] for r in ok][:limit]


def run_verify(
    pool: ProcessPoolExecutor, folder: Path, history: Path
) -> tuple[dict[str, object], dict[str, tuple[Candidate, Design | None]]]:
    """Tier 2 on every finalist (ordered); check.json per finalist; accepted ones kept."""
    b, req = load_folder(folder)
    payload_in = json.loads((folder / FINALISTS_FILE).read_text(encoding="utf-8"))
    finalists = payload_in["finalists"]
    r = payload_randomness(payload_in, folder / FINALISTS_FILE)
    genomes = np.array([f["genome"] for f in finalists]).reshape(len(finalists), -1)
    fn = functools.partial(verify_candidate, brief=b, req=req)
    checked = list(pool.map(fn, genomes, chunksize=1))
    names = reference_names()
    refs = np.array([decoded_desc(reference_genome(n), req) for n in names])
    known = np.vstack([refs, read_history(history)])
    rows, out = [], {}
    for entry, (c, design) in zip(finalists, checked, strict=True):
        target = folder / "finalists" / entry["id"] / CHECK_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, c.report.model_dump() if c.report is not None else {"error": c.reason})
        if design is not None:
            write_design(target.parent / DESIGN_FILE, design)
        rows.append(_verified_row(entry, c, known))
        out[entry["id"]] = (c, design)
    passing = accepted(rows, accept_other_failures_max(r), int(number("search_winners")))
    order = passing or [
        e["id"]
        for e, _ in sorted(
            zip(finalists, checked, strict=True),
            key=lambda ec: (ec[1][0].bar_violation, ec[0]["id"]),
        )
    ]
    payload = {"finalists": rows, "passing": passing, "order": order, "fallback": not passing}
    write_json(folder / VERIFIED_FILE, payload)
    return payload, out


def verify_cmd(folder: Path, history: Path, workers: int) -> Summary:
    """`verify`: check.json per finalist, verified.json with the accepted finalists."""
    with worker_pool(workers) as pool:
        payload, _ = run_verify(pool, folder, history)
    return {
        "cmd": "verify",
        "passing": payload["passing"],
        "fallback": payload["fallback"],
        "failed": {r["id"]: r["failed"] for r in payload["finalists"]},
        "verified": str(folder / VERIFIED_FILE),
    }
