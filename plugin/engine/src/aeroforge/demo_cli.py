"""`demo`: derive a brief, then N chained search → verify → build runs, one gallery of winners.

Each run's winner joins a run-local history, so later runs must be d_min away from it; the
user's ~/.aero-forge/history.jsonl is neither read nor written (it records deliveries)."""

import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aeroforge.brief import effective_randomness, load_brief
from aeroforge.candidate import Candidate
from aeroforge.constants import number
from aeroforge.derive import InfeasibleError, derive
from aeroforge.gallery import Tile, gallery_html
from aeroforge.geom import GeometryError, build_parts
from aeroforge.mesh import MeshError, tessellate
from aeroforge.models import Design
from aeroforge.physics_cli import BRIEF_FILE, EXIT_CHECK_FAILED, EXIT_OK, REQUIREMENTS_FILE
from aeroforge.render import render_png
from aeroforge.search_cli import (
    HISTORY_FILE,
    append_history,
    brief_hash,
    run_search,
    run_verify,
    worker_pool,
    write_json,
)

WINNER_FILE, GALLERY_FILE = "winner.json", "gallery.html"
BuildFn = Callable[[Design, Path, str], tuple[str, str]]

Summary = dict[str, object]


def render_views(design: Design, tiles_dir: Path, tile_id: str) -> tuple[str, str]:
    """Build, mesh and render iso + top views; paths relative to the gallery page."""
    meshes = tessellate(build_parts(design))
    px = int(number("gallery_tile_px"))
    paths = []
    for view in ("iso", "top"):
        name = f"{tile_id}_{view}.png"
        (tiles_dir / name).write_bytes(render_png(meshes, view, px))
        paths.append(f"{tiles_dir.name}/{name}")
    return paths[0], paths[1]


def pick_winner(
    order: list[str],
    checked: dict[str, tuple[Candidate, Design | None]],
    build: BuildFn,
    tiles_dir: Path,
    tile_id: str,
) -> tuple[str | None, tuple[str, str] | None, list[str]]:
    """First finalist in order whose design builds; CAD failures move on, never raise."""
    errors = []
    for fid in order:
        _, design = checked[fid]
        if design is None:
            errors.append(f"{fid}: no tier-2 design")
            continue
        try:
            return fid, build(design, tiles_dir, tile_id), errors
        except (GeometryError, MeshError) as exc:
            errors.append(f"{fid}: {type(exc).__name__}: {exc}")
    return None, None, errors


def _tile(
    run_id: str,
    seed: int,
    fid: str | None,
    views: tuple[str, str] | None,
    c: Candidate | None,
    info: dict,
    errors: list[str],
) -> Tile:
    """Gallery tile of one run: images, caption and every tier-2 check row."""
    if fid is None or views is None or c is None or c.report is None:
        return Tile(run_id, (), None, None, "; ".join(errors) or "no finalist", None)
    rows = tuple((k.id, k.passed, k.margin) for k in c.report.checks)
    caption = (
        f"seed {seed} · {fid} · {info['lineage']} · novelty {info['novelty']:.3f} · "
        f"style {c.style:.2f} · {'passable' if c.passable else 'NOT passable'}"
        f"{' · fallback' if info['fallback'] else ''}"
    )
    return Tile(run_id, c.genome, views[0], views[1], None, c.style, caption, rows)


def _winner_payload(c: Candidate, design: Design, info: dict) -> dict[str, object]:
    """winner.json: genome, adjustments, delivered design and its tier-2 report."""
    if c.report is None or c.adjust is None:
        raise ValueError("a winner needs a tier-2 report")
    return {
        **info,
        "genome": list(c.genome),
        "desc": list(c.desc) if c.desc is not None else None,
        "style": c.style,
        "adjust": c.adjust.model_dump(),
        "design": design.model_dump(),
        "report": c.report.model_dump(),
    }


def _one_run(
    pool: ProcessPoolExecutor,
    out: Path,
    i: int,
    seed: int,
    budget: int,
    build: BuildFn,
    randomness: float,
) -> tuple[Tile, dict[str, object]]:
    """Search, verify and build run i in out/run<i>; winner appended to the run-local history."""
    run_dir = out / f"run{i:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (BRIEF_FILE, REQUIREMENTS_FILE):
        shutil.copyfile(out / name, run_dir / name)
    history = out / HISTORY_FILE
    payload, wall = run_search(pool, run_dir, seed, budget, history, randomness)
    verified, checked = run_verify(pool, run_dir, history)
    tiles_dir = out / "tiles"
    fid, views, errors = pick_winner(verified["order"], checked, build, tiles_dir, f"run{i:02d}")
    entry = next((f for f in payload["finalists"] if f["id"] == fid), None)
    info = {
        "run": i,
        "seed": seed,
        "randomness": randomness,
        "finalist": fid,
        "lineage": entry["lineage"] if entry else None,
        "novelty": entry["novelty"] if entry else None,
        "fallback": bool(payload["fallback"] or verified["fallback"]),
    }
    c, design = checked[fid] if fid is not None else (None, None)
    summary = {**info, "wall_time_s": round(wall, 1), "funnel": payload["funnel"], "cad": errors}
    if c is not None and design is not None:
        write_json(run_dir / WINNER_FILE, _winner_payload(c, design, info))
        entry_id, b = f"run{i:02d}", load_brief(out / BRIEF_FILE)
        record = {"id": entry_id, "seed": seed, "brief_hash": brief_hash(b), "desc": c.desc}
        append_history(history, record)
    return _tile(f"run{i:02d}", seed, fid, views, c, info, errors), summary


def demo_cmd(
    brief_path: Path,
    runs: int,
    out: Path,
    budget: int,
    seed: int,
    workers: int,
    randomness: float | None,
) -> tuple[Summary, int]:
    """`demo`: derive, then `runs` chained searches; gallery of the winners with check rows."""
    b = load_brief(brief_path)
    r = effective_randomness(b, randomness)
    try:
        req = derive(b)
    except InfeasibleError as exc:
        return {
            "cmd": "demo",
            "feasible": False,
            "check": exc.check,
            "error": str(exc),
        }, EXIT_CHECK_FAILED
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(brief_path, out / BRIEF_FILE)
    text = req.model_dump_json(indent=1)
    (out / REQUIREMENTS_FILE).write_text(text, encoding="utf-8", newline="\n")
    (out / HISTORY_FILE).unlink(missing_ok=True)
    tiles, runs_out = [], []
    start = time.perf_counter()
    with worker_pool(workers) as pool:
        for i in range(runs):
            tile, summary = _one_run(pool, out, i, seed + i, budget, render_views, r)
            tiles.append(tile)
            runs_out.append(summary)
    page = out / GALLERY_FILE
    page.write_text(gallery_html(tiles, seed, brief_path.stem), encoding="utf-8")
    return {
        "cmd": "demo",
        "runs": [
            {k: r[k] for k in ("run", "finalist", "novelty", "fallback", "wall_time_s")}
            for r in runs_out
        ],
        "wall_time_s": round(time.perf_counter() - start, 1),
        "gallery": str(page),
    }, EXIT_OK
