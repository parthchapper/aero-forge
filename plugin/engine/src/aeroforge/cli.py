"""Command line: `python -m aeroforge <cmd>`; stdout is a JSON summary, artifacts go to disk."""

import argparse
import json
import logging
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError
from threadpoolctl import threadpool_limits

from aeroforge.brief import BRIEFS, brief_requirements
from aeroforge.constants import number
from aeroforge.deliver_cli import build_cmd, render_cmd, report_cmd
from aeroforge.demo_cli import demo_cmd
from aeroforge.finish import fin_capped
from aeroforge.gallery import Tile, gallery_html
from aeroforge.genetable import pin_choice
from aeroforge.genome import DecodeError, decode, random_genomes
from aeroforge.geom import GeometryError, build_parts, exact_volumes_m3
from aeroforge.mesh import MeshError, tessellate
from aeroforge.models import Requirements
from aeroforge.physics_cli import INPUT_ERRORS, check_cmd, derive_cmd, validate_cmd
from aeroforge.render import render_png
from aeroforge.search_cli import DEFAULT_HISTORY, search_cmd, verify_cmd
from aeroforge.style import (
    FEATURES,
    NUISANCE,
    StyleModel,
    features,
    fin_terms,
    packaged_model,
    scores,
)
from aeroforge.tune import (
    Marks,
    TuneError,
    deployed,
    fit_logistic,
    gate_good_in_top,
    kfold_accuracy,
    labelled,
    model_table,
    top_and_random,
)

EXIT_OK, EXIT_BAD_INPUT = 0, 2
STYLE_HEADER = (
    "# Style model: score = sigmoid(sum(weight * (f - mean) / std) + bias)."
    " Written by `aeroforge tune-style`.\n"
)

log = logging.getLogger("aeroforge")


class InputError(ValueError):
    """A command-line input file is unusable; reported as JSON with exit code 2."""


def _parser() -> argparse.ArgumentParser:
    """Argument parser with one sub-command per action."""
    parser = argparse.ArgumentParser(prog="aeroforge")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gallery = sub.add_parser("gallery", help="random genomes -> gallery.html")
    gallery.add_argument("--n", type=int, required=True, help="number of tiles")
    gallery.add_argument("--seed", type=int, required=True, help="genome RNG seed")
    gallery.add_argument("--out", type=Path, default=Path("gallery"), help="output directory")
    gallery.add_argument(
        "--brief", choices=BRIEFS, default="stub", help="stand-in brief until derive (step 3)"
    )
    gallery.add_argument(
        "--body-style",
        choices=("smooth", "faceted"),
        default=None,
        help="pin the body_style gene of every genome (default: drawn at random)",
    )
    gallery.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="GENE=LABEL",
        help="pin a choice gene of every genome (repeatable); airfoil_facets=faceted gives the "
        "tiles 3, 4, ... 8 facets in turn",
    )
    gallery.add_argument(
        "--top-by-style", action="store_true", help="half top style scores of a pool, half random"
    )
    tune = sub.add_parser("tune-style", help="fit style weights to marks.json files")
    tune.add_argument("marks", type=Path, nargs="+", help="exported marks.json files")
    tune.add_argument(
        "--brief",
        choices=BRIEFS,
        action="append",
        help="brief(s) for files that carry none (repeatable); a file's own brief wins",
    )
    tune.add_argument("--out", type=Path, default=None, help="style.yaml path (default: packaged)")
    validate = sub.add_parser("validate", help="brief.yaml -> exit code")
    validate.add_argument("brief", type=Path)
    derive = sub.add_parser("derive", help="brief.yaml -> requirements.json")
    derive.add_argument("brief", type=Path)
    derive.add_argument(
        "--out", type=Path, default=None, help="default: requirements.json beside it"
    )
    check = sub.add_parser(
        "check", help="design.yaml (+ brief.yaml, requirements.json) -> check.json"
    )
    check.add_argument("design", type=Path)
    check.add_argument("--tier", type=int, choices=(1, 2), default=2, help="aero fidelity tier")
    _search_parsers(sub)
    _deliver_parsers(sub)
    return parser


def _deliver_parsers(sub: argparse._SubParsersAction) -> None:
    """build, render and report sub-commands (SPEC §11, §12)."""
    build = sub.add_parser("build", help="design dir -> parts/*.step, assembly.step, assembly.glb")
    build.add_argument("folder", type=Path)
    build.add_argument("--full", action="store_true", help="add internals (SPEC §10)")
    preview = "preview one verified finalist in finalists/<id>/ (no adopt)"
    build.add_argument("--finalist", default=None, help=preview)
    render = sub.add_parser("render", help="design dir -> views/{iso,top,front,side}.png")
    render.add_argument("folder", type=Path)
    render.add_argument("--finalist", default=None, help="render finalists/<id>/ instead")
    report = sub.add_parser("report", help="design dir -> report.md, viewer.html")
    report.add_argument("folder", type=Path)
    report.add_argument("--history", type=Path, default=DEFAULT_HISTORY)


def _search_parsers(sub: argparse._SubParsersAction) -> None:
    """search, verify and demo sub-commands (SPEC §12)."""
    workers = int(number("search_workers"))
    search = sub.add_parser("search", help="design dir -> finalists.json")
    search.add_argument("folder", type=Path, help="dir with brief.yaml + requirements.json")
    search.add_argument("--seed", type=int, default=None, help="default: brief seed, else fresh")
    search.add_argument(
        "--randomness", type=float, default=None, help="0 safe .. 1 wild; overrides the brief"
    )
    search.add_argument("--budget", type=int, default=int(number("search_budget")))
    search.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    search.add_argument("--workers", type=int, default=workers)
    verify = sub.add_parser("verify", help="design dir -> check.json per finalist")
    verify.add_argument("folder", type=Path)
    verify.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    verify.add_argument("--workers", type=int, default=workers)
    demo = sub.add_parser("demo", help="brief.yaml -> gallery of N search winners")
    demo.add_argument("brief", type=Path)
    demo.add_argument("--runs", type=int, required=True)
    demo.add_argument("--out", type=Path, default=None, help="default: demo/ beside the brief")
    demo.add_argument("--budget", type=int, default=int(number("search_budget")))
    demo.add_argument("--seed", type=int, default=0, help="run i uses seed + i")
    demo.add_argument(
        "--randomness", type=float, default=None, help="0 safe .. 1 wild; overrides the brief"
    )
    demo.add_argument("--workers", type=int, default=workers)


def _tile(
    index: int, genome: np.ndarray, req: Requirements, tiles_dir: Path, score: float | None
) -> Tile:
    """Decode, build, render one genome; a failure becomes a tile with the reason."""
    tile_id = f"g{index:03d}"
    try:
        design = decode(genome, req)
        meshes = tessellate(build_parts(design), exact_volumes_m3(design))
    except (DecodeError, GeometryError, MeshError) as exc:
        log.warning("%s failed: %s", tile_id, exc)
        err = f"{type(exc).__name__}: {exc}"
        return Tile(tile_id, tuple(map(float, genome)), None, None, err, score)
    px = int(number("gallery_tile_px"))
    paths = []
    for view in ("iso", "top"):
        name = f"{tile_id}_{view}.png"
        (tiles_dir / name).write_bytes(render_png(meshes, view, px))
        paths.append(f"{tiles_dir.name}/{name}")
    log.info("%s rendered (%d parts)", tile_id, len(meshes))
    return Tile(tile_id, tuple(map(float, genome)), paths[0], paths[1], None, score)


def _style_scores(genomes: np.ndarray, req: Requirements, model: StyleModel) -> np.ndarray:
    """Style score per genome; a genome that fails to decode scores -inf."""
    out = np.full(len(genomes), -np.inf)
    for i, g in enumerate(genomes):
        try:
            out[i] = scores(model, features(decode(g, req))[None, :])[0]
        except DecodeError as exc:
            log.warning("pool genome %d failed to decode: %s", i, exc)
    return out


def _pinned(genomes: np.ndarray, pins: dict[str, str]) -> np.ndarray:
    """genomes with each choice gene in pins set to its label; airfoil_facets=faceted cycles the
    counts 3..8 over the rows."""
    out = genomes
    for gene, label in pins.items():
        if (gene, label) == ("airfoil_facets", "faceted"):
            out = np.array(
                [pin_choice(r[None, :], gene, str(3 + i % 6))[0] for i, r in enumerate(out)]
            )
        else:
            out = pin_choice(out, gene, label)
    return out


def _picks(
    n: int, seed: int, req: Requirements, top_by_style: bool, pins: dict[str, str] | None = None
) -> tuple[np.ndarray, list[float | None]]:
    """Genomes to render and their style scores (None for a plain random gallery); pins sets
    those choice genes of every drawn genome."""
    rng = np.random.default_rng(seed)

    def draw(count: int) -> np.ndarray:
        return _pinned(random_genomes(rng, count), pins or {})

    if not top_by_style:
        return draw(n), [None] * n
    pool = draw(int(number("style_pool_size")))
    pool_scores = _style_scores(pool, req, packaged_model())
    decoded = np.isfinite(pool_scores)
    pool, pool_scores = pool[decoded], pool_scores[decoded]
    rows = top_and_random(pool_scores, n, rng)
    return pool[rows], [float(s) for s in pool_scores[rows]]


def _gallery(
    n: int, seed: int, out: Path, brief: str, top_by_style: bool, pins: dict[str, str]
) -> dict[str, object]:
    """Render n genomes into out/gallery.html."""
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    req = brief_requirements(brief)
    genomes, tile_scores = _picks(n, seed, req, top_by_style, pins)
    tiles = [
        _tile(i, g, req, tiles_dir, s)
        for i, (g, s) in enumerate(zip(genomes, tile_scores, strict=True))
    ]
    page = out / "gallery.html"
    page.write_text(gallery_html(tiles, seed, brief), encoding="utf-8")
    failed = [t.tile_id for t in tiles if t.error is not None]
    return {
        "cmd": "gallery",
        "seed": seed,
        "brief": brief,
        "n": n,
        "top_by_style": top_by_style,
        "pins": pins,
        "built": n - len(failed),
        "failed": failed,
        "gallery": str(page),
    }


def _load_marks(path: Path) -> Marks:
    """Validated marks.json."""
    try:
        return Marks.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise InputError(f"{path}: {exc}") from exc


def _rows(
    sets: list[tuple[Marks, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Feature + fin nuisance rows, labels (good 1), genome groups, stale marks dropped.

    A mark made before the fin cap on a design whose fin the cap now shrinks judged a shape that
    no longer exists, so it is dropped.
    """
    x, y, groups, group_of, stale = [], [], [], {}, 0
    for marks, briefs in sets:
        for brief in briefs:
            req = brief_requirements(brief)
            for it in labelled(marks):
                try:
                    d = decode(np.array(it.genome), req)
                except DecodeError as exc:
                    log.warning("%s (%s) skipped: %s", it.id, brief, exc)
                    continue
                if not marks.fin_cap and fin_capped(d):
                    stale += 1
                    continue
                x.append(np.concatenate([features(d), fin_terms(d)]))
                y.append(1.0 if it.mark == "good" else 0.0)
                groups.append(group_of.setdefault(it.genome, len(group_of)))
    x_arr = np.array(x).reshape(-1, len(FEATURES) + len(NUISANCE))
    return x_arr, np.array(y), np.array(groups), stale


def _tune(paths: list[Path], briefs: list[str] | None, out: Path) -> dict[str, object]:
    """Fit style.yaml to marks; summary has k-fold accuracy and the majority baseline."""
    sets = []
    for path in paths:
        marks = _load_marks(path)
        use = [marks.brief] if marks.brief else (briefs or [])
        if not use or any(b not in BRIEFS for b in use):
            raise InputError(f"{path}: brief {marks.brief!r} unknown; pass --brief {BRIEFS}")
        sets.append((marks, use))
    x, y, groups, stale = _rows(sets)
    l2 = number("style_l2")
    full = fit_logistic(x, y, l2)
    model = deployed(full, len(FEATURES))
    rng = np.random.default_rng(int(number("style_fold_seed")))
    acc = kfold_accuracy(x, y, groups, int(number("style_k_folds")), l2, rng)
    source = f"tune-style on {', '.join(p.name for p in paths)}; briefs {briefs or 'from files'}"
    out.write_text(
        STYLE_HEADER + yaml.safe_dump(model_table(model, source), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    top_n = int(number("style_gate_top_n"))
    return {
        "cmd": "tune-style",
        "rows": len(y),
        "stale_marks_dropped": stale,
        "genomes": len(np.unique(groups)),
        "good": int(y.sum()),
        "bad": int(len(y) - y.sum()),
        "kfold_accuracy": round(acc, 4),
        "majority_baseline": round(float(max(y.mean(), 1.0 - y.mean())), 4),
        "train_accuracy": round(
            float(np.mean((scores(model, x[:, : len(FEATURES)]) >= 0.5) == (y == 1))), 4
        ),
        "nuisance_weights": {
            n: round(float(w), 4)
            for n, w in zip(NUISANCE, full.weights[len(FEATURES) :], strict=True)
        },
        "weights": {n: round(float(w), 4) for n, w in zip(FEATURES, model.weights, strict=True)},
        f"good_in_top_{top_n}": {
            str(p): gate_good_in_top(m, top_n) for p, (m, _) in zip(paths, sets, strict=True)
        },
        "style": str(out),
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    """Dispatch one parsed command to (summary, exit code); raises on unusable input."""
    if args.cmd == "validate":
        return validate_cmd(args.brief)
    if args.cmd == "derive":
        return derive_cmd(args.brief, args.out)
    if args.cmd == "check":
        return check_cmd(args.design, args.tier)
    if args.cmd in ("search", "verify", "demo"):
        return _search_run(args)
    if args.cmd == "build":
        return build_cmd(args.folder, args.full, args.finalist)
    if args.cmd == "render":
        return render_cmd(args.folder, args.finalist)
    if args.cmd == "report":
        return report_cmd(args.folder, args.history)
    if args.cmd == "gallery":
        if args.n < 1:
            raise InputError("--n must be >= 1")
        pins = dict(p.split("=", 1) for p in args.pin)
        if args.body_style is not None:
            pins["body_style"] = args.body_style
        return _gallery(args.n, args.seed, args.out, args.brief, args.top_by_style, pins), EXIT_OK
    out = args.out or Path(str(files("aeroforge.data").joinpath("style.yaml")))
    return _tune(args.marks, args.brief, out), EXIT_OK


def _search_run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    """Dispatch search, verify and demo; counts must be positive."""
    if args.workers < 1 or getattr(args, "budget", 1) < 1 or getattr(args, "runs", 1) < 1:
        raise InputError("--workers, --budget and --runs must be >= 1")
    if args.cmd == "search":
        summary = search_cmd(
            args.folder, args.seed, args.budget, args.history, args.workers, args.randomness
        )
    elif args.cmd == "verify":
        summary = verify_cmd(args.folder, args.history, args.workers)
    else:
        out = args.out or args.brief.parent / "demo"
        summary = demo_cmd(
            args.brief, args.runs, out, args.budget, args.seed, args.workers, args.randomness
        )
    return summary, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run one command; returns the process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    try:
        with threadpool_limits(limits=int(number("blas_threads")), user_api="blas"):
            summary, code = _run(args)
    except (InputError, TuneError, *INPUT_ERRORS) as exc:
        print(json.dumps({"cmd": args.cmd, "error": str(exc)}))
        return EXIT_BAD_INPUT
    print(json.dumps(summary, indent=1))
    return code
