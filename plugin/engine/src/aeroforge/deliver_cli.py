"""CLI handlers for build, render and report (SPEC §11, §12) on a design dir.

build picks the delivered design (design.yaml, else the first verified finalist whose CAD builds)
and writes parts/*.step, assembly.step and assembly.glb; --full adds the internals and checks
their completeness and containment. render draws the four views from assembly.glb; report
writes report.md and viewer.html."""

import json
import shutil
from pathlib import Path

import cadquery as cq
import numpy as np

from aeroforge.candidate import decoded_desc
from aeroforge.checks import CheckReport
from aeroforge.constants import number
from aeroforge.descriptor import median_nearest
from aeroforge.geom import GeometryError, build_parts
from aeroforge.hardpoints import containment_failures, hardpoints
from aeroforge.internals import missing
from aeroforge.internals_geom import internal_parts
from aeroforge.mesh import MeshError, PartMesh, read_glb, tessellate, write_glb
from aeroforge.models import Brief, Design
from aeroforge.physics_cli import (
    CHECK_FILE,
    DESIGN_FILE,
    EXIT_CHECK_FAILED,
    EXIT_OK,
    SpecError,
    load_design,
    write_design,
)
from aeroforge.reference import reference_genome, reference_names
from aeroforge.render import render_png
from aeroforge.report import Novelty, report_md
from aeroforge.search_cli import (
    FINALISTS_FILE,
    VERIFIED_FILE,
    append_history,
    brief_hash,
    load_folder,
    payload_randomness,
    read_history,
    write_json,
)
from aeroforge.viewer import part_table, viewer_html

GENOME_FILE, PARTS_DIR, VIEWS_DIR = "genome.json", "parts", "views"
ASSEMBLY_STEP, ASSEMBLY_GLB = "assembly.step", "assembly.glb"
REPORT_FILE, VIEWER_FILE = "report.md", "viewer.html"
CRITIQUE_FILE, FINALISTS_DIR = "critique.json", "finalists"
VIEWS = ("iso", "top", "front", "side")
STEP_ROOT = "aeroforge"  # root node name; never a part name (parts have kind prefixes)
STEP_UNIT = "M"  # geometry is in metres; without it CadQuery labels the numbers millimetres

Summary = dict[str, object]


def _read_json(path: Path) -> dict:
    """Parsed JSON file, SpecError when missing or broken."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"{path}: {exc}") from exc


def _order(folder: Path) -> list[str]:
    """Verified finalist order, with critique.json's pick (the visual critic's) moved first."""
    order = list(_read_json(folder / VERIFIED_FILE)["order"])
    if not (folder / CRITIQUE_FILE).is_file():
        return order
    pick = _read_json(folder / CRITIQUE_FILE).get("pick")
    if pick not in order:
        raise SpecError(f"{folder / CRITIQUE_FILE}: pick {pick!r} is not in verified order {order}")
    return [pick, *(f for f in order if f != pick)]


def _candidates(folder: Path) -> list[tuple[str | None, Design]]:
    """design.yaml when present, else verified finalists in order (each with its design.yaml)."""
    if (folder / DESIGN_FILE).is_file():
        return [(None, load_design(folder / DESIGN_FILE))]
    out = []
    for fid in _order(folder):
        path = folder / FINALISTS_DIR / fid / DESIGN_FILE
        if path.is_file():
            out.append((fid, load_design(path)))
    if not out:
        raise SpecError(f"{folder}: no design.yaml and no verified finalist design (run verify)")
    return out


def _adopt(folder: Path, fid: str, design: Design) -> None:
    """Make finalist fid the dir's design: design.yaml, check.json, genome.json."""
    write_design(folder / DESIGN_FILE, design)
    shutil.copyfile(folder / FINALISTS_DIR / fid / CHECK_FILE, folder / CHECK_FILE)
    payload = _read_json(folder / FINALISTS_FILE)
    entry = next(f for f in payload["finalists"] if f["id"] == fid)
    report = _read_json(folder / CHECK_FILE)
    adjust = {
        "body_shift_m": report["body_shift_m"],
        "fin_scale": report["fin_scale"],
        "add_fin": report["added_fin"],
        "incidence_deg": report["incidence_deg"],
    }
    genome = {
        "finalist": fid,
        "seed": payload["seed"],
        "randomness": payload_randomness(payload, folder / FINALISTS_FILE),
        "genome": entry["genome"],
        "adjust": adjust,
    }
    write_json(folder / GENOME_FILE, genome)


def write_step(path: Path, parts: dict[str, cq.Solid]) -> None:
    """STEP assembly with one named solid per part, in metres."""
    assy = cq.Assembly(name=STEP_ROOT)
    for name, solid in parts.items():
        assy.add(solid, name=name)
    assy.export(str(path), "STEP", unit=STEP_UNIT)


def _write_cad(folder: Path, parts: dict[str, cq.Solid], meshes: list[PartMesh]) -> None:
    """parts/<name>.step, assembly.step and assembly.glb (stale part files removed first)."""
    parts_dir = folder / PARTS_DIR
    parts_dir.mkdir(exist_ok=True)
    for old in parts_dir.glob("*.step"):
        old.unlink()
    for name, solid in parts.items():
        write_step(parts_dir / f"{name}.step", {name: solid})
    write_step(folder / ASSEMBLY_STEP, parts)
    write_glb(folder / ASSEMBLY_GLB, meshes)


def _finalist_dir(folder: Path, fid: str) -> Path:
    """finalists/<fid>/ of a verified design dir (SpecError when verify left no design there)."""
    if not fid or Path(fid).name != fid or fid in (".", ".."):
        raise SpecError(f"invalid finalist id {fid!r}")
    target = folder / FINALISTS_DIR / fid
    if not (target / DESIGN_FILE).is_file():
        raise SpecError(f"{target / DESIGN_FILE} missing (run verify; ids are in verified.json)")
    return target


def preview_cmd(folder: Path, fid: str) -> tuple[Summary, int]:
    """`build --finalist`: outer-mould assembly.glb of one finalist in its own dir, no adopt."""
    target = _finalist_dir(folder, fid)
    try:
        meshes = tessellate(build_parts(load_design(target / DESIGN_FILE)))
    except (GeometryError, MeshError) as exc:
        error = f"{fid}: {type(exc).__name__}: {exc}"
        summary = {"cmd": "build", "finalist": fid, "built": False, "cad_errors": [error]}
        return summary, EXIT_CHECK_FAILED
    write_glb(target / ASSEMBLY_GLB, meshes)
    return {"cmd": "build", "finalist": fid, "built": True, "parts": len(meshes)}, EXIT_OK


def build_cmd(folder: Path, full: bool, finalist: str | None = None) -> tuple[Summary, int]:
    """`build`: CAD of the delivered design; --full adds internals (completeness, containment)."""
    if finalist is not None:
        if full:
            raise SpecError("--finalist previews the outer mould only; drop --full")
        return preview_cmd(folder, finalist)
    _, req = load_folder(folder)
    errors: list[str] = []
    for fid, design in _candidates(folder):
        try:
            items = hardpoints(design, req.recovery) if full else ()
            parts = dict(sorted({**build_parts(design), **internal_parts(items)}.items()))
            meshes = tessellate(parts)
        except (GeometryError, MeshError) as exc:
            errors.append(f"{fid}: {type(exc).__name__}: {exc}")
            continue
        if fid is not None:
            _adopt(folder, fid, design)
        _write_cad(folder, parts, meshes)
        absent, outside = missing(design, req.recovery, items), containment_failures(design, items)
        return {
            "cmd": "build",
            "full": full,
            "finalist": fid,
            "parts": len(parts),
            "internals": len(items),
            "missing": absent if full else None,
            "not_contained": outside if full else None,
            "cad_errors": errors,
            "assembly": str(folder / ASSEMBLY_STEP),
        }, EXIT_CHECK_FAILED if full and (absent or outside) else EXIT_OK
    return {"cmd": "build", "built": False, "cad_errors": errors}, EXIT_CHECK_FAILED


def render_cmd(folder: Path, finalist: str | None = None) -> tuple[Summary, int]:
    """`render`: views/{iso,top,front,side}.png from assembly.glb (of finalists/<id>/ if given)."""
    if finalist is not None:
        folder = _finalist_dir(folder, finalist)
    meshes = read_glb(_glb_path(folder))
    views = folder / VIEWS_DIR
    views.mkdir(exist_ok=True)
    px = int(number("view_px"))
    for view in VIEWS:
        (views / f"{view}.png").write_bytes(render_png(meshes, view, px))
    return {"cmd": "render", "parts": len(meshes), "views": str(views), "px": px}, EXIT_OK


def _glb_path(folder: Path) -> Path:
    """assembly.glb of a built dir."""
    path = folder / ASSEMBLY_GLB
    if not path.is_file():
        raise SpecError(f"{path} missing (run build first)")
    return path


def _history_ids(path: Path) -> list[str]:
    """Ids of history.jsonl rows, in the order read_history returns their descs."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [str(json.loads(line).get("id")) for line in lines if line]


def plain_desc(genome: np.ndarray, folder: Path) -> np.ndarray:
    """Descriptor of the plain decoded genome under the dir's requirements."""
    _, req = load_folder(folder)
    x = decoded_desc(genome, req)
    if x is None:
        raise SpecError(f"{folder / GENOME_FILE}: genome does not decode")
    return np.array(x, dtype=float)


def design_novelty(
    genome: np.ndarray, folder: Path, history: Path, own_id: str | None = None
) -> Novelty:
    """Distance of the genome's desc to every reference and past design except own_id's."""
    _, req = load_folder(folder)
    x = plain_desc(genome, folder)
    names = reference_names()
    refs = np.array([decoded_desc(reference_genome(n), req) for n in names])
    d_ref = np.linalg.norm(refs - x, axis=1)
    k = int(np.argmin(d_ref))
    ids = _history_ids(history)
    keep = [i for i, h in enumerate(ids) if h != own_id]
    if not keep:
        return Novelty(median_nearest(refs), names[k], float(d_ref[k]), None, None)
    d_past = np.linalg.norm(read_history(history)[keep] - x, axis=1)
    j = int(np.argmin(d_past))
    past = ids[keep[j]]
    return Novelty(median_nearest(refs), names[k], float(d_ref[k]), past, float(d_past[j]))


def _deliver(folder: Path, b: Brief, seed: int, x: np.ndarray, history: Path) -> bool:
    """Append the delivered design to history.jsonl once per design dir; True when appended."""
    if folder.name in _history_ids(history):
        return False
    record = {"id": folder.name, "seed": seed, "brief_hash": brief_hash(b), "desc": x.tolist()}
    append_history(history, record)
    return True


def report_cmd(folder: Path, history: Path) -> tuple[Summary, int]:
    """`report`: report.md and viewer.html of a built design dir."""
    b, req = load_folder(folder)
    design = load_design(folder / DESIGN_FILE)
    gen = _read_json(folder / GENOME_FILE)
    try:
        report = CheckReport.model_validate_json((folder / CHECK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"{folder / CHECK_FILE}: {exc}") from exc
    glb_path = _glb_path(folder)
    genome = np.array(gen["genome"], dtype=float)
    novelty = design_novelty(genome, folder, history, folder.name)
    name = str(gen.get("finalist") or folder.name)
    text = report_md(
        b, req, design, report, novelty, int(gen["seed"]), float(gen["randomness"]), name
    )
    (folder / REPORT_FILE).write_text(text, encoding="utf-8", newline="\n")
    names = [m.name for m in read_glb(glb_path)]
    table = part_table(design, hardpoints(design, req.recovery), names)
    page = viewer_html(f"aero-forge {folder.name} ({name})", glb_path.read_bytes(), table)
    (folder / VIEWER_FILE).write_text(page, encoding="utf-8", newline="\n")
    appended = _deliver(folder, b, int(gen["seed"]), plain_desc(genome, folder), history)
    return {
        "cmd": "report",
        "feasible": report.feasible,
        "history_appended": appended,
        "nearest_reference": [novelty.reference, round(novelty.reference_distance, 4)],
        "nearest_past": [novelty.past, novelty.past_distance],
        "report": str(folder / REPORT_FILE),
        "viewer": str(folder / VIEWER_FILE),
    }, EXIT_OK
