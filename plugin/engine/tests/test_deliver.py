import base64
import json
import re
from pathlib import Path

import cadquery as cq
import numpy as np
import pytest
from PIL import Image

from aeroforge.cli import main
from aeroforge.hardpoints import hardpoints
from aeroforge.internals_geom import internal_parts
from aeroforge.mesh import read_glb, tessellate
from aeroforge.models import Design, Requirements, instance_names
from aeroforge.physics_cli import load_design, write_design

WINNERS = Path(__file__).parent / "fixtures" / "winners"
pytestmark = pytest.mark.slow


def _winner(i: int) -> dict:
    return json.loads((WINNERS / f"winner_{i}.json").read_text(encoding="utf-8"))


def _verified_dir(folder: Path) -> None:
    """A design dir as search + verify leave it: one finalist f00 = saved winner 0."""
    w = _winner(0)
    for name in ("brief.yaml", "requirements.json"):
        (folder / name).write_bytes((WINNERS / name).read_bytes())
    finalist = {"id": "f00", "genome": w["genome"], "novelty": w["novelty"]}
    (folder / "finalists.json").write_text(
        json.dumps({"seed": 7, "randomness": 0.5, "finalists": [finalist]})
    )
    (folder / "verified.json").write_text(json.dumps({"order": ["f00"]}))
    target = folder / "finalists" / "f00"
    target.mkdir(parents=True)
    write_design(target / "design.yaml", Design.model_validate(w["design"]))
    (target / "check.json").write_text(json.dumps(w["report"]))


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int]]:
    folder = tmp_path_factory.mktemp("design")
    _verified_dir(folder)
    history = folder / "history.jsonl"
    history.write_text(json.dumps({"id": "past_run", "desc": _winner(1)["desc"]}) + "\n")
    codes = [
        main(["build", str(folder), "--full"]),
        main(["render", str(folder)]),
        main(["report", str(folder), "--history", str(history)]),
    ]
    return folder, codes


def _expected_parts(folder: Path) -> list[str]:
    design = load_design(folder / "design.yaml")
    req = Requirements.model_validate_json((folder / "requirements.json").read_text())
    names: list[str] = []
    for group in (design.surfaces, design.bodies, design.propulsors):
        names += [n for c in group for n in instance_names(c.name, c.mirrored)]
    for it in hardpoints(design, req.recovery):
        names += instance_names(it.name, it.mirrored)
    return sorted(names)


def test_build_render_report_exit_ok(built: tuple[Path, list[int]]) -> None:
    assert built[1] == [0, 0, 0]


def test_build_adopts_the_first_verified_finalist(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    genome = json.loads((folder / "genome.json").read_text())
    assert genome["finalist"] == "f00" and load_design(
        folder / "design.yaml"
    ) == Design.model_validate(_winner(0)["design"])


def test_assembly_step_reloads_with_the_expected_solids_and_names(
    built: tuple[Path, list[int]],
) -> None:
    folder, _ = built
    assy = cq.Assembly.importStep(str(folder / "assembly.step"), unit="M")
    names = sorted(c.name for c in assy.children)
    solids = sum(len(c.obj.Solids()) for c in assy.children)
    assert names == _expected_parts(folder) and solids == len(names)


def test_assembly_step_is_in_metres(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    assy = cq.Assembly.importStep(str(folder / "assembly.step"), unit="M")
    bb = assy.toCompound().BoundingBox()
    verts = np.vstack([m.vertices for m in read_glb(folder / "assembly.glb")])
    assert np.allclose([bb.xlen, bb.ylen, bb.zlen], np.ptp(verts, axis=0), atol=2e-3)


def test_one_step_file_per_part(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    assert sorted(p.stem for p in (folder / "parts").glob("*.step")) == _expected_parts(folder)


def test_views_are_four_1024_px_pngs(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    sizes = {p.stem: Image.open(p).size for p in (folder / "views").glob("*.png")}
    assert sizes == {v: (1024, 1024) for v in ("iso", "top", "front", "side")}


def _script(page: str, element_id: str) -> str:
    return re.search(rf'<script id="{element_id}"[^>]*>(.*?)</script>', page, re.S).group(1)


def test_viewer_embeds_the_glb(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    page = (folder / "viewer.html").read_text(encoding="utf-8")
    assert base64.b64decode(_script(page, "glb")) == (folder / "assembly.glb").read_bytes()


def test_viewer_part_table_lists_every_part_with_fields(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    table = json.loads(_script((folder / "viewer.html").read_text(encoding="utf-8"), "parts"))
    assert sorted(table) == _expected_parts(folder) and all(r["fields"] for r in table.values())


def test_viewer_loads_three_from_jsdelivr_via_importmap(built: tuple[Path, list[int]]) -> None:
    page = (built[0] / "viewer.html").read_text(encoding="utf-8")
    block = re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1)
    imports = json.loads(block)["imports"]
    assert all(u.startswith("https://cdn.jsdelivr.net/npm/three@") for u in imports.values())


def test_report_names_the_nearest_past_design(built: tuple[Path, list[int]]) -> None:
    text = (built[0] / "report.md").read_text(encoding="utf-8")
    assert "Nearest past design: past_run" in text


@pytest.mark.parametrize("i", range(5))
def test_internals_of_every_saved_winner_build_and_mesh(i: int) -> None:
    design = Design.model_validate(_winner(i)["design"])
    req = Requirements.model_validate_json((WINNERS / "requirements.json").read_text())
    items = hardpoints(design, req.recovery)
    meshes = tessellate(internal_parts(items))
    assert sorted(m.name for m in meshes) == sorted(
        n for it in items for n in instance_names(it.name, it.mirrored)
    )


def _two_finalist_dir(folder: Path) -> None:
    """Verified dir with finalists f00 (winner 0) and f01 (winner 1)."""
    _verified_dir(folder)
    w = _winner(1)
    payload = json.loads((folder / "finalists.json").read_text())
    payload["finalists"].append({"id": "f01", "genome": w["genome"], "novelty": w["novelty"]})
    (folder / "finalists.json").write_text(json.dumps(payload))
    (folder / "verified.json").write_text(json.dumps({"order": ["f00", "f01"]}))
    (folder / "finalists" / "f01").mkdir()
    write_design(folder / "finalists" / "f01" / "design.yaml", Design.model_validate(w["design"]))
    (folder / "finalists" / "f01" / "check.json").write_text(json.dumps(w["report"]))


def test_finalist_preview_builds_and_renders_in_its_own_dir_without_adopting(
    tmp_path: Path,
) -> None:
    _two_finalist_dir(tmp_path)
    codes = [
        main(["build", str(tmp_path), "--finalist", "f01"]),
        main(["render", str(tmp_path), "--finalist", "f01"]),
    ]
    views = sorted(p.name for p in (tmp_path / "finalists" / "f01" / "views").glob("*.png"))
    assert (codes, views, (tmp_path / "design.yaml").exists()) == (
        [0, 0],
        ["front.png", "iso.png", "side.png", "top.png"],
        False,
    )


def test_finalist_preview_of_an_unverified_id_is_bad_input(tmp_path: Path) -> None:
    _two_finalist_dir(tmp_path)
    assert main(["build", str(tmp_path), "--finalist", "f09"]) == 2


def test_build_adopts_the_critique_pick_over_the_verified_order(tmp_path: Path) -> None:
    _two_finalist_dir(tmp_path)
    (tmp_path / "critique.json").write_text(json.dumps({"pick": "f01"}))
    main(["build", str(tmp_path)])
    assert json.loads((tmp_path / "genome.json").read_text())["finalist"] == "f01"


def test_build_rejects_a_critique_pick_outside_the_verified_order(tmp_path: Path) -> None:
    _two_finalist_dir(tmp_path)
    (tmp_path / "critique.json").write_text(json.dumps({"pick": "f07"}))
    assert main(["build", str(tmp_path)]) == 2


def test_report_appends_the_delivery_to_history_once(built: tuple[Path, list[int]]) -> None:
    folder, _ = built
    history = folder / "history.jsonl"
    main(["report", str(folder), "--history", str(history)])
    ids = [json.loads(line)["id"] for line in history.read_text().splitlines() if line]
    assert ids == ["past_run", folder.name]
