import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import aeroforge.render as render
from aeroforge.brief import load_brief
from aeroforge.candidate import decoded_desc
from aeroforge.checks import CheckReport
from aeroforge.deliver_cli import design_novelty
from aeroforge.descriptor import novelty
from aeroforge.hardpoints import containment_failures, hardpoints
from aeroforge.internals import Internal, expected_names, missing, outline
from aeroforge.mesh import PartMesh
from aeroforge.models import Design, Requirements, Spar
from aeroforge.physics_cli import load_design, write_design
from aeroforge.reference import reference_genome, reference_names
from aeroforge.report import report_md
from aeroforge.search_cli import load_folder
from aeroforge.shapes import inside_loft

WINNERS = Path(__file__).parent / "fixtures" / "winners"
N_WINNERS = 5


def _winner(i: int) -> dict:
    return json.loads((WINNERS / f"winner_{i}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def winners() -> list[Design]:
    return [Design.model_validate(_winner(i)["design"]) for i in range(N_WINNERS)]


@pytest.fixture(scope="module")
def winner_req() -> Requirements:
    return Requirements.model_validate_json(
        (WINNERS / "requirements.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def items(winners: list[Design], winner_req: Requirements) -> list[tuple[Internal, ...]]:
    return [hardpoints(d, winner_req.recovery) for d in winners]


def test_completeness_list_is_satisfied_on_the_saved_winners(
    winners: list[Design], winner_req: Requirements, items: list[tuple[Internal, ...]]
) -> None:
    assert [missing(d, winner_req.recovery, it) for d, it in zip(winners, items, strict=True)] == [
        []
    ] * 5


def test_completeness_names_every_spar_joiner_control_mount_bay_and_recovery(
    winners: list[Design], winner_req: Requirements
) -> None:
    d = winners[0]
    names = expected_names(d, winner_req.recovery)
    joiners = sum(s.n_structural - 1 for s in d.surfaces)
    controls = sum(s.control is not None for s in d.surfaces)
    count = len(d.surfaces) + joiners + 2 * controls + len(d.propulsors) + 2 * len(d.boxes) + 1
    assert len(names) == count


def test_a_missing_internal_is_reported(
    winners: list[Design], winner_req: Requirements, items: list[tuple[Internal, ...]]
) -> None:
    dropped = tuple(it for it in items[0] if it.item != "hatch")
    assert sorted(missing(winners[0], winner_req.recovery, dropped)) == sorted(
        f"hatch_{b.name}" for b in winners[0].boxes
    )


def test_containment_holds_on_the_saved_winners(
    winners: list[Design], items: list[tuple[Internal, ...]]
) -> None:
    assert [containment_failures(d, it) for d, it in zip(winners, items, strict=True)] == [[]] * 5


def test_an_oversized_spar_fails_containment(
    winners: list[Design], winner_req: Requirements
) -> None:
    d = winners[0]
    s = d.surfaces[0]
    chord = min(sec.chord_m for sec in s.sections[: s.n_structural])
    huge = Spar(od_m=0.5 * chord, id_m=0.4 * chord, chord_frac=s.spar.chord_frac)
    bad = d.model_copy(update={"surfaces": (s.model_copy(update={"spar": huge}), *d.surfaces[1:])})
    assert f"spar_{s.name}" in containment_failures(bad, hardpoints(bad, winner_req.recovery))


def test_spar_axis_lies_inside_the_3d_loft_of_untwisted_surfaces(
    winners: list[Design], items: list[tuple[Internal, ...]]
) -> None:
    checked = 0
    for d, group in zip(winners, items, strict=True):
        surfaces = {s.name: s for s in d.surfaces if s.incidence_deg == 0.0}
        for it in (x for x in group if x.item == "spar" and x.owner in surfaces):
            s = surfaces[it.owner]
            path = np.asarray(it.path_m)
            w = np.array([0.01, 0.5, 0.99])[:, None, None]  # end points sit on the cap planes
            pts = ((1 - w) * path[:-1] + w * path[1:]).reshape(-1, 3)
            assert inside_loft(s.sections, outline(s), pts, s.mirrored).all()
            checked += 1
    assert checked > 0


def test_hardpoints_are_deterministic(winners: list[Design], winner_req: Requirements) -> None:
    assert hardpoints(winners[1], winner_req.recovery) == hardpoints(
        winners[1], winner_req.recovery
    )


def test_recovery_item_follows_the_brief(winners: list[Design]) -> None:
    names = {it.name for it in hardpoints(winners[2], "net") if it.item == "recovery"}
    assert names == {"recovery_net"}


def test_design_yaml_round_trips_exactly(winners: list[Design], tmp_path: Path) -> None:
    write_design(tmp_path / "design.yaml", winners[3])
    assert load_design(tmp_path / "design.yaml") == winners[3]


def _report_text(tmp_path: Path) -> tuple[str, CheckReport]:
    w = _winner(0)
    for name in ("brief.yaml", "requirements.json"):
        (tmp_path / name).write_bytes((WINNERS / name).read_bytes())
    b = load_brief(WINNERS / "brief.yaml")
    req = Requirements.model_validate_json((WINNERS / "requirements.json").read_text())
    report = CheckReport.model_validate(w["report"])
    novelty = design_novelty(np.array(w["genome"]), tmp_path, tmp_path / "none.jsonl")
    return report_md(
        b, req, Design.model_validate(w["design"]), report, novelty, w["seed"], 0.5, "f00"
    ), report


def test_report_has_every_check_row_and_the_fidelity_statement(tmp_path: Path) -> None:
    text, report = _report_text(tmp_path)
    rows = [line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| ")]
    assert {c.id for c in report.checks} <= set(rows) and "## Fidelity" in text


def test_report_lists_the_decoder_notes_such_as_a_dropped_tail_upsweep(tmp_path: Path) -> None:
    w = _winner(0)
    note = "tail upsweep dropped: a root chord sits on the rising tail cone"
    design = Design.model_validate(w["design"]).model_copy(update={"notes": (note,)})
    for name in ("brief.yaml", "requirements.json"):
        (tmp_path / name).write_bytes((WINNERS / name).read_bytes())
    req = Requirements.model_validate_json((WINNERS / "requirements.json").read_text())
    novelty = design_novelty(np.array(w["genome"]), tmp_path, tmp_path / "none.jsonl")
    text = report_md(
        load_brief(WINNERS / "brief.yaml"),
        req,
        design,
        CheckReport.model_validate(w["report"]),
        novelty,
        w["seed"],
        0.5,
        "f00",
    )
    assert "## Decoder notes" in text and f"- {note}" in text
    plain = _report_text(tmp_path)[0]
    assert "## Decoder notes" in plain  # always present: "- none" or the notes


def test_report_shows_the_search_seed_and_randomness(tmp_path: Path) -> None:
    text, _ = _report_text(tmp_path)
    assert f"Search seed {_winner(0)['seed']}, randomness 0.5." in text


def test_report_gives_the_three_shadow_measures(tmp_path: Path) -> None:
    text, _ = _report_text(tmp_path)
    assert "## Shadow areas" in text
    assert "Top-down shadow area" in text and "Frontal shadow area" in text


def test_report_nearest_reference_is_no_closer_than_the_search_novelty(tmp_path: Path) -> None:
    w = _winner(0)
    for name in ("brief.yaml", "requirements.json"):
        (tmp_path / name).write_bytes((WINNERS / name).read_bytes())
    n = design_novelty(np.array(w["genome"]), tmp_path, tmp_path / "none.jsonl")
    # The fixture's stored novelty is a v1 (16-number) value; U4 compares in descriptor v2, so the
    # search's own novelty is recomputed from the genome against the same references.
    _, req = load_folder(tmp_path)
    x = np.array(decoded_desc(np.array(w["genome"]), req))
    refs = np.array([decoded_desc(reference_genome(r), req) for r in reference_names()])
    assert n.reference_distance >= novelty(x, refs) - 1e-9


def _thin_tube_mesh() -> PartMesh:
    tube = trimesh.creation.cylinder(radius=0.004, height=1.2, sections=24)
    tube.apply_transform(trimesh.transformations.rotation_matrix(1.1, [0.3, 1.0, 0.2]))
    return PartMesh("spar_test", "internal", np.asarray(tube.vertices), np.asarray(tube.faces))


def test_scanline_rasterisation_matches_the_box_test(monkeypatch: pytest.MonkeyPatch) -> None:
    meshes = [_thin_tube_mesh()]
    fast = render.render_rgb(meshes, "iso", 160)
    original = render.number
    monkeypatch.setattr(
        render,
        "number",
        lambda k, table="constants": 1e9 if k == "render_scan_min_px" else original(k, table),
    )
    assert np.array_equal(fast, render.render_rgb(meshes, "iso", 160))


PROPERTY_DESIGNS = 200
MAX_MISFIT_SHARE = 0.02  # SPEC §5 CAD success bar (98 %), applied to internals containment


@pytest.fixture(scope="module")
def property_items(
    decoded: list[Design], brief: Requirements
) -> list[tuple[Design, tuple[Internal, ...]]]:
    return [(d, hardpoints(d, brief.recovery)) for d in decoded[:PROPERTY_DESIGNS]]


def test_random_designs_always_get_a_complete_internals_list(
    property_items: list[tuple[Design, tuple[Internal, ...]]], brief: Requirements
) -> None:
    assert all(not missing(d, brief.recovery, it) for d, it in property_items)


def test_random_designs_keep_internals_contained_at_the_cad_success_rate(
    property_items: list[tuple[Design, tuple[Internal, ...]]],
) -> None:
    misfits = sum(bool(containment_failures(d, it)) for d, it in property_items)
    assert misfits <= MAX_MISFIT_SHARE * len(property_items)
