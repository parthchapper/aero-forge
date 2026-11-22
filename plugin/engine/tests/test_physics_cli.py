import json
from pathlib import Path

import pytest
from conftest import SAMPLE_BRIEF

from aeroforge.cli import main
from aeroforge.models import Requirements

EXIT_OK, EXIT_CHECK_FAILED, EXIT_BAD_INPUT = 0, 1, 2


def _design_dir(tmp_path: Path, req: Requirements, design_yaml: str) -> Path:
    (tmp_path / "brief.yaml").write_text(SAMPLE_BRIEF.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "requirements.json").write_text(req.model_dump_json(), encoding="utf-8")
    spec = tmp_path / "design.yaml"
    spec.write_text(design_yaml, encoding="utf-8")
    return spec


def test_validate_accepts_the_sample_brief(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SAMPLE_BRIEF)]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_validate_rejects_a_brief_without_launch(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text("payload_mass_kg: 1\nspan_max_m: 2\n", encoding="utf-8")
    assert main(["validate", str(path)]) == EXIT_BAD_INPUT


@pytest.mark.slow  # user (step 3 gate r4): tier-2 conventional tests leave the fast suite
def test_check_writes_check_json_and_passes_the_conventional_at_tier_2(
    tmp_path: Path, sample_req: Requirements
) -> None:
    spec = _design_dir(tmp_path, sample_req, "reference: conventional\n")
    assert main(["check", str(spec), "--tier", "2"]) == EXIT_OK
    report = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert report["feasible"] is True and report["tier"] == 2


def test_check_exit_code_follows_feasibility_at_tier_1(
    tmp_path: Path, sample_req: Requirements
) -> None:
    spec = _design_dir(tmp_path, sample_req, "reference: conventional\n")
    code = main(["check", str(spec), "--tier", "1"])
    report = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert code == (EXIT_OK if report["feasible"] else EXIT_CHECK_FAILED)


def test_check_rejects_a_design_yaml_with_an_unknown_gene(
    tmp_path: Path, sample_req: Requirements
) -> None:
    spec = _design_dir(tmp_path, sample_req, "genes: {no_such_gene: 1.0}\n")
    assert main(["check", str(spec)]) == EXIT_BAD_INPUT


@pytest.mark.slow
def test_derive_writes_requirements_json(tmp_path: Path) -> None:
    brief = tmp_path / "brief.yaml"
    brief.write_text(SAMPLE_BRIEF.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["derive", str(brief)]) == EXIT_OK
    req = Requirements.model_validate_json((tmp_path / "requirements.json").read_text("utf-8"))
    assert len(req.points) == 6


def test_derive_exits_1_and_names_stall_for_ten_km_in_one_hour(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief = tmp_path / "brief.yaml"
    text = SAMPLE_BRIEF.read_text(encoding="utf-8").replace(
        "cruise_speed_ms: 18.0", "cruise_speed_ms: 2.78"
    )
    brief.write_text(text.replace("endurance_min: 60", "range_km: 10"), encoding="utf-8")
    assert main(["derive", str(brief)]) == EXIT_CHECK_FAILED
    assert json.loads(capsys.readouterr().out)["check"] == "stall"
