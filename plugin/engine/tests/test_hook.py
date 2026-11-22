import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[2]
HOOK = PLUGIN / "hooks" / "validate_brief.py"
SAMPLE = Path(__file__).parent / "fixtures" / "brief_sample.yaml"
HOOK_LIMIT_S = 2.0  # SPEC §13


def _run(file_path: Path) -> subprocess.CompletedProcess:
    event = {"hook_event_name": "PostToolUse", "tool_name": "Write"}
    event["tool_input"] = {"file_path": str(file_path)}
    return subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(event), capture_output=True, text=True
    )


def _brief(root: Path, text: str, *parts: str) -> Path:
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_invalid_brief_under_designs_exits_2(tmp_path: Path) -> None:
    path = _brief(tmp_path, "payload_mass_kg: -1\nlaunch: rocket\n", "designs", "uav", "brief.yaml")
    assert _run(path).returncode == 2


def test_invalid_brief_message_is_at_most_5_lines(tmp_path: Path) -> None:
    path = _brief(tmp_path, "payload_mass_kg: -1\nlaunch: rocket\n", "designs", "uav", "brief.yaml")
    lines = _run(path).stderr.strip().splitlines()
    assert 1 <= len(lines) <= 5


def test_valid_brief_under_designs_exits_0(tmp_path: Path) -> None:
    path = _brief(tmp_path, SAMPLE.read_text(encoding="utf-8"), "designs", "uav", "brief.yaml")
    assert _run(path).returncode == 0


@pytest.mark.parametrize(
    "parts",
    [("brief.yaml",), ("notes", "brief.yaml"), ("designs", "uav", "design.yaml")],
)
def test_invalid_yaml_elsewhere_exits_0(tmp_path: Path, parts: tuple[str, ...]) -> None:
    path = _brief(tmp_path, "launch: rocket\n", *parts)
    assert _run(path).returncode == 0


def test_event_without_a_file_path_exits_0() -> None:
    proc = subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps({"tool_input": {}}), text=True
    )
    assert proc.returncode == 0


def test_hook_on_a_brief_runs_under_2_s(tmp_path: Path) -> None:
    path = _brief(tmp_path, "launch: rocket\n", "designs", "uav", "brief.yaml")
    start = time.perf_counter()
    _run(path)
    assert time.perf_counter() - start < HOOK_LIMIT_S


def test_hooks_json_runs_the_script_after_write_and_edit() -> None:
    config = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    entry = config["hooks"]["PostToolUse"][0]
    command = entry["hooks"][0]["command"]
    assert (entry["matcher"], "${CLAUDE_PLUGIN_ROOT}/hooks/validate_brief.py" in command) == (
        "Write|Edit",
        True,
    )
