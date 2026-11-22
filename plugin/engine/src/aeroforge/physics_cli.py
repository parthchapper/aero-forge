"""CLI handlers for validate, derive and check (SPEC §12); each returns (summary, exit code)."""

import json
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError

from aeroforge.aero import AeroError
from aeroforge.brief import BriefError, load_brief
from aeroforge.checks import evaluate
from aeroforge.derive import InfeasibleError, derive
from aeroforge.genome import DecodeError, decode
from aeroforge.models import Design, Requirements
from aeroforge.reference import ReferenceError, encode, reference_genome

EXIT_OK, EXIT_CHECK_FAILED = 0, 1
BRIEF_FILE, REQUIREMENTS_FILE, CHECK_FILE = "brief.yaml", "requirements.json", "check.json"
DESIGN_FILE = "design.yaml"

Summary = dict[str, object]


class SpecError(ValueError):
    """A design.yaml or requirements.json is unusable."""


def validate_cmd(brief_path: Path) -> tuple[Summary, int]:
    """Exit 0 when the brief validates (BriefError otherwise)."""
    b = load_brief(brief_path)
    return {"cmd": "validate", "brief": str(brief_path), "valid": True, "launch": b.launch}, EXIT_OK


def derive_cmd(brief_path: Path, out: Path | None) -> tuple[Summary, int]:
    """Write requirements.json next to the brief (or to out); an infeasible brief exits 1."""
    b = load_brief(brief_path)
    try:
        req = derive(b)
    except InfeasibleError as exc:
        return {
            "cmd": "derive",
            "feasible": False,
            "check": exc.check,
            "error": str(exc),
        }, EXIT_CHECK_FAILED
    target = out or brief_path.with_name(REQUIREMENTS_FILE)
    target.write_text(req.model_dump_json(indent=1), encoding="utf-8", newline="\n")
    return {
        "cmd": "derive",
        "feasible": True,
        "mtow_kg": round(req.mtow_kg, 4),
        "lifting_area_min_m2": round(req.lifting_area_min_m2, 4),
        "battery_wh": round(req.battery_wh, 3),
        "ref_length_m": round(req.ref_length_m, 4),
        "points_ms": {p.id: round(p.speed_ms, 3) for p in req.points},
        "baseline_cruise_ld": [req.baseline_cruise_ld_tier1, req.baseline_cruise_ld_tier2],
        "requirements": str(target),
    }, EXIT_OK


def _design_genome(spec_path: Path) -> np.ndarray:
    """Genome named by design.yaml: `reference: <name>` or `genes: {name: value}`."""
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"{spec_path}: {exc}") from exc
    if not isinstance(spec, dict) or len(spec) != 1 or not ({"reference", "genes"} & set(spec)):
        raise SpecError(
            f"{spec_path}: expected exactly one of `reference: <name>` or `genes: {{...}}`"
        )
    if "reference" in spec:
        return reference_genome(str(spec["reference"]))
    if not isinstance(spec["genes"], dict):
        raise SpecError(f"{spec_path}: genes must map gene names to values")
    return encode(spec["genes"])


def write_design(path: Path, design: Design) -> None:
    """Delivered design.yaml (SPEC §11): the full Design; PyYAML floats round-trip exactly."""
    text = yaml.safe_dump(design.model_dump(mode="json"), sort_keys=False)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_design(path: Path) -> Design:
    """Validated delivered design.yaml."""
    try:
        return Design.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise SpecError(f"{path}: {exc}") from exc


def _requirements(path: Path) -> Requirements:
    """Validated requirements.json."""
    try:
        return Requirements.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise SpecError(f"{path}: {exc}") from exc


def check_cmd(spec_path: Path, tier: int) -> tuple[Summary, int]:
    """Check the design in design.yaml against brief.yaml + requirements.json beside it."""
    folder = spec_path.parent
    b = load_brief(folder / BRIEF_FILE)
    req = _requirements(folder / REQUIREMENTS_FILE)
    try:
        genome = _design_genome(spec_path)
        decode(genome, req)
    except (ReferenceError, DecodeError) as exc:
        raise SpecError(f"{spec_path}: {exc}") from exc
    try:
        _, report = evaluate(genome, b, req, tier)
    except AeroError as exc:
        return {
            "cmd": "check",
            "tier": tier,
            "feasible": False,
            "error": str(exc),
        }, EXIT_CHECK_FAILED
    out = folder / CHECK_FILE
    out.write_text(json.dumps(report.model_dump(), indent=1), encoding="utf-8", newline="\n")
    summary: Summary = {
        "cmd": "check",
        "tier": tier,
        "feasible": report.feasible,
        "violation": round(report.violation, 4),
        "passable": report.passable,
        "failed": [c.id for c in report.checks if not c.passed],
        "margins": {c.id: round(c.margin, 4) for c in report.checks},
        "check": str(out),
    }
    return summary, EXIT_OK if report.feasible else EXIT_CHECK_FAILED


INPUT_ERRORS = (BriefError, SpecError, ReferenceError)
