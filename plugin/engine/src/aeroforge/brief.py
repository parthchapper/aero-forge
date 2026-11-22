"""Brief loading, validation and defaults (SPEC §4 brief table)."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from aeroforge.constants import constant, load_table, number
from aeroforge.genetable import GenomeError
from aeroforge.models import Brief, Requirements


class BriefError(ValueError):
    """A brief file is unreadable or invalid."""


def load_brief(path: Path) -> Brief:
    """Validated Brief from a YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Brief.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise BriefError(f"{path}: {exc}") from exc


def launch_value(table: str, launch: str) -> float:
    """Per-launch entry of a checks.yaml table."""
    values = constant("checks", table)
    if not isinstance(values, dict) or launch not in values:
        raise BriefError(f"checks.yaml {table} has no {launch!r} entry")
    return float(values[launch])


def resolve(b: Brief) -> Brief:
    """Brief with every defaulted field filled from checks.yaml (SPEC §4 defaults)."""
    return b.model_copy(
        update={
            "stall_max_ms": b.stall_max_ms or launch_value("launch_stall_max_ms", b.launch),
            "dash_speed_ms": b.dash_speed_ms
            or number("dash_cruise_ratio", "checks") * b.cruise_speed_ms,
            "n_limit": b.n_limit or number("n_limit_default", "checks"),
            "gust_ms": b.gust_ms if b.gust_ms is not None else number("gust_ms_default", "checks"),
            "altitude_m": b.altitude_m
            if b.altitude_m is not None
            else number("altitude_m_default", "checks"),
            "efficiency_floor": b.efficiency_floor or number("efficiency_floor_default", "checks"),
        }
    )


def effective_randomness(b: Brief, override: float | None = None) -> float:
    """Search randomness r in [0, 1]: the CLI override, else the brief's, else the default."""
    r = override if override is not None else b.randomness
    if r is None:
        return number("randomness_default")
    if not 0.0 <= r <= 1.0:
        raise BriefError(f"randomness must be within [0, 1], got {r}")
    return float(r)


def launch_speed_ms(b: Brief) -> float:
    """Launch (release / catapult end / rotation) speed of the brief's launch method [m/s]."""
    return launch_value("launch_speed_ms", b.launch)


BRIEFS = ("stub", "small_boxes")


def brief_requirements(name: str) -> Requirements:
    """Fixed Requirements from data/requirements_<name>.yaml (stand-in until derive)."""
    if name not in BRIEFS:
        raise GenomeError(f"unknown stand-in brief {name!r}; expected one of {BRIEFS}")
    raw = load_table(f"requirements_{name}")
    values = {
        k: (tuple(e["value"]) if isinstance(e["value"], list) else e["value"])
        for k, e in raw.items()
    }
    return Requirements.model_validate(values)
