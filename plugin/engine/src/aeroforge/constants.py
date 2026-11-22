"""Single loader for the engine's data/*.yaml tables."""

from functools import cache
from importlib.resources import files
from typing import Any

import yaml

DATA_PACKAGE = "aeroforge.data"


class DataFileError(Exception):
    """A data/*.yaml file is missing or not a mapping."""


@cache
def load_table(name: str) -> dict[str, Any]:
    """Return the parsed mapping in data/<name>.yaml (read once, cached)."""
    resource = files(DATA_PACKAGE).joinpath(f"{name}.yaml")
    if not resource.is_file():
        raise DataFileError(f"data/{name}.yaml not found")
    table = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(table, dict):
        raise DataFileError(f"data/{name}.yaml must contain a mapping")
    return table


def constant(table: str, key: str) -> object:
    """Return the `value` of entry `key` in data/<table>.yaml; entries carry value, unit, source."""
    entries = load_table(table)
    if key not in entries:
        raise DataFileError(f"data/{table}.yaml has no entry {key!r}")
    entry = entries[key]
    if not isinstance(entry, dict) or "value" not in entry:
        raise DataFileError(f"data/{table}.yaml entry {key!r} lacks a value field")
    return entry["value"]


def number(key: str, table: str = "constants") -> float:
    """Return a numeric `value` from data/<table>.yaml as float."""
    value = constant(table, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataFileError(f"data/{table}.yaml entry {key!r} is not a number")
    return float(value)


def knob(key: str, r: float, table: str = "constants") -> float:
    """Randomness knob k(r) = r0 + r (r1 - r0) from the entry's r0 and r1 values, r in [0, 1]."""
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"randomness must be within [0, 1], got {r}")
    entry = load_table(table).get(key)
    if not isinstance(entry, dict):
        raise DataFileError(f"data/{table}.yaml has no entry {key!r}")
    r0, r1 = entry.get("r0"), entry.get("r1")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (r0, r1)):
        raise DataFileError(f"data/{table}.yaml entry {key!r} needs numeric r0 and r1 fields")
    return float(r0) + r * (float(r1) - float(r0))


def numbers(key: str, table: str = "constants") -> tuple[float, ...]:
    """Return a list-of-numbers `value` from data/<table>.yaml as a float tuple."""
    value = constant(table, key)
    if not isinstance(value, list) or not all(isinstance(v, int | float) for v in value):
        raise DataFileError(f"data/{table}.yaml entry {key!r} is not a list of numbers")
    return tuple(float(v) for v in value)
