"""Reference configurations: sparse named gene overrides → genome (SPEC §9 reference set)."""

import numpy as np

from aeroforge.constants import load_table
from aeroforge.genetable import gene_table

BASELINE = "conventional"


class ReferenceError(KeyError):
    """Unknown reference name, unknown gene, or a value outside its gene range."""


def encode(overrides: dict[str, object]) -> np.ndarray:
    """Genome from named physical values / choice labels; unlisted genes take their default.

    A float gene defaults to its genes.yaml `ref_default`, else the middle of its range; a choice
    gene to its first label; a choice maps to the middle of its label's bin so decoding returns
    the same label.
    """
    known = {g.name for g in gene_table()}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ReferenceError(f"unknown genes {unknown}")
    genome = []
    for g in gene_table():
        if g.kind == "choice":
            label = str(overrides.get(g.name, g.choices[0]))
            if label not in g.choices:
                raise ReferenceError(f"{g.name}: {label!r} not in {g.choices}")
            genome.append((g.choices.index(label) + 0.5) / len(g.choices))
            continue
        value = overrides.get(g.name)
        if value is None:
            value = g.ref_default
        if value is None:
            genome.append(0.5)
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ReferenceError(f"{g.name}: {value!r} is not a number")
        u = (float(value) - g.lo) / (g.hi - g.lo)
        if not 0.0 <= u <= 1.0:
            raise ReferenceError(f"{g.name}: {value} outside [{g.lo}, {g.hi}]")
        genome.append(u)
    return np.array(genome)


BASE_KEY = "base"


def _overrides(table: dict[str, object], name: str, chain: tuple[str, ...]) -> dict[str, object]:
    """Gene overrides of an entry, its `base` entry's first (recursively)."""
    if name not in table:
        raise ReferenceError(f"no reference configuration {name!r}")
    if name in chain:
        raise ReferenceError(f"reference base cycle {' -> '.join((*chain, name))}")
    entry = table[name]
    if not isinstance(entry, dict):
        raise ReferenceError(f"reference {name!r} must map gene names to values")
    own = {k: v for k, v in entry.items() if k != BASE_KEY}
    if BASE_KEY not in entry:
        return own
    return {**_overrides(table, str(entry[BASE_KEY]), (*chain, name)), **own}


def reference_genome(name: str) -> np.ndarray:
    """Genome of the named entry in data/reference.yaml (a `base` entry's overrides first)."""
    return encode(_overrides(load_table("reference"), name, ()))


def reference_names() -> tuple[str, ...]:
    """Every reference configuration name, sorted."""
    return tuple(sorted(load_table("reference")))
