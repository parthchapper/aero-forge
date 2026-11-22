"""Genome table: genes.yaml rows, unpacking a genome vector into physical values (SPEC §4)."""

from dataclasses import dataclass
from functools import cache

import numpy as np

from aeroforge.constants import load_table, number
from aeroforge.models import Family


class GenomeError(ValueError):
    """Genome vector has the wrong length or values outside [0, 1]."""


@dataclass(frozen=True)
class Gene:
    """One genes.yaml row."""

    name: str
    kind: str
    lo: float
    hi: float
    choices: tuple[str, ...]
    ref_default: float | None = None


@dataclass(frozen=True)
class Genes:
    """Unpacked genome: physical floats and choice labels by gene name."""

    num: dict[str, float]
    pick: dict[str, str]


@cache
def gene_table() -> tuple[Gene, ...]:
    """Rows of data/genes.yaml in genome order."""
    rows = load_table("genes")["genes"]
    return tuple(
        Gene(
            name=str(r["name"]),
            kind=str(r["kind"]),
            lo=float(r.get("lo", 0.0)),
            hi=float(r.get("hi", 1.0)),
            choices=tuple(str(c) for c in r.get("choices", ())),
            ref_default=float(r["ref_default"]) if "ref_default" in r else None,
        )
        for r in rows
    )


def genome_length() -> int:
    """Genome dimension D."""
    return len(gene_table())


def surface_count() -> int:
    """Number of surface slots defined in genes.yaml."""
    return sum(1 for g in gene_table() if g.name.endswith("_root_x_frac"))


def _apply_layout(num: dict[str, float], pick: dict[str, str]) -> None:
    """A named layout replaces the slot roles of s1-s3 (data/layouts.yaml): choice genes take its
    labels, float genes its value shifted by layout_tune_frac of the gene's own offset from the
    range middle, so the slot genes still fine-tune it. `free` changes nothing."""
    name = pick["layout"]
    if name == "free":
        return
    tune = number("layout_tune_frac")
    rows = {g.name: g for g in gene_table()}
    for key, value in load_table("layouts")["layouts"][name].items():
        gene = rows[key]
        if gene.kind == "choice":
            pick[key] = str(value)
        else:
            shifted = float(value) + tune * (num[key] - (gene.lo + gene.hi) / 2)
            num[key] = float(np.clip(shifted, gene.lo, gene.hi))


def unpack(genome: np.ndarray) -> Genes:
    """Map u in [0,1]^D to physical gene values and choice labels."""
    arr = np.asarray(genome, dtype=float)
    if arr.shape != (genome_length(),) or not np.all(np.isfinite(arr)):
        raise GenomeError(f"genome must be a finite vector of length {genome_length()}")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise GenomeError("genome values must lie in [0, 1]")
    num: dict[str, float] = {}
    pick: dict[str, str] = {}
    for gene, u in zip(gene_table(), arr, strict=True):
        if gene.kind == "choice":
            pick[gene.name] = gene.choices[min(int(u * len(gene.choices)), len(gene.choices) - 1)]
        else:
            num[gene.name] = gene.lo + float(u) * (gene.hi - gene.lo)
    _apply_layout(num, pick)
    return Genes(num, pick)


def family_of(v: Genes) -> Family:
    """Design-language values from the global genes."""
    airfoil = v.pick["fam_airfoil"]
    return Family(
        sweep_deg=v.num["fam_sweep_deg"],
        taper=v.num["fam_taper"],
        dihedral_deg=v.num["fam_dihedral_deg"],
        tip_style=v.pick["fam_tip_style"],  # type: ignore[arg-type]
        camber_frac=int(airfoil[4]) / 100,
        camber_loc_frac=int(airfoil[5]) / 10,
        tc=v.num["fam_tc"],
        body_fineness=v.num["fam_body_fineness"],
        body_section_exp=v.num["fam_body_section_exp"],
        body_nose_frac=v.num["body_nose_frac"],
        body_tail_frac=v.num["body_tail_frac"],
        body_width_ratio=v.num["body_width_ratio"],
        body_tail_upsweep_deg=v.num["body_tail_upsweep_deg"],
        box_layout=v.pick["box_layout"],  # type: ignore[arg-type]
        layout=v.pick["layout"],  # type: ignore[arg-type]
    )
