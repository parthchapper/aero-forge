"""Genome table: genes.yaml rows, unpacking a genome vector into physical values (SPEC §4)."""

from dataclasses import dataclass
from functools import cache

import numpy as np

from aeroforge.coarse import coarse_family, level
from aeroforge.constants import load_table, number
from aeroforge.models import BodyShape, Family


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


def pin_choice(genomes: np.ndarray, gene: str, label: str) -> np.ndarray:
    """Copy of genomes (rows) with the choice gene `gene` set to the middle of `label`'s cell."""
    names = [g.name for g in gene_table()]
    row = gene_table()[names.index(gene)]
    if row.kind != "choice" or label not in row.choices:
        raise GenomeError(f"{gene}: no choice {label!r}")
    out = np.array(genomes, dtype=float, copy=True)
    out[..., names.index(gene)] = (row.choices.index(label) + 0.5) / len(row.choices)
    return out


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


def _count(value: float, lo: int, hi: int) -> int:
    """Integer gene: floor of the value (a tiny lift absorbs round-off), clipped to lo..hi."""
    return int(np.clip(np.floor(value + number("facet_geom_tol_m")), lo, hi))


def shape_of(v: Genes) -> BodyShape:
    """Body style genes (SPEC §4, U1)."""
    return BodyShape(
        style=v.pick["body_style"],  # type: ignore[arg-type]
        sides=_count(v.num["body_section_sides"], 3, int(number("facet_max_sides"))),
        stations=_count(v.num["body_stations"], 1, int(number("facet_max_stations"))),
        chine_z_frac=v.num["chine_z_frac"],
        belly_flat_frac=v.num["belly_flat_frac"],
        roof_pitch_deg=v.num["roof_pitch_deg"],
        side_tumblehome_deg=v.num["side_tumblehome_deg"],
        lower_flare_deg=v.num["lower_flare_deg"],
        nose=v.pick["nose_style"],  # type: ignore[arg-type]
        tail=v.pick["tail_style"],  # type: ignore[arg-type]
        tail_flatten=v.num["tail_flatten"],
        tail_end_size_frac=v.num["tail_end_size_frac"],
        max_section_x_frac=v.num["max_section_x_frac"],
    )


def facets_of(v: Genes) -> int:
    """Flat segments per side of a wing section (0 = smooth) from the airfoil_facets gene."""
    label = v.pick["airfoil_facets"]
    return 0 if label == "smooth" else int(label)


def family_of(v: Genes) -> Family:
    """Design-language values from the global genes."""
    airfoil = v.pick["fam_airfoil"]
    fam = Family(
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
        shape=shape_of(v),
        airfoil_facets=facets_of(v),
        edge_alignment=v.pick["edge_alignment"],  # type: ignore[arg-type]
        facet_coarseness=level(v.pick["facet_coarseness"]),
    )
    return coarse_family(fam)
