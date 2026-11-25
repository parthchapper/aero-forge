"""Search prior of the facet coarseness (SPEC §9, U3b): a random genome with a faceted part draws
its facet_coarseness from the weights off / low / mid / high of constants.yaml (assumed: mostly
high) instead of uniformly. Applied where genomes are drawn (genome.random_genomes: the search's
starting population and the gallery); the gene's value -> label map is unchanged, so a genome
with the gene at its no-op default still means off. Pure; the rows keep their other genes."""

import numpy as np

from aeroforge.constants import number

KEYS = ("coarse_prior_off", "coarse_prior_low", "coarse_prior_mid", "coarse_prior_high")


def _label(genes: list, name: str, u: np.ndarray) -> np.ndarray:
    gene = next(g for g in genes if g.name == name)
    idx = np.minimum((u * len(gene.choices)).astype(int), len(gene.choices) - 1)
    return np.array(gene.choices)[idx]


def weighted(genomes: np.ndarray, genes: list) -> np.ndarray:
    """genomes (n, D) with facet_coarseness redrawn by the prior in every row with a faceted wing
    or body (the row's own uniform draw is mapped through the weights, so nothing else moves)."""
    names = [g.name for g in genes]
    col = {n: names.index(n) for n in ("airfoil_facets", "body_style", "facet_coarseness")}
    out = genomes.copy()
    faceted = (_label(genes, "airfoil_facets", out[:, col["airfoil_facets"]]) != "smooth") | (
        _label(genes, "body_style", out[:, col["body_style"]]) == "faceted"
    )
    cum = np.cumsum([number(k) for k in KEYS])
    pick = np.minimum(np.searchsorted(cum / cum[-1], out[:, col["facet_coarseness"]], "right"), 3)
    centre = (pick + 0.5) / len(KEYS)
    out[faceted, col["facet_coarseness"]] = centre[faceted]
    return out
