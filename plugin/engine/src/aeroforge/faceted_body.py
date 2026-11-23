"""Body models for the layout: the tail-end size both body kinds share and the flat-panel body."""

from aeroforge.constants import number
from aeroforge.facets import facet_body
from aeroforge.models import Body, Family


def tail_radius_frac(fam: Family) -> float:
    """Tail-end radius / max radius of a smooth body: the gene, floored off a degenerate ring."""
    return max(fam.shape.tail_end_size_frac, number("facet_tail_end_min_frac"))


def faceted_body(
    fam: Family, length_m: float, axes_m: tuple[float, float], name: str, upsweep_deg: float = 0.0
) -> Body:
    """Body of flat panels; its smooth-law fields give the same extent and envelope (widest
    station at nose_frac, boat-tail after it) to consumers that are not facet-aware yet."""
    m = fam.shape.max_section_x_frac
    return Body(
        name=name,
        mirrored=False,
        nose_m=(0.0, 0.0, 0.0),
        length_m=length_m,
        half_width_m=axes_m[0],
        half_height_m=axes_m[1],
        section_exp=fam.body_section_exp,
        nose_frac=m,
        tail_frac=1.0 - m,
        tail_radius_frac=tail_radius_frac(fam),
        tail_upsweep_deg=upsweep_deg,
        facets=facet_body(fam.shape, length_m, axes_m, upsweep_deg),
    )
