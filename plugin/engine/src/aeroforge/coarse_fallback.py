"""Containment check after coarsening: a coarse body that fails the fit checks is given up."""

from aeroforge.fitcheck import fit_margins
from aeroforge.models import Design, Requirements


def coarse_body_fails(d: Design, req: Requirements) -> str | None:
    """The failing fit margins of a design with a coarsened body (box, root, spar, prop or ground
    line below 0), None when the body is not coarse or holds everything."""
    if d.family.shape.coarseness <= 0.0:
        return None
    bad = {k: v for k, v in fit_margins(d, req).items() if v < 0.0}
    return ", ".join(f"{k} {v:.3f}" for k, v in bad.items()) or None
