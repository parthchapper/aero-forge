"""Where the tail upsweep may start relative to the roots on the centre body (#52, U2 follow-up)."""

import numpy as np

from aeroforge.assemble import Context, Layout, host_name, root_reach_m
from aeroforge.constants import number
from aeroforge.planform import SurfaceGenes
from aeroforge.shapes import axis_z_at_x


def _body_roots(
    specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context
) -> list[tuple[float, float]]:
    """(le x, te x) [m] of the root chord of every surface hosted on the centre body."""
    return [
        (pf.sections[0].le_m[0], pf.sections[0].le_m[0] + pf.sections[0].chord_m)
        for i, pf in lay.planforms.items()
        if host_name(specs[i], ctx, lay.pod) == ctx.body.name
    ]


def root_over_rise(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context) -> bool:
    """A body-hosted root chord that the tail upsweep would lift out of the body."""
    return any(root_reach_m(ctx.body, le, te) < 0.0 for le, te in _body_roots(specs, lay, ctx))


def rise_start_frac(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context) -> float:
    """Length fraction from which the tail upsweep rises without lifting any body-hosted root chord
    out of the body: for each failing root the rise may begin as far aft as its height allows
    (the chord's usable height u h is its reach plus half the rise across it, and a rise of tan a
    per metre needs te - start <= 2 u h / tan a), so the rise starts behind the rearmost root when
    that root has no height to spare."""
    body = ctx.body
    slope = float(np.tan(np.radians(body.tail_upsweep_deg)))
    start = 0.0
    for le, te in _body_roots(specs, lay, ctx):
        reach = root_reach_m(body, le, te)
        if reach >= 0.0:
            continue
        across = abs(axis_z_at_x(body, te) - axis_z_at_x(body, le))
        usable = reach + across / 2.0
        run_m = 2.0 * max(usable, 0.0) * (1.0 - number("rise_fit_margin_frac")) / slope
        start = max(start, (te - run_m - body.nose_m[0]) / body.length_m)
    return start
