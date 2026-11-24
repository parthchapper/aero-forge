"""Where the tail upsweep may start relative to the roots on the centre body (#52, U2 follow-up).

A flat-panel body shifts whole stations, so its axis is straight between stations: a rise that
starts between two stations tilts the whole cell it starts in (facets.axis_rise_between_m is that
axis). The start is the earliest one at which no body-hosted root chord is lifted further out of
the body than it would be without the rise."""

import numpy as np

from aeroforge.assemble import Context, Layout, host_name, root_reach_m
from aeroforge.constants import number
from aeroforge.facet_query import axis_half_height_m
from aeroforge.facets import axis_rise_between_m
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


def _terms(ctx: Context, roots: list[tuple[float, float]]) -> list[tuple[np.ndarray, float, float]]:
    """Per root chord: (length fractions of its two ends, usable height [m], the axis lift across
    the chord that the rise is not responsible for [m]). The usable height is the share of the
    smallest half-height over the chord a root may use; the lift not due to the rise is the axis
    height difference across the chord less the present rise's. Computed once per call: they do
    not depend on where the rise would start."""
    body, shape = ctx.body, ctx.fam.shape
    nose, usable_frac = body.nose_m[0], number("root_z_usable_frac")
    out = []
    for le, te in roots:
        t = np.array([le - nose, te - nose]) / body.length_m
        rise = axis_rise_between_m(t, shape, body.length_m, body.tail_upsweep_deg, body.rise_x_frac)
        lift = axis_z_at_x(body, te) - axis_z_at_x(body, le)
        half_h = axis_half_height_m(body.facets, le - nose, te - nose)
        out.append((t, usable_frac * half_h, float(lift - (rise[1] - rise[0]))))
    return out


def _reach(ctx: Context, term: tuple[np.ndarray, float, float], start_frac: float | None) -> float:
    """Reach of a root chord [m] (as assemble.root_reach_m) were the rise to start at start_frac."""
    t, usable, shift = term
    body = ctx.body
    rise = axis_rise_between_m(t, ctx.fam.shape, body.length_m, body.tail_upsweep_deg, start_frac)
    return usable - abs(shift + float(rise[1] - rise[0])) / 2.0


def _floor_m(term: tuple[np.ndarray, float, float]) -> float:
    """Lowest reach [m] the rise may leave a root: zero, or the root's own reach without the rise
    when that is already negative (the rise must not make it worse)."""
    _, usable, shift = term
    return min(0.0, usable - abs(shift) / 2.0) - number("facet_geom_tol_m")


def root_over_rise(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context) -> bool:
    """A body-hosted root chord that the tail upsweep would lift out of the body (a flat-panel
    body: further out than it lies without the rise; a smooth body: reach below zero)."""
    roots = _body_roots(specs, lay, ctx)
    if ctx.body.facets is None:
        return any(root_reach_m(ctx.body, le, te) < 0.0 for le, te in roots)
    return any(
        _reach(ctx, term, ctx.body.rise_x_frac) < _floor_m(term) for term in _terms(ctx, roots)
    )


def rise_start_frac(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context) -> float:
    """Length fraction of the earliest rise start (from the widest station on, in rise_start_steps
    steps) at which no body-hosted root chord is lifted out of the body; 1 when none is. A
    flat-panel body only."""
    terms = _terms(ctx, _body_roots(specs, lay, ctx))
    lo, steps = ctx.fam.shape.max_section_x_frac, int(number("rise_start_steps"))
    for start in np.linspace(lo, 1.0, steps + 1):
        if all(_reach(ctx, term, float(start)) >= _floor_m(term) for term in terms):
            return float(start)
    return 1.0
