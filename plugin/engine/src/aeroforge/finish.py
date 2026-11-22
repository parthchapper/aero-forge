"""Final decode stage: surface models, boxes, propellers and the nose-origin shift."""

import numpy as np

from aeroforge.assemble import Context, Layout, host_name, surface_area
from aeroforge.constants import constant, number, numbers
from aeroforge.layout import boxes_in_body, boxes_in_wing, host_end, place_prop, prop_diameter
from aeroforge.models import ControlSurface, Design, Propulsor, Spar, Surface, Vec3
from aeroforge.planform import Planform, SurfaceGenes, joined_sweep_deg
from aeroforge.shapes import component_points, prop_points

PROP_COUNT = {"centre": 1, "pods": 2, "centre_pods": 3}


class LayoutError(RuntimeError):
    """A finished layout lacks something an earlier stage guaranteed."""


def _joined_shape(pf: Planform) -> tuple[float, float, float]:
    """(sweep_deg, cant_deg, taper) implied by a joined panel's root and tip."""
    root, tip = pf.sections[0], pf.sections[-1]
    return joined_sweep_deg(pf), root.cant_deg, tip.chord_m / root.chord_m


def _surface_model(
    g: SurfaceGenes, pf: Planform, fit: tuple[float, Spar], ctx: Context, lay: Layout
) -> Surface:
    """Surface model from its genes, planform and spar fit."""
    tc, spar = fit
    true, projected = surface_area(pf, g)
    span_lo, span_hi = numbers("control_span_frac")
    control = None
    if g.control_frac >= number("control_min_chord_frac"):
        control = ControlSurface(
            chord_frac=g.control_frac, span_start_frac=span_lo, span_end_frac=span_hi
        )
    sweep, cant, taper = g.sweep_deg, g.cant_deg, g.taper
    if g.join_target is not None:
        sweep, cant, taper = _joined_shape(pf)
    return Surface(
        name=f"surface{g.index}",
        host=host_name(g, ctx, lay.pod),
        mirrored=g.mirrored,
        camber_frac=g.camber_frac,
        camber_loc_frac=g.camber_loc_frac,
        tc=tc,
        sweep_deg=sweep,
        cant_deg=cant,
        taper=taper,
        aspect_ratio=pf.aspect_ratio,
        tip_style=pf.tip_style,
        joined_to=None if g.join_target is None else f"surface{g.join_target}",
        sections=pf.sections,
        n_structural=pf.n_structural,
        area_m2=true,
        projected_area_m2=projected,
        control=control,
        spar=spar,
    )


def _props(
    ctx: Context, lay: Layout, pusher: bool, pattern: str, obstacles: np.ndarray
) -> tuple[Propulsor, ...]:
    """Propellers of the pattern, each placed clear of obstacles."""
    d = prop_diameter(ctx.req, PROP_COUNT[pattern])
    recoveries = constant("constants", "ground_line_recoveries")
    if not isinstance(recoveries, list):
        raise TypeError("ground_line_recoveries must be a list")
    ground = ctx.req.recovery in recoveries
    out = []
    if pattern in ("centre", "centre_pods"):
        if ctx.body is not None:
            host, end = ctx.body.name, host_end(ctx.body, pusher)
        else:
            root = lay.planforms[0].sections[0]
            le = np.asarray(root.le_m)
            host, end = (
                "surface0",
                (le + np.array([root.chord_m if pusher else 0.0, 0.0, 0.0]), 0.0),
            )
        out.append(place_prop("prop0", host, False, end, d, pusher, ctx.genes, obstacles, ground))
    if pattern in ("pods", "centre_pods"):
        if lay.pod is None:
            raise LayoutError("pod propellers requested without a pod")
        end = host_end(lay.pod, pusher)
        out.append(
            place_prop("prop1", lay.pod.name, True, end, d, pusher, ctx.genes, obstacles, ground)
        )
    return tuple(out)


def _boxes(ctx: Context, lay: Layout, fits: dict[int, tuple[float, Spar]]) -> tuple:
    """Internal boxes in the centre body or surface0's centre section."""
    if ctx.body is not None and not ctx.wing_boxes:
        return boxes_in_body(ctx.body, ctx.req, ctx.fam.box_layout)
    s0 = lay.planforms[0]
    boxes = boxes_in_wing(s0.sections[: s0.n_structural], ctx.fam, fits[0][0], ctx.req)
    if boxes is None:
        raise LayoutError("boxes do not fit surface0 although the repair loop accepted it")
    return boxes


def finish_design(
    ctx: Context,
    specs: dict[int, SurfaceGenes],
    lay: Layout,
    fits: dict[int, tuple[float, Spar]],
    pusher: bool,
    pattern: str,
    notes: tuple[str, ...],
) -> Design:
    """Complete Design with x measured from the nose-most point."""
    surfaces = tuple(
        _surface_model(specs[i], lay.planforms[i], fits[i], ctx, lay) for i in sorted(specs)
    )
    bodies = tuple(b for b in (ctx.body, lay.pod) if b is not None)
    bare = Design(
        family=ctx.fam,
        lifting_area_m2=ctx.lifting_area_m2,
        surfaces=surfaces,
        bodies=bodies,
        propulsors=(),
        boxes=_boxes(ctx, lay, fits),
        notes=notes,
    )
    obstacles = np.vstack(list(component_points(bare, include_props=False).values()))
    design = bare.model_copy(update={"propulsors": _props(ctx, lay, pusher, pattern, obstacles)})
    # Mirroring keeps x, so the obstacle cloud plus one instance per prop holds the nose-most point.
    ring = int(number("body_ring_points"))
    x_min = min(
        float(obstacles[:, 0].min()),
        *(float(prop_points(p, ring)[:, 0].min()) for p in design.propulsors),
    )
    return translate_x(design, -x_min)


def _sx(p: Vec3, dx_m: float) -> Vec3:
    """Point shifted by dx_m along x."""
    return (p[0] + dx_m, p[1], p[2])


def translate_x(design: Design, dx_m: float) -> Design:
    """Design moved by dx_m along x."""
    surfaces = tuple(
        s.model_copy(
            update={
                "sections": tuple(
                    c.model_copy(update={"le_m": _sx(c.le_m, dx_m)}) for c in s.sections
                )
            }
        )
        for s in design.surfaces
    )
    bodies = tuple(b.model_copy(update={"nose_m": _sx(b.nose_m, dx_m)}) for b in design.bodies)
    props = tuple(
        p.model_copy(update={"hub_m": _sx(p.hub_m, dx_m), "attach_m": _sx(p.attach_m, dx_m)})
        for p in design.propulsors
    )
    boxes = tuple(b.model_copy(update={"centre_m": _sx(b.centre_m, dx_m)}) for b in design.boxes)
    return design.model_copy(
        update={"surfaces": surfaces, "bodies": bodies, "propulsors": props, "boxes": boxes}
    )
