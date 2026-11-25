"""Genome table and decode(genome, req) -> Design (SPEC §4, §5)."""

from dataclasses import dataclass, replace

import numpy as np

from aeroforge.align import strict_family, strict_notes
from aeroforge.assemble import Context, Layout, host_name, solve_layout, surface_area
from aeroforge.body_align import BodySnap, initial_shape, next_shape, uncoarsened
from aeroforge.constants import number
from aeroforge.facet_ends import FacetError
from aeroforge.finish import finish_design
from aeroforge.genetable import Genes, family_of, surface_count, unpack
from aeroforge.genetable import random_genomes as random_genomes  # re-export
from aeroforge.layout import boxes_in_wing
from aeroforge.models import Adjust, Body, Design, Family, Requirements, Spar
from aeroforge.planform import SurfaceGenes, added_fin_spec, fit_tc
from aeroforge.pod_fit import pod_y_bounds, prop_diameter
from aeroforge.prop_end import PROP_COUNT, centre_body_for_prop
from aeroforge.rise import rise_start_frac, root_over_rise
from aeroforge.shift import Anchor, Growth, growth

NO_ADJUST = Adjust()


class DecodeError(RuntimeError):
    """The decode repair loop could not reach a valid design."""


@dataclass(frozen=True)
class State:
    """Decode repair state; every change is monotone so the loop terminates."""

    body_on: bool
    pods_on: bool
    active: frozenset[int]
    tc: tuple[float, ...]
    s0_taper: float
    joins_off: frozenset[int]
    rehosted: frozenset[int]
    wing_boxes: bool
    flat_tail: bool
    rise_x_frac: float | None
    body_shape: BodySnap | None  # strict alignment: the body's end shape so far
    body_min_length_m: float
    shift: Growth
    notes: tuple[str, ...]


def _initial_state(v: Genes, fam: Family, adjust: Adjust) -> State:
    """Repair state straight from the genes."""
    n = surface_count()
    active = frozenset([0] + [i for i in range(1, n) if v.pick[f"s{i}_active"] == "on"])
    active |= {n} if adjust.add_fin else set()
    return State(
        body_on=v.pick["body_on"] == "on",
        pods_on=v.pick["pods_on"] == "on",
        active=active,
        tc=(fam.tc,) * (n + 1),
        s0_taper=fam.taper,
        joins_off=frozenset(range(n)) if fam.edge_alignment == "strict" else frozenset(),
        rehosted=frozenset(),
        wing_boxes=fam.box_layout == "wing_root",
        flat_tail=False,
        rise_x_frac=None,
        body_shape=initial_shape(fam),
        body_min_length_m=0.0,
        shift=Growth(0.0, 0.0, adjust.body_shift_m),
        notes=(*strict_notes(v, fam), *fam.facet_notes),
    )


def _host(v: Genes, state: State, i: int) -> str:
    """Effective host kind of surface i."""
    if i == 0 or v.pick[f"s{i}_host"] == "centre" or not state.pods_on or i in state.rehosted:
        return "centre"
    return "pod"


def _cant(v: Genes, fam: Family, i: int) -> float:
    """Inner-panel cant of surface i with the family dihedral [deg]."""
    return float(np.clip(v.num[f"s{i}_cant_deg"] + fam.dihedral_deg, -90.0, 90.0))


def _mirrored(v: Genes, fam: Family, state: State, i: int) -> bool:
    """False for a centre-hosted surface steep enough to be one centreline fin."""
    fin = _host(v, state, i) == "centre" and abs(_cant(v, fam, i)) >= number("fin_snap_cant_deg")
    return not fin


def _join_target(v: Genes, fam: Family, state: State, i: int) -> int | None:
    """Valid lower-index tip-join target of surface i, if any."""
    if i == 0 or i in state.joins_off or v.pick[f"s{i}_tip_join"] == "none":
        return None
    j = int(v.pick[f"s{i}_tip_join"].removeprefix("surface"))
    ok = j < i and j in state.active and _mirrored(v, fam, state, i) and _mirrored(v, fam, state, j)
    return j if ok else None


def _spec(v: Genes, fam: Family, state: State, i: int, targets: set[int]) -> SurfaceGenes:
    """Per-surface genes with the family applied and decoder snaps."""
    p = f"s{i}_"
    sign = 1.0 if v.pick[p + "sweep_sign"] == "aft" else -1.0
    cap = number("sweep_max_deg")
    sweep = float(np.clip(sign * fam.sweep_deg + v.num[p + "sweep_delta_deg"], -cap, cap))
    cant = _cant(v, fam, i)
    mirrored = _mirrored(v, fam, state, i)
    brk = v.num[p + "break_frac"]
    outer_cant = float(np.clip(cant + v.num[p + "outer_cant_delta_deg"], -90.0, 90.0))
    outer_sweep = float(np.clip(sweep + v.num[p + "outer_sweep_delta_deg"], -cap, cap))
    if not mirrored:
        cant = float(np.copysign(90.0, cant))
    if brk >= number("break_none_frac") or not mirrored:
        brk, outer_cant, outer_sweep = 1.0, cant, sweep
    fin = abs(cant) >= number("fin_snap_cant_deg")
    return SurfaceGenes(
        index=i,
        host=_host(v, state, i),
        mirrored=mirrored,
        root_x_frac=v.num[p + "root_x_frac"],
        root_z_frac=v.num[p + "root_z_frac"],
        weight=v.num[p + "area_weight"],
        aspect_ratio=v.num[p + "aspect_ratio"],
        sweep_deg=sweep,
        cant_deg=cant,
        break_frac=brk,
        outer_sweep_deg=outer_sweep,
        outer_cant_deg=outer_cant,
        taper=state.s0_taper if i == 0 else fam.taper,
        tip_style="square" if i in targets or fam.edge_angles_deg else fam.tip_style,
        camber_frac=0.0 if fin else fam.camber_frac,
        camber_loc_frac=fam.camber_loc_frac,
        tc=state.tc[i],
        control_frac=v.num[p + "control_frac"],
        join_target=_join_target(v, fam, state, i),
        root_chord_scale=v.num[p + "root_chord_scale"],
        twist_deg=v.num[p + "twist_deg"] if mirrored else 0.0,  # a twisted centreline fin would toe
        airfoil_facets=fam.airfoil_facets,
        align_angles_deg=fam.edge_angles_deg,
        coarseness=fam.facet_coarseness,
    )


def surface_specs(v: Genes, fam: Family, state: State) -> dict[int, SurfaceGenes]:
    """Specs of all active surfaces by index."""
    n = surface_count()
    genes = [i for i in state.active if i < n]
    targets = {t for i in genes if (t := _join_target(v, fam, state, i)) is not None}
    return {
        i: _spec(v, fam, state, i, targets) if i < n else added_fin_spec(fam, state.tc[i], i)
        for i in sorted(state.active)
    }


def prop_pattern(v: Genes, state: State) -> str:
    """Propeller pattern, falling back to the centre when there are no pods."""
    pattern = v.pick["prop_pattern"]
    return pattern if (state.pods_on or pattern == "centre") else "centre"


def _root_fixes(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context, state: State) -> State:
    """Roots without room: a bare surface0 gets a body, pod roots move to the centre, body0
    grows (up to body_grow_max_ref_frac) or else the crowded root is dropped."""
    hosts = {i: host_name(specs[i], ctx, lay.pod) for i in (*lay.long_roots, *lay.crowded_roots)}
    if "surface0" in hosts.values():
        return replace(
            state, body_on=True, notes=(*state.notes, "body forced on: no room for a root")
        )
    on_pod = {i for i, h in hosts.items() if h == "pod"}
    lengths = [lay.planforms[i].root_chord_m for i in lay.long_roots if hosts[i] == "body0"]
    cap = number("body_grow_max_ref_frac") * ctx.req.ref_length_m
    drop = set()
    for i in (i for i in lay.crowded_roots if ctx.body is not None and hosts[i] == "body0"):
        chord = lay.planforms[i].root_chord_m
        grown = ctx.body.length_m + chord * (1 + 2 * number("surface_gap_chord_frac"))
        if grown <= cap:
            lengths.append(grown)
        else:
            drop.add(i)
    return replace(
        state,
        rehosted=state.rehosted | on_pod,
        active=state.active - drop,
        body_min_length_m=max([state.body_min_length_m, *lengths]),
        notes=(*state.notes, f"roots moved, body grown or dropped (no room): {sorted(hosts)}"),
    )


def _rise_start(specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context, state: State) -> State:
    """State with the tail upsweep starting at the earliest station that lifts no body-hosted root
    chord out of the body; dropped (flat tail) when none leaves a tail to rise over or the start
    is already there. A smooth body keeps the 0.3 rule (drop): its decode is a frozen baseline."""
    old = state.rise_x_frac
    rear = 1.0 if ctx.body.facets is None else rise_start_frac(specs, lay, ctx)
    if rear >= number("rise_start_max_frac") or (old is not None and rear <= old):
        notes = (*state.notes, "tail upsweep dropped: a root chord sits on the rising tail cone")
        return replace(state, flat_tail=True, notes=notes)
    notes = (
        *state.notes,
        f"tail upsweep starts at {rear:.0%} of the body: no root chord sits on the rise",
    )
    return replace(state, rise_x_frac=rear, notes=notes)


def _repair(
    specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context, state: State
) -> State | dict[int, tuple[float, Spar]]:
    """Next repair state, or the spar fits when the layout is final."""
    if ctx.body is not None and ctx.body.tail_upsweep_deg > 0.0 and root_over_rise(specs, lay, ctx):
        return _rise_start(specs, lay, ctx, state)
    if lay.failed_joins:
        return replace(
            state,
            joins_off=state.joins_off | set(lay.failed_joins),
            notes=(*state.notes, f"joins dropped (too swept): {sorted(lay.failed_joins)}"),
        )
    if lay.pod_failed:
        return replace(
            state, pods_on=False, notes=(*state.notes, "pods dropped: no room on surface0")
        )
    if lay.long_roots or lay.crowded_roots:
        return _root_fixes(specs, lay, ctx, state)
    total = sum(surface_area(lay.planforms[i], g)[1] for i, g in specs.items())
    if abs(total - ctx.lifting_area_m2) > number("area_check_rel_tol") * ctx.lifting_area_m2:
        joined = {i for i, g in specs.items() if g.join_target is not None}
        return replace(
            state,
            joins_off=state.joins_off | joined,
            notes=(*state.notes, "joins dropped: area solve"),
        )
    small = [
        i
        for i, g in specs.items()
        if i
        and surface_area(lay.planforms[i], g)[0] < number("min_area_share") * ctx.lifting_area_m2
    ]
    if small:
        return replace(
            state,
            active=state.active - set(small),
            notes=(*state.notes, f"dropped small surfaces {small}"),
        )
    fits: dict[int, tuple[float, Spar]] = {}
    for i, g in specs.items():
        fit = fit_tc(g, lay.planforms[g.index])
        if fit is None:
            return _no_spar(state, i)
        fits[i] = fit
    return fits


def _no_spar(state: State, i: int) -> State:
    """Drop a surface no tube fits; surface0 instead gets a larger tip chord."""
    if i:
        return replace(
            state,
            active=state.active - {i},
            notes=(*state.notes, f"surface{i} dropped: no spar fits"),
        )
    if state.s0_taper >= 1.0:
        raise DecodeError("surface0 cannot carry the smallest spar even untapered at tc_max")
    taper = min(1.0, state.s0_taper + number("taper_step"))
    return replace(state, s0_taper=taper, notes=(*state.notes, "surface0 taper raised for spar"))


def _shift_repair(
    specs: dict[int, SurfaceGenes], lay: Layout, ctx: Context, state: State
) -> State | None:
    """Next state when the shifted body0 must grow (or the shift shrink) to hold its roots."""
    if ctx.anchor is None or ctx.body is None:
        return None
    roots = [
        (specs[i].root_x_frac, pf.sections[0].chord_m)
        for i, pf in lay.planforms.items()
        if host_name(specs[i], ctx, lay.pod) == ctx.body.name
    ]
    g = growth(roots, ctx.anchor.length_m, state.shift.shift_m, ctx.req)
    old = state.shift
    if max(g.front_m - old.front_m, g.rear_m - old.rear_m, abs(g.shift_m - old.shift_m)) <= 1e-9:
        return None
    capped = g.shift_m != old.shift_m
    notes = (*state.notes, f"body shift capped at {g.shift_m:.4f} m") if capped else state.notes
    keep = Growth(max(g.front_m, old.front_m), max(g.rear_m, old.rear_m), g.shift_m)
    return replace(state, shift=keep, notes=notes)


def design_area_m2(v: Genes, req: Requirements) -> float:
    """Lifting area S = wing_area_scale x S_min [m^2], held to the span limit: S stops growing where
    the span would force the aspect ratio under derive_min_aspect_ratio (never below S_min)."""
    cap_m2 = max(req.span_max_m**2 / number("derive_min_aspect_ratio"), req.lifting_area_min_m2)
    return min(v.num["wing_area_scale"] * req.lifting_area_min_m2, cap_m2)


def _centre(
    v: Genes, fam: Family, req: Requirements, state: State
) -> tuple[Body | None, Anchor | None, Family]:
    """The centre body of this pass (None when off) and its root anchor; a body whose boxes ride
    in the wing is a plain body of the fineness."""
    if not state.body_on:
        return None, None, fam
    length_frac = v.num["body_length_frac"]
    if state.flat_tail:  # a root sat on the rising tail cone: the upsweep was dropped
        fam = fam.model_copy(update={"body_tail_upsweep_deg": 0.0})
    elif state.rise_x_frac is not None:  # the upsweep starts behind the rearmost root
        fam = fam.model_copy(update={"body_rise_x_frac": state.rise_x_frac})
    return centre_body_for_prop(
        req,
        fam,
        length_frac,
        state.body_min_length_m,
        state.shift,
        not state.wing_boxes,
        (prop_pattern(v, state), v.pick["prop_direction"] == "pusher"),
    )


def _body_forced(state: State) -> State:
    """Boxes do not fit surface0: the centre body goes on and holds them."""
    notes = (*state.notes, "body forced on: boxes do not fit surface0")
    return replace(state, body_on=True, wing_boxes=False, notes=notes)


def _attempt(
    v: Genes, fam: Family, req: Requirements, state: State, adjust: Adjust
) -> Design | State:
    """One decode pass: a finished design or the next repair state."""
    if state.body_shape is not None:
        fam = fam.model_copy(update={"shape": state.body_shape.shape})
    body, anchor, centre_fam = _centre(v, fam, req, state)
    # the end shape and the rise start are kept (a dropped upsweep stays a gene value, as before)
    fam = fam.model_copy(
        update={"shape": centre_fam.shape, "body_rise_x_frac": centre_fam.body_rise_x_frac}
    )
    if body and state.body_shape and (better := next_shape(fam, body, state.body_shape)):
        notes = (*state.notes, *better.notes)
        return replace(state, body_shape=better, rise_x_frac=None, flat_tail=False, notes=notes)
    pattern = prop_pattern(v, state)
    half_w = body.half_width_m if body else 0.0
    pusher = v.pick["prop_direction"] == "pusher"
    mount = (prop_diameter(req, PROP_COUNT[pattern]), pusher) if pattern != "centre" else None
    bounds = pod_y_bounds(v.num, fam, req, half_w, pattern, mount)
    ctx = Context(
        req,
        fam,
        v.num,
        body,
        state.pods_on,
        bounds,
        design_area_m2(v, req),
        adjust.fin_scale,
        anchor,
        state.wing_boxes,
        mount,
    )
    specs = surface_specs(v, fam, state)
    lay = solve_layout(specs, ctx)
    grown = _shift_repair(specs, lay, ctx, state)
    if grown is not None:
        return grown
    result = _repair(specs, lay, ctx, state)
    if isinstance(result, State):
        return result
    if body is None or state.wing_boxes:
        s0 = lay.planforms[0]
        if boxes_in_wing(s0.sections[: s0.n_structural], fam, result[0][0], req) is None:
            return _body_forced(state)
    notes = (
        (*state.notes, f"{FIN_CAP_NOTE}: {list(lay.capped_fins)}")
        if lay.capped_fins
        else state.notes
    )
    return finish_design(ctx, specs, lay, result, pusher, pattern, notes)


def decode(genome: np.ndarray, req: Requirements, adjust: Adjust = NO_ADJUST) -> Design:
    """Decode a genome (with balance adjustments) into a complete, invariant-satisfying Design."""
    v = unpack(genome)
    fam = strict_family(v, family_of(v))
    state = _initial_state(v, fam, adjust)
    for _ in range(int(number("decode_max_passes"))):
        try:
            outcome = _attempt(v, fam, req, state, adjust)
        except FacetError as e:  # the coarse body cannot hold the design: the genes' body instead
            if state.body_shape is None or state.body_shape.shape.coarseness <= 0.0:
                raise
            snap = uncoarsened(fam, state.body_shape, str(e))
            outcome = replace(state, body_shape=snap, notes=(*state.notes, *snap.notes))
        if isinstance(outcome, Design):
            return outcome
        state = outcome
    raise DecodeError("decode repair loop did not settle")


FIN_CAP_NOTE = "fins capped"


def fin_capped(d: Design) -> bool:
    """The decoder shrank a fin to the fin area or height cap (step 2 round 2)."""
    return any(n.startswith(FIN_CAP_NOTE) for n in d.notes)
