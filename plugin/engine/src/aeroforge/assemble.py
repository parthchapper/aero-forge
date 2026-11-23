"""Surface layout: roots snapped to hosts, pods on surface0, area scale solved to S."""

from dataclasses import dataclass, replace

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_query import axis_half_height_m
from aeroforge.layout import place_pod
from aeroforge.models import Body, Family, Requirements, Vec3
from aeroforge.planform import (
    Planform,
    SurfaceGenes,
    build_joined,
    build_planform,
    join_too_swept,
    joined_root_chord,
    section_areas,
)
from aeroforge.shapes import axis_z_at_x, body_radius_frac
from aeroforge.shift import Anchor, anchored_x_m

# Placed roots per host name: (root leading-edge x [m], trailing-edge x [m], cant [deg], index).
Root = tuple[float, float, float, int]
Placed = dict[str, list[Root]]


@dataclass(frozen=True)
class Context:
    """Fixed inputs of one layout attempt."""

    req: Requirements
    fam: Family
    genes: dict[str, float]
    body: Body | None
    pods_on: bool
    pod_y_bounds: tuple[float, float]
    lifting_area_m2: float  # this design's S (wing_area_scale x S_min)
    fin_scale: float = 1.0
    anchor: Anchor | None = None
    wing_boxes: bool = False  # the boxes ride in surface0, not in the centre body


@dataclass(frozen=True)
class Layout:
    """Planforms by surface index and the right pod, with the verdicts at this area scale.

    failed_joins: too swept or degenerate; pod_failed: pod past surface0's inner panel or no
    room at any scale; long_roots: root chord longer than its host; crowded_roots: root still
    clashing with an earlier similar-cant root on the same host; capped_fins: fins held at the
    fin area or height cap.
    """

    planforms: dict[int, Planform]
    pod: Body | None
    failed_joins: tuple[int, ...]
    pod_failed: bool
    long_roots: tuple[int, ...]
    crowded_roots: tuple[int, ...]
    capped_fins: tuple[int, ...]


def instances(g: SurfaceGenes) -> int:
    """Number of instances of a surface."""
    return 2 if g.mirrored else 1


def surface_area(pf: Planform, g: SurfaceGenes) -> tuple[float, float]:
    """(true, projected) area of all instances [m^2]."""
    true, projected = section_areas(pf.sections)
    return instances(g) * true, instances(g) * projected


def host_name(g: SurfaceGenes, ctx: Context, pod: Body | None) -> str:
    """Name of the part a surface root sits on."""
    if g.host == "pod" and pod is not None:
        return pod.name
    if ctx.body is not None:
        return ctx.body.name
    return "centre" if g.index == 0 else "surface0"


def _fit_x(
    x_m: float,
    chord_m: float,
    cant_deg: float,
    placed: list[Root],
    span_m: tuple[float, float],
) -> tuple[float, bool]:
    """Root mid-chord x [m] nearest x_m with the whole chord on the host span and clear of
    similar-cant roots; the host centre and False when the chord is longer than the host.

    A host with no free slot keeps the clamped x; the layout's crowding check flags it.
    """
    lo, hi = span_m
    half = chord_m / 2
    if chord_m > hi - lo:
        return (lo + hi) / 2, False
    x_lo, x_hi = lo + half, hi - half
    want = float(np.clip(x_m, x_lo, x_hi))
    blocked = [
        (le - gap - half, te + gap + half)
        for le, te, other_cant, _ in placed
        if abs(other_cant - cant_deg) < number("surface_separation_cant_deg")
        for gap in (number("surface_gap_chord_frac") * max(chord_m, te - le),)
    ]
    # The nearest free x is the wanted x itself or the edge of a blocked or host interval.
    candidates = [want, x_lo, x_hi, *(edge for interval in blocked for edge in interval)]
    free = [
        c
        for c in candidates
        if x_lo <= c <= x_hi and not any(b_lo < c < b_hi for b_lo, b_hi in blocked)
    ]
    if not free:
        return want, True
    return min(free, key=lambda c: (abs(c - want), c)), True


def _clash(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Two roots (le x, te x [m], cant [deg]) of similar cant closer than the separation gap."""
    if abs(a[2] - b[2]) >= number("surface_separation_cant_deg"):
        return False
    gap = number("surface_gap_chord_frac") * max(a[1] - a[0], b[1] - b[0])
    tol = number("surface_gap_tol_m")
    return a[1] + gap > b[0] + tol and b[1] + gap > a[0] + tol


def _crowded(placed: Placed) -> tuple[int, ...]:
    """Indices of roots that clash with an earlier root on the same host."""
    out = {
        i
        for roots in placed.values()
        for k, (le, te, cant, i) in enumerate(roots)
        if any(_clash((le, te, cant), other[:3]) for other in roots[:k])
    }
    return tuple(sorted(out))


def root_reach_m(body: Body, x_lo_m: float, x_hi_m: float) -> float:
    """Height [m] a root chord over x_lo_m..x_hi_m can use above and below its centring on the
    body axis: the usable share of the smallest section half-height there, less half the rise of
    a tail upsweep across the chord. Negative: the chord cannot stay inside the body."""
    t = (np.array([x_lo_m, x_hi_m]) - body.nose_m[0]) / body.length_m
    if body.facets is not None:
        half_h = axis_half_height_m(body.facets, x_lo_m - body.nose_m[0], x_hi_m - body.nose_m[0])
    else:
        half_h = body.half_height_m * float(body_radius_frac(body, t).min())
    ends = np.array([axis_z_at_x(body, x_lo_m), axis_z_at_x(body, x_hi_m)])
    return number("root_z_usable_frac") * half_h - abs(float(ends[1] - ends[0])) / 2


def _body_root(
    g: SurfaceGenes,
    root_m: tuple[float, float],
    body: Body,
    placed: list[Root],
    anchor: Anchor | None,
) -> tuple[Vec3, bool]:
    """Root mid-chord point with the whole root chord inside body (anchored when shifted)."""
    chord_m, cant_deg = root_m
    lo = body.nose_m[0]
    want = (
        anchored_x_m(g.root_x_frac, chord_m, anchor)
        if anchor is not None
        else g.root_x_frac * body.length_m
    )
    x, fits = _fit_x(lo + want, chord_m, cant_deg, placed, (lo, lo + body.length_m))
    # The radius law rises, holds, then falls, so its minimum over the chord is at an end; a tail
    # upsweep lifts the axis along the chord, so the root is centred on its middle.
    ends = np.array([axis_z_at_x(body, x - chord_m / 2), axis_z_at_x(body, x + chord_m / 2)])
    reach = root_reach_m(body, x - chord_m / 2, x + chord_m / 2)
    z = float(ends.mean()) + g.root_z_frac * max(reach, 0.0)
    return (x, body.nose_m[1], z), fits


def _root(
    g: SurfaceGenes,
    root_m: tuple[float, float],
    ctx: Context,
    pod: Body | None,
    s0: Planform | None,
    placed: Placed,
) -> tuple[Vec3, bool]:
    """Root mid-chord point on the host for root_m = (chord [m], cant [deg]); whether it fits."""
    chord_m, cant_deg = root_m
    here = placed.get(host_name(g, ctx, pod), [])
    if g.host == "pod" and pod is not None:
        return _body_root(g, root_m, pod, here, None)
    if ctx.body is not None:
        return _body_root(g, root_m, ctx.body, here, ctx.anchor)
    if s0 is None:
        return (0.0, 0.0, 0.0), True
    root = s0.sections[0]
    span = (root.le_m[0], root.le_m[0] + root.chord_m)
    x, fits = _fit_x(span[0] + g.root_x_frac * root.chord_m, chord_m, cant_deg, here, span)
    return (x, 0.0, root.le_m[2]), fits


def _joined_cant_estimate(target_tip_m: Vec3, y0_m: float, ctx: Context) -> float:
    """Cant [deg] of a joined panel from its host axis to the target tip (root z not yet known)."""
    z0 = ctx.body.nose_m[2] if ctx.body is not None else 0.0
    return float(np.degrees(np.arctan2(target_tip_m[2] - z0, target_tip_m[1] - y0_m)))


def scaled_spec(g: SurfaceGenes, scale: float) -> SurfaceGenes:
    """An unjoined surface whose only steep (> vertical_cant_deg) panel is the outer one grows
    that panel by moving its break inboard (user Deviation, step 3 gate r4)."""
    steep = number("vertical_cant_deg")
    outer_only = abs(g.cant_deg) <= steep < abs(g.outer_cant_deg) and g.break_frac < 1.0
    if scale == 1.0 or g.join_target is not None or not outer_only:
        return g
    outer = min(scale * (1.0 - g.break_frac), 1.0 - number("vertical_break_min_frac"))
    return replace(g, break_frac=1.0 - outer)


def scales_by_area(g: SurfaceGenes) -> bool:
    """Unjoined surface (not surface0) with a root panel steeper than vertical_cant_deg."""
    return g.join_target is None and g.index != 0 and abs(g.cant_deg) > number("vertical_cant_deg")


def is_fin_spec(g: SurfaceGenes) -> bool:
    """An unjoined surface whose inner panel is steeper than fin_cant_deg."""
    return g.join_target is None and abs(g.cant_deg) >= number("fin_cant_deg")


def _root_y(g: SurfaceGenes, pod: Body | None) -> float:
    """Root station y [m]: the right pod's for pod-hosted surfaces, else the centreline."""
    return pod.nose_m[1] if (g.host == "pod" and pod is not None) else 0.0


def _fin_capped_area(g: SurfaceGenes, area_m2: float, ctx: Context, root_m: Vec3) -> float:
    """Fin true area [m^2] within fin_area_max_frac x S and fin_height_max_ref_frac x ref length.

    Fins project ~0 area, so S alone leaves them unbounded (user marks: tall fins read badly).
    Height without the span limit grows exactly like sqrt(area), so one rescale lands on the
    cap; the span limit only shortens a fin (a clipped height ignores area, so rescaling the
    clipped height would not converge).
    """
    area = min(area_m2, number("fin_area_max_frac") * ctx.lifting_area_m2)
    h_max = number("fin_height_max_ref_frac") * ctx.req.ref_length_m
    unclipped = build_planform(g, area, root_m, np.inf, False)
    height = float(np.ptp([s.le_m[2] for s in unclipped.sections]))
    return area if height <= h_max else area * (h_max / height) ** 2


def _build_one(
    g: SurfaceGenes,
    area_m2: float,
    ctx: Context,
    pod: Body | None,
    built: dict[int, Planform],
    placed: Placed,
) -> tuple[Planform | None, bool]:
    """Planform at true area area_m2 (None for a degenerate join) and whether its root fits."""
    lim = ctx.req.span_max_m / 2
    y0 = _root_y(g, pod)
    mitre = g.index == 0 and ctx.body is None
    outer_max = number("outer_panel_height_max_ref_frac") * ctx.req.ref_length_m
    # the cap changes the clipped semi-span, so the placement chord must come from a capped build
    dry = build_planform(g, area_m2, (0.0, y0, 0.0), lim, mitre, outer_max)
    if g.join_target is None:
        mid, fits = _root(g, (dry.root_chord_m, g.cant_deg), ctx, pod, built.get(0), placed)
        return build_planform(g, area_m2, mid, lim, mitre, outer_max), fits
    if g.join_target not in built:
        return None, True
    target = built[g.join_target]
    tip = target.sections[target.n_structural - 1]
    root_m = (joined_root_chord(g, area_m2), _joined_cant_estimate(tip.le_m, y0, ctx))
    mid, fits = _root(g, root_m, ctx, pod, built.get(0), placed)
    return build_joined(g, area_m2, mid, tip), fits


def _pod(g0: SurfaceGenes, s0: Planform, ctx: Context, placed: Placed) -> tuple[Body | None, bool]:
    """Right pod on surface0 with surface0's chord there registered as a pod root; failed flag."""
    placement = place_pod(
        (s0.sections[0], s0.sections[1]), ctx.fam, ctx.req, ctx.genes, ctx.pod_y_bounds
    )
    if placement is None:
        return None, True
    pod, s = placement
    a, b = s0.sections[0], s0.sections[1]
    le_x = a.le_m[0] + s * (b.le_m[0] - a.le_m[0])
    chord = a.chord_m + s * (b.chord_m - a.chord_m)
    placed[pod.name] = [(le_x, le_x + chord, g0.cant_deg, 0)]
    return pod, s > 1.0


def build_layout(specs: dict[int, SurfaceGenes], scale_m2: float, ctx: Context) -> Layout:
    """All planforms at true area scale_m2 x weight, plus the pod on surface0."""
    built: dict[int, Planform] = {}
    failed: list[int] = []
    long_roots: list[int] = []
    capped: list[int] = []
    placed: Placed = {}
    pod: Body | None = None
    pod_failed = False
    for i in sorted(specs):
        g = scaled_spec(specs[i], ctx.fin_scale)
        area = scale_m2 * g.weight
        if g.area_share is not None:
            area = g.area_share * ctx.lifting_area_m2
        if is_fin_spec(g):
            fin_area = _fin_capped_area(g, area, ctx, (0.0, _root_y(g, pod), 0.0))
            capped += [i] if fin_area < area else []
            area = fin_area
        if scales_by_area(g):
            area *= ctx.fin_scale
        pf, fits = _build_one(g, area, ctx, pod, built, placed)
        if not fits:
            long_roots.append(i)
        if pf is None or (g.join_target is not None and join_too_swept(pf)):
            failed.append(i)
        if pf is None:
            continue
        built[i] = pf
        root = pf.sections[0]
        cant = g.cant_deg if g.join_target is None else root.cant_deg
        here = placed.setdefault(host_name(g, ctx, pod), [])
        here.append((root.le_m[0], root.le_m[0] + root.chord_m, cant, i))
        if i == 0 and ctx.pods_on:
            pod, pod_failed = _pod(g, pf, ctx, placed)
    return Layout(
        built, pod, tuple(failed), pod_failed, tuple(long_roots), _crowded(placed), tuple(capped)
    )


def _projected_total(specs: dict[int, SurfaceGenes], layout: Layout) -> float:
    """Sum of projected areas of all built surfaces [m^2]."""
    return sum(surface_area(layout.planforms[i], specs[i])[1] for i in layout.planforms)


def solve_layout(specs: dict[int, SurfaceGenes], ctx: Context) -> Layout:
    """Layout whose projected lifting area equals S; its verdicts hold at that scale only."""
    target = ctx.lifting_area_m2
    unit = build_layout(specs, 1.0, ctx)
    if ctx.pods_on and unit.pod is None:
        return unit
    proj = {i: surface_area(unit.planforms[i], specs[i])[1] for i in unit.planforms}
    plain = [i for i, g in specs.items() if g.join_target is None and not is_fin_spec(g)]
    free = sum(proj[i] for i in plain)
    k_hi = target / free  # upper bracket: joins and fins only add projected area
    fin_proj = sum(proj[i] for i, g in specs.items() if is_fin_spec(g))
    linear = fin_proj <= number("area_solve_rel_tol") * free  # centreline fins project ~0
    if linear and all(g.join_target is None for g in specs.values()):
        lay = build_layout(specs, k_hi, ctx)
        # the outer-panel height cap is absolute, so planforms are not similar across scales
        if abs(_projected_total(specs, lay) - target) <= number("area_check_rel_tol") * target:
            return lay
    return _illinois(specs, ctx, k_hi * number("area_solve_bracket_frac"), k_hi)


def _illinois(
    specs: dict[int, SurfaceGenes], ctx: Context, k_lo_m2: float, k_hi_m2: float
) -> Layout:
    """Regula falsi (Illinois) on the area scale [m^2]; joined areas make the sum nonlinear."""
    target = ctx.lifting_area_m2
    tol = number("area_solve_rel_tol") * target
    k_lo, k_hi = k_lo_m2, k_hi_m2

    def residual(k_m2: float) -> tuple[float, Layout]:
        lay = build_layout(specs, k_m2, ctx)
        return _projected_total(specs, lay) - target, lay

    f_lo, lay = residual(k_lo)
    f_hi, lay = residual(k_hi)
    max_iter = int(number("area_solve_max_iter"))
    for _ in range(max_iter):  # the height cap can leave the unit-scale estimate k_hi too low
        if f_hi >= 0:
            break
        k_lo, f_lo = k_hi, f_hi
        k_hi *= number("area_solve_bracket_growth")
        f_hi, lay = residual(k_hi)
    side = 0
    for _ in range(max_iter):
        k = (k_lo * f_hi - k_hi * f_lo) / (f_hi - f_lo) if f_hi != f_lo else k_hi
        f, lay = residual(k)
        if abs(f) <= tol:
            return lay
        if (f > 0) == (f_hi > 0):
            k_hi, f_hi = k, f
            f_lo = f_lo / 2 if side == 1 else f_lo
            side = 1
        else:
            k_lo, f_lo = k, f
            f_hi = f_hi / 2 if side == -1 else f_hi
            side = -1
    return lay
