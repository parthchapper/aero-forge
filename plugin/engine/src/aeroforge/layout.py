"""Bodies, internal boxes, pods and propellers placed around the surfaces."""

from itertools import product

import numpy as np

from aeroforge.airfoil import naca_outline
from aeroforge.constants import constant, number, numbers
from aeroforge.facet_metrics import end_radius_m
from aeroforge.facet_query import fit_scale
from aeroforge.faceted_body import faceted_body, tail_radius_frac
from aeroforge.facets import FacetError, axis_rise_m, facet_body
from aeroforge.models import Body, Box, Family, Propulsor, Requirements, Section, Vec3
from aeroforge.pod_fit import pod_for_prop
from aeroforge.shapes import axis_z_at_x, body_axis_z, body_radius_frac, inside_loft


def _vec(a: np.ndarray) -> Vec3:
    """numpy 3-vector → model tuple."""
    return (float(a[0]), float(a[1]), float(a[2]))


def box_sizes(req: Requirements) -> dict[str, Vec3]:
    """Envelope [l, w, h] of payload, battery and avionics boxes [m]."""
    volume = req.battery_wh / number("battery_energy_density_wh_per_m3")
    prop = np.array(numbers("battery_box_proportions"))
    battery = prop * (volume / float(np.prod(prop))) ** (1 / 3)
    return {
        "payload": req.payload_box_m,
        "battery": _vec(battery),
        "avionics": _vec(np.array(numbers("avionics_box_m"))),
    }


def _box_order() -> list[str]:
    """Front-to-back stacking order of boxes."""
    order = constant("constants", "box_order")
    if not isinstance(order, list):
        raise TypeError("box_order must be a list")
    return [str(name) for name in order]


def _stack_length(sizes: dict[str, Vec3]) -> float:
    """Length of all boxes stacked along x with gaps [m]."""
    return sum(s[0] for s in sizes.values()) + (len(sizes) + 1) * number("box_gap_m")


def arrange_boxes(sizes: dict[str, Vec3], layout: str) -> tuple[Vec3, dict[str, Vec3]]:
    """Envelope [l, w, h] of the boxes and each box centre's offset from the envelope centre [m].

    inline stacks along x (wing_root packs like inline: it only applies once the wing cannot hold
    the boxes), side_by_side lines them up across y, stacked piles them up along z. Gaps sit
    between boxes and at both ends of the packing axis (user, novelty plan step 10).
    """
    gap, order = number("box_gap_m"), _box_order()
    axis = {"side_by_side": 1, "stacked": 2}.get(layout, 0)
    env = [max(sizes[n][k] for n in order) for k in range(3)]
    env[axis] = sum(sizes[n][axis] for n in order) + (len(order) + 1) * gap
    if axis != 0:
        env[0] += 2 * gap
    offsets, at = {}, -env[axis] / 2 + gap
    for name in order:
        along = sizes[name][axis]
        off = [0.0, 0.0, 0.0]
        off[axis] = at + along / 2
        offsets[name] = _vec(np.array(off))
        at += along + gap
    return _vec(np.array(env)), offsets


def _section_axes(half_w: float, half_h: float, ratio: float) -> tuple[float, float]:
    """Half axes with width / height multiplied by ratio: the body grows (wider, or taller when
    ratio < 1) so the boxes inside still fit; ratio 1 returns the inputs unchanged."""
    if ratio == 1.0:
        return half_w, half_h
    aspect = half_w / half_h * ratio
    return (aspect * half_h, half_h) if ratio > 1.0 else (half_w, half_w / aspect)


def _box_corners(req: Requirements, layout: str, x_centre_m: float) -> np.ndarray:
    """All corners of the packed boxes with the packing centred at x_centre_m on the axis [m]."""
    sizes = box_sizes(req)
    _, offsets = arrange_boxes(sizes, layout)
    corners = []
    for name, off in offsets.items():
        half = np.array(sizes[name]) / 2
        centre = np.array([x_centre_m, 0.0, 0.0]) + np.array(off)
        corners += [centre + half * np.array(sign) for sign in product((-1.0, 1.0), repeat=3)]
    return np.array(corners)


def _faceted_fit(
    fam: Family, req: Requirements, layout: str, length_m: float, axes_m: tuple[float, float]
) -> float:
    """Even cross-section scale at which the packed boxes fit a faceted body of this length and
    base axes (closed form, facets.fit_scale)."""
    poly = facet_body(fam.shape, length_m, axes_m)
    corners = _box_corners(req, layout, fam.shape.max_section_x_frac * length_m)
    # an upswept body is the level one sheared in z: fit the boxes relative to the axis, with the
    # wall margin a smooth body keeps (it also leaves room for the hatch above the boxes)
    t = corners[:, 0] / length_m
    corners[:, 2] -= axis_rise_m(t, fam.shape, length_m, fam.body_tail_upsweep_deg)
        "hatch_thickness_m"
    )  # the hatch lies on the top boxes
    return fit_scale(poly, corners) * (1.0 + number("body_wall_margin_frac"))


def size_faceted_body(
    req: Requirements, fam: Family, floor_m: float, holds_boxes: bool
) -> tuple[float, tuple[float, float]]:
    """(length, half-width and half-height) [m] of a faceted body at least floor_m long, at the
    fineness (length / max diameter, measured on the inline packing's fit: decision B1). It holds
    the boxes' packing when holds_boxes: the cross-section never drops below the closed-form fit."""
    ratio, fineness = fam.body_width_ratio, fam.body_fineness
    if not holds_boxes:
        axes = _section_axes(1.0, 1.0, ratio)
        k = floor_m / (fineness * 2 * max(axes))
        return floor_m, (axes[0] * k, axes[1] * k)
    env, _ = arrange_boxes(box_sizes(req), fam.box_layout)
    axes = _section_axes(env[1] / 2, env[2] / 2, ratio)
    inline, _ = arrange_boxes(box_sizes(req), "inline")
    # a stack centred on the widest station must lie within the inner part of the shorter side
    shorter = min(fam.shape.max_section_x_frac, 1.0 - fam.shape.max_section_x_frac)
    length = max(
        floor_m, max(env[0], inline[0]) / (2.0 * number("facet_box_length_frac") * shorter)
    )
    in_axes = _section_axes(inline[1] / 2, inline[2] / 2, ratio)
    # lengthen until the boxes fit at all and the inline packing's fit reads at most the fineness
    # (B1); then the packing in use fits too, and the section widens to the fineness exactly
    cap_m = number("facet_length_max_ref_mult") * req.ref_length_m
    while True:
        if length > cap_m:
            raise FacetError(f"the boxes need a faceted body longer than {cap_m:.2f} m")
        k_in = _faceted_fit(fam, req, "inline", length, in_axes)
        k_fit = _faceted_fit(fam, req, fam.box_layout, length, axes)
        if np.isfinite(k_fit) and fineness * 2 * max(in_axes) * k_in <= length:
            break
        length *= number("facet_length_growth")
    k = max(k_fit, length / (fineness * 2 * max(axes)))
    return length, (axes[0] * k, axes[1] * k)


def size_centre_body(
    req: Requirements,
    fam: Family,
    length_frac: float,
    min_length_m: float,
    holds_boxes: bool = True,
) -> Body:
    """Centre body at the origin, at least min_length_m long, at the fineness (length / max
    diameter, after the width ratio).

    It holds the boxes' envelope unless they ride in the wing (holds_boxes False): then it is a
    plain body of the fineness and the length floors. The fineness is measured on the diameter
    inline packing would need (user decision B1, upgrade U0), so side_by_side / stacked bodies
    are not lengthened by their wider packing; their section never shrinks below the box fit.
    """
    nose_frac, tail_frac = fam.body_nose_frac, fam.body_tail_frac
    floor_m = max(length_frac * req.ref_length_m, min_length_m)
    if fam.shape.style == "faceted":
        length, axes = size_faceted_body(req, fam, floor_m, holds_boxes)
        return faceted_body(fam, length, axes, "body0", fam.body_tail_upsweep_deg)
    if holds_boxes:
        env, _ = arrange_boxes(box_sizes(req), fam.box_layout)
        fit = 2 ** (1 / fam.body_section_exp) * (1 + number("body_wall_margin_frac"))
        half_w, half_h = _section_axes(fit * env[1] / 2, fit * env[2] / 2, fam.body_width_ratio)
        floor_m = max(floor_m, env[0] / (1 - nose_frac - tail_frac))
        inline, _ = arrange_boxes(box_sizes(req), "inline")
        ref_w, ref_h = _section_axes(fit * inline[1] / 2, fit * inline[2] / 2, fam.body_width_ratio)
    else:
        half_w, half_h = ref_w, ref_h = _section_axes(1.0, 1.0, fam.body_width_ratio)
    min_diameter = 2 * max(half_w, half_h)
    ref_diameter = 2 * max(ref_w, ref_h)
    length = max(floor_m, fam.body_fineness * ref_diameter) if holds_boxes else floor_m
    grow = length / (fam.body_fineness * min_diameter)
    if holds_boxes:
        grow = max(1.0, grow)  # a packed section never drops below the box fit
    return Body(
        name="body0",
        mirrored=False,
        nose_m=(0.0, 0.0, 0.0),
        length_m=length,
        half_width_m=half_w * grow,
        half_height_m=half_h * grow,
        section_exp=fam.body_section_exp,
        nose_frac=nose_frac,
        tail_frac=tail_frac,
        tail_radius_frac=tail_radius_frac(fam),
        tail_upsweep_deg=fam.body_tail_upsweep_deg,
    )


def boxes_in_body(
    body: Body, req: Requirements, layout: str, widest_x_frac: float | None = None
) -> tuple[Box, ...]:
    """Boxes packed by layout around the centre of the body's constant-section middle; on a
    flat-panel body (widest_x_frac = the max_section_x_frac gene) around its widest station, on
    the axis there."""
    sizes = box_sizes(req)
    _, offsets = arrange_boxes(sizes, layout)
    if body.facets is not None:
        if widest_x_frac is None:
            raise FacetError("a flat-panel body needs the widest station to place its boxes")
        mid_x = body.nose_m[0] + body.length_m * widest_x_frac
        centre = np.array([mid_x, body.nose_m[1], axis_z_at_x(body, mid_x)])
    else:
        mid_x = body.nose_m[0] + body.length_m * (body.nose_frac + (1 - body.tail_frac)) / 2
        centre = np.array([mid_x, body.nose_m[1], body.nose_m[2]])
    return tuple(
        Box(
            name=name,
            host=body.name,
            centre_m=_vec(centre + np.array(offsets[name])),
            size_m=sizes[name],
        )
        for name in _box_order()
    )


def _contour_band(
    root: Section, fam: Family, tc: float, x0_m: float, x1_m: float
) -> tuple[float, float]:
    """(highest lower-surface z, lowest upper-surface z) [m] of the root section over x0_m..x1_m."""
    xy = naca_outline(
        fam.camber_frac, fam.camber_loc_frac, tc, int(number("airfoil_points_per_side"))
    )
    split = len(xy) // 2
    upper, lower = xy[: split + 1][::-1], xy[split:]
    n = int(number("box_contour_samples"))
    xc = np.linspace(x0_m - root.le_m[0], x1_m - root.le_m[0], n) / root.chord_m
    z_lo = root.le_m[2] + root.chord_m * float(np.interp(xc, lower[:, 0], lower[:, 1]).max())
    z_hi = root.le_m[2] + root.chord_m * float(np.interp(xc, upper[:, 0], upper[:, 1]).min())
    return z_lo, z_hi


def box_lattice(box: Box, n_per_axis: int) -> np.ndarray:
    """n_per_axis^3 points spanning the box, corners included [m]."""
    u = np.linspace(-0.5, 0.5, n_per_axis)
    grid = np.stack(np.meshgrid(u, u, u, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(box.centre_m) + grid * np.asarray(box.size_m)


def boxes_in_wing(
    sections: tuple[Section, ...], fam: Family, tc: float, req: Requirements
) -> tuple[Box, ...] | None:
    """Boxes inside surface0's mitred centre section across their full width, else None.

    sections are surface0's structural sections; z is centred in the root contour band and the
    whole box is then checked against the real ruled loft of both halves.
    """
    root, semi_y_m = sections[0], sections[-1].le_m[1]
    n_af = int(number("airfoil_points_per_side"))
    xy = naca_outline(fam.camber_frac, fam.camber_loc_frac, tc, n_af)
    n_lattice = int(number("box_lattice_points"))
    sizes = box_sizes(req)
    if _stack_length(sizes) > number("wing_box_chord_frac") * root.chord_m:
        return None
    if max(s[1] for s in sizes.values()) > number("wing_box_span_frac") * 2 * semi_y_m:
        return None
    gap = number("box_gap_m")
    x = root.le_m[0] + 0.5 * root.chord_m - _stack_length(sizes) / 2 + gap
    boxes = []
    for name in _box_order():
        size = sizes[name]
        z_lo, z_hi = _contour_band(root, fam, tc, x, x + size[0])
        if size[2] > number("wing_box_thickness_frac") * (z_hi - z_lo):
            return None
        box = Box(
            name=name,
            host="surface0",
            centre_m=(x + size[0] / 2, 0.0, (z_lo + z_hi) / 2),
            size_m=size,
        )
        if not inside_loft(sections, xy, box_lattice(box, n_lattice), mirrored=True).all():
            return None
        boxes.append(box)
        x += size[0] + gap
    return tuple(boxes)


def place_pod(
    inner: tuple[Section, Section],
    fam: Family,
    req: Requirements,
    genes: dict[str, float],
    y_bounds: tuple[float, float],
    mount: tuple[float, bool] | None = None,
) -> tuple[Body, float] | None:
    """Right pod within y_bounds on the line of surface0's inner panel, and its panel fraction.

    A fraction above 1 lies past the panel end (judged by the caller at the final scale);
    None when no scale can fit a pod.
    """
    root, outer = (np.asarray(s.le_m) for s in inner)
    y_lo, y_hi = y_bounds
    if y_lo > y_hi or outer[1] - root[1] < number("pod_panel_min_y_m"):
        return None
    y_want = root[1] + genes["pod_y_frac"] * (outer[1] - root[1])
    s = (float(np.clip(y_want, y_lo, y_hi)) - root[1]) / (outer[1] - root[1])
    le = root + s * (outer - root)
    length = genes["pod_length_frac"] * req.ref_length_m
    fam, radius = pod_for_prop(fam, length, length / (2 * fam.body_fineness), mount)
    nose = (float(le[0] - genes["pod_x_frac"] * length), float(le[1]), float(le[2]))
    if fam.shape.style == "faceted":
        body = faceted_body(fam, length, (radius, radius), "pod")
        return body.model_copy(update={"mirrored": True, "nose_m": nose}), s
    pod = Body(
        name="pod",
        mirrored=True,
        nose_m=nose,
        length_m=length,
        half_width_m=radius,
        half_height_m=radius,
        section_exp=fam.body_section_exp,
        nose_frac=number("body_nose_frac"),
        tail_frac=number("body_tail_frac"),
        tail_radius_frac=tail_radius_frac(fam),
    )
    return pod, s


def host_end(body: Body, pusher: bool) -> tuple[np.ndarray, float]:
    """Attach point at the nose (tractor) or tail (pusher) of a body, and the local radius [m]: of
    the tail cap of a smooth body, of the end section of a flat-panel one (0 where it closes to a
    point or an edge)."""
    nose = np.asarray(body.nose_m)
    if not pusher:
        return nose, 0.0
    if body.facets is not None:
        radius = end_radius_m(body.facets, body.length_m)
    else:
        radius = float(body_radius_frac(body, np.array([1.0]))[0]) * min(
            body.half_width_m, body.half_height_m
        )
    if body.facets is not None:
        rise = axis_z_at_x(body, body.nose_m[0] + body.length_m) - body.nose_m[2]
    else:
        rise = float(body_axis_z(body, np.array([1.0]))[0]) - body.nose_m[2]
    return nose + np.array([body.length_m, 0.0, rise]), radius


def place_prop(
    name: str,
    host: str,
    mirrored: bool,
    end: tuple[np.ndarray, float],
    diameter_m: float,
    pusher: bool,
    genes: dict[str, float],
    obstacles: np.ndarray,
    ground_line: bool,
    motor_m: np.ndarray | None = None,
) -> Propulsor:
    """Hub off the host end, raised clear of the ground line if any, slid clear of obstacles; the
    shaft starts at motor_m (the firewall, forward of a slim tail end) when given."""
    attach, end_radius = end
    margin = number("prop_clearance_m")
    # Clearances are kept from sampled obstacles plus a pad covering the gaps between samples.
    padded = margin + number("prop_sample_pad_m")
    gap_lo, gap_hi = numbers("prop_gap_frac")
    sign = 1.0 if pusher else -1.0
    gap = margin + (gap_lo + genes["prop_gap_frac"] * (gap_hi - gap_lo)) * diameter_m
    hub = attach + np.array([sign * gap, 0.0, genes["prop_z_frac"] * end_radius])
    radius = diameter_m / 2
    if ground_line:
        hub[2] = max(hub[2], float(obstacles[:, 2].min()) + padded + radius)
    rho = np.hypot(obstacles[:, 1] - hub[1], obstacles[:, 2] - hub[2])
    near = obstacles[rho <= radius + padded, 0]
    if near.size:
        hub[0] = max(hub[0], near.max() + padded) if pusher else min(hub[0], near.min() - padded)
    return Propulsor(
        name=name,
        host=host,
        mirrored=mirrored,
        hub_m=_vec(hub),
        attach_m=_vec(attach if motor_m is None else motor_m),
        diameter_m=diameter_m,
        pusher=pusher,
    )
