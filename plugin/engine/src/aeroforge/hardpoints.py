"""Design.hardpoints (SPEC §4, §10): every internal of a design, placed from Design fields only.

Bay frames wrap the boxes; hatches and the recovery plate are pushed against the host skin by
bisection on the analytic outer mould (never measured from meshes)."""

from collections.abc import Callable

import numpy as np

from aeroforge.constants import number, numbers
from aeroforge.facet_hatch import hatch_rect_xy
from aeroforge.facet_metrics import vertical_extent_m
from aeroforge.facet_query import slice_section
from aeroforge.internals import (
    Internal,
    contained,
    inside_part,
    probe_points,
    surface_items,
)
from aeroforge.models import Body, Box, Design, Propulsor, Recovery
from aeroforge.shapes import axis_z_at_x, body_radius_frac

Z_HAT = np.array([0.0, 0.0, 1.0])


def _outermost(fits: Callable[[float], bool], reach_m: float) -> float:
    """Largest offset in [0, reach_m] at which fits holds (0 when it fails already at 0)."""
    if not fits(0.0):
        return 0.0
    if fits(reach_m):
        return reach_m
    lo, hi = 0.0, reach_m
    for _ in range(int(number("outline_search_steps"))):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def _plate(
    design: Design, name: str, item: str, host: str, base: Internal, sign: float
) -> Internal:
    """base moved along sign * z until it touches the host's skin from inside."""

    def at(dz: float) -> Internal:
        c = np.asarray(base.centre_m) + sign * dz * Z_HAT
        return base.model_copy(update={"centre_m": (float(c[0]), float(c[1]), float(c[2]))})

    def fits(dz: float) -> bool:
        # the meshed skin sags inside the analytic one: keep a clearance so plates stay hidden
        return bool(inside_part(design, host, probe_points(at(dz + clearance))).all())

    clearance = number("plate_skin_clearance_m")

    reach = 2.0 * max(base.size_m) + _host_height(design, host)
    plate = at(_outermost(fits, reach))
    return plate.model_copy(update={"name": name, "item": item})


def _vec(a: np.ndarray) -> tuple[float, float, float]:
    """numpy 3-vector as a plain float tuple."""
    return (float(a[0]), float(a[1]), float(a[2]))


def _mount_base(design: Design, p: Propulsor, inset_m: float) -> Internal:
    """Firewall disc normal to x (not to the shaft: a raised, tilted shaft would swing the rim out
    of the host end), inset_m inside the host end along the shaft; radius 0 until sized."""
    axis = np.eye(3)[0] * (1.0 if p.pusher else -1.0)
    t = number("mount_plate_thickness_m")
    centre = np.asarray(p.attach_m) - axis * (inset_m + t / 2)
    host = next((b for b in design.bodies if b.name == p.host), None)
    if host is not None:  # a raised tail axis lies lower at the plate than at the attach point
        drop = axis_z_at_x(host, p.attach_m[0]) - axis_z_at_x(host, float(centre[0]))
        centre[2] -= drop
    u = np.cross(axis, Z_HAT)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.eye(3)[1]
    return Internal(
        name=f"mount_{p.name}",
        item="mount",
        owner=p.name,
        mirrored=p.mirrored,
        shape="disc",
        centre_m=_vec(centre),
        size_m=(t, 0.0, 0.0),
        axes=(_vec(axis), _vec(u), _vec(np.cross(axis, u))),
        hole_d_m=number("mount_bolt_d_m"),
        within=(p.host,),
    )


def _mount_radius(design: Design, p: Propulsor, base: Internal, full_m: float) -> float:
    """Largest radius (up to full_m) whose rim stays in the host; 0 when the centre is outside."""

    def fits(r_m: float) -> bool:
        sized = base.model_copy(update={"size_m": (base.size_m[0], r_m, 0.0)})
        return bool(inside_part(design, p.host, probe_points(sized)).all())

    return _outermost(fits, full_m)


def _mount(design: Design, p: Propulsor) -> Internal:
    """Firewall disc inside the host end with the motor bolt circle; its radius is the largest (up
    to mount_plate_radius_frac x D) whose rim stays in the host. On a flat-panel host a plate that
    cannot reach mount_plate_min_frac of the full radius keeps the full radius, so containment
    reports it: layout.firewall_attach already placed the motor inside a slim end, and a mount that
    still does not fit is a misfit, not a smaller plate. (Smooth hosts keep the 0.3 rule: any
    fitting plate; a centre outside the host keeps the full plate.)"""
    full = number("mount_plate_radius_frac") * p.diameter_m
    base = _mount_base(design, p, number("mount_plate_inset_m"))
    fitted = _mount_radius(design, p, base, full)
    host = next((b for b in design.bodies if b.name == p.host), None)
    small = fitted < number("mount_plate_min_frac") * full
    radius = full if small and host is not None and host.facets is not None else (fitted or full)
    centre = np.asarray(base.centre_m)
    _, u, v = (np.asarray(a) for a in base.axes)
    n = int(number("mount_bolt_count"))
    rb = number("mount_bolt_circle_frac") * radius
    angles = 2 * np.pi * np.arange(n) / n + np.pi / n
    holes = [centre + rb * (np.cos(a) * u + np.sin(a) * v) for a in angles]
    sized = base.model_copy(update={"size_m": (base.size_m[0], radius, 0.0)})
    return sized.model_copy(update={"holes_m": tuple(map(_vec, holes))})


def _host_height(design: Design, host: str) -> float:
    """Upper bound on the host's height [m] (bisection reach)."""
    for b in design.bodies:
        if b.name == host:
            return vertical_extent_m(b.facets) if b.facets is not None else 2.0 * b.half_height_m
    s = next(x for x in design.surfaces if x.name == host)
    return max(sec.chord_m for sec in s.sections)


def _hatch_footprint_m(design: Design, box: Box, top_m: float) -> tuple[float, float, float, float]:
    """Hatch centre x, y and length, width [m]: the bay footprint, on a flat-panel body cut back
    to the edges of the part of the roof panel that covers most of it and lies above the hatch
    (its top face at top_m + half a hatch thickness; the planes are exact, so no sag clearance).
    A hatch smaller than hatch_panel_min_frac of the bay footprint, or whose centre lies more than
    hatch_centre_tol_frac of the footprint off the bay's, is not a usable hatch: the whole
    footprint is returned then, so containment reports the misfit instead of a token hatch."""
    body = next((b for b in design.bodies if b.name == box.host), None)
    x, y = box.centre_m[0], box.centre_m[1]
    length, width = box.size_m[0], box.size_m[1]
    if body is None or body.facets is None:
        return x, y, length, width
    nose = np.asarray(body.nose_m)
    z_m = top_m + number("hatch_thickness_m") / 2 - nose[2]  # flat panels: no meshed-skin sag
    (cx, cy), (hx, hy) = hatch_rect_xy(
        body.facets, (x - nose[0], y - nose[1]), (length / 2, width / 2), z_m
    )
    usable = 4.0 * hx * hy >= number("hatch_panel_min_frac") * length * width
    tol = number("hatch_centre_tol_frac")
    centred = abs(cx - (x - nose[0])) <= tol * length and abs(cy - (y - nose[1])) <= tol * width
    if not (usable and centred):
        return x, y, length, width
    return cx + float(nose[0]), cy + float(nose[1]), 2.0 * hx, 2.0 * hy


def _bay_and_hatch(design: Design, box: Box) -> list[Internal]:
    """Open-top frame around the box and a hatch outline in the skin above it."""
    bay = Internal(
        name=f"bay_{box.name}",
        item="bay",
        owner=box.name,
        mirrored=False,
        shape="box",
        centre_m=box.centre_m,
        size_m=box.size_m,
        hollow="open_top",
        wall_m=number("bay_frame_wall_m"),
        within=(box.host,),
    )
    t = number("hatch_thickness_m")
    top = box.centre_m[2] + box.size_m[2] / 2 + t / 2
    cx, cy, length, width = _hatch_footprint_m(design, box, top)
    base = bay.model_copy(
        update={
            "centre_m": (cx, cy, top),
            "size_m": (length, width, t),
            "hollow": "ring",
            "wall_m": number("hatch_rim_m"),
        }
    )
    hatch = _plate(design, f"hatch_{box.name}", "hatch", box.host, base, 1.0)
    return [bay, hatch]


def _body_width_m(body: Body, x_m: float) -> float:
    """Full width [m] of the body's outer mould at world x_m."""
    if body.facets is not None:
        return slice_section(body.facets, x_m - body.nose_m[0])[0]
    r = float(body_radius_frac(body, np.array([(x_m - body.nose_m[0]) / body.length_m]))[0])
    return 2.0 * r * body.half_width_m


def _recovery(design: Design, recovery: Recovery) -> Internal:
    """Belly skid strip, skid / gear mount or (net) top hook plate under or over the box stack."""
    host = design.boxes[0].host  # the body or the wing that holds the boxes
    length, width, thick = numbers("recovery_plate_m")
    x = float(np.mean([b.centre_m[0] for b in design.boxes]))
    z = float(np.mean([b.centre_m[2] for b in design.boxes]))
    body = next((b for b in design.bodies if b.name == host), None)
    if body is not None:
        width = min(width, number("recovery_fill_frac") * _body_width_m(body, x))
        z = body.nose_m[2]
    base = Internal(
        name="recovery",
        item="recovery",
        owner=host,
        mirrored=False,
        shape="box",
        centre_m=(x, 0.0, z),
        size_m=(length, width, thick),
        within=(host,),
    )
    sign = 1.0 if recovery == "net" else -1.0
    return _plate(design, f"recovery_{recovery}", "recovery", host, base, sign)


def hardpoints(design: Design, recovery: Recovery) -> tuple[Internal, ...]:
    """All internals of the delivered design, sorted by name."""
    items = surface_items(design) + [_mount(design, p) for p in design.propulsors]
    for box in design.boxes:
        items += _bay_and_hatch(design, box)
    items.append(_recovery(design, recovery))
    return tuple(sorted(items, key=lambda it: it.name))


def containment_failures(design: Design, items: tuple[Internal, ...]) -> list[str]:
    """Names of internals whose section leaves the local outer mould."""
    return [it.name for it in items if not contained(design, it)]
