"""Design.hardpoints (SPEC §4, §10): every internal of a design, placed from Design fields only.

Bay frames wrap the boxes; hatches and the recovery plate are pushed against the host skin by
bisection on the analytic outer mould (never measured from meshes)."""

from collections.abc import Callable

import numpy as np

from aeroforge.constants import number, numbers
from aeroforge.internals import (
    Internal,
    contained,
    inside_part,
    probe_points,
    surface_items,
)
from aeroforge.models import Box, Design, Propulsor, Recovery
from aeroforge.shapes import body_radius_frac

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


def _mount(design: Design, p: Propulsor) -> Internal:
    """Firewall disc inside the host end (normal to x) with the motor bolt circle; its radius is
    the largest (up to mount_plate_radius_frac x D) whose rim stays in the host."""
    # normal to x, not to the shaft: a raised, tilted shaft would swing the rim out of the host end
    axis = np.eye(3)[0] * (1.0 if p.pusher else -1.0)
    t = number("mount_plate_thickness_m")
    centre = np.asarray(p.attach_m) - axis * (number("mount_plate_inset_m") + t / 2)
    u = np.cross(axis, Z_HAT)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.eye(3)[1]
    v = np.cross(axis, u)
    base = Internal(
        name=f"mount_{p.name}",
        item="mount",
        owner=p.name,
        mirrored=p.mirrored,
        shape="disc",
        centre_m=_vec(centre),
        size_m=(t, 0.0, 0.0),
        axes=(_vec(axis), _vec(u), _vec(v)),
        hole_d_m=number("mount_bolt_d_m"),
        within=(p.host,),
    )

    def sized(r_m: float) -> Internal:
        return base.model_copy(update={"size_m": (t, r_m, 0.0)})

    def fits(r_m: float) -> bool:
        return bool(inside_part(design, p.host, probe_points(sized(r_m))).all())

    full = number("mount_plate_radius_frac") * p.diameter_m
    # a centre outside the host keeps the full plate so containment reports it (and CAD works)
    radius = _outermost(fits, full) or full
    n = int(number("mount_bolt_count"))
    rb = number("mount_bolt_circle_frac") * radius
    angles = 2 * np.pi * np.arange(n) / n + np.pi / n
    holes = [centre + rb * (np.cos(a) * u + np.sin(a) * v) for a in angles]
    return sized(radius).model_copy(update={"holes_m": tuple(map(_vec, holes))})


def _host_height(design: Design, host: str) -> float:
    """Upper bound on the host's height [m] (bisection reach)."""
    for b in design.bodies:
        if b.name == host:
            return 2.0 * b.half_height_m
    s = next(x for x in design.surfaces if x.name == host)
    return max(sec.chord_m for sec in s.sections)


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
    base = bay.model_copy(
        update={
            "centre_m": (box.centre_m[0], box.centre_m[1], top),
            "size_m": (box.size_m[0], box.size_m[1], t),
            "hollow": "ring",
            "wall_m": number("hatch_rim_m"),
        }
    )
    hatch = _plate(design, f"hatch_{box.name}", "hatch", box.host, base, 1.0)
    return [bay, hatch]


def _recovery(design: Design, recovery: Recovery) -> Internal:
    """Belly skid strip, skid / gear mount or (net) top hook plate under or over the box stack."""
    host = design.boxes[0].host  # the body or the wing that holds the boxes
    length, width, thick = numbers("recovery_plate_m")
    x = float(np.mean([b.centre_m[0] for b in design.boxes]))
    z = float(np.mean([b.centre_m[2] for b in design.boxes]))
    body = next((b for b in design.bodies if b.name == host), None)
    if body is not None:
        r = float(body_radius_frac(body, np.array([(x - body.nose_m[0]) / body.length_m]))[0])
        width = min(width, 2.0 * number("recovery_fill_frac") * r * body.half_width_m)
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
