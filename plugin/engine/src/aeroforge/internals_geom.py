"""Internals -> one named CadQuery solid per instance (metres); separate from the outer mould."""

import cadquery as cq
import numpy as np

from aeroforge.geom import GeometryError, checked_solid
from aeroforge.internals import Internal
from aeroforge.models import instance_names


def _v(p: object) -> cq.Vector:
    """cq.Vector from a 3-sequence."""
    x, y, z = map(float, p)  # type: ignore[call-overload]
    return cq.Vector(x, y, z)


def _cylinder(r_m: float, a: np.ndarray, b: np.ndarray) -> cq.Solid:
    """Solid cylinder of radius r_m from a to b."""
    d = b - a
    return cq.Solid.makeCylinder(r_m, float(np.linalg.norm(d)), pnt=_v(a), dir=_v(d))


def _tube(it: Internal) -> cq.Shape:
    """Hollow (id_m > 0) or solid rod along the path; inner kinks overlap by one radius."""
    path = np.asarray(it.path_m)
    pieces = []
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        d = (b - a) / np.linalg.norm(b - a)
        a_ext = a - d * it.od_m / 2 if k > 0 else a
        b_ext = b + d * it.od_m / 2 if k < len(path) - 2 else b
        outer = _cylinder(it.od_m / 2, a_ext, b_ext)
        if it.id_m > 0.0:
            outer = outer.cut(_cylinder(it.id_m / 2, a_ext - d * it.od_m, b_ext + d * it.od_m))
        pieces.append(outer)
    shape = pieces[0]
    return shape.fuse(*pieces[1:]) if len(pieces) > 1 else shape


def _located(solid: cq.Shape, it: Internal) -> cq.Shape:
    """Solid built about the origin in the item's (length, width, height) frame, moved in place."""
    ax = it.axes
    plane = cq.Plane(origin=_v(it.centre_m), xDir=_v(ax[0]), normal=_v(ax[2]))
    return solid.moved(cq.Location(plane))


def _box(it: Internal) -> cq.Shape:
    """Box, open-top tray or ring frame centred on the item."""
    length, width, height = it.size_m
    outer = cq.Solid.makeBox(
        length, width, height, pnt=cq.Vector(-length / 2, -width / 2, -height / 2)
    )
    if it.hollow != "solid":
        w = it.wall_m
        lift = w if it.hollow == "open_top" else -height
        inner = cq.Solid.makeBox(
            length - 2 * w,
            width - 2 * w,
            3 * height,
            pnt=cq.Vector(-length / 2 + w, -width / 2 + w, -height / 2 + lift),
        )
        outer = outer.cut(inner)
    return _located(outer, it)


def _disc(it: Internal) -> cq.Shape:
    """Plate disc normal to axes[0] with through holes at the bolt centres."""
    thick, radius = it.size_m[0], it.size_m[1]
    n = np.asarray(it.axes[0])
    c = np.asarray(it.centre_m)
    plate: cq.Shape = _cylinder(radius, c - n * thick / 2, c + n * thick / 2)
    for h in it.holes_m:
        p = np.asarray(h)
        plate = plate.cut(_cylinder(it.hole_d_m / 2, p - n * thick, p + n * thick))
    return plate


BUILDERS = {"tube": _tube, "box": _box, "disc": _disc}


def internal_parts(items: tuple[Internal, ...]) -> dict[str, cq.Solid]:
    """Every internal instance by stable name (`spar_surface0_R`, `bay_battery`, ...)."""
    parts: dict[str, cq.Solid] = {}
    for it in items:
        names = instance_names(it.name, it.mirrored)
        try:
            solid = checked_solid(names[-1], BUILDERS[it.shape](it).clean())
        except GeometryError:
            raise
        except Exception as exc:  # OCC raises untyped errors; surface them as the typed one.
            raise GeometryError(it.name, f"{type(exc).__name__}: {exc}") from exc
        parts[names[-1]] = solid
        if it.mirrored:
            parts[names[0]] = checked_solid(names[0], solid.mirror("XZ"))
    return dict(sorted(parts.items()))
