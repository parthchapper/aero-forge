"""Design -> one named CadQuery solid per part instance (metres)."""

import logging

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.gp import gp_Pnt

from aeroforge.constants import number
from aeroforge.models import Body, Design, Propulsor, Surface, instance_names
from aeroforge.shapes import axis_sign, blade_rings, body_rings, body_stations, section_outline

log = logging.getLogger(__name__)


class GeometryError(RuntimeError):
    """A part could not be built as one valid solid."""

    def __init__(self, part: str, reason: str) -> None:
        super().__init__(f"{part}: {reason}")
        self.part = part
        self.reason = reason


def _wire(points: np.ndarray) -> cq.Wire:
    """Closed periodic spline wire through points [m]."""
    vectors = [cq.Vector(*map(float, p)) for p in points]
    return cq.Wire.assembleEdges([cq.Edge.makeSpline(vectors, periodic=True)])


def checked_solid(part: str, shape: cq.Shape) -> cq.Solid:
    """The single valid solid in shape, else GeometryError."""
    solids = shape.Solids()
    if len(solids) != 1:
        raise GeometryError(part, f"expected 1 solid, got {len(solids)}")
    solid = solids[0]
    if not solid.isValid() or solid.Volume() <= 0.0:
        raise GeometryError(part, "invalid or empty solid")
    return solid


def surface_solid(surface: Surface) -> cq.Solid:
    """Ruled loft through all sections (exact for straight-tapered panels)."""
    n_af = int(number("airfoil_points_per_side"))
    loft = BRepOffsetAPI_ThruSections(True, True)
    # OCC's compatibility pass re-seats the seams of strongly canted sections and twists the
    # ruled panels (CHANGELOG Decisions 1b); our outlines already share one parametrisation.
    loft.CheckCompatibility(False)
    for s in surface.sections:
        loft.AddWire(_wire(section_outline(s, surface, n_af)).wrapped)
    loft.Build()
    if not loft.IsDone():
        raise GeometryError(surface.name, "surface loft failed")
    return checked_solid(surface.name, cq.Shape.cast(loft.Shape()))


def body_solid(body: Body) -> cq.Solid:
    """Loft from a nose vertex through super-ellipse rings to a flat tail cap."""
    stations = body_stations(int(number("body_loft_cad_stations")))[1:]
    rings = body_rings(body, stations, int(number("body_loft_ring_points")))
    loft = BRepOffsetAPI_ThruSections(True, False)
    loft.AddVertex(BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, body.nose_m))).Vertex())
    for ring in rings:
        loft.AddWire(_wire(ring).wrapped)
    loft.Build()
    if not loft.IsDone():
        raise GeometryError(body.name, "body loft failed")
    return checked_solid(body.name, cq.Shape.cast(loft.Shape()))


def prop_solid(prop: Propulsor) -> cq.Solid:
    """Two pitched blades, spinner cone and shaft fused into one solid."""
    d = prop.diameter_m
    hub = cq.Vector(*prop.hub_m)
    x_dir = cq.Vector(axis_sign(prop), 0.0, 0.0)
    blade = cq.Solid.makeLoft(
        [_wire(r) for r in blade_rings(prop, int(number("body_ring_points")))]
    )
    other = blade.rotate(hub, hub + cq.Vector(1.0, 0.0, 0.0), 180.0)
    spinner = cq.Solid.makeCone(
        number("spinner_radius_frac") * d,
        0.0,
        number("spinner_length_frac") * d,
        pnt=hub,
        dir=x_dir,
    )
    shape = blade.fuse(other, spinner)
    shaft_vec = hub - cq.Vector(*prop.attach_m)
    if shaft_vec.Length > 0.0:
        axis = shaft_vec.normalized()
        embed = number("shaft_embed_frac") * d
        start = cq.Vector(*prop.attach_m) - axis * embed
        shaft = cq.Solid.makeCylinder(
            number("shaft_radius_frac") * d, shaft_vec.Length + embed, pnt=start, dir=axis
        )
        shape = shape.fuse(shaft)
    return checked_solid(prop.name, shape.clean())


def build_parts(design: Design) -> dict[str, cq.Solid]:
    """All part instances by stable name (`surface0_L`, `body0`, `pod_R`, `prop1_L`, ...)."""
    builders: list[tuple[str, bool, cq.Solid]] = []
    try:
        builders += [(s.name, s.mirrored, surface_solid(s)) for s in design.surfaces]
        builders += [(b.name, b.mirrored, body_solid(b)) for b in design.bodies]
        builders += [(p.name, p.mirrored, prop_solid(p)) for p in design.propulsors]
    except GeometryError:
        raise
    except Exception as exc:  # OCC raises untyped errors; surface them as the typed one.
        raise GeometryError("design", f"{type(exc).__name__}: {exc}") from exc
    parts: dict[str, cq.Solid] = {}
    for name, mirrored, solid in builders:
        names = instance_names(name, mirrored)
        parts[names[-1]] = solid
        if mirrored:
            parts[names[0]] = checked_solid(names[0], solid.mirror("XZ"))
    log.debug("junctions: plain overlapping parts, no blends (CHANGELOG Deviations)")
    return dict(sorted(parts.items()))
