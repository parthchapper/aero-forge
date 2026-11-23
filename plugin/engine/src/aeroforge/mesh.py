"""Named solids -> per-part triangle meshes -> trimesh Scene, GLB and STL."""

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps

from aeroforge.constants import number

MIRROR_Y = np.array([1.0, -1.0, 1.0])
KIND_PREFIXES = (
    ("surface", "surface"),
    ("body", "body"),
    ("pod", "pod"),
    ("prop", "prop"),
    *(
        (p, "internal")
        for p in ("spar", "joiner", "hinge", "servo", "mount", "bay", "hatch", "recovery")
    ),
)


class MeshError(RuntimeError):
    """A part could not be tessellated to a mesh matching its solid."""


@dataclass(frozen=True)
class PartMesh:
    """Triangle mesh of one named part [m]; vertices are not shared across CAD faces."""

    name: str
    kind: str
    vertices: np.ndarray
    faces: np.ndarray


def part_kind(name: str) -> str:
    """Part family from its stable name prefix."""
    for prefix, kind in KIND_PREFIXES:
        if name.startswith(prefix):
            return kind
    raise MeshError(f"unknown part name {name!r}")


def solid_volume_m3(solid: cq.Solid) -> float:
    """Adaptive OCC volume [m^3]; plain Solid.Volume() under-reports ruled lofts."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props, number("volume_integration_eps"))
    return float(props.Mass())


def _signed_volume_m3(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Volume [m^3] enclosed by a closed, outward-oriented triangle soup."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _tessellate_part(name: str, solid: cq.Solid, exact_m3: float | None = None) -> PartMesh:
    """Mesh at a tolerance scaled to the part's cross-section, sqrt(volume / diagonal), so thin
    wings get as fine a mesh as fat bodies; OCC's mesher sometimes folds an apex loft at a given
    tolerance, so a mesh whose volume disagrees with the solid is redone finer. exact_m3 is a
    volume known in closed form (a flat-panel body's polyhedron), used instead of the solid's."""
    exact = solid_volume_m3(solid) if exact_m3 is None else exact_m3
    diagonal_m = solid.BoundingBox().DiagonalLength
    if part_kind(name) == "prop":
        # Thin blades would drive the section rule to ~0.1 mm; prop volume barely depends on it.
        tol = number("tess_prop_tolerance_frac") * diagonal_m
    else:
        tol = number("tess_tolerance_frac") * float(np.sqrt(abs(exact) / diagonal_m))
    angular = number("tess_angular_rad")
    for _ in range(int(number("tess_attempts"))):
        BRepTools.Clean_s(solid.wrapped)  # OCC otherwise reuses the first triangulation
        verts, tris = solid.tessellate(tol, angular)
        vertices = np.array([v.toTuple() for v in verts], dtype=np.float64)
        faces = np.array(tris, dtype=np.int64).reshape(-1, 3)
        if len(faces) and abs(_signed_volume_m3(vertices, faces) - exact) <= number(
            "tess_volume_rel_tol"
        ) * abs(exact):
            return PartMesh(name, part_kind(name), vertices, faces)
        tol *= number("tess_retry_factor")
        angular *= number(
            "tess_retry_factor"
        )  # flat boxy bodies sag at the corners, not the chords
    raise MeshError(f"{name}: no tessellation matched the solid volume")


def _mirrored(mesh: PartMesh, name: str) -> PartMesh:
    """Mesh of the y-mirrored twin: y negated, winding reversed to keep normals outward."""
    return PartMesh(name, mesh.kind, mesh.vertices * MIRROR_Y, mesh.faces[:, ::-1].copy())


def tessellate(
    parts: dict[str, cq.Solid], exact_m3: dict[str, float] | None = None
) -> list[PartMesh]:
    """Volume-checked meshes of all parts, sorted by name; `_L` parts mirror their `_R` mesh.
    exact_m3 gives closed-form volumes [m^3] by part name (see geom.exact_volumes_m3)."""
    known = exact_m3 or {}
    meshes = {
        n: _tessellate_part(n, parts[n], known.get(n))
        for n in parts
        if not _has_right_twin(n, parts)
    }
    for n in parts:
        if _has_right_twin(n, parts):
            meshes[n] = _mirrored(meshes[n[:-1] + "R"], n)
    return [meshes[n] for n in sorted(meshes)]


def _has_right_twin(name: str, parts: dict[str, cq.Solid]) -> bool:
    """A `_L` part whose `_R` twin is meshed and mirrored instead."""
    return name.endswith("_L") and name[:-1] + "R" in parts


def export_mesh(mesh: PartMesh, merge: bool) -> trimesh.Trimesh:
    """Trimesh without zero-area apex triangles (OCC loft vertices); merge closes seams."""
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.update_faces(tm.nondegenerate_faces())
    if merge:
        tm.merge_vertices()
        tm.update_faces(tm.nondegenerate_faces())
    return tm


def to_scene(meshes: list[PartMesh]) -> trimesh.Scene:
    """Scene with one node and geometry per part, both named after the part (creases kept)."""
    scene = trimesh.Scene()
    for m in meshes:
        scene.add_geometry(export_mesh(m, merge=False), node_name=m.name, geom_name=m.name)
    return scene


def glb_bytes(meshes: list[PartMesh]) -> bytes:
    """Binary glTF of the assembly."""
    return to_scene(meshes).export(file_type="glb")


def stl_bytes(mesh: PartMesh) -> bytes:
    """Binary STL of one part, watertight."""
    return export_mesh(mesh, merge=True).export(file_type="stl")


def write_glb(path: Path, meshes: list[PartMesh]) -> None:
    """Write assembly.glb."""
    path.write_bytes(glb_bytes(meshes))


def write_stls(directory: Path, meshes: list[PartMesh]) -> None:
    """Write <part>.stl for every part into directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for m in meshes:
        (directory / f"{m.name}.stl").write_bytes(stl_bytes(m))


def read_glb(path: Path) -> list[PartMesh]:
    """Part meshes back from assembly.glb (node names are the part names), sorted by name."""
    scene = trimesh.load(path, force="scene")
    out = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        vertices = trimesh.transformations.transform_points(geom.vertices, transform)
        out.append(PartMesh(node, part_kind(node), np.asarray(vertices), np.asarray(geom.faces)))
    return sorted(out, key=lambda m: m.name)
