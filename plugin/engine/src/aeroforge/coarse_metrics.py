"""How coarse the faceted parts of a design are (SPEC §11, U3b): sheets, crease angles and face
directions, measured on the finished polyhedra, and the report.md section and decoder notes built
from them. Pure numpy; a sheet is a physical piece (coplanar faces merged: what the cut list and
the CAD solid hold), nothing is grouped further. Wing caps (root and tip) are not skin."""

from dataclasses import dataclass

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_aero import facet_drag_area_m2
from aeroforge.facet_decimate import min_crease_deg
from aeroforge.facet_pieces import PieceError, pieces
from aeroforge.facet_wing import _part_length, wing_geometry, wing_polyhedron
from aeroforge.facets import facet_body_info
from aeroforge.models import Body, Design, Polyhedron, Surface

SHARP_DEG = 10.0


@dataclass(frozen=True)
class Measure:
    """Coarseness of one faceted part. sheets: skin sheets of the whole part (a mirrored wing:
    one half; wing caps left out); upper / lower: the sheets of each side (normal up or down);
    creases_deg: the angle across every shared edge of two skin sheets; directions: 3 deg clusters
    of the sheet normals, in all and on the upper / lower side."""

    part: str
    kind: str
    sheets: int
    upper: int
    lower: int
    creases_deg: np.ndarray
    directions: int
    dir_upper: int
    dir_lower: int

    @property
    def side(self) -> int:
        """Sheets on the busier side."""
        return max(self.upper, self.lower)

    @property
    def dir_side(self) -> int:
        """Face directions on the busier side."""
        return max(self.dir_upper, self.dir_lower)


def _directions(normals: np.ndarray) -> int:
    reps: list[np.ndarray] = []
    tol = np.cos(np.radians(number("coarse_cluster_deg")))
    for n in normals:
        if not any(n @ r >= tol for r in reps):
            reps.append(n)
    return len(reps)


def measure(
    part: str, kind: str, poly: Polyhedron, cap_faces: tuple[int, ...] = ()
) -> Measure | None:
    """Measure of a polyhedron (sheets holding a face of cap_faces are caps), None when its
    sheets cannot be formed."""
    try:
        sheets = pieces(poly)
    except PieceError:
        return None
    skin = [i for i, s in enumerate(sheets) if not set(s.faces) & set(cap_faces)]
    normals = np.array([sheets[i].normal for i in skin]).reshape(-1, 3)
    up = normals[:, 2] >= 0.0
    dots = [
        float(sheets[i].normal @ sheets[j].normal)
        for i in skin
        for j in sheets[i].across
        if j > i
        if j in set(skin)
    ]
    return Measure(
        part,
        kind,
        len(skin),
        int(up.sum()),
        int((~up).sum()),
        np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))),
        _directions(normals),
        _directions(normals[up]),
        _directions(normals[~up]),
    )


def design_measures(design: Design) -> list[Measure]:
    """Measures of every faceted body, then every faceted wing."""
    out = [measure(b.name, "body", b.facets) for b in design.bodies if b.facets is not None]
    for s in design.surfaces:
        if s.airfoil_facets:
            poly = wing_polyhedron(s)
            out.append(measure(s.name, "wing", poly, (0, len(poly.faces) - 1)))
    return [m for m in out if m is not None]


def _budget(m: Measure) -> float:
    key = "coarse_wing_side_panels_max" if m.kind == "wing" else "coarse_body_panels_max"
    return number(key)


def _count(m: Measure) -> int:
    """The count the panel budget is read on: a wing's busier side, a body's sheets."""
    return m.side if m.kind == "wing" else m.sheets


def coarse_rows(design: Design) -> list[str]:
    """Report lines: the coarseness, one row per faceted part, the aero stand-in's cost."""
    fam = design.family
    least = min_crease_deg(fam.facet_coarseness)
    rows = [
        f"- facet_coarseness {fam.facet_coarseness:.2f}: smallest crease {least:.0f} deg. Sheets "
        "are the physical pieces of panels.csv (a mirrored wing: one half, caps left out); sides "
        "are upper / lower; directions are clusters of "
        f"{number('coarse_cluster_deg'):g} deg of the sheet normals.",
        "",
        "| part | sheets | upper / lower | directions | upper / lower | creases under 10 deg "
        "| under 15 deg | smallest [deg] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in design_measures(design):
        c = m.creases_deg
        low10 = float((c < SHARP_DEG).mean()) if len(c) else 0.0
        low15 = float((c < 15.0).mean()) if len(c) else 0.0
        rows.append(
            f"| {m.part} | {m.sheets} | {m.upper} / {m.lower} | {m.directions} "
            f"| {m.dir_upper} / {m.dir_lower} | {low10:.0%} | {low15:.0%} "
            f"| {float(c.min()) if len(c) else 0.0:.1f} |"
        )
    drag = facet_drag_area_m2(design)
    area = sum(s.area_m2 for s in design.surfaces if s.airfoil_facets) or 1.0
    rows.append(
        "\n"
        f"- aero cost against the smooth parent section: profile drag area +{drag:.5f} m^2 "
        f"(Cd {drag / area:+.4f} on the faceted wings), clmax change 0 as modelled (the parent "
        "section's clmax is kept: optimistic for a polygon, Known issue)."
    )
    return rows


def planar_notes(surface: Surface, c: float) -> list[str]:
    """Notes on the vertex moves that made the twisted quads of a faceted wing flat."""
    planar = wing_geometry(surface)[1]
    if planar is None and c > 0.0 and len(surface.sections) > 1:
        return [
            f"coarseness {c:.2f}: {surface.name}: no strip merge was accepted (it squeezed an "
            "edge, flipped or bent a face, or left an end cap unflat): the section vertices stay"
        ]
    if planar is None or (not planar.merged and not planar.blocked):
        return []
    part = _part_length(surface)
    out = [
        f"coarseness {c:.2f}: {surface.name}: {len(planar.merged)} strip quads made flat by moving "
        f"vertices up to {planar.move_m * 1000:.1f} mm ({planar.move_m / part:.1%} of the part "
        f"length, cap {number('coarse_move_frac'):.0%})"
    ]
    if planar.blocked:
        out.append(
            f"coarseness {c:.2f}: {surface.name}: {planar.blocked} twisted strip quads stay two "
            "triangles (a merge would move vertices over the cap): blocked by the cap"
        )
    return out


def body_merge_notes(design: Design, body: Body, c: float) -> list[str]:
    """Notes on the vertex moves that made neighbouring body sheets under the smallest crease
    coplanar (the body's own length and size set the cap, so a pod has its own note)."""
    shape = design.family.shape.model_copy(update={"coarseness": c})
    axes = (body.half_width_m, body.half_height_m)
    info = facet_body_info(shape, body.length_m, axes, body.tail_upsweep_deg, body.rise_x_frac)[1]
    if info is None or (not info.merged and not info.blocked):
        return []
    out = [
        f"coarseness {c:.2f}: {body.name}: {info.merged} pairs of neighbouring sheets under "
        f"{min_crease_deg(c):.0f} deg made coplanar by moving vertices up to "
        f"{info.move_m * 1000:.1f} mm ({info.move_m / body.length_m:.1%} of the body length, cap "
        f"{number('coarse_move_frac'):.0%})"
    ]
    if info.blocked:
        out.append(
            f"coarseness {c:.2f}: {body.name}: {info.blocked} pairs stay creases (the merge would "
            "move vertices over the cap or conflict with another): blocked by the cap"
        )
    return out


def coarse_notes(design: Design) -> tuple[str, ...]:
    """Decoder notes for parts still over the budgets or under the smallest crease, measured on
    the finished design (nothing the coarsening could not reach goes unreported)."""
    c = design.family.facet_coarseness
    if c <= 0.0:
        return ()
    out = [n for s in design.surfaces if s.airfoil_facets for n in planar_notes(s, c)]
    out += [
        n for b in design.bodies if b.facets is not None for n in body_merge_notes(design, b, c)
    ]
    for m in design_measures(design):
        shallow = int((m.creases_deg < min_crease_deg(c) - 1e-6).sum())
        dirs = m.dir_side > number("coarse_directions_max")
        if shallow or _count(m) > _budget(m) or dirs:
            out.append(
                f"coarseness {c:.2f}: {m.part} has {m.sheets} sheets ({m.upper} upper / "
                f"{m.lower} lower; budget {_budget(m):g} on the busier "
                f"{'side' if m.kind == 'wing' else 'part'}), {m.dir_side} face directions on its "
                f"busier side (budget {number('coarse_directions_max'):g}), {shallow} of "
                f"{len(m.creases_deg)} creases under {min_crease_deg(c):.0f} deg"
            )
    return tuple(out)
