"""How coarse the faceted parts of a design are (SPEC §11, U3b): sheets, crease angles and face
directions, measured on the finished polyhedra, and the report.md section and decoder notes built
from them. Pure numpy; the sheets are the physical pieces (coplanar faces merged)."""

from dataclasses import dataclass

import numpy as np

from aeroforge.coarse import min_crease_deg
from aeroforge.constants import number
from aeroforge.facet_aero import facet_drag_area_m2
from aeroforge.facet_pieces import PieceError, pieces
from aeroforge.facet_wing import wing_polyhedron
from aeroforge.models import Design, Polyhedron

SHARP_DEG = 10.0
WING_CAPS = 2  # root and tip cap of a wing: not a side


@dataclass(frozen=True)
class Measure:
    """Coarseness of one faceted part: sheets, the crease angle [deg] across every shared edge of
    two sheets, and the face directions (3 deg clusters of the sheet normals) in all and the
    larger count of the upper and lower sides."""

    part: str
    kind: str
    sheets: int
    creases_deg: np.ndarray
    directions: int
    directions_side: int
    sheets_raw: int


def _directions(normals: np.ndarray) -> int:
    reps: list[np.ndarray] = []
    tol = np.cos(np.radians(number("coarse_cluster_deg")))
    for n in normals:
        if not any(n @ r >= tol for r in reps):
            reps.append(n)
    return len(reps)


def _panel_groups(sheets: list, tol_deg: float) -> list[int]:
    """Panel number of every sheet: sheets across a crease of at most tol_deg are one panel (the
    two triangles of a warped quad are one panel to the eye and the builder)."""
    parent = list(range(len(sheets)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cos = np.cos(np.radians(tol_deg))
    for i, s in enumerate(sheets):
        for j in s.across:
            if float(s.normal @ sheets[j].normal) >= cos:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(sheets))]


def measure(part: str, kind: str, poly: Polyhedron) -> Measure | None:
    """Measure of a polyhedron, None when its sheets cannot be formed. Panels merge sheets across
    a crease of at most coarse_cluster_deg (the warped halves of a twisted strip)."""
    try:
        sheets = pieces(poly)
    except PieceError:
        return None
    group = _panel_groups(sheets, number("coarse_cluster_deg"))
    normals = np.array([s.normal for s in sheets])
    dots = [
        float(s.normal @ sheets[j].normal)
        for i, s in enumerate(sheets)
        for j in s.across
        if j > i and group[i] != group[j]
    ]
    upper, lower = normals[normals[:, 2] >= 0.0], normals[normals[:, 2] < 0.0]
    return Measure(
        part,
        kind,
        len(set(group)),
        np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))),
        _directions(normals),
        max(_directions(upper), _directions(lower)),
        len(sheets),
    )


def design_measures(design: Design) -> list[Measure]:
    """Measures of every faceted body, then every faceted wing."""
    out = [measure(b.name, "body", b.facets) for b in design.bodies if b.facets is not None]
    out += [
        measure(s.name, "wing", wing_polyhedron(s)) for s in design.surfaces if s.airfoil_facets
    ]
    return [m for m in out if m is not None]


def _sheets_per_side(m: Measure) -> float:
    return (m.sheets - WING_CAPS) / 2.0 if m.kind == "wing" else float(m.sheets)


def _budget(m: Measure) -> float:
    key = "coarse_wing_side_panels_max" if m.kind == "wing" else "coarse_body_panels_max"
    return number(key)


def coarse_rows(design: Design) -> list[str]:
    """Report lines: the coarseness, one row per faceted part, the aero stand-in's cost."""
    fam = design.family
    least = min_crease_deg(fam.facet_coarseness)
    rows = [
        f"- facet_coarseness {fam.facet_coarseness:.2f}: smallest crease {least:.0f} deg, wing "
        f"sections of {fam.airfoil_facets} flat segments per side. A panel joins sheets across "
        f"creases of {number('coarse_cluster_deg'):g} deg or less (the halves of a twisted strip).",
        "",
        "| part | panels (wing: per side) | sheets | face directions (per side) "
        "| creases under 10 deg | smallest crease [deg] |",
        "|---|---|---|---|---|---|",
    ]
    for m in design_measures(design):
        low = float((m.creases_deg < SHARP_DEG).mean()) if len(m.creases_deg) else 0.0
        smallest = float(m.creases_deg.min()) if len(m.creases_deg) else 0.0
        rows.append(
            f"| {m.part} | {_sheets_per_side(m):g} | {m.sheets_raw} "
            f"| {m.directions} ({m.directions_side}) | {low:.0%} | {smallest:.1f} |"
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


def coarse_notes(design: Design) -> tuple[str, ...]:
    """Decoder notes for parts still over the budgets or under the smallest crease, measured on
    the finished design (nothing the coarsening could not reach goes unreported)."""
    c = design.family.facet_coarseness
    if c <= 0.0:
        return ()
    out = []
    for m in design_measures(design):
        shallow = int((m.creases_deg < min_crease_deg(c) - 1e-6).sum())
        over = _sheets_per_side(m) > _budget(m)
        dirs = m.directions_side > number("coarse_directions_max")
        if shallow or over or dirs:
            out.append(
                f"coarseness {c:.2f}: {m.part} has {_sheets_per_side(m):g} sheets"
                f"{' per side' if m.kind == 'wing' else ''} (budget {_budget(m):g}), "
                f"{m.directions_side} face directions per side (budget "
                f"{number('coarse_directions_max'):g}), {shallow} of {len(m.creases_deg)} creases "
                f"under {min_crease_deg(c):.0f} deg"
            )
    return tuple(out)
