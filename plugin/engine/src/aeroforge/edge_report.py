"""report.md section on the top-view outline edges (SPEC §11, U3): every measured edge with its
angle and gap to the angle set, the angles shared across parts, and the tip tilt of washed-out
tips. Pure text from a Design; the angles are measured on the finished geometry (edges.py)."""

from aeroforge.constants import number
from aeroforge.edges import (
    Edge,
    angle_gap_deg,
    design_edges,
    edge_gap_deg,
    main_sweep_deg,
    tip_tilt_deg,
)
from aeroforge.models import Design


def shared_angles(edges: list[Edge]) -> list[tuple[float, list[str]]]:
    """Groups of edges of at least two different parts whose angles chain within
    edge_group_tol_deg: (mean angle, "part kind" labels). Measured, not assumed."""
    tol = number("edge_group_tol_deg")
    ordered = sorted(edges, key=lambda e: e.sweep_deg)
    groups: list[list[Edge]] = []
    for e in ordered:
        if groups and angle_gap_deg(e.sweep_deg, groups[-1][-1].sweep_deg) <= tol:
            groups[-1].append(e)
        else:
            groups.append([e])
    out = []
    for g in groups:
        if len({e.part for e in g}) >= 2:
            out.append((sum(e.sweep_deg for e in g) / len(g), [f"{e.part} {e.kind}" for e in g]))
    return out


def edge_rows(design: Design) -> list[str]:
    """Report lines: alignment mode, edge table, shared angles, tip tilt."""
    fam = design.family
    edges = design_edges(design)
    tol = number("edge_align_tol_deg")
    rows = [
        f"- edge_alignment: {fam.edge_alignment}; main wing sweep {main_sweep_deg(design):.1f} deg;"
        + (
            f" wing angle set {[round(a, 1) for a in fam.edge_angles_deg]} deg, body set"
            f" {[round(a, 1) for a in fam.edge_body_angles_deg]} deg."
            if fam.edge_angles_deg
            else " angles are measured against the set the main sweep would give."
        ),
        "",
        "| part | edge | sweep from y [deg] | length [m] | gap to set [deg] | within tolerance |",
        "|---|---|---|---|---|---|",
    ]
    for e in edges:
        gap = edge_gap_deg(e, design)
        rows.append(
            f"| {e.part} | {e.kind} | {e.sweep_deg:+.2f} | {e.length_m:.3f} | {gap:.2f} |"
            f" {'yes' if gap <= tol else 'no'} |"
        )
    aligned = sum(edge_gap_deg(e, design) <= tol for e in edges)
    rows.append(f"\n- {aligned} of {len(edges)} measured edges are within {tol:g} deg of the set.")
    shared = shared_angles(edges)
    rows += [
        f"- shared angle {a:+.2f} deg ({number('edge_group_tol_deg'):g} deg chain): {', '.join(m)}"
        for a, m in shared
    ] or ["- no angle is shared by edges of two different parts."]
    tilts = {s.name: tip_tilt_deg(s) for s in design.surfaces if s.mirrored}
    if tilts:
        rows.append(
            "- tip chord tilt from washout, seen from above: "
            + ", ".join(f"{k} {v:.2f} deg" for k, v in tilts.items())
        )
    return rows
