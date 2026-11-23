"""Panel cut list of flat-panel bodies (SPEC §11): panels.csv rows and the panels.svg layout.

Pure data in, text out; the writer is `write_panels`. A row is one physical sheet: neighbouring
faces in one plane are merged (facet_pieces), so a body of N faces has N or fewer rows. Sheets are
numbered `<body>-F###` in the order of their first face. Bevel = angle [deg] between the outward
normals of a sheet and its neighbour across an edge (the saw angle for a mitre is half of it)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeroforge.constants import number
from aeroforge.facet_pieces import pieces
from aeroforge.models import Body, Design

CSV_HEADER = "id,part,count,area_m2,edge_lengths_m,neighbour_ids,bevel_deg"
PANELS_DIR, CSV_FILE, SVG_FILE = "panels", "panels.csv", "panels.svg"
MM = 1000.0


@dataclass(frozen=True)
class Panel:
    """One flat sheet of a body: its corners [m] in loop order and its edge data."""

    id: str
    part: str
    count: int
    area_m2: float
    corners_m: np.ndarray
    edge_lengths_m: tuple[float, ...]
    neighbour_ids: tuple[str, ...]
    bevel_deg: tuple[float, ...]


def body_panels(body: Body) -> list[Panel]:
    """Panels (physical sheets) of one faceted body, empty for a smooth one: neighbouring faces in
    one plane are one sheet, so the rows are the pieces to cut. Bevel = angle [deg] between the
    outward normals of a sheet and its neighbour across an edge."""
    poly = body.facets
    if poly is None:
        return []
    v = np.asarray(poly.vertices_m)
    sheets = pieces(poly)
    ids = [f"{body.name}-F{k + 1:03d}" for k in range(len(sheets))]
    panels = []
    for k, sheet in enumerate(sheets):
        corners = v[list(sheet.loop)]
        bevel = [
            float(np.degrees(np.arccos(np.clip(sheet.normal @ sheets[j].normal, -1.0, 1.0))))
            for j in sheet.across
        ]
        panels.append(
            Panel(
                id=ids[k],
                part=body.name,
                count=2 if body.mirrored else 1,
                area_m2=sheet.area_m2,
                corners_m=corners,
                edge_lengths_m=tuple(
                    float(np.linalg.norm(corners[i] - corners[(i + 1) % len(corners)]))
                    for i in range(len(corners))
                ),
                neighbour_ids=tuple(ids[j] for j in sheet.across),
                bevel_deg=tuple(bevel),
            )
        )
    return panels


def design_panels(design: Design) -> list[Panel]:
    """Panels of every faceted body of the design, in body order."""
    return [p for b in design.bodies for p in body_panels(b)]


def panels_csv(panels: list[Panel]) -> str:
    """panels.csv text: one row per flat face; list columns are `;`-joined in edge order (edge i
    runs from corner i to corner i + 1)."""
    rows = [CSV_HEADER]
    for p in panels:
        lengths = ";".join(f"{x:.6f}" for x in p.edge_lengths_m)
        bevels = ";".join(f"{x:.3f}" for x in p.bevel_deg)
        neighbours = ";".join(p.neighbour_ids)
        rows.append(f"{p.id},{p.part},{p.count},{p.area_m2:.9f},{lengths},{neighbours},{bevels}")
    return "\n".join(rows) + "\n"


def flat_outline_mm(p: Panel) -> np.ndarray:
    """The face laid flat as seen from outside: (n, 2) in mm, first edge along +x, min corner at
    the origin."""
    c = p.corners_m
    n = np.cross(c, np.roll(c, -1, axis=0)).sum(axis=0)
    n /= np.linalg.norm(n)
    e1 = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
    e2 = np.cross(n, e1)
    flat = np.stack([(c - c[0]) @ e1, (c - c[0]) @ e2], axis=1) * MM
    flat[:, 1] *= -1.0  # SVG y points down
    return flat - flat.min(axis=0)


def _face_svg(p: Panel, at: np.ndarray, font: float) -> str:
    """One <g> with the outline, the face number and a rotated length label on every edge."""
    flat = flat_outline_mm(p) + at
    mid = flat.mean(axis=0)
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in flat)
    number_text = p.id.rsplit("-", 1)[1]
    out = [
        f'<g id="{p.id}"><title>{p.id} {p.area_m2 * 1e4:.1f} cm2</title>',
        f'<polygon points="{pts}" fill="none" stroke="black" stroke-width="0.4"/>',
        f'<text x="{mid[0]:.2f}" y="{mid[1]:.2f}" font-size="{font * 1.5:.2f}" '
        f'text-anchor="middle">{number_text}</text>',
    ]
    for i, length in enumerate(p.edge_lengths_m):
        a, b = flat[i], flat[(i + 1) % len(flat)]
        inward = mid - (a + b) / 2
        m = (a + b) / 2 + font * 1.1 * inward / max(float(np.linalg.norm(inward)), 1e-9)
        angle = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        if abs(angle) > 90.0:  # keep the text upright
            angle += 180.0 if angle < 0.0 else -180.0
        out.append(
            f'<text x="{m[0]:.2f}" y="{m[1]:.2f}" font-size="{font:.2f}" text-anchor="middle" '
            f'dominant-baseline="central" '
            f'transform="rotate({angle:.1f} {m[0]:.2f} {m[1]:.2f})">{length * MM:.1f}</text>'
        )
    return "".join(out) + "</g>"


def panels_svg(panels: list[Panel]) -> str:
    """panels.svg text: every face laid flat in true size [mm], packed in rows, numbered, each
    edge dimensioned (length in mm)."""
    width, gap, font = (
        number("panels_sheet_width_mm"),
        number("panels_gap_mm"),
        number("panels_font_mm"),
    )
    x = y = row_h = 0.0
    groups = []
    for p in panels:
        w, h = flat_outline_mm(p).max(axis=0)
        if x > 0.0 and x + w > width:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        groups.append(_face_svg(p, np.array([x, y]), font))
        x, row_h = x + w + gap, max(row_h, h)
    total = max(y + row_h, 1.0)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}mm" height="{total:.0f}mm" '
        f'viewBox="0 0 {width:.0f} {total:.0f}" font-family="sans-serif">'
    )
    return head + "".join(groups) + "</svg>\n"


def write_panels(folder: Path, design: Design) -> int:
    """Write panels/panels.csv and panels/panels.svg for the design's faceted bodies (nothing,
    and no stale files, for a smooth design); returns the number of panels."""
    target = folder / PANELS_DIR
    panels = design_panels(design)
    for name in (CSV_FILE, SVG_FILE):
        (target / name).unlink(missing_ok=True)
    if not panels:
        return 0
    target.mkdir(exist_ok=True)
    (target / CSV_FILE).write_text(panels_csv(panels), encoding="utf-8", newline="\n")
    (target / SVG_FILE).write_text(panels_svg(panels), encoding="utf-8", newline="\n")
    return len(panels)
