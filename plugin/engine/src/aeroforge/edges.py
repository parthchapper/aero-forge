"""Top-view outline edges and their angles (SPEC §4, §7, U3): measured from the finished geometry
(section outlines, polyhedron vertices), never from the sweep genes, because cant and twist change
the projected angles. Pure numpy, metres.

Angle convention: the sweep angle of an edge is measured from the y axis in the xy plane, oriented
inboard -> outboard, so +30 is a trailing-back edge and +90 / -90 run along x (folded to +90).
Mirror pairs share one angle; the right / centre instance is measured."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial import ConvexHull

from aeroforge.constants import number, numbers
from aeroforge.models import Body, Design, Surface
from aeroforge.shapes import section_outline


@dataclass(frozen=True)
class Edge:
    """One straight outline edge seen from above."""

    part: str
    kind: str
    sweep_deg: float
    length_m: float


def sweep_deg(inboard_xy: np.ndarray, outboard_xy: np.ndarray) -> float:
    """Sweep angle [deg] of the edge from its inboard to its outboard end, in (-90, 90]."""
    dx = float(outboard_xy[0] - inboard_xy[0])
    dy = abs(float(outboard_xy[1])) - abs(float(inboard_xy[1]))
    if abs(dy) < 1e-12:
        return 90.0
    return float(np.degrees(np.arctan2(dx, abs(dy))))


def angle_gap_deg(a: float, b: float) -> float:
    """Smallest difference [deg] of two sweep angles (they repeat every 180)."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def allowed_angles_deg(main_sweep_deg: float) -> tuple[float, ...]:
    """The angle set strict alignment snaps to: the fixed angles of constants.yaml
    (edge_fixed_angles_deg) and the main wing's leading-edge sweep and its complement, each with
    both signs, folded into (-90, 90], sorted."""
    a = abs(main_sweep_deg)
    raw = [*numbers("edge_fixed_angles_deg")]
    raw += [s * v for v in (a, 90.0 - a) for s in (1.0, -1.0)]
    folded = {round(90.0 - ((90.0 - v) % 180.0), 9) for v in raw}  # into (-90, 90]
    return tuple(sorted(folded))


def body_angles_deg(main_sweep_deg: float) -> tuple[float, ...]:
    """The angle set for flat-panel body edges: the wing set plus the steep angles 90 - a / 2^k
    (k = 1..edge_body_halvings, a the main sweep). A body is long and thin, so its nose and tail
    edges run within a few degrees of the x axis, where the wing set has nothing."""
    a = abs(main_sweep_deg)
    extra = [90.0 - a / 2**k for k in range(1, int(number("edge_body_halvings")) + 1)]
    return tuple(sorted({*allowed_angles_deg(main_sweep_deg), *extra, *(-v for v in extra)}))


def nearest_allowed(angle_deg: float, allowed: tuple[float, ...]) -> float:
    """The allowed angle closest to angle_deg."""
    return min(allowed, key=lambda v: angle_gap_deg(angle_deg, v))


def surface_edges(s: Surface) -> list[Edge]:
    """Leading, trailing and tip-chord edges of a surface's right / centre instance on the planform
    (section leading edge, chord along x): the washout rotation of a section about its leading
    edge is left out here and measured by tip_tilt_deg. Empty when the surface is seen edge-on (a
    centreline fin); edges shorter than edge_min_length_m are left out."""
    if not s.mirrored:
        return []
    le = [np.array(c.le_m[:2]) for c in s.sections]
    te = [p + np.array([c.chord_m, 0.0]) for p, c in zip(le, s.sections, strict=True)]
    out = []
    for kind, pts in (("leading", le), ("trailing", te)):
        out += [
            Edge(s.name, kind, sweep_deg(a, b), float(np.linalg.norm(b - a)))
            for a, b in pairwise(pts)
        ]
    out.append(
        Edge(s.name, "tip", sweep_deg(le[-1], te[-1]), float(np.linalg.norm(te[-1] - le[-1])))
    )
    return [e for e in out if e.length_m >= number("edge_min_length_m")]


def tip_tilt_deg(s: Surface) -> float:
    """Angle [deg] between the tip section's real chord line (leading to trailing-edge vertex of its
    outline) and the x axis, seen from above: the washout rotation of a canted tip tilts it."""
    row = section_outline(s.sections[-1], s, int(number("airfoil_points_per_side")))
    d = ((row[0] + row[-1]) / 2 - row[len(row) // 2])[:2]
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))


def body_edges(b: Body) -> list[Edge]:
    """Outline edges of a flat-panel body seen from above: the right half of the convex hull of its
    vertices projected on xy (nose and tail edges, cap edges across y included); none for a smooth
    body, whose outline has no straight edge."""
    if b.facets is None:
        return []
    v = np.asarray(b.facets.vertices_m)[:, :2] + np.asarray(b.nose_m)[:2]
    hull = v[ConvexHull(v).vertices]
    out = []
    for p, q in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        if (p[1] + q[1]) / 2.0 < -1e-9:
            continue
        a, c = (p, q) if abs(p[1]) <= abs(q[1]) else (q, p)
        out.append(Edge(b.name, "outline", sweep_deg(a, c), float(np.linalg.norm(q - p))))
    return [e for e in out if e.length_m >= number("edge_min_length_m")]


def design_edges(d: Design) -> list[Edge]:
    """Every measured outline edge of the design: bodies, then surfaces."""
    return [
        *(e for b in d.bodies for e in body_edges(b)),
        *(e for s in d.surfaces for e in surface_edges(s)),
    ]


def main_sweep_deg(d: Design) -> float:
    """Leading-edge sweep [deg] of surface0's first panel (the angle set's base)."""
    s0 = d.surfaces[0]
    a, b = s0.sections[0], s0.sections[1]
    return sweep_deg(np.array(a.le_m[:2]), np.array(b.le_m[:2]))


def edge_gap_deg(e: Edge, d: Design) -> float:
    """Gap [deg] of an edge to the design's angle set (a body edge to the body set); 0 for a design
    that was not strictly aligned only if a set exists, else the gap to the set it would have had
    (the wing set from the measured main sweep)."""
    body = e.kind == "outline"
    fam = d.family
    given = fam.edge_body_angles_deg if body else fam.edge_angles_deg
    if not given:
        m = main_sweep_deg(d)
        given = body_angles_deg(m) if body else allowed_angles_deg(m)
    return min(angle_gap_deg(e.sweep_deg, a) for a in given)


def alignment_misses(d: Design) -> list[tuple[Edge, float]]:
    """(edge, gap) of every measured edge farther than edge_align_tol_deg from its angle set."""
    tol = number("edge_align_tol_deg")
    return [(e, g) for e in design_edges(d) if (g := edge_gap_deg(e, d)) > tol]


def alignment_score(d: Design) -> float:
    """Length-weighted mean over the measured edges of exp(-(gap / edge_soft_sigma_deg)^2): 1 when
    every edge is on its angle set, 0 when there is no edge."""
    edges = design_edges(d)
    total = sum(e.length_m for e in edges)
    if total == 0.0:
        return 0.0
    sigma = number("edge_soft_sigma_deg")
    return (
        sum(e.length_m * float(np.exp(-((edge_gap_deg(e, d) / sigma) ** 2))) for e in edges) / total
    )


def soft_style_bonus(d: Design) -> float:
    """Added to the style score of a design whose edge_alignment gene is soft; 0 otherwise."""
    if d.family.edge_alignment != "soft":
        return 0.0
    return number("edge_soft_weight") * alignment_score(d)
