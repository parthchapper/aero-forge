"""`fit` check: SPEC §5 invariants 5-7 re-measured on a Design as normalised margins."""

import numpy as np

from aeroforge.airfoil import local_thickness_frac, wing_outline
from aeroforge.constants import constant, number
from aeroforge.layout import box_lattice
from aeroforge.models import Box, Design, Propulsor, Requirements
from aeroforge.shapes import MIRROR_Y, component_points, inside_body, inside_loft


def box_points_inside(d: Design, box: Box) -> np.ndarray:
    """Per lattice point of box: inside its host body or surface loft."""
    pts = box_lattice(box, int(number("box_lattice_points")))
    for b in d.bodies:
        if b.name == box.host:
            return inside_body(b, pts)
    s = next(s for s in d.surfaces if s.name == box.host)
    xy = wing_outline(
        s.camber_frac,
        s.camber_loc_frac,
        s.tc,
        int(number("airfoil_points_per_side")),
        s.airfoil_facets,
    )
    return inside_loft(s.sections[: s.n_structural], xy, pts, mirrored=s.mirrored)


def box_margin(d: Design) -> float:
    """0 when every box lattice point is inside its host, else minus the share outside (inv 5)."""
    inside = [box_points_inside(d, box) for box in d.boxes]
    return sum(int(ok.sum()) for ok in inside) / sum(len(ok) for ok in inside) - 1.0


def spar_margin(d: Design) -> float:
    """Min over surfaces and structural sections of (room - tube OD) / OD (inv 6)."""
    n_af = int(number("airfoil_points_per_side"))
    skins = 2 * number("spar_skin_thickness_m")
    worst = np.inf
    for s in d.surfaces:
        t = local_thickness_frac(
            s.camber_frac, s.camber_loc_frac, s.tc, s.spar.chord_frac, n_af, s.airfoil_facets
        )
        room = min(t * c.chord_m for c in s.sections[: s.n_structural]) - skins
        worst = min(worst, (room - s.spar.od_m) / s.spar.od_m)
    return float(worst)


def _discs(d: Design) -> list[tuple[np.ndarray, float]]:
    """(hub, radius) of every propeller instance [m]."""
    out = []
    for p in d.propulsors:
        hub = np.asarray(p.hub_m)
        out += [(hub, p.diameter_m / 2)] + (
            [(hub * MIRROR_Y, p.diameter_m / 2)] if p.mirrored else []
        )
    return out


def prop_margin(d: Design, req: Requirements) -> float:
    """Min of (axial gap - margin) / margin over structure inside each disc's radial band, and of
    the ground-line gap for wheels/skid recovery (inv 7)."""
    margin = number("prop_clearance_m")
    pts = np.vstack(list(component_points(d, include_props=False).values()))
    worst = np.inf
    for hub, r in _discs(d):
        rho = np.hypot(pts[:, 1] - hub[1], pts[:, 2] - hub[2])
        band = rho <= r + margin
        if band.any():
            worst = min(worst, (float(np.abs(pts[band, 0] - hub[0]).min()) - margin) / margin)
    recoveries = constant("constants", "ground_line_recoveries")
    if isinstance(recoveries, list) and req.recovery in recoveries:
        ground = float(pts[:, 2].min())
        worst = min([worst, *((hub[2] - r - ground - margin) / margin for hub, r in _discs(d))])
    return float(worst)


def _shaft_points(p: Propulsor) -> np.ndarray:
    """Samples of the shaft from attach point to hub, both instances when mirrored [m]."""
    t = np.linspace(0.0, 1.0, int(number("shaft_samples")))[:, None]
    attach = np.asarray(p.attach_m)
    pts = attach + t * (np.asarray(p.hub_m) - attach)
    return np.vstack([pts, pts * MIRROR_Y]) if p.mirrored else pts


def shaft_margin(d: Design) -> float:
    """0 when no propeller shaft passes through a surface other than its host, else minus the
    share of shaft samples (outside every body) that lie inside one (user, step 4)."""
    n_af = int(number("airfoil_points_per_side"))
    hits, total = 0, 0
    for p in d.propulsors:
        pts = _shaft_points(p)
        for b in d.bodies:
            pts = pts[~inside_body(b, pts)]
        inside = np.zeros(len(pts), dtype=bool)
        for s in d.surfaces:
            if s.name != p.host:
                xy = wing_outline(s.camber_frac, s.camber_loc_frac, s.tc, n_af, s.airfoil_facets)
                inside |= inside_loft(s.sections, xy, pts, mirrored=s.mirrored)
        hits, total = hits + int(inside.sum()), total + len(pts)
    return -hits / total if total else 0.0


def fit_margins(d: Design, req: Requirements) -> dict[str, float]:
    """Normalised margins of invariants 5, 6 and 7."""
    return {"boxes": box_margin(d), "spar": spar_margin(d), "prop_clearance": prop_margin(d, req)}
