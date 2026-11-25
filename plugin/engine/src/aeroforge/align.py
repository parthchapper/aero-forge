"""Strict edge alignment of wing planforms (SPEC §4, U3): the leading and trailing edge of every
panel is moved to the nearest angle of the allowed set (edges.allowed_angles_deg) by shifting the
section leading edges and chords along x only. Span positions (y, z) stay, so span and aspect
ratio keep their values; the root chord is re-solved so the projected area keeps its value
(invariant 3). Pure numpy, metres."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from aeroforge.body_align import coercion_notes
from aeroforge.constants import constant, number
from aeroforge.edges import allowed_angles_deg, angle_gap_deg, body_angles_deg
from aeroforge.genetable import Genes, surface_count
from aeroforge.models import Family, Section


@dataclass(frozen=True)
class Snap:
    """Result of snapping one surface: the sections and a note ('' when every edge was snapped)."""

    sections: tuple[Section, ...]
    note: str


def chord_floor_m() -> float:
    """Smallest section chord [m] a stock spar still fits at tc_max (with strict_spar_room_frac of
    the thickness usable), so a snapped planform never leaves the spar repair without a way out."""
    tubes = constant("constants", "spar_tubes_m")
    if not isinstance(tubes, list):
        raise TypeError("spar_tubes_m must be a list")
    smallest = min(float(od) for od, _ in tubes)
    room = number("strict_spar_room_frac") * number("tc_max")
    return (smallest + 2 * number("spar_skin_thickness_m")) / room


def _candidates(angle_deg: float, allowed: tuple[float, ...]) -> list[float]:
    """The strict_candidate_count allowed angles nearest to angle_deg, 90 (along x) excluded: a
    panel with span cannot have an edge that runs along x."""
    finite = [a for a in allowed if a < 90.0 - 1e-9]
    ranked = sorted(finite, key=lambda a: angle_gap_deg(angle_deg, a))
    return ranked[: int(number("strict_candidate_count"))]


def _solve(
    tan_le: list[float], tan_te: list[float], dy: np.ndarray, half_area_m2: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """(LE x offsets from the root LE, chords) for the edge slopes, root chord from the area."""
    k = np.concatenate([[0.0], np.cumsum((np.array(tan_te) - np.array(tan_le)) * dy)])
    c0 = (half_area_m2 - float((dy * (k[:-1] + k[1:]) / 2).sum())) / float(dy.sum())
    chords = c0 + k
    if chords.min() < chord_floor_m():
        return None
    return np.concatenate([[0.0], np.cumsum(np.array(tan_le) * dy)]), chords


def snap_sections(
    sections: tuple[Section, ...],
    allowed: tuple[float, ...],
    mid_chord_x_m: float,
) -> Snap:
    """Sections whose panel edges lie on allowed angles, the best feasible combination (least
    total angular move, weighted by edge length); the natural sections and a note when no
    combination keeps every chord above chord_floor_m."""
    pts = np.array([c.le_m for c in sections])
    dy = np.abs(np.diff(pts[:, 1]))
    arc = np.hypot(dy, np.diff(pts[:, 2]))
    le_x = pts[:, 0]
    chords0 = np.array([c.chord_m for c in sections])
    target_m2 = float((dy * (chords0[:-1] + chords0[1:]) / 2).sum())  # projected area: invariant 3
    te_x = le_x + np.array([c.chord_m for c in sections])
    if dy.sum() < number("edge_min_length_m"):  # seen edge-on altogether: no edge to snap
        return Snap(sections, "")
    narrow = dy < number("edge_min_length_m")  # seen edge-on (a near-vertical winglet panel)
    nat_le = np.degrees(np.arctan2(np.diff(le_x), dy))
    nat_te = np.degrees(np.arctan2(np.diff(te_x), dy))
    options = [
        [[0.0, 0.0]]  # an edge-on panel keeps its chord and loses its sweep: a line in plan
        if edge_on
        else [list(p) for p in product(_candidates(a, allowed), _candidates(b, allowed))]
        for a, b, edge_on in zip(nat_le, nat_te, narrow, strict=True)
    ]
    best: tuple[float, tuple[np.ndarray, np.ndarray]] | None = None
    for combo in product(*options):
        le, te = [p[0] for p in combo], [p[1] for p in combo]
        cost = sum(
            ln * (angle_gap_deg(n1, a) + angle_gap_deg(n2, b))
            for ln, n1, n2, a, b in zip(arc, nat_le, nat_te, le, te, strict=True)
        )
        if best is not None and cost >= best[0]:
            continue
        solved = _solve(
            [float(np.tan(np.radians(a))) for a in le],
            [float(np.tan(np.radians(b))) for b in te],
            dy,
            target_m2,
        )
        if solved is not None:
            best = (cost, solved)
    if best is None:
        return Snap(sections, "strict alignment: no allowed edge angles keep every chord usable")
    x_off, chords = best[1]
    x0 = mid_chord_x_m - 0.5 * chords[0]
    out = tuple(
        c.model_copy(update={"le_m": (float(x0 + dx), c.le_m[1], c.le_m[2]), "chord_m": float(w)})
        for c, dx, w in zip(sections, x_off, chords, strict=True)
    )
    note = "strict alignment: edge-on panel(s) lost their sweep" if narrow.any() else ""
    return Snap(out, note)


def main_sweep_gene_deg(v: Genes, fam: Family) -> float:
    """Top-view sweep [deg] of surface0's inner panel from the genes: its quarter-chord sweep seen
    from above (a canted panel's projected sweep is steeper), the angle set's base."""
    sign = 1.0 if v.pick["s0_sweep_sign"] == "aft" else -1.0
    cap = number("sweep_max_deg")
    sweep = float(np.clip(sign * fam.sweep_deg + v.num["s0_sweep_delta_deg"], -cap, cap))
    cant = float(np.clip(v.num["s0_cant_deg"] + fam.dihedral_deg, -90.0, 90.0))
    return float(
        np.degrees(np.arctan(np.tan(np.radians(sweep)) / max(np.cos(np.radians(cant)), 1e-6)))
    )


def strict_family(v: Genes, fam: Family) -> Family:
    """fam with the allowed angle set when edge_alignment is strict, else unchanged."""
    if fam.edge_alignment != "strict":
        return fam
    a = main_sweep_gene_deg(v, fam)
    return fam.model_copy(
        update={
            "edge_angles_deg": allowed_angles_deg(a),
            "edge_body_angles_deg": body_angles_deg(a),
        }
    )


def strict_notes(v: Genes, fam: Family) -> tuple[str, ...]:
    """Decoder notes for what strict alignment overrides before any planform is built: round or
    raked tips become square (their outline is not straight), tip joins are dropped (a joined
    panel's edges are set by the target tip), a smooth body has no straight edges to snap."""
    if fam.edge_alignment != "strict":
        return ()
    out = [f"strict alignment: edges snapped to {list(fam.edge_angles_deg)} deg"]
    if fam.tip_style != "square":
        out.append(f"strict alignment: {fam.tip_style} tips made square (their outline is curved)")
    joins = [i for i in range(1, surface_count()) if v.pick[f"s{i}_tip_join"] != "none"]
    if joins:
        out.append(f"strict alignment: tip joins of surfaces {joins} dropped")
    if fam.shape.style == "smooth":
        out.append("strict alignment: a smooth body has no straight outline edge to snap")
    else:
        out += coercion_notes(fam.shape)
    return tuple(out)
