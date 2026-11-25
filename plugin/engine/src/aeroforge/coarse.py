"""Facet coarseness of wings (SPEC §6, U3b): fewer, larger, sharper flat panels.

The coarseness c in [0, 1] (gene facet_coarseness: off, low, mid, high) sets a smallest crease
angle (c x coarse_min_crease_deg) and a limit on the flat segments per side of a wing section.
A wing section keeps the largest segment count at or under the limit whose interior corners all
turn by at least the smallest crease, as long as the polygon moves by at most coarse_move_frac of
the chord; when no count does, the closest allowed one is kept and the shortfall is a note. The
washout becomes whole steps (stepped_levels). Pure; unit chord, degrees."""

import numpy as np

from aeroforge.airfoil import facet_outline
from aeroforge.constants import number
from aeroforge.models import Family

LABELS = {"off": 0.0, "low": "coarse_level_low", "mid": "coarse_level_mid", "high": 1.0}
MIN_FACETS = 2


def level(label: str) -> float:
    """Coarseness of a facet_coarseness gene label."""
    v = LABELS[label]
    return v if isinstance(v, float) else number(v)


def lerp(c: float, loose: str, tight: str) -> float:
    """Constant `loose` at coarseness 0 to constant `tight` at 1."""
    return (1.0 - c) * number(loose) + c * number(tight)


def min_crease_deg(c: float) -> float:
    """Smallest allowed crease between adjacent sheets at coarseness c."""
    return c * number("coarse_min_crease_deg")


def side_turns_deg(outline: np.ndarray) -> np.ndarray:
    """Turn angles [deg] at the interior corners of both sides of a section polygon (the leading
    and trailing edge vertices are meant to be sharp and are left out)."""
    split = len(outline) // 2
    out = []
    for side in (outline[: split + 1], outline[split:]):
        d = np.diff(side, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        out.append(np.abs(np.degrees(np.diff(np.unwrap(ang)))))
    return np.concatenate(out)


def section_move(fam: Family, k0: int, k: int) -> float:
    """Largest distance [chord] between the k0-facet and the k-facet section polygon, read at the
    k0 polygon's vertices."""
    if k0 == k:
        return 0.0
    args = (fam.camber_frac, fam.camber_loc_frac, fam.tc, int(number("airfoil_points_per_side")))
    a, b = facet_outline(*args, k0), facet_outline(*args, k)
    split = len(a) // 2
    move = 0.0
    for sa, sb in (
        (a[: split + 1][::-1], b[: len(b) // 2 + 1][::-1]),
        (a[split:], b[len(b) // 2 :]),
    ):
        move = max(move, float(np.abs(sa[:, 1] - np.interp(sa[:, 0], sb[:, 0], sb[:, 1])).max()))
    return move


def wing_facets(fam: Family) -> tuple[int, tuple[str, ...]]:
    """Flat segments per side the coarseness leaves a wing section, and the notes on what moved."""
    k0, c = fam.airfoil_facets, fam.facet_coarseness
    if not k0 or c <= 0.0:
        return k0, ()
    cap = round(lerp(c, "coarse_wing_facets_loose", "coarse_wing_facets_tight"))
    least, room = min_crease_deg(c), number("coarse_move_frac")
    args = (fam.camber_frac, fam.camber_loc_frac, fam.tc, int(number("airfoil_points_per_side")))
    ok = [k for k in range(MIN_FACETS, k0 + 1) if section_move(fam, k0, k) <= room]
    within = [k for k in ok if k <= cap]
    sharp = [k for k in within if side_turns_deg(facet_outline(*args, k)).min() >= least]
    pick = max(sharp) if sharp else max(within, default=k0)
    notes: list[str] = []
    if pick != k0:
        notes.append(
            f"coarseness {c:.2f}: wing sections {k0} -> {pick} flat segments per side "
            f"(largest vertex move {section_move(fam, k0, pick):.1%} of the chord, cap {room:.0%})"
        )
    turn = float(side_turns_deg(facet_outline(*args, pick)).min())
    if turn < least:
        notes.append(
            f"coarseness {c:.2f}: wing section corners turn by as little as {turn:.1f} deg "
            f"(smallest crease {least:.1f}): "
            + (
                f"{MIN_FACETS} segments is the floor"
                if pick == MIN_FACETS
                else f"a coarser polygon would move it over {room:.0%} of the chord, merge skipped"
            )
        )
    return pick, tuple(notes)


def coarse_family(fam: Family) -> Family:
    """fam with the wing-section facet count the coarseness allows."""
    k, notes = wing_facets(fam)
    return fam if not notes else fam.model_copy(update={"airfoil_facets": k, "facet_notes": notes})


def stepped_levels(twist_deg: float, c: float, strips: int) -> int:
    """Number of washout steps (0: leave the washout linear): the tip twist in steps of c x
    coarse_twist_step_deg, at most one fewer than the strips (so some strips are flat quads) and
    at least one."""
    if c <= 0.0 or twist_deg == 0.0:
        return 0
    n = round(abs(twist_deg) / (c * number("coarse_twist_step_deg")))
    return max(1, min(n, max(strips - 1, 1)))
