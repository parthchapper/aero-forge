"""Facet coarseness of wings (SPEC §6, U3b): fewer, larger, sharper flat panels.

The coarseness c in [0, 1] (gene facet_coarseness: off, low, mid, high) sets a smallest crease
angle (c x coarse_min_crease_deg) and a limit on the flat segments per side of a wing section.
A wing section drops its flattest corners (facet_decimate, flattest first) until every interior
corner turns by at least the smallest crease and the segment budget is met, while the polygon stays
within coarse_move_frac of the chord of the gene polygon; a corner the cap keeps is a note. The
washout becomes whole steps (stepped_levels). Pure; unit chord, degrees."""

from aeroforge.airfoil import facet_sides
from aeroforge.constants import number
from aeroforge.facet_decimate import min_crease_deg, pack, unpack
from aeroforge.models import Family

LABELS = {"off": 0.0, "low": "coarse_level_low", "mid": "coarse_level_mid", "high": 1.0}


def level(label: str) -> float:
    """Coarseness of a facet_coarseness gene label."""
    v = LABELS[label]
    return v if isinstance(v, float) else number(v)


def lerp(c: float, loose: str, tight: str) -> float:
    """Constant `loose` at coarseness 0 to constant `tight` at 1."""
    return (1.0 - c) * number(loose) + c * number(tight)


def wing_notes(fam: Family, code: int) -> tuple[str, ...]:
    """Decoder notes on what decimation did to the wing section of facet code `code`."""
    k0, c = unpack(code)
    _, _, info = facet_sides(
        fam.camber_frac, fam.camber_loc_frac, fam.tc, int(number("airfoil_points_per_side")), code
    )
    if info is None:
        return ()
    du, dl = info
    up, lo = len(du.keep) - 1, len(dl.keep) - 1
    cap, least = number("coarse_move_frac"), min_crease_deg(c)
    out = []
    if (up, lo) != (k0, k0):
        out.append(
            f"coarseness {c:.2f}: wing sections {k0} -> {up} upper / {lo} lower flat segments "
            f"(flattest corners dropped; largest vertex move {max(du.move, dl.move):.1%} of the "
            f"chord, cap {cap:.0%})"
        )
    if du.blocked or dl.blocked:
        sides = " and ".join(n for n, d in (("upper", du), ("lower", dl)) if d.blocked)
        out.append(
            f"coarseness {c:.2f}: wing section {sides} side: a corner turning under {least:.0f} "
            f"deg stays (dropping it would move vertices over {cap:.0%} of the chord): merge "
            "blocked by the cap"
        )
    return tuple(out)


def coarse_family(fam: Family) -> Family:
    """fam with the facet code of the coarseness (the wing-section polygon is decimated where
    the airfoil is built) and the notes on what that did."""
    k0, c = fam.airfoil_facets, fam.facet_coarseness
    if not k0 or c <= 0.0:
        return fam
    code = pack(k0, c)
    return fam.model_copy(update={"airfoil_facets": code, "facet_notes": wing_notes(fam, code)})


def stepped_levels(twist_deg: float, c: float, strips: int) -> int:
    """Number of washout steps (0: leave the washout linear): the tip twist in steps of c x
    coarse_twist_step_deg, at most one fewer than the strips (so some strips are flat quads) and
    at least one."""
    if c <= 0.0 or twist_deg == 0.0:
        return 0
    n = round(abs(twist_deg) / (c * number("coarse_twist_step_deg")))
    return max(1, min(n, max(strips - 1, 1)))
