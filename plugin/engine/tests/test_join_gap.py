"""Step 7 guard: a tip-joined surface's tip is pulled inboard in the VLM geometry only."""

import numpy as np

from aeroforge.airplane import _tip_gapped, aero_sections
from aeroforge.constants import number
from aeroforge.models import Design


def _joined_surfaces(designs: list[Design]) -> list:
    return [s for d in designs for s in d.surfaces if s.joined_to is not None]


def test_the_vlm_gap_pulls_each_joined_tip_inboard_by_the_gap_share_of_its_last_panel(
    designs: list[Design],
) -> None:
    joined = _joined_surfaces(designs)
    assert joined, "seeded genomes should include tip-joined surfaces"
    k = number("vlm_join_gap_frac")
    for s in joined:
        secs = aero_sections(s)
        gapped = _tip_gapped(secs)
        inner, tip, moved = (
            np.array(secs[-2].le_m),
            np.array(secs[-1].le_m),
            np.array(gapped[-1].le_m),
        )
        assert np.allclose(moved - tip, k * (inner - tip))
        assert gapped[:-1] == secs[:-1]


def test_the_design_itself_keeps_the_shared_tip(designs: list[Design]) -> None:
    """The gap is a solver device: the delivered geometry still meets at one tip point."""
    for d in designs:
        by_name = {s.name: s for s in d.surfaces}
        for s in d.surfaces:
            if s.joined_to is None:
                continue
            target = by_name[s.joined_to]
            own = np.array(s.sections[-1].le_m)
            other = np.array(target.sections[target.n_structural - 1].le_m)
            assert np.allclose(own, other, atol=1e-9)
