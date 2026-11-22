import numpy as np
import pytest

from aeroforge.brief import BRIEFS, brief_requirements
from aeroforge.descriptor import DESC_SIZE, desc, median_nearest, novelty
from aeroforge.fitcheck import shaft_margin
from aeroforge.genome import decode
from aeroforge.models import Design, Propulsor, Requirements
from aeroforge.reference import ReferenceError, reference_genome, reference_names

SLOT_DESIGNS = 200
MIN_REFERENCES = 40


def test_desc_is_identical_under_slot_reordering(designs: list[Design]) -> None:
    rng = np.random.default_rng(4)
    for d in designs[:SLOT_DESIGNS]:
        order = rng.permutation(len(d.surfaces))
        shuffled = d.model_copy(update={"surfaces": tuple(d.surfaces[i] for i in order)})
        assert np.array_equal(desc(shuffled), desc(d))


def test_desc_is_sixteen_numbers_in_unit_range(designs: list[Design]) -> None:
    x = np.array([desc(d) for d in designs[:SLOT_DESIGNS]])
    assert x.shape == (SLOT_DESIGNS, DESC_SIZE)
    assert np.isfinite(x).all() and x.min() >= 0.0 and x.max() <= 1.0


def test_reference_set_has_forty_entries_that_decode_on_every_brief(
    sample_req: Requirements,
) -> None:
    names = reference_names()
    assert len(names) >= MIN_REFERENCES
    for req in (sample_req, *(brief_requirements(b) for b in BRIEFS)):
        for n in names:
            decode(reference_genome(n), req)


def test_conventional_canard_and_flying_wing_are_pairwise_beyond_d_min(
    sample_req: Requirements,
) -> None:
    refs = {n: desc(decode(reference_genome(n), sample_req)) for n in reference_names()}
    d_min = median_nearest(np.array(list(refs.values())))
    trio = [refs[n] for n in ("conventional", "canard", "flying_wing")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.linalg.norm(trio[i] - trio[j]) > d_min


def test_novelty_is_the_nearest_distance() -> None:
    known = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert novelty(np.array([3.0, 0.0]), known) == pytest.approx(3.0)
    assert novelty(np.array([1.0, 1.0]), np.zeros((0, 2))) == float("inf")


def test_reference_base_cycle_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {"a": {"base": "b"}, "b": {"base": "a"}}
    monkeypatch.setattr("aeroforge.reference.load_table", lambda _: table)
    with pytest.raises(ReferenceError):
        reference_genome("a")


def _pusher_through_fin(d: Design) -> Design:
    """d with a pusher shaft running fore-aft through the fin at half its height."""
    fin = next(s for s in d.surfaces if not s.mirrored)
    root, tip = fin.sections[0], fin.sections[-1]
    z = (root.le_m[2] + tip.le_m[2]) / 2
    x = (root.le_m[0] + tip.le_m[0]) / 2
    chord = (root.chord_m + tip.chord_m) / 2
    attach, hub = (x - chord / 2, 0.0, z), (x + 3 * chord / 2, 0.0, z)
    prop = d.propulsors[0].model_copy(update={"attach_m": attach, "hub_m": hub, "pusher": True})
    return d.model_copy(update={"propulsors": (prop,)})


def test_shaft_through_a_fin_has_a_negative_margin(conventional: Design) -> None:
    assert shaft_margin(conventional) == 0.0
    assert shaft_margin(_pusher_through_fin(conventional)) < 0.0


def test_shaft_check_ignores_the_shaft_inside_bodies(conventional: Design) -> None:
    p: Propulsor = conventional.propulsors[0]
    body = next(b for b in conventional.bodies if b.name == "body0")
    hub = tuple(float(v) for v in np.add(body.nose_m, (body.length_m / 2, 0.0, 0.0)))
    inside = p.model_copy(update={"hub_m": hub})
    assert shaft_margin(conventional.model_copy(update={"propulsors": (inside,)})) == 0.0
