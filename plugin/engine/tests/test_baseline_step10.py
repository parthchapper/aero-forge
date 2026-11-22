"""Novelty plan step 10 guards: what the decoder produced before any new gene or the area interface.

The frozen JSON in fixtures/baseline_pre_step10/ was written by the step-9 code. These tests must
pass on the untouched decoder and stay green through every stage of step 10 (a Design may gain
fields; shared fields are compared with a float tolerance).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from aeroforge.brief import brief_requirements
from aeroforge.genome import decode
from aeroforge.models import Design, Requirements
from aeroforge.reference import encode, reference_genome, reference_names

FROZEN = Path(__file__).parent / "fixtures" / "baseline_pre_step10"
REL = 1e-9
ABS = 1e-12


def _load(name: str) -> dict:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def _renamed(raw: dict) -> dict:
    """Frozen Requirements JSON through the step-10 field rename (lifting_area_m2 → _min_m2)."""
    return {("lifting_area_min_m2" if k == "lifting_area_m2" else k): v for k, v in raw.items()}


def _migrated_requirements(raw: dict) -> Requirements:
    return Requirements.model_validate(_renamed(raw))


FAMILY_DEFAULTS = {
    "body_nose_frac": 0.22,
    "body_tail_frac": 0.40,
    "body_width_ratio": 1.0,
    "body_tail_upsweep_deg": 0.0,
    "box_layout": "inline",
    "layout": "free",
}


def _migrated_design(raw: dict, lifting_area_m2: float) -> Design:
    """Frozen Design JSON with what step 10 added: the Family fields at their reference defaults
    and the design's own lifting area (the old requirements' S: wing_area_scale 1)."""
    family = {**FAMILY_DEFAULTS, **raw["family"]}
    return Design.model_validate({**raw, "family": family, "lifting_area_m2": lifting_area_m2})


def _same(old: object, new: object, path: str = "") -> None:
    """Every field of the frozen value is present and equal in the new one; new keys are fine."""
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        for key, value in old.items():
            assert key in new, f"{path}.{key} is missing"
            _same(value, new[key], f"{path}.{key}")
    elif isinstance(old, list):
        assert isinstance(new, list) and len(old) == len(new), path
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            _same(a, b, f"{path}[{i}]")
    elif isinstance(old, float):
        assert new == pytest.approx(old, rel=REL, abs=ABS), path
    else:
        assert old == new, path


def _requirements(label: str) -> Requirements:
    return _migrated_requirements(_load(f"requirements_{label}.json"))


@pytest.mark.parametrize("label", ["sample", "stub"])
def test_every_reference_decodes_to_the_same_design_as_before_step_10(label: str) -> None:
    req = _requirements(label)
    frozen = _load(f"references_{label}.json")["designs"]
    assert set(frozen) == set(reference_names())
    for name in reference_names():
        design = decode(reference_genome(name), req)
        _same(frozen[name], design.model_dump(mode="json"), name)


def test_the_stub_requirements_are_unchanged() -> None:
    _same(
        _renamed(_load("requirements_stub.json")),
        brief_requirements("stub").model_dump(mode="json"),
    )


def test_derive_on_the_sample_brief_reproduces_the_frozen_requirements(
    sample_req: Requirements,
) -> None:
    _same(_renamed(_load("requirements_sample.json")), sample_req.model_dump(mode="json"))


def test_random_genomes_keep_their_body_length_when_the_new_genes_sit_at_their_defaults() -> None:
    """The frozen physical values re-encoded into today's table decode to the frozen body0 length
    (after step 10: new genes at ref_default, box_layout inline, wing_area_scale 1)."""
    frozen = _load("random20_stub.json")
    req = brief_requirements("stub")
    for physical, length in zip(frozen["physical"], frozen["body0_length_m"], strict=True):
        design = decode(encode({**physical["num"], **physical["pick"]}), req)
        body0 = next(b for b in design.bodies if b.name == "body0")
        assert body0.length_m == pytest.approx(length, rel=REL)


def test_frozen_designs_validate_as_designs() -> None:
    sample = _load("references_stub.json")["designs"]["conventional"]
    assert isinstance(_migrated_design(sample, _requirements("stub").lifting_area_min_m2), Design)
    assert np.isfinite(_load("random20_stub.json")["body0_length_m"]).all()
