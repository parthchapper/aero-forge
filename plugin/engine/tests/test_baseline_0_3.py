"""Upgrade 1.0 guard: what the 0.3 decoder produced after U0 (decisions A1 and B1 applied).

fixtures/baseline_0.3/ was written by the U0 code. Through U1-U9 every reference must decode to
the same Design, and the frozen random genomes (one [0, 1] value per 0.3 gene name; genes added
later take their ref_default) must too. A Design may gain fields: only the frozen keys are compared, floats
with a tolerance. The yaml files are snapshots of the 0.3 tables, kept for reference only.
"""

import json
from pathlib import Path

import pytest

from aeroforge.brief import brief_requirements
from aeroforge.genetable import gene_table
from aeroforge.genome import decode
from aeroforge.models import Requirements
from aeroforge.reference import encode, reference_genome, reference_names


def gene_names() -> list[str]:
    return [g.name for g in gene_table()]


FROZEN = Path(__file__).parent / "fixtures" / "baseline_0.3"
REL = 1e-9
ABS = 1e-12


def _load(name: str) -> dict:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def _same(old: object, new: object, path: str = "") -> None:
    """Every frozen field is present and equal in the new value; new keys are fine."""
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


@pytest.mark.parametrize("label", ["sample", "stub"])
def test_every_reference_decodes_to_the_same_design_as_in_0_3(label: str) -> None:
    req = Requirements.model_validate(_load(f"requirements_{label}.json"))
    frozen = _load(f"references_{label}.json")["designs"]
    assert set(frozen) == set(reference_names())
    for name in reference_names():
        design = decode(reference_genome(name), req)
        _same(frozen[name], design.model_dump(mode="json"), name)


def test_random_genomes_given_as_physical_values_decode_to_the_same_designs_as_in_0_3() -> None:
    frozen = _load("random20_stub.json")
    req = brief_requirements("stub")
    for i, (physical, design) in enumerate(zip(frozen["physical"], frozen["designs"], strict=True)):
        genome = encode({})  # genes added after 0.3 sit at their ref_default
        for k, name in enumerate(gene_names()):
            if name in physical:
                genome[k] = physical[name]
        decoded = decode(genome, req)
        _same(design, decoded.model_dump(mode="json"), f"random[{i}]")


def test_the_stub_requirements_are_unchanged_from_0_3() -> None:
    _same(_load("requirements_stub.json"), brief_requirements("stub").model_dump(mode="json"))


def test_derive_on_the_sample_brief_reproduces_the_0_3_requirements(
    sample_req: Requirements,
) -> None:
    _same(_load("requirements_sample.json"), sample_req.model_dump(mode="json"))
