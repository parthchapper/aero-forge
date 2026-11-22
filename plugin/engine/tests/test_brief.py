from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from aeroforge.brief import BriefError, load_brief, resolve
from aeroforge.constants import constant, load_table, number
from aeroforge.genetable import unpack
from aeroforge.models import Brief
from aeroforge.reference import BASELINE, ReferenceError, encode, reference_genome

BOTH_MISSIONS = {
    "payload_mass_kg": 0.5,
    "payload_box_m": [0.1, 0.1, 0.1],
    "span_max_m": 2.0,
    "cruise_speed_ms": 15.0,
    "launch": "hand",
    "endurance_min": 30,
    "range_km": 10,
}


def test_sample_brief_resolves_spec_defaults(sample_brief: Brief) -> None:
    b = resolve(sample_brief)
    ratio = number("dash_cruise_ratio", "checks")
    assert b.dash_speed_ms == pytest.approx(ratio * b.cruise_speed_ms)
    assert b.n_limit == number("n_limit_default", "checks")
    assert b.gust_ms == number("gust_ms_default", "checks")
    assert b.efficiency_floor == number("efficiency_floor_default", "checks")
    stall = constant("checks", "launch_stall_max_ms")
    assert isinstance(stall, dict)
    assert b.stall_max_ms == stall[b.launch]


def test_brief_needs_exactly_one_of_endurance_and_range() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        Brief.model_validate(BOTH_MISSIONS)


def test_invalid_brief_file_raises_brief_error(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text("payload_mass_kg: -1\n", encoding="utf-8")
    with pytest.raises(BriefError):
        load_brief(path)


def test_reference_genome_decodes_back_to_its_overrides() -> None:
    genes = unpack(reference_genome(BASELINE))
    for name, value in load_table("reference")[BASELINE].items():
        if name in genes.pick:
            assert genes.pick[name] == str(value)
        else:
            assert genes.num[name] == pytest.approx(value)


def test_unknown_gene_or_out_of_range_value_is_rejected() -> None:
    with pytest.raises(ReferenceError):
        encode({"no_such_gene": 1.0})
    with pytest.raises(ReferenceError):
        encode({"fam_tc": 0.5})


def test_unlisted_genes_take_in_range_defaults() -> None:
    genome = encode({})
    assert np.all((genome >= 0.0) & (genome <= 1.0))
