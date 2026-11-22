import pytest
from pydantic import ValidationError

from aeroforge.brief import BRIEFS, brief_requirements
from aeroforge.genetable import GenomeError
from aeroforge.models import Requirements


@pytest.mark.parametrize("name", BRIEFS)
def test_every_stand_in_brief_validates(name: str) -> None:
    assert isinstance(brief_requirements(name), Requirements)


def test_unknown_stand_in_brief_raises_typed_error() -> None:
    with pytest.raises(GenomeError):
        brief_requirements("no_such_brief")


@pytest.mark.parametrize(
    "field", ["mtow_kg", "lifting_area_min_m2", "span_max_m", "ref_length_m", "battery_wh"]
)
@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_requirements_reject_non_positive_or_non_finite_sizes(
    req: Requirements, field: str, bad: float
) -> None:
    with pytest.raises(ValidationError):
        Requirements.model_validate({**req.model_dump(), field: bad})


def test_requirements_reject_a_degenerate_payload_box(req: Requirements) -> None:
    with pytest.raises(ValidationError):
        Requirements.model_validate({**req.model_dump(), "payload_box_m": (0.1, 0.0, 0.05)})


def test_requirements_reject_an_unknown_recovery(req: Requirements) -> None:
    with pytest.raises(ValidationError):
        Requirements.model_validate({**req.model_dump(), "recovery": "parachute"})


def test_recovery_defaults_to_belly(req: Requirements) -> None:
    raw = req.model_dump(exclude={"recovery"})
    assert Requirements.model_validate(raw).recovery == "belly"
