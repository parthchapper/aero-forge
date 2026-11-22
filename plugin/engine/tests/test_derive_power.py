"""Step 8 (novelty plan): derive stops a brief whose baseline cannot approach the dash speed."""

from pathlib import Path

import pytest

from aeroforge.brief import load_brief
from aeroforge.derive import InfeasibleError, derive

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.slow  # one derive closure per brief (20-30 s each)


def test_the_500ms_brief_stops_at_the_power_check_before_any_search() -> None:
    with pytest.raises(InfeasibleError) as info:
        derive(load_brief(FIXTURES / "brief_dash_500ms.yaml"))
    assert info.value.check == "power"
    assert "installed" in str(info.value)


def test_the_80ms_brief_passes_the_power_guard() -> None:
    req = derive(load_brief(FIXTURES / "brief_dash_80ms.yaml"))
    assert req.lifting_area_min_m2 > 0.0 and req.mtow_kg > 0.0
