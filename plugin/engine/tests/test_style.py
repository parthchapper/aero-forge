import numpy as np
import pytest

from aeroforge.models import Design
from aeroforge.style import (
    FEATURES,
    StyleError,
    features,
    fin_terms,
    is_fin,
    model_from_table,
    packaged_model,
    panels,
    score,
    wing_surfaces,
)


def test_features_are_finite_on_every_design(decoded: list[Design]) -> None:
    x = np.array([features(d) for d in decoded])
    assert x.shape == (len(decoded), len(FEATURES)) and np.isfinite(x).all()


def test_removing_fin_surfaces_leaves_features_unchanged(designs: list[Design]) -> None:
    with_fins = [d for d in designs if fin_terms(d)[1] > 0]
    assert with_fins, "seeded genomes should include fins"
    for d in with_fins:
        bare = d.model_copy(update={"surfaces": wing_surfaces(d)})
        assert np.array_equal(features(bare), features(d))


def test_fin_terms_are_zero_without_fin_panels(designs: list[Design]) -> None:
    for d in designs:
        no_fin = tuple(s for s in d.surfaces if not any(is_fin(p) for p in panels(s)))
        if no_fin:
            bare = d.model_copy(update={"surfaces": no_fin})
            assert np.array_equal(fin_terms(bare), [0.0, 0.0])


def test_packaged_model_scores_inside_unit_interval(designs: list[Design]) -> None:
    model = packaged_model()
    s = np.array([score(features(d), model) for d in designs[:200]])
    assert np.all((s > 0.0) & (s < 1.0))


def test_model_table_must_list_the_feature_names_in_order() -> None:
    rows = [{"name": n, "mean": 0.0, "std": 1.0, "weight": 0.0} for n in reversed(FEATURES)]
    with pytest.raises(StyleError):
        model_from_table({"bias": 0.0, "features": rows})


def test_model_table_rejects_non_positive_std() -> None:
    rows = [{"name": n, "mean": 0.0, "std": 0.0, "weight": 0.0} for n in FEATURES]
    with pytest.raises(StyleError):
        model_from_table({"bias": 0.0, "features": rows})
