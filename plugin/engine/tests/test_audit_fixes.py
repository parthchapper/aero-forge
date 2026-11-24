"""Regression tests for audit fixes, edge guards, CLI error handling, and aero optimizations."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from pydantic import ValidationError

from aeroforge.aero import buildup_pitch_updater, buildup_samples, cn_beta_buildup, point_speeds
from aeroforge.balance import ON_TARGET, _fits, balance
from aeroforge.brief import brief_requirements
from aeroforge.checks import Inputs, check_stall_order, evaluate
from aeroforge.cli import main
from aeroforge.constants import knob
from aeroforge.derive import _aitken
from aeroforge.facet_ends import FacetError
from aeroforge.fitcheck import box_points_inside, shaft_margin
from aeroforge.genetable import DecodeError
from aeroforge.genome import decode, random_genomes
from aeroforge.geom import build_parts
from aeroforge.internals import inside_part
from aeroforge.layout import box_lattice
from aeroforge.mass import BATTERY, cg_range, mass_items, move_item
from aeroforge.models import Brief, Requirements
from aeroforge.prepare import _incidence_deg, prepare
from aeroforge.reference import BASELINE, encode, reference_genome
from aeroforge.search import SearchResult
from aeroforge.search_cli import search_payload
from aeroforge.trim import StallOrder, lateral, tier1
from aeroforge.viewer import _script_json

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("layout", ["side_by_side", "stacked"])
def test_balance_slides_boxes_on_side_by_side_and_stacked_layouts(
    layout: str, sample_brief: Brief, sample_req: Requirements
) -> None:
    g = encode({"body_on": "on", "box_layout": layout, "s0_root_x_frac": 0.35})
    d = decode(g, sample_req)
    assert _fits(d)
    aft = cg_range(mass_items(d, sample_req, sample_brief.payload_mass_kg))[1]
    speeds = point_speeds(sample_req)
    cruise = next(p.speed_ms for p in sample_req.points if p.id == "P3")
    sm = buildup_samples(d, sample_req, speeds, cruise, aft)
    bal = balance(d, sample_req, sample_brief.payload_mass_kg, buildup_samples, sm)
    assert bal.outcome == ON_TARGET
    assert abs(bal.battery_shift_m) + abs(bal.stack_shift_m) > 1e-3


def test_negative_control_deflection_adds_drag_and_never_gives_negative_cd(
    sample_brief: Brief, sample_req: Requirements
) -> None:
    p = prepare(reference_genome(BASELINE), sample_req, sample_brief.payload_mass_kg)
    items = mass_items(p.design, sample_req, sample_brief.payload_mass_kg)
    base_res = tier1(p.design, sample_req, cg_range(items))
    fwd_res = tier1(p.design, sample_req, cg_range(move_item(items, BATTERY, -0.35)))
    base_p3 = next(t for t in base_res.trims if t.point == "P3" and t.cg == "fore")
    fwd_p3 = next(t for t in fwd_res.trims if t.point == "P3" and t.cg == "fore")
    assert fwd_p3.deflection_deg < base_p3.deflection_deg < 0.0
    assert fwd_p3.cd > base_p3.cd > 0.0
    rng = np.random.default_rng(777)
    for row in random_genomes(rng, 25):
        _, rep = evaluate(row, sample_brief, sample_req, 1)
        eff = next(c for c in rep.checks if c.id == "efficiency")
        ene = next(c for c in rep.checks if c.id == "energy")
        assert float(eff.detail["cruise_ld"]) > 0.0
        assert float(ene.detail["electric_power_w"]) > 0.0


def test_mirrored_pod_containment_and_shaft_margin_are_left_right_symmetric() -> None:
    req = brief_requirements("stub")
    g = encode(
        {
            "body_on": "on",
            "pods_on": "on",
            "prop_pattern": "pods",
            "prop_direction": "tractor",
            "pod_x_frac": 0.10,
            "pod_length_frac": 0.25,
        }
    )
    d = decode(g, req)
    pod = next(b for b in d.bodies if b.name == "pod")
    right = np.array([[pod.nose_m[0] + pod.length_m * 0.5, pod.nose_m[1], pod.nose_m[2]]])
    left = right * np.array([1.0, -1.0, 1.0])
    assert inside_part(d, "pod", right).all()
    assert inside_part(d, "pod", left).all()
    assert shaft_margin(d) == 0.0


def test_check_stall_order_fails_on_non_positive_fore_ratio(
    sample_brief: Brief, sample_req: Requirements
) -> None:
    d = decode(reference_genome(BASELINE), sample_req)
    items = mass_items(d, sample_req, sample_brief.payload_mass_kg)
    base = tier1(d, sample_req, cg_range(items))
    for fore in (-0.2, 0.0):
        aero = replace(
            base,
            tier=2,
            stall_order=(StallOrder(fore_ratio=fore, other_ratio=0.9, peak_span_frac=0.3),),
        )
        res = check_stall_order(Inputs(d, sample_brief, sample_req, items, aero))
        assert not res.passed and res.margin < 0.0


def _occ_contains(solid: object, pts: np.ndarray) -> np.ndarray:
    clf = BRepClass3d_SolidClassifier(solid.wrapped)  # type: ignore[attr-defined]
    out = []
    for p in pts:
        clf.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-7)
        out.append(clf.State() in (TopAbs_IN, TopAbs_ON))
    return np.array(out)


def test_inside_loft_matches_real_cad_on_tilted_wing() -> None:
    req = brief_requirements("small_boxes")
    g = encode(
        {
            "body_on": "off",
            "box_layout": "wing_root",
            "s0_aspect_ratio": 4.0,
            "s0_twist_deg": -6.0,
            "fam_tc": 0.15,
            "wing_area_scale": 2.5,
            "s0_root_chord_scale": 2.0,
            "s1_active": "off",
            "s2_active": "off",
            "s3_active": "off",
        }
    )
    d = decode(g, req)
    tilted = d.model_copy(
        update={"surfaces": (d.surfaces[0].model_copy(update={"incidence_deg": 6.0}),)}
    )
    parts = build_parts(tilted)
    avionics = next(b for b in tilted.boxes if b.name == "avionics")
    pts = box_lattice(avionics, 5)
    analytic = box_points_inside(tilted, avionics)
    cad = _occ_contains(parts["surface0_R"], pts) | _occ_contains(parts["surface0_L"], pts)
    assert not analytic.all()
    assert np.mean(analytic == cad) > 0.98


def test_aitken_never_extrapolates_non_positive() -> None:
    assert _aitken(100.0, 60.0, 25.0) == 25.0
    assert _aitken(10.0, 4.0, 1.0) == 1.0
    assert _aitken(10.0, 6.0, 4.0) == pytest.approx(2.0)


def test_brief_rejects_inconsistent_speed_ordering(sample_brief: Brief) -> None:
    base = sample_brief.model_dump()
    with pytest.raises(ValidationError, match="dash_speed_ms"):
        Brief.model_validate({**base, "dash_speed_ms": base["cruise_speed_ms"] - 1.0})
    with pytest.raises(ValidationError, match="stall_max_ms"):
        Brief.model_validate({**base, "stall_max_ms": base["cruise_speed_ms"]})


def test_facet_error_is_a_decode_error() -> None:
    assert issubclass(FacetError, DecodeError)
    assert issubclass(FacetError, ValueError)


def test_incidence_deg_returns_nan_on_singular_system(
    sample_brief: Brief, sample_req: Requirements
) -> None:
    d = decode(reference_genome(BASELINE), sample_req)
    items = mass_items(d, sample_req, sample_brief.payload_mass_kg)
    aft = cg_range(items)[1]
    sm = buildup_samples(d, sample_req, point_speeds(sample_req), 18.0, aft)
    singular = replace(sm, surface_force_n=np.zeros_like(sm.surface_force_n))
    assert np.isnan(_incidence_deg(d, sample_req, items, singular))


def test_cli_edge_cases_return_clean_exit_codes(tmp_path: Path, sample_req: Requirements) -> None:
    code = main(
        [
            "demo",
            str(FIXTURES / "brief_dash_500ms.yaml"),
            "--runs",
            "1",
            "--out",
            str(tmp_path / "d"),
        ]
    )
    assert code == 1
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "brief.yaml").write_bytes((FIXTURES / "brief_sample.yaml").read_bytes())
    (folder / "requirements.json").write_text(sample_req.model_dump_json(), encoding="utf-8")
    assert (
        main(["verify", str(folder), "--history", str(tmp_path / "h.jsonl"), "--workers", "1"]) == 2
    )
    (folder / "finalists.json").write_text('{"seed": 1, "randomness": 0.5, "finalists": []}')
    assert (
        main(["verify", str(folder), "--history", str(tmp_path / "h.jsonl"), "--workers", "1"]) == 0
    )
    assert main(["build", str(folder), "--finalist", "../escape"]) == 2
    assert "<" not in _script_json({"x": "<!-- </script> -->"})


def test_fast_aero_helpers_match_full_buildup_samples(
    sample_brief: Brief, sample_req: Requirements
) -> None:
    p = prepare(reference_genome(BASELINE), sample_req, sample_brief.payload_mass_kg)
    d = p.design
    aft = cg_range(mass_items(d, sample_req, sample_brief.payload_mass_kg))[1]
    speeds = point_speeds(sample_req)
    cruise = next(pt.speed_ms for pt in sample_req.points if pt.id == "P3")
    full = buildup_samples(d, sample_req, speeds, cruise, aft)
    si = int(np.flatnonzero(speeds == cruise)[0])
    from aeroforge.aero import flight_state

    want_cnb = lateral(full, si, aft, flight_state(d, sample_req), cruise)[0]
    assert cn_beta_buildup(d, sample_req, cruise, aft) == pytest.approx(
        want_cnb, rel=1e-12, abs=1e-12
    )
    sm0, updater = buildup_pitch_updater(d, sample_req, speeds, cruise, aft)
    assert np.allclose(sm0.force_n, full.force_n, atol=1e-12)
    tilted_surfaces = tuple(
        s.model_copy(update={"incidence_deg": s.incidence_deg + 2.0})
        if s.name == full.pitch_control
        else s
        for s in d.surfaces
    )
    tilted = d.model_copy(update={"surfaces": tilted_surfaces})
    fast_t = updater(tilted, aft)
    full_t = buildup_samples(tilted, sample_req, speeds, cruise, aft)
    assert np.allclose(fast_t.force_n, full_t.force_n, atol=1e-12)
    assert np.allclose(fast_t.moment_nm, full_t.moment_nm, atol=1e-12)
    assert np.allclose(fast_t.profile_n, full_t.profile_n, atol=1e-12)


def test_search_payload_carries_critic_strangeness_weight(
    sample_brief: Brief, sample_req: Requirements
) -> None:
    res = SearchResult((), (), 0, False, True, 0.2, 0.6, 0.75)
    payload = search_payload(res, sample_brief, sample_req, 1, 10)
    assert payload["critic_strangeness_weight"] == pytest.approx(
        knob("knob_critic_strangeness_weight", 0.75)
    )
