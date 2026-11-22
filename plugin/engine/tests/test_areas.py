"""Step 9 (novelty plan): top-down and frontal shadow areas against closed forms and a raster."""

import statistics
import time
from itertools import pairwise
from math import gamma

import numpy as np
import pytest

from aeroforge.areas import areas, body_top_share, frontal_area_m2, top_area_m2
from aeroforge.models import Body, Design, Section, Surface
from aeroforge.shapes import body_radius_frac

TOL = 0.02  # the plan's accuracy for closed-form shapes
RASTER_TOL = 0.01  # measured gaps vs the grids: <= 0.08 % (6 top views, 4 frontal views)
CHORD_M, HALF_SPAN_M, TC = 0.2, 0.6, 0.1
RASTER_CELL_M = 0.002
RASTER_DESIGNS = 6
FRONT_DESIGNS = 4
RASTER_FRONT_CELL_M = 0.001
BODY_SECTIONS = 100
WINDOW_PAD_M = 0.02
MIN_PANEL_M = 1.0e-6
TIMED_DESIGNS = 300
MS_BUDGET = 10.0


def _body(a: float, b: float, n: float, **kw: float) -> Body:
    fields = dict(
        name="body0",
        mirrored=False,
        nose_m=(0.0, 0.0, 0.0),
        length_m=1.0,
        half_width_m=a,
        half_height_m=b,
        section_exp=n,
        nose_frac=0.2,
        tail_frac=0.3,
        tail_radius_frac=0.0,
    )
    return Body(**{**fields, **kw})


def _wing(
    base: Design,
    x0: float = 0.0,
    cant: float = 0.0,
    half_span: float = HALF_SPAN_M,
    mirrored: bool = True,
) -> Surface:
    """A rectangular wing (chord CHORD_M) from the centre line out to half_span [m]."""
    right = (half_span * np.cos(np.radians(cant)), half_span * np.sin(np.radians(cant)))
    secs = (
        Section(le_m=(x0, 0.0, 0.0), chord_m=CHORD_M, cant_deg=cant),
        Section(le_m=(x0, right[0], right[1]), chord_m=CHORD_M, cant_deg=cant),
    )
    return base.surfaces[0].model_copy(
        update={
            "sections": secs,
            "n_structural": 2,
            "mirrored": mirrored,
            "joined_to": None,
            "tc": TC,
            "name": f"wing{x0:g}",
        }
    )


def _design(
    base: Design, surfaces: tuple[Surface, ...] = (), bodies: tuple[Body, ...] = ()
) -> Design:
    return base.model_copy(update={"surfaces": surfaces, "bodies": bodies})


@pytest.mark.parametrize(("n", "a", "b"), [(2.0, 0.10, 0.07), (2.5, 0.12, 0.12), (4.0, 0.10, 0.05)])
def test_a_super_ellipse_body_frontal_area_matches_the_closed_form(
    conventional: Design, n: float, a: float, b: float
) -> None:
    exact = 4 * a * b * gamma(1 + 1 / n) ** 2 / gamma(1 + 2 / n)
    got = frontal_area_m2(_design(conventional, bodies=(_body(a, b, n),)))
    assert got == pytest.approx(exact, rel=TOL)


def test_a_body_top_area_matches_a_numeric_integral_of_its_width_law(conventional: Design) -> None:
    body = _body(0.1, 0.07, 2.2, tail_radius_frac=0.15)
    t = np.linspace(0.0, 1.0, 20001)
    exact = 2 * body.half_width_m * body.length_m * np.trapezoid(body_radius_frac(body, t), t)
    d = _design(conventional, bodies=(body,))
    assert top_area_m2(d) == pytest.approx(exact, rel=TOL)


def test_a_flat_rectangular_wing_has_top_area_span_times_chord_and_thin_frontal_area(
    conventional: Design,
) -> None:
    d = _design(conventional, (_wing(conventional),))
    assert top_area_m2(d) == pytest.approx(2 * HALF_SPAN_M * CHORD_M, rel=TOL)
    assert frontal_area_m2(d) == pytest.approx(2 * HALF_SPAN_M * TC * CHORD_M, rel=TOL)


def test_a_wing_canted_to_90_degrees_has_almost_no_top_area_but_its_full_frontal_height(
    conventional: Design,
) -> None:
    flat = top_area_m2(_design(conventional, (_wing(conventional),)))
    fin = _design(conventional, (_wing(conventional, cant=90.0, half_span=0.3, mirrored=False),))
    assert top_area_m2(fin) < 0.01 * flat
    assert frontal_area_m2(fin) == pytest.approx(0.3 * TC * CHORD_M, rel=0.05)


def test_overlapping_parts_count_once(conventional: Design) -> None:
    one = _design(conventional, (_wing(conventional),))
    twin = _design(conventional, (_wing(conventional), _wing(conventional)))
    shifted = _design(conventional, (_wing(conventional), _wing(conventional, x0=CHORD_M / 2)))
    assert top_area_m2(twin) == pytest.approx(top_area_m2(one), rel=1e-6)
    assert top_area_m2(shifted) == pytest.approx(2 * HALF_SPAN_M * 1.5 * CHORD_M, rel=TOL)
    assert frontal_area_m2(twin) == pytest.approx(frontal_area_m2(one), rel=1e-6)


def test_a_body_overlapping_a_wing_adds_less_than_its_own_area(conventional: Design) -> None:
    body = _body(0.1, 0.07, 2.0, nose_m=(-0.2, 0.0, 0.0))
    wing = _wing(conventional, x0=0.1)
    alone = top_area_m2(_design(conventional, bodies=(body,)))
    both = top_area_m2(_design(conventional, (wing,), (body,)))
    wing_alone = top_area_m2(_design(conventional, (wing,)))
    assert max(alone, wing_alone) < both < alone + wing_alone


def test_body_top_share_is_one_for_a_bare_body_zero_without_one_and_the_ratio_otherwise(
    conventional: Design,
) -> None:
    body = _body(0.1, 0.07, 2.0, nose_m=(-1.5, 0.0, 0.0))  # ends before the wing starts
    wing = _wing(conventional)
    assert body_top_share(_design(conventional, bodies=(body,))) == pytest.approx(1.0)
    assert body_top_share(_design(conventional, (wing,))) == 0.0
    b, w = (
        top_area_m2(_design(conventional, **{k: v}))
        for k, v in (("bodies", (body,)), ("surfaces", (wing,)))
    )
    assert body_top_share(_design(conventional, (wing,), (body,))) == pytest.approx(
        b / (b + w), rel=1e-3
    )


def test_areas_agree_with_the_single_measures_and_stay_in_range(designs: list[Design]) -> None:
    for d in designs[:60]:
        got = areas(d)
        assert got.top_m2 == pytest.approx(top_area_m2(d))
        assert got.frontal_m2 == pytest.approx(frontal_area_m2(d))
        assert got.body_top_share == pytest.approx(body_top_share(d))
        assert got.top_m2 > 0.0 and got.frontal_m2 > 0.0 and 0.0 <= got.body_top_share <= 1.0


def _in_quads(x: np.ndarray, y: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Point-in-polygon by sign of cross products (convex quads, either winding)."""
    hit = np.zeros(x.shape, dtype=bool)
    for q in quads:
        signs = []
        for (x0, y0), (x1, y1) in zip(q, np.roll(q, -1, axis=0), strict=True):
            signs.append((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0))
        s = np.array(signs)
        hit |= (s >= 0).all(axis=0) | (s <= 0).all(axis=0)
    return hit


def _raster_top(d: Design) -> float:
    """Brute-force top-view area: grid cells whose centre is in a panel or a body outline."""
    from aeroforge.style import panels

    quads = []
    for s in d.surfaces:
        for p in panels(s):
            q = p.pts_m[[0, 1, 3, 2]][:, :2]  # (x, y)
            quads.append(q)
            if s.mirrored:
                quads.append(q * np.array([1.0, -1.0]))
    pts = np.vstack(
        [q for q in quads]
        + [
            [
                [b.nose_m[0], b.nose_m[1] + sgn * b.half_width_m],
                [b.nose_m[0] + b.length_m, b.nose_m[1]],
            ]
            for b in d.bodies
            for sgn in (1, -1)
        ]
        or [np.zeros((1, 2))]
    )
    lo, hi = pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05
    xs = np.arange(lo[0], hi[0], RASTER_CELL_M) + RASTER_CELL_M / 2
    ys = np.arange(lo[1], hi[1], RASTER_CELL_M) + RASTER_CELL_M / 2
    x, y = np.meshgrid(xs, ys)
    inside = _in_quads(x, y, quads) if quads else np.zeros(x.shape, dtype=bool)
    for b in d.bodies:
        t = (x - b.nose_m[0]) / b.length_m
        for y0 in [b.nose_m[1], -b.nose_m[1]] if b.mirrored else [b.nose_m[1]]:
            width = b.half_width_m * body_radius_frac(b, t)
            inside |= (t >= 0.0) & (t <= 1.0) & (np.abs(y - y0) <= width)
    return float(inside.sum() * RASTER_CELL_M**2)


@pytest.mark.slow  # a 2 mm grid over 6 designs, about 6 s
def test_top_area_agrees_with_a_brute_force_grid_on_random_designs(designs: list[Design]) -> None:
    for d in designs[:RASTER_DESIGNS]:
        assert top_area_m2(d) == pytest.approx(_raster_top(d), rel=RASTER_TOL)


def _front_bands(d: Design) -> np.ndarray:
    """Thickness bands (y, z) written from the definition: tc x chord segments along each
    section normal, joined between sections; mirrored instances flipped in y."""
    quads = []
    for s in d.surfaces:
        for a, b in pairwise(s.sections):
            if np.hypot(b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]) < MIN_PANEL_M:
                continue
            ends = []
            for sec in (a, b):
                n = np.array([-np.sin(np.radians(sec.cant_deg)), np.cos(np.radians(sec.cant_deg))])
                c = np.array(sec.le_m[1:])
                ends.append((c - n * s.tc * sec.chord_m / 2, c + n * s.tc * sec.chord_m / 2))
            q = np.array([ends[0][0], ends[0][1], ends[1][1], ends[1][0]])
            quads += [q, q * np.array([-1.0, 1.0])] if s.mirrored else [q]
    return np.array(quads)


def _raster_front(d: Design) -> float:
    """Brute-force frontal area: a (y, z) grid; bands by point-in-quad, bodies by testing the
    super-ellipse of 200 sections along x."""
    bands = _front_bands(d)
    # the window follows the design: a strongly canted wing plus a tall outer panel leaves any
    # fixed height behind (the first version clipped such designs and read 2 % low)
    corners = [bands.reshape(-1, 2)]
    for b in d.bodies:
        reach = b.tail_frac * b.length_m * np.tan(np.radians(b.tail_upsweep_deg))
        for y0 in (b.nose_m[1], -b.nose_m[1]):
            corners.append([[y0 - b.half_width_m, b.nose_m[2] - b.half_height_m]])
            corners.append([[y0 + b.half_width_m, b.nose_m[2] + b.half_height_m + reach]])
    box = np.vstack(corners)
    lo, hi = box.min(axis=0) - WINDOW_PAD_M, box.max(axis=0) + WINDOW_PAD_M
    ys = np.arange(lo[0], hi[0], RASTER_FRONT_CELL_M) + RASTER_FRONT_CELL_M / 2
    zs = np.arange(lo[1], hi[1], RASTER_FRONT_CELL_M) + RASTER_FRONT_CELL_M / 2
    y, z = np.meshgrid(ys, zs)
    inside = _in_quads(y, z, bands)
    for b in d.bodies:
        for y0 in [b.nose_m[1], -b.nose_m[1]] if b.mirrored else [b.nose_m[1]]:
            j0, j1 = np.searchsorted(ys, [y0 - b.half_width_m, y0 + b.half_width_m])
            # the tail cone's axis rises linearly at tail_upsweep_deg (written out here, not
            # shapes.body_axis_z), so the section window reaches up by the full rise
            slope = np.tan(np.radians(b.tail_upsweep_deg))
            max_rise = b.tail_frac * b.length_m * slope
            i0, i1 = np.searchsorted(
                zs, [b.nose_m[2] - b.half_height_m, b.nose_m[2] + b.half_height_m + max_rise]
            )
            sy, sz = y[i0:i1, j0:j1], z[i0:i1, j0:j1]
            for t in np.linspace(0.0, 1.0, BODY_SECTIONS):
                r = float(body_radius_frac(b, np.array([t]))[0])
                if r <= 0.0:
                    continue
                centre_z = b.nose_m[2] + max(t - (1.0 - b.tail_frac), 0.0) * b.length_m * slope
                level = (np.abs(sy - y0) / (b.half_width_m * r)) ** b.section_exp + (
                    np.abs(sz - centre_z) / (b.half_height_m * r)
                ) ** b.section_exp
                inside[i0:i1, j0:j1] |= level <= 1.0
    return float(inside.sum() * RASTER_FRONT_CELL_M**2)


def _has_pod_and_canted_outer_panel(d: Design) -> bool:
    outer = [s for s in d.surfaces if s.n_structural >= 3 and abs(s.sections[-1].cant_deg) > 20.0]
    return any(b.mirrored for b in d.bodies) and bool(outer)


@pytest.mark.slow  # a 1 mm (y, z) grid with 100 body sections per body, about 30 s
def test_frontal_area_agrees_with_a_brute_force_grid_on_designs_with_pods_and_canted_panels(
    designs: list[Design],
) -> None:
    picked = [d for d in designs if _has_pod_and_canted_outer_panel(d)][:FRONT_DESIGNS]
    assert len(picked) == FRONT_DESIGNS, "seeded genomes should include such designs"
    for d in picked:
        assert frontal_area_m2(d) == pytest.approx(_raster_front(d), rel=RASTER_TOL)


@pytest.mark.slow  # user decision (upgrade U0): host-speed dependent, kept out of the fast suite
def test_areas_cost_under_ten_milliseconds_per_design(designs: list[Design]) -> None:
    """The median over the sample (user decision, upgrade U0): one slow design (up to ~50 ms)
    must not decide it; the limit stays 10 ms."""
    per_design = [_time([d]) for d in designs[:TIMED_DESIGNS]]
    assert statistics.median(per_design) * 1000.0 < MS_BUDGET


def _time(batch: list[Design]) -> float:
    start = time.perf_counter()
    for d in batch:
        areas(d)
    return time.perf_counter() - start
