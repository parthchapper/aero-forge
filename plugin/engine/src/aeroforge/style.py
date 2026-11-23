"""Style features and score (SPEC §7); fin panels are invisible to every feature."""

from dataclasses import dataclass
from itertools import pairwise
from math import atan2, degrees, hypot

import numpy as np

from aeroforge.constants import load_table, number
from aeroforge.facet_metrics import largest_section_half_extents_m
from aeroforge.models import Body, Design, Surface

FEATURES = (
    "body_fineness",
    "bodyless",
    "log_ar_main",
    "log_ar_rest",
    "area_balance",
    "sweep_spread_deg",
    "taper_spread",
    "root_cant_mean_deg",
    "crease_max_deg",
    "part_count",
    "tip_join_share",
    "planform_compactness",
    "side_height_ratio",
)


NUISANCE = ("fin_area_share", "fin_count")


class StyleError(ValueError):
    """A style model mapping does not match the feature list."""


@dataclass(frozen=True)
class Panel:
    """Straight span segment between two sections of the right/centre instance."""

    pts_m: np.ndarray  # (4, 3): LE a, TE a, LE b, TE b
    cant_deg: float
    projected_m2: float
    true_m2: float


def panels(s: Surface) -> list[Panel]:
    """Panels of a surface; zero-length segments (coincident join stations) skipped."""
    out: list[Panel] = []
    for a, b in pairwise(s.sections):
        dy, dz = b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]
        if hypot(dy, dz) < number("style_panel_min_m"):
            continue
        te_a, te_b = np.add(a.le_m, (a.chord_m, 0, 0)), np.add(b.le_m, (b.chord_m, 0, 0))
        pts = np.array([a.le_m, te_a, b.le_m, te_b])
        mean_chord = (a.chord_m + b.chord_m) / 2
        cant = degrees(atan2(dz, abs(dy)))
        out.append(Panel(pts, cant, abs(dy) * mean_chord, hypot(dy, dz) * mean_chord))
    return out


def is_fin(p: Panel) -> bool:
    """A panel steeper than fin_cant_deg."""
    return abs(p.cant_deg) >= number("fin_cant_deg")


def wing_surfaces(d: Design) -> tuple[Surface, ...]:
    """Surfaces whose root panel is not a fin; fin surfaces are ignored by the style score."""
    out = []
    for s in d.surfaces:
        ps = panels(s)
        if ps and not is_fin(ps[0]):
            out.append(s)
    return tuple(out)


def _instances(mirrored: bool) -> int:
    """Instance count of a mirrored or centre component."""
    return 2 if mirrored else 1


def _body_points(b: Body) -> np.ndarray:
    """Bounding-box corners of a body, both instances when mirrored [m]; a flat-panel body gives
    its own vertices (the tail upsweep is part of its height)."""
    x0, y0, z0 = b.nose_m
    ys = [y0, -y0] if b.mirrored else [y0]
    if b.facets is not None:
        v = np.asarray(b.facets.vertices_m) + np.asarray(b.nose_m)
        return np.vstack([v] + ([v * np.array([1.0, -1.0, 1.0])] if b.mirrored else []))
    return np.array(
        [
            (x, y + sy * b.half_width_m, z0 + sz * b.half_height_m)
            for x in (x0, x0 + b.length_m)
            for y in ys
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )


def _spread(values: list[float]) -> float:
    """Population standard deviation; 0 for fewer than two values."""
    return float(np.std(values)) if len(values) > 1 else 0.0


def _geometry(d: Design, wings: tuple[Surface, ...]) -> tuple[float, float]:
    """(planform compactness, side-view height / length) of bodies and non-fin panels."""
    pts, area = [], 0.0
    for s in wings:
        for p in panels(s):
            if is_fin(p):
                continue
            area += _instances(s.mirrored) * p.projected_m2
            pts.append(p.pts_m)
            if s.mirrored:
                pts.append(p.pts_m * (1, -1, 1))
    pts.extend(_body_points(b) for b in d.bodies)
    ext = np.ptp(np.concatenate(pts), axis=0)
    if ext[0] <= 0.0:
        return 0.0, 0.0
    compact = area / (ext[0] * ext[1]) if ext[1] > 0.0 else 0.0
    return float(compact), float(ext[2] / ext[0])


def _body_terms(d: Design) -> tuple[float, float]:
    """(centre-body fineness, bodyless flag); fineness 0 without a centre body."""
    centre = [b for b in d.bodies if not b.mirrored]
    if not centre:
        return 0.0, 1.0
    b = centre[0]
    half_w, half_h = (
        largest_section_half_extents_m(b.facets)
        if b.facets is not None
        else (b.half_width_m, b.half_height_m)
    )
    return b.length_m / (2.0 * float(np.sqrt(half_w * half_h))), 0.0


def _root_mid_x(s: Surface) -> float:
    """Root mid-chord station [m]."""
    return s.sections[0].le_m[0] + s.sections[0].chord_m / 2


def _layout_terms(wings: tuple[Surface, ...]) -> tuple[float, float]:
    """(area-weighted log AR of non-main surfaces, fore-minus-aft projected-area share of them).

    Log AR: joined panels can reach AR ~ 90; a log keeps them from dominating the fit.
    """
    rest = wings[1:]
    total = sum(s.projected_area_m2 for s in rest)
    if total <= 0.0:
        return 0.0, 0.0
    x_main = _root_mid_x(wings[0])
    ar = sum(float(np.log(s.aspect_ratio)) * s.projected_area_m2 for s in rest) / total
    signed = sum((1 if _root_mid_x(s) < x_main else -1) * s.projected_area_m2 for s in rest)
    return ar, signed / total


def _crease_max(wings: tuple[Surface, ...]) -> float:
    """Largest cant change between adjacent non-fin panels [deg]."""
    worst = 0.0
    for s in wings:
        ps = panels(s)
        for a, b in pairwise(ps):
            if not (is_fin(a) or is_fin(b)):
                worst = max(worst, abs(b.cant_deg - a.cant_deg))
    return worst


def features(d: Design) -> np.ndarray:
    """Style feature vector in FEATURES order (ratios, degrees, counts)."""
    wings = wing_surfaces(d)
    fineness, bodyless = _body_terms(d)
    ar_rest, balance = _layout_terms(wings)
    compact, height = _geometry(d, wings)
    parts = (
        sum(_instances(b.mirrored) for b in d.bodies)
        + sum(_instances(s.mirrored) for s in wings)
        + sum(_instances(p.mirrored) for p in d.propulsors)
    )
    values = (
        fineness,
        bodyless,
        float(np.log(wings[0].aspect_ratio)),
        ar_rest,
        balance,
        _spread([s.sweep_deg for s in wings]),
        _spread([s.taper for s in wings]),
        float(np.mean([abs(panels(s)[0].cant_deg) for s in wings])),
        _crease_max(wings),
        float(parts),
        sum(s.joined_to is not None for s in wings) / len(wings),
        compact,
        height,
    )
    return np.array(values, dtype=float)


def fin_terms(d: Design) -> np.ndarray:
    """(fin-panel true-area share, surfaces with a fin panel): fitted as nuisance, never scored.

    Panels, not surfaces: steep outer panels on wing surfaces are fins to the eye too.
    """
    fin_area, total, with_fin = 0.0, 0.0, 0
    for s in d.surfaces:
        n = _instances(s.mirrored)
        ps = panels(s)
        total += n * sum(p.true_m2 for p in ps)
        fin_area += n * sum(p.true_m2 for p in ps if is_fin(p))
        with_fin += any(is_fin(p) for p in ps)
    return np.array([fin_area / total, float(with_fin)])


@dataclass(frozen=True)
class StyleModel:
    """Logistic model on standardised features."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float


def model_from_table(table: dict[str, object]) -> StyleModel:
    """StyleModel from a parsed style.yaml mapping."""
    rows = table.get("features")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != list(FEATURES):
        raise StyleError(f"style model features must list exactly {FEATURES}")
    col = {k: np.array([float(r[k]) for r in rows]) for k in ("mean", "std", "weight")}
    if np.any(col["std"] <= 0.0):
        raise StyleError("style model feature std must be > 0")
    return StyleModel(col["mean"], col["std"], col["weight"], float(table["bias"]))


def packaged_model() -> StyleModel:
    """The model in data/style.yaml."""
    return model_from_table(load_table("style"))


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Logistic function without exp overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def scores(model: StyleModel, x: np.ndarray) -> np.ndarray:
    """sigmoid(w · standardised x + b) per row of x (n, F), in (0, 1)."""
    return sigmoid(((x - model.mean) / model.std) @ model.weights + model.bias)


def score(f: np.ndarray, model: StyleModel) -> float:
    """Style score of one feature vector, in (0, 1)."""
    return float(scores(model, f[None, :])[0])
