"""A flat-panel pod that hosts a motor (U2 follow-up): its end and, when the end is not enough, its
radius are raised until a firewall plate fits, so the pod is never slimmer than its motor mount."""

from collections.abc import Callable

import numpy as np

from aeroforge.constants import number
from aeroforge.faceted_body import faceted_body, plate_fits, prop_end_family
from aeroforge.models import Body, Family, Requirements


def grow_to_fit(
    fam: Family,
    build_at: Callable[[float, Family], Body],
    mount: tuple[float, bool],
    k_max: float,
) -> tuple[Family, float] | None:
    """(family, section scale k in 1..k_max) of the smallest flat-panel host whose firewall plate
    fits: for each scale (steps of prop_grow_factor) the end is raised first (prop_end_family),
    then the whole section grows. build_at(k, family) is the host with its section scaled by k;
    mount = (prop diameter [m], pusher). None when no scale up to k_max holds a plate."""
    diameter, pusher = mount
    k = 1.0
    for _ in range(int(number("prop_grow_steps")) + 1):
        grown = prop_end_family(fam, lambda f, k=k: build_at(k, f), diameter, pusher)
        if plate_fits(build_at(k, grown), diameter, pusher):
            return grown, k
        if k >= k_max:
            break
        k = min(k * number("prop_grow_factor"), k_max)
    return None


def pod_for_prop(
    fam: Family, length_m: float, radius_m: float, mount: tuple[float, bool] | None
) -> tuple[Family, float]:
    """(family, radius [m]) of a pod of length_m whose firewall plate fits; mount = (prop diameter
    [m], pusher). The inputs when the pod is smooth, has no motor, or no radius up to
    pod_radius_max_length_frac x length helps."""
    if mount is None or fam.shape.style != "faceted":
        return fam, radius_m
    k_max = max(number("pod_radius_max_length_frac") * length_m / radius_m, 1.0)
    found = grow_to_fit(
        fam, lambda k, f: faceted_body(f, length_m, (radius_m * k,) * 2, "pod"), mount, k_max
    )
    return (fam, radius_m) if found is None else (found[0], radius_m * found[1])


def prop_diameter(req: Requirements, count: int) -> float:
    """Diameter [m] from static thrust share and disc loading."""
    thrust = number("static_thrust_to_weight") * req.mtow_kg * number("gravity_m_s2")
    area = thrust / count / number("prop_disc_loading_n_m2")
    return float(np.sqrt(4 * area / np.pi))


def pod_y_bounds(
    genes: dict[str, float],
    fam: Family,
    req: Requirements,
    body_half_width_m: float,
    pattern: str,
    mount: tuple[float, bool] | None = None,
) -> tuple[float, float]:
    """Allowed right-pod centre y range [m] (clear of body, props and span limit)."""
    length = genes["pod_length_frac"] * req.ref_length_m
    pod_r = pod_for_prop(fam, length, length / (2 * fam.body_fineness), mount)[1]
    clear, margin = number("pod_body_clearance_m"), number("prop_clearance_m")
    lo = body_half_width_m + pod_r + clear
    hi = req.span_max_m / 2 - pod_r - clear
    if pattern != "centre":
        d = prop_diameter(req, 3 if pattern == "centre_pods" else 2)
        lo = max(lo, d / 2 + (d / 2 + margin if pattern == "centre_pods" else margin / 2))
        hi = min(hi, req.span_max_m / 2 - d / 2 - clear)
    return lo, hi
