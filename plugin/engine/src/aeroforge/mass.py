"""Component masses with positions, total mass and CG range (SPEC §8 `mass`)."""

from itertools import pairwise

import numpy as np

from aeroforge.constants import number, numbers
from aeroforge.facet_query import area_centroid_m, surface_area_m2
from aeroforge.models import Body, Design, Frozen, Requirements, Surface, Vec3
from aeroforge.shapes import body_rings, body_stations

BATTERY, PAYLOAD = "battery", "payload"


class MassItem(Frozen):
    """One component: mass [kg] at its centroid [m]."""

    name: str
    mass_kg: float
    xyz_m: Vec3


def _vec(a: np.ndarray) -> Vec3:
    """numpy 3-vector → model tuple."""
    return (float(a[0]), float(a[1]), float(a[2]))


def _instances(mirrored: bool) -> int:
    """Instance count of a mirrored or centre component."""
    return 2 if mirrored else 1


def _on_centreline(xyz: np.ndarray, mirrored: bool) -> np.ndarray:
    """Combined centroid of a mirrored pair lies on y = 0."""
    return xyz * np.array([1.0, 0.0, 1.0]) if mirrored else xyz


def _surface_area_centroid(s: Surface) -> tuple[float, np.ndarray]:
    """True area [m²] and centroid [m] of one instance of s."""
    frac = number("surface_centroid_chord_frac")
    area, moment = 0.0, np.zeros(3)
    for a, b in pairwise(s.sections):
        span = float(np.hypot(b.le_m[1] - a.le_m[1], b.le_m[2] - a.le_m[2]))
        mean_chord = (a.chord_m + b.chord_m) / 2
        mid = (np.asarray(a.le_m) + np.asarray(b.le_m)) / 2 + np.array([frac * mean_chord, 0, 0])
        area += span * mean_chord
        moment += span * mean_chord * mid
    return area, moment / area


def spar_points(s: Surface) -> np.ndarray:
    """Spar line points on the structural sections of one instance [m]."""
    return np.array(
        [
            np.add(c.le_m, (s.spar.chord_frac * c.chord_m, 0, 0))
            for c in s.sections[: s.n_structural]
        ]
    )


def spar_length_m(s: Surface) -> float:
    """Spar length of one instance along its spar line [m]."""
    return float(np.linalg.norm(np.diff(spar_points(s), axis=0), axis=1).sum())


def _surface_items(s: Surface) -> list[MassItem]:
    """Skin/core, spar and servos of a surface."""
    n = _instances(s.mirrored)
    area, centroid = _surface_area_centroid(s)
    base, per_tc = numbers("surface_wetted_coeffs")
    wetted = area * (base + per_tc * s.tc)
    where = _vec(_on_centreline(centroid, s.mirrored))
    items = [
        MassItem(
            name=s.name, mass_kg=n * wetted * number("surface_areal_density_kg_m2"), xyz_m=where
        )
    ]
    spar_pts = spar_points(s)
    length = spar_length_m(s)
    tube = np.pi / 4 * (s.spar.od_m**2 - s.spar.id_m**2) * length * number("spar_density_kg_m3")
    spar_mid = _on_centreline((spar_pts[0] + spar_pts[-1]) / 2, s.mirrored)
    items.append(MassItem(name=f"{s.name}_spar", mass_kg=n * tube, xyz_m=_vec(spar_mid)))
    if s.control is not None:
        items.append(
            MassItem(name=f"{s.name}_servo", mass_kg=n * number("servo_mass_kg"), xyz_m=where)
        )
    return items


def _smooth_wetted(b: Body) -> tuple[float, np.ndarray]:
    """Wetted area [m^2] and its centroid [m, from the nose; on the axis line at its x] of a
    smooth body (ring perimeters integrated along x)."""
    t = body_stations(int(number("body_loft_stations")))
    rings = body_rings(b, t, int(number("body_ring_points")))
    closed = np.concatenate([rings, rings[:, :1]], axis=1)
    perimeter = np.linalg.norm(np.diff(closed, axis=1), axis=2).sum(axis=1)
    x = t * b.length_m
    strip = np.diff(x) * (perimeter[:-1] + perimeter[1:]) / 2
    wetted = float(strip.sum())
    return wetted, np.array([float((strip * (x[:-1] + x[1:]) / 2).sum() / wetted), 0.0, 0.0])


def _body_item(b: Body) -> MassItem:
    """Shell mass of a body from its wetted area: for flat panels the exact sum of the faces times
    sheet thickness times sheet density, centred on the area centroid (z follows the upsweep)."""
    if b.facets is not None:
        wetted, offset = surface_area_m2(b.facets), area_centroid_m(b.facets)
        areal = number("faceted_panel_thickness_m") * number("faceted_panel_density_kg_m3")
    else:
        wetted, offset = _smooth_wetted(b)
        areal = number("body_areal_density_kg_m2")
    centre = _on_centreline(np.asarray(b.nose_m) + offset, b.mirrored)
    mass = _instances(b.mirrored) * wetted * areal
    return MassItem(name=b.name, mass_kg=mass, xyz_m=_vec(centre))


def installed_power_w(req: Requirements) -> float:
    """Total motor shaft power [W]."""
    return number("installed_power_w_per_kg") * req.mtow_kg


def _propulsion_items(d: Design, req: Requirements) -> list[MassItem]:
    """Motor + ESC at each attach point, propeller at each hub."""
    count = sum(_instances(p.mirrored) for p in d.propulsors)
    per_motor = installed_power_w(req) / count
    drive = per_motor / number("motor_specific_power_w_per_kg") + per_motor / number(
        "esc_specific_power_w_per_kg"
    )
    items = []
    for p in d.propulsors:
        n = _instances(p.mirrored)
        attach = _on_centreline(np.asarray(p.attach_m), p.mirrored)
        hub = _on_centreline(np.asarray(p.hub_m), p.mirrored)
        prop = number("prop_mass_per_diameter_sq_kg_m2") * p.diameter_m**2
        items.append(MassItem(name=f"{p.name}_drive", mass_kg=n * drive, xyz_m=_vec(attach)))
        items.append(MassItem(name=f"{p.name}_prop", mass_kg=n * prop, xyz_m=_vec(hub)))
    return items


def mass_items(d: Design, req: Requirements, payload_mass_kg: float) -> tuple[MassItem, ...]:
    """Every component mass of d (SPEC §8 `mass` list)."""
    items = [it for s in d.surfaces for it in _surface_items(s)]
    items += [_body_item(b) for b in d.bodies]
    items += _propulsion_items(d, req)
    box = {b.name: b.centre_m for b in d.boxes}
    battery_kg = req.battery_wh / number("battery_specific_energy_wh_per_kg")
    items += [
        MassItem(name=BATTERY, mass_kg=battery_kg, xyz_m=box[BATTERY]),
        MassItem(name="avionics", mass_kg=number("avionics_mass_kg"), xyz_m=box["avionics"]),
        MassItem(name=PAYLOAD, mass_kg=payload_mass_kg, xyz_m=box[PAYLOAD]),
    ]
    return tuple(items)


def total_mass_kg(items: tuple[MassItem, ...]) -> float:
    """Sum of component masses [kg]."""
    return float(sum(it.mass_kg for it in items))


def centre_of_gravity(items: tuple[MassItem, ...]) -> np.ndarray:
    """Mass-weighted centroid [m]."""
    m = np.array([it.mass_kg for it in items])
    xyz = np.array([it.xyz_m for it in items])
    return (m[:, None] * xyz).sum(axis=0) / m.sum()


def move_item(items: tuple[MassItem, ...], name: str, dx_m: float) -> tuple[MassItem, ...]:
    """Items with the named one shifted by dx_m along x."""
    return tuple(
        it.model_copy(update={"xyz_m": (it.xyz_m[0] + dx_m, it.xyz_m[1], it.xyz_m[2])})
        if it.name == name
        else it
        for it in items
    )


def cg_range(items: tuple[MassItem, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(fore, aft) CG [m] over payload on/off and the battery position tolerance."""
    tol = number("battery_x_tol_m")
    cgs = []
    for payload_on in (True, False):
        kept = items if payload_on else tuple(it for it in items if it.name != PAYLOAD)
        cgs += [centre_of_gravity(move_item(kept, BATTERY, dx)) for dx in (-tol, tol)]
    arr = np.array(cgs)
    return arr[np.argmin(arr[:, 0])], arr[np.argmax(arr[:, 0])]
