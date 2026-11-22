from aeroforge.mass import BATTERY, centre_of_gravity, cg_range, mass_items, move_item
from aeroforge.models import Design, Requirements

PAYLOAD_KG = 0.5
BATTERY_SHIFT_M = 0.05
MASS_DESIGNS = 50


def test_every_component_has_a_positive_mass(designs: list[Design], req: Requirements) -> None:
    for d in designs[:MASS_DESIGNS]:
        items = mass_items(d, req, PAYLOAD_KG)
        names = {it.name for it in items}
        assert {"battery", "payload", "avionics"} <= names
        assert {s.name for s in d.surfaces} | {f"{s.name}_spar" for s in d.surfaces} <= names
        assert {b.name for b in d.bodies} <= names
        assert all(it.mass_kg > 0 for it in items)


def test_cg_range_has_the_fore_cg_ahead_of_the_aft_cg(
    designs: list[Design], req: Requirements
) -> None:
    for d in designs[:MASS_DESIGNS]:
        fore, aft = cg_range(mass_items(d, req, PAYLOAD_KG))
        assert fore[0] <= aft[0]


def test_moving_the_battery_aft_moves_the_cg_aft(designs: list[Design], req: Requirements) -> None:
    items = mass_items(designs[0], req, PAYLOAD_KG)
    moved = move_item(items, BATTERY, BATTERY_SHIFT_M)
    assert centre_of_gravity(moved)[0] > centre_of_gravity(items)[0]
