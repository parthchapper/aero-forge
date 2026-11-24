"""How the internals sit (report.md): what the placement did to a motor plate, a hatch and a
body-hosted servo tray, so a fit that passes containment is not silent about a small plate, a
shrunk or off-centre hatch, or a servo far from its hinge (U2 audit)."""

from aeroforge.constants import number
from aeroforge.hardpoints import hardpoints
from aeroforge.internals import Internal
from aeroforge.models import Design, Recovery

MM = 1000.0


def _mount_row(design: Design, mount: Internal) -> str:
    """Plate radius of one firewall against its nominal radius."""
    p = next(x for x in design.propulsors if x.name == mount.owner)
    nominal = number("mount_plate_radius_frac") * p.diameter_m
    radius = mount.size_m[1]
    return (
        f"- {mount.name}: plate radius {radius * MM:.1f} mm of {nominal * MM:.1f} mm nominal "
        f"({radius / nominal:.0%})"
    )


def _hatch_row(hatch: Internal, bay: Internal) -> str:
    """Hatch size against its bay footprint and its centre's offset from the bay centre."""
    area = hatch.size_m[0] * hatch.size_m[1] / (bay.size_m[0] * bay.size_m[1])
    dx = abs(hatch.centre_m[0] - bay.centre_m[0]) / bay.size_m[0]
    dy = abs(hatch.centre_m[1] - bay.centre_m[1]) / bay.size_m[1]
    return (
        f"- {hatch.name}: {area:.0%} of its bay footprint, centre off by "
        f"{dx:.0%} (length) / {dy:.0%} (width)"
    )


def _servo_row(design: Design, tray: Internal) -> str:
    """Offset of a body-hosted servo tray from the mid chord of its surface's root, in chords."""
    root = next(x for x in design.surfaces if x.name == tray.owner).sections[0]
    slide = (tray.centre_m[0] - (root.le_m[0] + root.chord_m / 2)) / root.chord_m
    return f"- {tray.name}: {slide:+.2f} root chords from mid chord, in the body"


def fit_rows(design: Design, recovery: Recovery) -> list[str]:
    """One line per motor plate, hatch and body-hosted servo tray; '- none' when there is none."""
    items = {it.name: it for it in hardpoints(design, recovery)}
    rows = []
    for it in items.values():
        if it.item == "mount":
            rows.append(_mount_row(design, it))
        elif it.item == "hatch":
            rows.append(_hatch_row(it, items["bay_" + it.owner]))
        elif it.item == "servo_tray" and it.surface is None:
            rows.append(_servo_row(design, it))
    return rows or ["- none"]
