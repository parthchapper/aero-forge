"""Frozen pydantic models shared by the engine (SI units, degrees, SPEC §4 frame)."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)

Vec3 = tuple[float, float, float]
TipStyle = Literal["square", "rounded", "raked"]
Recovery = Literal["belly", "skid", "wheels", "net"]
Launch = Literal["hand", "catapult", "runway"]
BoxLayout = Literal["inline", "side_by_side", "stacked", "wing_root"]
LayoutName = Literal[
    "free", "conventional", "canard", "tandem", "three_surface", "flying_wing", "joined"
]
BodyStyle = Literal["smooth", "faceted"]
EdgeAlignment = Literal["off", "soft", "strict"]
NoseStyle = Literal["point", "chisel", "blunt"]
TailStyle = Literal["point", "knife", "flat"]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class Frozen(BaseModel):
    """Base for immutable, strict models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Brief(Frozen):
    """User constraints (SPEC §4); unset optional fields take checks.yaml defaults in derive."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    payload_mass_kg: PositiveFloat
    payload_box_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    span_max_m: PositiveFloat
    cruise_speed_ms: PositiveFloat
    endurance_min: PositiveFloat | None = None
    range_km: PositiveFloat | None = None
    launch: Launch
    recovery: Recovery = "belly"
    stall_max_ms: PositiveFloat | None = None
    dash_speed_ms: PositiveFloat | None = None
    n_limit: PositiveFloat | None = None
    gust_ms: NonNegativeFloat | None = None
    altitude_m: NonNegativeFloat | None = None
    mass_max_kg: PositiveFloat | None = None
    efficiency_floor: PositiveFloat | None = None
    seed: NonNegativeInt | None = None
    randomness: UnitInterval | None = None  # search randomness r; None = constants.yaml default

    @model_validator(mode="after")
    def _one_mission(self) -> "Brief":
        """Exactly one of endurance_min and range_km."""
        if (self.endurance_min is None) == (self.range_km is None):
            raise ValueError("give exactly one of endurance_min and range_km")
        return self


class FlightPoint(Frozen):
    """SPEC §8 flight point: speed and load factor; a gust point adds its load per design."""

    id: str
    speed_ms: PositiveFloat
    load_factor: PositiveFloat
    gust_ms: NonNegativeFloat = 0.0


class Requirements(Frozen):
    """Sizing targets: the decoder reads the first block; derive fills points and baselines."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    mtow_kg: PositiveFloat
    lifting_area_min_m2: PositiveFloat
    span_max_m: PositiveFloat
    ref_length_m: PositiveFloat
    battery_wh: PositiveFloat
    payload_box_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    recovery: Recovery = "belly"
    altitude_m: NonNegativeFloat = 0.0
    points: tuple[FlightPoint, ...] = ()
    baseline_cruise_ld_tier1: PositiveFloat | None = None
    baseline_cruise_ld_tier2: PositiveFloat | None = None


class Adjust(Frozen):
    """Decode adjustments found by the balance (user Deviations, step 3 gate r3).

    body_shift_m moves body0 with its boxes and props along +x relative to the surfaces;
    fin_scale multiplies every fin surface's area past the step-2 fin cap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    body_shift_m: float = 0.0
    fin_scale: PositiveFloat = 1.0
    add_fin: bool = False
    incidence_deg: float = 0.0


class BodyShape(Frozen):
    """Body style genes (SPEC §4, U1): a smooth super-ellipse or a body of flat panels.

    Defaults are the reference values, at which a smooth body is exactly the 0.3 body.
    """

    style: BodyStyle = "smooth"
    sides: int = 8
    stations: int = 3
    chine_z_frac: float = 0.0
    belly_flat_frac: float = 0.0
    roof_pitch_deg: float = 45.0
    side_tumblehome_deg: float = 0.0
    lower_flare_deg: float = 0.0
    nose: NoseStyle = "point"
    tail: TailStyle = "point"
    tail_flatten: float = 0.0
    tail_end_size_frac: float = 0.18
    max_section_x_frac: float = 0.41
    linear_edges: bool = False  # U3 strict alignment: linear nose and tail radius law


class Family(Frozen):
    """Design-language values every surface and body inherits."""

    sweep_deg: float
    taper: float
    dihedral_deg: float
    tip_style: TipStyle
    camber_frac: float
    camber_loc_frac: float
    tc: float
    body_fineness: float
    body_section_exp: float
    body_nose_frac: float
    body_tail_frac: float
    body_width_ratio: float
    body_tail_upsweep_deg: float
    body_rise_x_frac: float | None = None  # decoder: where the tail upsweep starts (None: default)
    box_layout: BoxLayout
    layout: LayoutName
    shape: BodyShape = BodyShape()
    airfoil_facets: int = 0  # U3: flat segments per side of wing sections (0 = smooth)
    edge_alignment: EdgeAlignment = "off"  # U3: top-view outline edges
    edge_angles_deg: tuple[float, ...] = ()  # U3: the angle set strict alignment snapped to
    edge_body_angles_deg: tuple[float, ...] = ()  # U3: the same for flat-panel body edges
    facet_coarseness: float = 0.0  # U3b: 0 = U3 panels, 1 = few large sharp panels (faceted parts)
    facet_notes: tuple[str, ...] = ()  # U3b: what coarseness changed in the wing sections


class Section(Frozen):
    """Airfoil station: leading edge, chord along +x, thickness axis set by cant."""

    le_m: Vec3
    chord_m: float
    cant_deg: float
    twist_deg: float = (
        0.0  # washout: nose-up about the leading edge, added to the surface incidence
    )


class ControlSurface(Frozen):
    """Trailing-edge control as fractions of chord and structural semi-span."""

    chord_frac: float
    span_start_frac: float
    span_end_frac: float


class Spar(Frozen):
    """Round tube along the spar line at a fixed chord fraction."""

    od_m: float
    id_m: float
    chord_frac: float


class Surface(Frozen):
    """Lifting or stabilising surface; sections describe the right/centre instance."""

    name: str
    host: str
    mirrored: bool
    camber_frac: float
    camber_loc_frac: float
    tc: float
    sweep_deg: float
    cant_deg: float
    taper: float
    aspect_ratio: float
    tip_style: TipStyle
    joined_to: str | None
    sections: tuple[Section, ...]
    n_structural: int
    area_m2: float
    projected_area_m2: float
    control: ControlSurface | None
    spar: Spar
    incidence_deg: float = 0.0
    airfoil_facets: int = 0  # U3: flat segments per side of the section polygon (0 = smooth)


class Polyhedron(Frozen):
    """Closed solid of flat faces in the body's own frame (origin at the nose point, metres).

    faces are vertex-index loops, outward counter-clockwise; every face is planar.
    """

    vertices_m: tuple[Vec3, ...]
    faces: tuple[tuple[int, ...], ...]


class Body(Frozen):
    """Super-ellipse body of revolution-like shape along +x from its nose point, or (facets set)
    a body of flat panels whose other fields give its extent and the envelope law consumers
    that are not yet facet-aware (CAD, mass, aero: U2) use."""

    name: str
    mirrored: bool
    nose_m: Vec3
    length_m: float
    half_width_m: float
    half_height_m: float
    section_exp: float
    nose_frac: float
    tail_frac: float
    tail_radius_frac: float
    tail_upsweep_deg: float = 0.0
    rise_x_frac: float | None = None  # length fraction where the rise starts (None: the default)
    facets: Polyhedron | None = None


class Propulsor(Frozen):
    """Propeller on a shaft from an attach point on its host to the hub."""

    name: str
    host: str
    mirrored: bool
    hub_m: Vec3
    attach_m: Vec3
    diameter_m: float
    pusher: bool


class Box(Frozen):
    """Axis-aligned internal component envelope (length along x, width y, height z)."""

    name: str
    host: str
    centre_m: Vec3
    size_m: Vec3


class Design(Frozen):
    """Decoded, physical design; everything downstream consumes only this."""

    family: Family
    lifting_area_m2: PositiveFloat
    surfaces: tuple[Surface, ...]
    bodies: tuple[Body, ...]
    propulsors: tuple[Propulsor, ...]
    boxes: tuple[Box, ...]
    notes: tuple[str, ...]


def instance_names(name: str, mirrored: bool) -> tuple[str, ...]:
    """Part names of a component: `<name>_L`, `<name>_R` when mirrored, else `<name>`."""
    return (f"{name}_L", f"{name}_R") if mirrored else (name,)
