"""U2: faceted CAD (watertight, exact volume, face count), skin mass by hand, upsweep through CAD /
mass / aero, the cut list's physical sheets, and containment of the internals by body style."""

import numpy as np
import pytest
from conftest import u2_random_genomes
from test_facets import SEED, random_shape

import aeroforge.facet_query as q
from aeroforge import facets as f
from aeroforge.airplane import fuselages
from aeroforge.brief import brief_requirements
from aeroforge.constants import number
from aeroforge.facet_metrics import (
    end_radius_m,
    inradius_on_axis_m,
    largest_section_half_extents_m,
    vertical_extent_m,
)
from aeroforge.facet_pieces import pieces
from aeroforge.genetable import gene_table, genome_length, pin_choice
from aeroforge.genome import decode
from aeroforge.geom import faceted_body_solid, prop_solid
from aeroforge.hardpoints import containment_failures, hardpoints
from aeroforge.internals import Internal
from aeroforge.layout import host_end
from aeroforge.mass import _body_item, mass_items
from aeroforge.mesh import solid_volume_m3
from aeroforge.models import Body, Design, Polyhedron, Requirements
from aeroforge.panels import body_panels, flat_outline_mm
from aeroforge.style import _body_points, _body_terms

CAD_BODIES = 12
PINNED_SEED = 20261112
MISFIT_BAR = 0.01  # F6: internals containment, per body style
STYLE_DESIGNS = 60
AXES_M = (0.09, 0.07)
LENGTH_M = 1.3
EXACT_REL = 1e-6


def _cuboid(length: float, width: float, height: float) -> Polyhedron:
    """Closed box of 6 quads, outward faces, nose face at x = 0, centred on the axis."""
    v = [
        (x, y, z)
        for x in (0.0, length)
        for y in (-width / 2, width / 2)
        for z in (-height / 2, height / 2)
    ]
    quads = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
    return Polyhedron(vertices_m=tuple(v), faces=tuple(f[::-1] for f in quads))


def _box_body(mirrored: bool = False) -> Body:
    poly = _cuboid(0.4, 0.2, 0.1)
    return Body(
        name="box",
        mirrored=mirrored,
        nose_m=(1.0, 0.0, 0.3),
        length_m=0.4,
        half_width_m=0.1,
        half_height_m=0.05,
        section_exp=2.0,
        nose_frac=0.5,
        tail_frac=0.5,
        tail_radius_frac=0.1,
        facets=poly,
    )


def test_box_body_skin_mass_is_area_times_thickness_times_density_by_hand() -> None:
    # faces: 2 (0.2 x 0.1) + 2 (0.4 x 0.1) + 2 (0.4 x 0.2) = 0.04 + 0.08 + 0.16 = 0.28 m^2
    area = 0.28
    assert q.surface_area_m2(_box_body().facets) == pytest.approx(area)  # type: ignore[arg-type]
    item = _body_item(_box_body())
    assert item.mass_kg == pytest.approx(area * 0.004 * 200.0)  # 0.224 kg
    assert item.mass_kg == pytest.approx(0.224)
    assert item.xyz_m == pytest.approx((1.2, 0.0, 0.3))  # the box's own centre
    assert _body_item(_box_body(mirrored=True)).mass_kg == pytest.approx(2 * 0.224)


def test_a_faceted_body_weighs_what_a_smooth_one_of_equal_wetted_area_weighs() -> None:
    assert number("faceted_panel_thickness_m") * number("faceted_panel_density_kg_m3") == (
        pytest.approx(number("body_areal_density_kg_m2"))
    )


def _upswept_design(req: Requirements, upsweep_deg: float) -> tuple[Design, Body]:
    """A decoded flat-panel design whose centre body keeps the asked tail upsweep (the decoder
    drops it when a root chord would sit on the rising tail)."""
    names = [g.name for g in gene_table()]
    for row in u2_random_genomes(SEED, 60):
        pinned = pin_choice(row[None, :], "body_style", "faceted")
        genome = pin_choice(pinned, "body_on", "on")[0]
        genome[names.index("body_tail_upsweep_deg")] = upsweep_deg / 15.0
        design = decode(genome, req)
        body = next((b for b in design.bodies if b.name == "body0"), None)
        if (
            body is not None
            and body.facets is not None
            and body.tail_upsweep_deg == pytest.approx(upsweep_deg)
        ):
            return design, body
    raise AssertionError(f"no genome kept a {upsweep_deg} degree upsweep")


@pytest.mark.parametrize("upsweep_deg", [5.0, 15.0])
def test_the_tail_upsweep_reaches_cad_mass_and_the_aero_section(
    upsweep_deg: float, req: Requirements
) -> None:
    design, body = _upswept_design(req, upsweep_deg)
    assert body.facets is not None
    rise = q.slice_section(body.facets, body.length_m)[3]
    law = f.axis_rise_m(
        np.array([1.0]), design.family.shape, body.length_m, upsweep_deg, body.rise_x_frac
    )[0]
    assert rise == pytest.approx(law, abs=1e-9)
    solid = faceted_body_solid(body, body.facets)
    box = solid.BoundingBox()
    assert box.zmax - box.zmin == pytest.approx(vertical_extent_m(body.facets), rel=1e-9)
    assert solid_volume_m3(solid) == pytest.approx(q.volume_m3(body.facets), rel=EXACT_REL)
    fus = fuselages(design)[0]
    for xs in fus.xsecs:
        _, _, _, zc = q.slice_section(body.facets, xs.xyz_c[0] - body.nose_m[0])
        assert xs.xyz_c[2] == pytest.approx(body.nose_m[2] + zc)
    level = q.area_centroid_m(
        f.facet_body(design.family.shape, body.length_m, (body.half_width_m, body.half_height_m))
    )
    mass_z = _body_item(body).xyz_m[2] - body.nose_m[2]
    assert mass_z == pytest.approx(q.area_centroid_m(body.facets)[2])
    assert mass_z > level[2]


def _faceted_bodies(n: int) -> list[Body]:
    rng = np.random.default_rng(SEED)
    out = []
    for i in range(n):
        shape = random_shape(rng)
        up = (0.0, 5.0, 15.0)[i % 3]
        axes = (AXES_M[0] * (1 + 0.1 * i), AXES_M[1])
        poly = f.facet_body(shape, LENGTH_M, axes, up)
        out.append(
            Body(
                name=f"b{i}",
                mirrored=False,
                nose_m=(0.1 * i, 0.0, 0.2),
                length_m=LENGTH_M,
                half_width_m=axes[0],
                half_height_m=axes[1],
                section_exp=2.0,
                nose_frac=shape.max_section_x_frac,
                tail_frac=1.0 - shape.max_section_x_frac,
                tail_radius_frac=0.1,
                tail_upsweep_deg=up,
                facets=poly,
            )
        )
    return out


def test_cad_solids_are_watertight_with_the_exact_volume_and_one_face_per_polyhedron_face() -> None:
    for b in _faceted_bodies(CAD_BODIES):
        assert b.facets is not None
        solid = faceted_body_solid(b, b.facets)
        assert solid.isValid()
        assert solid_volume_m3(solid) == pytest.approx(q.volume_m3(b.facets), rel=EXACT_REL)
        assert len(solid.Faces()) == len(b.facets.faces)
        assert len(solid.Shells()) == 1
        # watertight: every edge is shared by exactly two of the solid's faces
        edges = [e for fc in solid.Faces() for e in fc.Edges()]
        assert len(edges) == 2 * len(solid.Edges())


def test_cad_face_count_equals_panel_count_of_the_plan() -> None:
    rng = np.random.default_rng(SEED + 5)
    for _ in range(CAD_BODIES):
        shape = random_shape(rng)
        plan = f.effective_plan(shape)
        poly = f.facet_body(shape, LENGTH_M, AXES_M)
        body = _faceted_bodies(1)[0].model_copy(update={"facets": poly, "nose_m": (0.0, 0.0, 0.0)})
        solid = faceted_body_solid(body, poly)
        assert len(solid.Faces()) == f.panel_count(plan, f.section_polygon(shape, plan.sides))


def _simple(outline: np.ndarray) -> bool:
    """No two non-adjacent edges of the closed polygon cross."""
    n = len(outline)

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = outline[j], outline[(j + 1) % n]
            if cross(a, b, c) * cross(a, b, d) < 0.0 and cross(c, d, a) * cross(c, d, b) < 0.0:
                return False
    return True


def test_a_split_cube_face_is_one_sheet_and_the_cube_has_six() -> None:
    cube = _cuboid(1.0, 1.0, 1.0)
    split = tuple(
        tri for fc in cube.faces for tri in ((fc[0], fc[1], fc[2]), (fc[0], fc[2], fc[3]))
    )
    sheets = pieces(cube.model_copy(update={"faces": split}))
    assert len(sheets) == 6
    assert all(len(s.faces) == 2 and len(s.loop) == 4 for s in sheets)
    assert sum(s.area_m2 for s in sheets) == pytest.approx(6.0)


def test_cut_list_rows_are_the_physical_sheets() -> None:
    merged = faces = 0
    for b in _faceted_bodies(CAD_BODIES):
        assert b.facets is not None
        sheets = body_panels(b)
        merged += len(sheets)
        faces += len(b.facets.faces)
        assert len(sheets) == len(pieces(b.facets))
        normals = [s.normal for s in pieces(b.facets)]
        # no two neighbouring sheets are coplanar any more
        for sheet, p in zip(pieces(b.facets), sheets, strict=True):
            for j, bevel in zip(sheet.across, p.bevel_deg, strict=True):
                assert bevel > 1e-4, (j, normals[j])
    assert merged <= faces


def test_each_sheet_laid_flat_has_its_3d_area_and_a_simple_outline() -> None:
    for b in _faceted_bodies(CAD_BODIES):
        for p in body_panels(b):
            flat = flat_outline_mm(p) / 1000.0
            x, y = flat[:, 0], flat[:, 1]
            area = abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2.0
            assert area == pytest.approx(p.area_m2, rel=1e-9)
            assert _simple(flat)
            lengths = np.linalg.norm(flat - np.roll(flat, -1, axis=0), axis=1)
            assert lengths == pytest.approx(np.array(p.edge_lengths_m), rel=1e-9)


def test_the_equivalent_style_size_of_a_faceted_body_comes_from_its_own_faces() -> None:
    for b in _faceted_bodies(CAD_BODIES):
        assert b.facets is not None
        v = np.asarray(b.facets.vertices_m) + np.asarray(b.nose_m)
        pts = _body_points(b)
        assert np.ptp(pts[:, 2]) == pytest.approx(np.ptp(v[:, 2]))  # upsweep is in the height
        half_w, half_h = largest_section_half_extents_m(b.facets)
        design = Design.model_construct(bodies=(b,))
        fineness, _ = _body_terms(design)
        assert fineness == pytest.approx(b.length_m / (2.0 * np.sqrt(half_w * half_h)))


def test_the_pusher_end_of_a_faceted_body_has_the_radius_of_its_end_section() -> None:
    for b in _faceted_bodies(CAD_BODIES):
        assert b.facets is not None
        attach, radius = host_end(b, pusher=True)
        assert radius == pytest.approx(end_radius_m(b.facets, b.length_m))
        assert attach[2] == pytest.approx(
            b.nose_m[2] + q.slice_section(b.facets, b.length_m)[3], abs=1e-9
        )


def _faceted_pusher_props(n: int) -> list[tuple[Design, object, Body]]:
    """(design, propulsor, host) of every pusher on a flat-panel host among n seeded genomes."""
    req = brief_requirements("stub")
    gs = pin_choice(u2_random_genomes(2024, n), "body_style", "faceted")
    out = []
    for g in gs:
        d = decode(g, req)
        bodies = {b.name: b for b in d.bodies}
        out += [
            (d, p, bodies[p.host])
            for p in d.propulsors
            if p.pusher and bodies[p.host].facets is not None
        ]
    return out


def test_the_inradius_of_a_box_section_is_half_its_short_side() -> None:
    cube = _cuboid(1.0, 0.4, 0.2)
    assert inradius_on_axis_m(cube, 0.5) == pytest.approx(0.1)
    assert inradius_on_axis_m(cube, 5.0) == pytest.approx(inradius_on_axis_m(cube, 1.0))


def test_a_pusher_firewall_sits_where_a_plate_fits_and_the_motor_and_shaft_go_with_it() -> None:
    req = brief_requirements("stub")
    moved = 0
    for d, p, host in _faceted_pusher_props(40):
        assert host.facets is not None
        end_x = host.nose_m[0] + host.length_m
        full = number("mount_plate_radius_frac") * p.diameter_m
        target = number("mount_plate_min_frac") * full
        plate_x = (
            p.attach_m[0] - number("mount_plate_inset_m") - number("mount_plate_thickness_m") / 2
        )
        fits = inradius_on_axis_m(host.facets, plate_x - host.nose_m[0]) >= target - 1e-9
        reach = number("mount_slide_body_frac") * host.length_m
        assert fits or end_x - p.attach_m[0] >= reach - 1e-6  # no fit only at the farthest place
        drive = next(m for m in mass_items(d, req, 0.5) if m.name == f"{p.name}_drive")
        assert drive.xyz_m[0] == pytest.approx(p.attach_m[0], abs=1e-9)  # motor at the firewall
        slide = end_x - p.attach_m[0]
        note = [n for n in d.notes if n.startswith(f"firewall of {p.name} moved inside")]
        assert bool(note) == (slide > 1e-9)
        if note:
            moved += 1
            assert f"{slide:.3f} m" in note[0]
    assert moved > 0


def test_a_firewall_plate_that_cannot_reach_half_its_radius_is_a_misfit() -> None:
    for d, p, host in _faceted_pusher_props(40):
        full = number("mount_plate_radius_frac") * p.diameter_m
        end_x = host.nose_m[0] + host.length_m
        slim = (
            inradius_on_axis_m(host.facets, host.length_m - 0.0115)
            < number("mount_plate_min_frac") * full
        )  # type: ignore[arg-type]
        if not slim:
            continue
        at_end = d.model_copy(
            update={
                "propulsors": tuple(
                    q.model_copy(update={"attach_m": (end_x, p.attach_m[1], p.attach_m[2])})
                    if q.name == p.name
                    else q
                    for q in d.propulsors
                )
            }
        )
        items = hardpoints(at_end, "belly")
        assert f"mount_{p.name}" in containment_failures(at_end, items)
        return
    raise AssertionError("no slim tail among the sample")


def test_every_contained_hatch_is_a_usable_one_centred_over_its_bay() -> None:
    req = brief_requirements("stub")
    gs = pin_choice(u2_random_genomes(31, 80), "body_style", "faceted")
    contained = tiny = 0
    for g in gs:
        d = decode(g, req)
        items = hardpoints(d, req.recovery)
        failed = set(containment_failures(d, items))
        for it in items:
            if it.item != "hatch" or it.name in failed:
                continue
            box = next(b for b in d.boxes if f"hatch_{b.name}" == it.name)
            if not any(b.facets is not None and b.name == box.host for b in d.bodies):
                continue
            contained += 1
            area = it.size_m[0] * it.size_m[1] / (box.size_m[0] * box.size_m[1])
            assert area >= number("hatch_panel_min_frac") - 1e-9
            tol = number("hatch_centre_tol_frac")
            assert abs(it.centre_m[0] - box.centre_m[0]) <= tol * box.size_m[0] + 1e-9
            assert abs(it.centre_m[1] - box.centre_m[1]) <= tol * box.size_m[1] + 1e-9
            tiny += area < 1.0
    assert contained > 0 and tiny > 0  # some hatches really are cut back, and still usable


def _pinned(
    brief: str, style: str, n: int, seed: int = PINNED_SEED
) -> tuple[Requirements, np.ndarray]:
    gs = pin_choice(u2_random_genomes(seed, n), "body_style", style)
    return brief_requirements(brief), gs


# (brief, genome index, internal, fits, seed): a motor firewall moved inside a pointed tail, servo
# trays slid along a knife tail, a hatch clipped to a flat roof; and a genome whose roof holds less
# than 25 % of its bay (or only off its centre), a misfit that is reported, not hidden (the genome
# seed 20261112 / 57 fits since the U2 follow-ups)
INTERNALS_CASES = [
    ("stub", 133, "mount_prop0", True, PINNED_SEED),
    ("stub", 31, "servo_surface2", True, PINNED_SEED),
    ("small_boxes", 5, "servo_surface3", True, PINNED_SEED),
    ("small_boxes", 30, "hatch_avionics", True, PINNED_SEED),
    ("small_boxes", 201, "hatch_avionics", False, 4242),
]


@pytest.mark.parametrize(("brief", "index", "what", "fits", "seed"), INTERNALS_CASES)
def test_the_internals_that_once_missed_a_faceted_body_fit_or_are_reported_as_misfits(
    brief: str, index: int, what: str, fits: bool, seed: int
) -> None:
    req, genomes = _pinned(brief, "faceted", index + 1, seed)
    design = decode(genomes[index], req)
    items = hardpoints(design, req.recovery)
    assert what in {it.name for it in items}
    assert (what in containment_failures(design, items)) == (not fits)


def test_containment_reports_an_internal_pushed_out_of_a_faceted_roof() -> None:
    req, genomes = _pinned("stub", "faceted", 6)
    design = decode(genomes[0], req)
    items = hardpoints(design, req.recovery)
    assert containment_failures(design, items) == []
    hatch = next(it for it in items if it.item == "hatch")
    lifted = hatch.model_copy(
        update={"centre_m": (hatch.centre_m[0], hatch.centre_m[1], hatch.centre_m[2] + 1.0)}
    )
    other = tuple(lifted if it.name == hatch.name else it for it in items)
    assert containment_failures(design, other) == [hatch.name]
    assert isinstance(lifted, Internal)


def _misfits(brief: str, style: str, n: int) -> int:
    req, genomes = _pinned(brief, style, n)
    count = 0
    for g in genomes:
        d = decode(g, req)
        count += bool(containment_failures(d, hardpoints(d, req.recovery)))
    return count


STYLES = ["faceted", "smooth"]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("brief", ["stub", "small_boxes"])
def test_internals_misfits_stay_under_one_percent_for_each_body_style(
    brief: str, style: str
) -> None:
    assert _misfits(brief, style, STYLE_DESIGNS) <= MISFIT_BAR * STYLE_DESIGNS


@pytest.mark.slow
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("brief", ["stub", "small_boxes"])
def test_internals_misfits_stay_under_one_percent_over_four_hundred_genomes(
    brief: str, style: str
) -> None:
    n = 400
    assert _misfits(brief, style, n) <= MISFIT_BAR * n


def test_genome_length_is_99_before_u3_and_u3_adds_exactly_two_genes() -> None:
    assert genome_length() == 99 + 2
    assert [g.name for g in gene_table()[-2:]] == ["airfoil_facets", "edge_alignment"]


def test_a_long_slightly_tilted_shaft_fuses_with_its_blades_into_one_solid() -> None:
    """Gallery genome 45 (seed 1, flat-panel): a firewall 0.56 m inside the tail left the shaft
    ending on the blade plane and the fuse gave two solids."""
    req = brief_requirements("stub")
    g = pin_choice(u2_random_genomes(1, 50), "body_style", "faceted")[45]
    design = decode(g, req)
    long_shafts = 0
    for p in design.propulsors:
        solid = prop_solid(p)
        assert len(solid.Solids()) == 1
        long_shafts += np.linalg.norm(np.subtract(p.hub_m, p.attach_m)) > 0.5
    assert long_shafts > 0
