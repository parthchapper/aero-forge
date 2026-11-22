# fix.md: faceted panel bodies, 100 ranked controls, mission-to-design

This plan adds three things on top of `plan.md` (steps 7–13):

1. **Faceted bodies.** The body is built from flat panels, the way the F-117 is built. The look is not copied, only the method: every face is a flat plate, every edge is a sharp line, and nothing is a curved, rounded tube. You choose the number of panels.
2. **100 controls, ranked.** A single table of 100 named parameters that you can set, pin or narrow. Rank 1 changes the design the most.
3. **Mission in, alien design out.** You type `/forge new I want the drone to [your mission]`. Claude turns the mission into a small set of controls (ranges, not shapes). Physics then sizes the aircraft, the search shapes it, and the novelty filters push it away from known aircraft.

The steps are named F1 to F7. They slot in between the remaining `plan.md` steps (see "Order" below). Follow `RULES.md`: one step per session, test, gate, then stop.

**Approval.** RULES.md says a change to an interface, a gate or a check band needs your approval. Your request for this file is that approval for the changes listed below. Log each one under Deviations in `CHANGELOG.md` with "user, fix.md" as the reason.

**Principle kept.** SPEC §1 says "size by physics, shape by search, geometry by code. Claude never writes geometry or genome values." Controls are requirements, ranges and filters. Claude writes an exact value only when you state one. Log this as a Deviation from SPEC §1 ("Claude may narrow gene ranges from the mission text").

---

## Order

| when | step |
|---|---|
| now | Finish `plan.md` step 10 (open test decision + your gallery gate). |
| 1 | **F1** Faceted body geometry |
| 2 | **F2** Faceted body in CAD, physics, internals, cut list |
| 3 | `plan.md` step 11 (novelty sees the body). Add the F1 facet terms listed there. |
| 4 | **F3** Faceted wings and edge alignment |
| 5 | `plan.md` step 12 (style and critic retune). Add the facet style features listed there. |
| 6 | **F4** Controls layer (the 100-row table) |
| 7 | **F5** Alienness and new filters and checks |
| 8 | **F6** Mission → controls in the forge skill |
| 9 | **F7** = `plan.md` step 13 proof run, extended |

Steps 11 and 12 come after F1–F3 so that the novelty measure and the style score are tuned on faceted bodies, not on the old round ones.

---

## What "faceted" means here

A faceted body is a closed solid made only of flat faces.

- The body has **S stations** along its length. At each station the cross-section is a **convex polygon with K sides**, mirror-symmetric about y = 0.
- Between two stations with the **same polygon shape**, the second polygon is the first scaled uniformly and shifted in z. Each joining face is then a flat trapezoid. That holds by construction, so no tolerance is needed.
- At a **shape break** (the polygon proportions change, for example a flatter tail), each joining face would twist. It is split into two flat triangles.
- The nose and tail close with flat faces too (see `nose_style`, `tail_style`).

**Panel count** (flat faces on one body, both sides, caps included):

$$N = K\,(S-1) + K\,B + n_{\text{nose}} + n_{\text{tail}}$$

where $B$ is the number of shape breaks (0–2), $n_{\text{nose}}$ is $K$ for `point` and `chisel` and 1 for `blunt`, and $n_{\text{tail}}$ is $K$ for `point` and `knife` and 1 for `flat`.

Example: K = 6, S = 3, one break, point nose, flat tail gives N = 12 + 6 + 6 + 1 = 25 panels.

**Choosing the count.** When you set `body_panel_count`, the decoder lists every (K, S, B, nose, tail) combination that gives exactly that count and picks the one closest to what the genes ask for. If no combination is exact, it takes the nearest one and records it in `notes`. With `panel_count_exact: true`, a count that can't be reached exactly is rejected in `derive`, before the search. That check depends on N only.

**Polygon shape.** The decoder builds one convex outline from the feature genes: flat belly width, lower flare, chine (the sharp widest edge), tumblehome (upper sides leaning in), and roof pitch. It then places K vertices on that outline. The chine points and the belly and roof corners are always vertices. The rest are spread by arc length. If K is smaller than the number of forced corners, the lowest-ranked features are dropped (roof ridge first, then belly corners). The outline must stay convex. Gene ranges are chosen so that it always is, and a test checks it.

**Why convex sections.** Each segment between two stations is then the convex hull of two parallel polygons. "Is this box inside the body?" becomes an exact half-space test, with no sampling. That keeps invariant 5 (boxes fit) exact.

---

## Step F1: Faceted body geometry (pure code, no CAD)

Read: SPEC §4, §5. `models.py`, `shapes.py`, `layout.py`, `areas.py`, `genes.yaml`, `genetable.py`, `reference.py`.

Build:
- New module `facets.py` (pure functions):
  - `section_polygon(genes, K)` → 2-D vertices (y, z), convex and symmetric.
  - `body_polyhedron(body)` → vertices [m] and faces (vertex-index tuples), using the rules above.
  - `panel_count(K, S, B, nose, tail)` and `panel_plan(N, wishes)` → the chosen (K, S, B, nose, tail).
  - `inside_faceted(body, points)` → half-space test per segment.
- `models.py`: add `Polyhedron(vertices_m, faces)` and an optional `Body.facets: Polyhedron | None`. `None` = today's smooth body. Every downstream user then works from explicit faces, as SPEC §4 asks ("Design: decoded, physical, complete").
- New genes (each gets a `genes.yaml` row, a `ref_default` and decoder code):

| gene | range | ref_default |
|---|---|---|
| `body_style` | smooth, faceted | smooth |
| `body_section_sides` | 3–12 | 8 |
| `body_stations` | 1–8 | 3 |
| `chine_z_frac` | −1 to 1 (bottom to top of the section) | 0 |
| `belly_flat_frac` | 0–1 | 0 |
| `roof_pitch_deg` | 10–60 | 45 |
| `side_tumblehome_deg` | 0–45 | 0 |
| `lower_flare_deg` | 0–45 | 0 |
| `nose_style` | point, chisel, blunt | point |
| `tail_style` | point, knife, flat | point |
| `tail_flatten` | 0–1 (tail polygon squashed in z; above 0 adds a break) | 0 |
| `tail_end_size_frac` | 0–0.5 (today the constant `tail_radius_frac`) | today's constant |
| `max_section_x_frac` | 0.2–0.7 (where the body is widest) | today's value |

  `ref_default: smooth` keeps all 44 references unchanged. `test_baseline_step10.py` already guards that.
- Pods follow `body_style`, so one design language holds (SPEC §5 invariant 9).
- `layout.py` `size_centre_body()`: for a faceted body, find the smallest uniform scale at which every box corner passes the half-space test. This is a closed-form max over corners and faces.
- `shapes.py`: `body_rings` and `inside_body` dispatch to `facets.py` when `facets` is set.
- `areas.py`: top and frontal shadows of a faceted body come from its polygons exactly (no super-ellipse formula).
- `genome.py` is close to the 400-line limit. Put the new decode code in `facets.py`, not in `genome.py`.

Tests:
- Every face is planar (max distance from its best-fit plane < 1e-9 × body length).
- The solid is closed: each edge is shared by exactly two faces, and V − E + F = 2.
- Volume from the divergence theorem matches the sum of segment volumes.
- Mirror symmetry about y = 0.
- `panel_count` equals the actual face count on 1,000 random genomes.
- For every N from 8 to 200, `panel_plan` returns an exact plan or the nearest one, and the "exact" flag rejects only N values that can't be reached.
- Boxes are inside on 1,000 random genomes.
- Every section polygon is convex.
- Every reference decodes to the same `Design` as before.

Gate (auto): tests pass. `areas.py` and `facets.py` together add less than 10 ms per design.

## Step F2: Faceted body in CAD, physics, internals and a panel cut list

Read: SPEC §6, §8, §10, §11. `geom.py`, `mesh.py`, `mass.py`, `airplane.py`, `aero.py`, `internals.py`, `hardpoints.py`.

Build:
- `geom.py`: build each faceted body as a solid from its planar faces (polygon wire → face → shell → solid). Check the result with the OCC shape checker. Per RULES "Truth", verify the CadQuery calls on the installed version and record them in Decisions. Bodies stay separate solids, as today.
- `mesh.py`: it already volume-checks each part. Compare against the exact polyhedron volume from F1.
- `mass.py`: structure mass uses the exact wetted area (sum of face areas).
- `airplane.py` / `aero.py`: AeroBuildup models fuselages as round or super-ellipse sections. Map each polygon to an equivalent section with the same width, height and area. Record under Known issues that the buildup does not model chine vortices or the extra drag of sharp edges, and add one line to the report's fidelity statement. Check the installed aerosandbox `FuselageXSec` arguments first (RULES "Truth").
- `internals.py`: containment for faceted bodies uses the exact half-space test. Hatch outlines follow the face edges.
- New deliverable `panels/` in `designs/<id>/`:
  - `panels.csv`: one row per face: id, part, area_m2, outline edge lengths, bevel angle to each neighbour.
  - `panels.svg`: each face laid flat, numbered, with dimensions.

  This is a flat-sheet cut list, which a faceted body makes possible. It is written by a writer, not by core code (RULES "Code"). Add it to SPEC §11 as a Deviation.

Tests:
- CAD builds succeed on at least 98 % of 100 random faceted genomes (`slow`).
- CAD volume within 1 % of the polyhedron volume.
- The equivalent aero section has the same area as the polygon.
- `panels.csv` has exactly N rows per faceted body, and the sum of the face areas equals the wetted area.
- Internals are contained on the 5 winner fixtures re-decoded with `body_style: faceted`.

User gate: `python -m aeroforge gallery --n 50 --seed 1` with `body_style` pinned to faceted. Pass = every body is clearly made of flat panels, no broken or floating parts, and panel counts match the request.

## Step F3: Faceted wings and edge alignment

Read: SPEC §4, §6, §8. `planform.py`, `airfoil.py`, `aero.py`, `geom.py`.

Build:
- Gene `airfoil_facets`: 0 (smooth, today) or 3–8 flat segments per side. The section outline becomes a polygon that follows the chosen airfoil's thickness. Section aero: send the polygon coordinates to NeuralFoil. Check on the installed version that it accepts sharp-cornered coordinates, and record that in Decisions. If it doesn't, use the parent airfoil's polars plus a drag increment from a new constant (with a source note) and log that under Known issues.
- When `airfoil_facets` > 0, wing panels are lofted ruled between sections. Faces that would twist (taper with twist) are split into flat triangles, as for bodies.
- Gene `edge_alignment`: off, soft, strict. **strict** snaps the plan-view edges (body nose and tail edges, wing leading and trailing edges, tail edges) to a small set of shared angles taken from the main wing's sweep. **soft** only adds a style feature that rewards alignment. This is a classic low-signature planform rule. Here it is used for looks, not for any radar claim.

Tests: faceted wing faces are planar. Area and span invariants (SPEC §5 #3) still hold. With strict alignment, every plan-view edge angle is within 0.5° of an allowed angle. Tier-1 check time grows by less than 20 %.

User gate: a gallery of 50 with `airfoil_facets` > 0 and `edge_alignment: strict`.

## Additions to plan.md steps 11 and 12

- **Step 11 (descriptor):** also add `facet_density` = panel count ÷ (body wetted area ÷ $S$), scaled to [0, 1]. Smooth bodies give 0. `DESC_SIZE` becomes 21, not 20.
- **Step 12 (style):** add the style features `chine_sharpness` (mean bevel angle at the chine), `edge_alignment_score`, and `panel_count_norm`. Your marks round then tells the style score whether you like faceted bodies, rather than that being guessed.

---

## Step F4: The controls layer

Read: SPEC §4, §12, §13. `models.py`, `brief.py`, `genetable.py`, `constants.py`, `search.py`, `hooks/validate_brief.py`.

Build:
- `data/controls.yaml`: one row per control (the 100 rows below): `name`, `rank`, `group`, `kind`, `target`, `lo`/`hi` or `choices`, `unit`, `default`, `meaning`, `cues` (mission words that suggest it).
  - `kind` is one of: **brief** (an existing Brief field), **gene** (pins or narrows a gene's range), **knob** (overrides an `r`-interpolated knob), **filter** (finalist rule in `select()`), **check** (a new pass/fail check).
- `Brief.controls: dict[str, value | [lo, hi]] | None`. Validation:
  - unknown names fail
  - values outside the control's range fail
  - `[lo, hi]` with lo > hi fails
  - a `brief`-kind control given both as a field and as a control fails

  `validate_brief.py` uses the same model, so the hook needs no change.
- **Gene controls never change the genome length.** The genome stays [0,1]^D. A control narrows the range the decoder maps into: value = lo′ + g × (hi′ − lo′). A single value pins the gene (lo′ = hi′). The narrowed bounds travel as data: `decode(genome, req, bounds)`, where `bounds` comes from the brief. This is an interface change, so log it.
- Choice genes: a control may list the allowed choices. The decoder maps g onto that subset.
- References and the baseline always decode with the full ranges, so novelty and `d_min` stay comparable across runs.
- `report.md` gets a "Controls" section: every control you set, its value, and where it came from (you, or Claude from the mission words).

Tests:
- Each of the 100 controls validates and round-trips.
- A pinned gene decodes to exactly its value on 100 random genomes.
- A narrowed gene stays inside [lo′, hi′].
- References are unchanged when controls are set.
- A brief with an unknown control fails the hook with a 1-line message.

Gate (auto): tests pass. A brief that pins `body_style: faceted`, `body_panel_count: 40` and `layout: [canard, tandem]` produces 10 finalists that all obey those controls.

## Step F5: Alienness, new filters and new checks

Read: SPEC §8, §9. `search.py`, `search_cli.py`, `checks.py`, `checks.yaml`, `layout.py`.

Build:
- **`alienness` (0–1)**: one master control. It moves several knobs together, and any knob you set yourself wins over it:

| knob | a = 0 | a = 0.5 | a = 1 |
|---|---|---|---|
| `randomness` (at least) | as set | 0.7 | 1.0 |
| `exclude_layouts` | none | conventional | conventional, canard, tandem, three_surface |
| `min_conventional_distance` (× d_min) | 0 | 3 | 6 |
| `bomb_cap` (body_top_share max) | 0.45 | 0.38 | 0.30 |
| `critic_strangeness_weight` | 1 | 2 | 3 |
| `novelty_style_weight` w | 0.2 | 0.6 | 1.0 |

- New **filters** in `select()`:
  - `exclude_layouts`: the finalist's nearest reference may not belong to these families.
  - `min_conventional_distance`: the finalist must be at least this far from the `conventional` reference.
  - `surfaces_allowed` [min, max]: the count of active surfaces.
  - `tip_joins_allowed`: true or false.
- `fin_policy` (gene control on the s1–s3 cant ranges):
  - `any`
  - `no_vertical`: |cant| ≤ 60°
  - `v_tail_only`: fins at 30–60°
  - `none`: no surface above 30° cant
- New **checks**:
  - `stowage`: the whole aircraft's bounding box fits inside `stowage_box_m` [l, w, h] (no folding is modelled). It goes on the passable bar only when you set it.
  - `payload_view`: down, forward or side. The payload box must touch that face of its host, so a camera can look out. `layout.py` places the box there, and the check confirms it.
- Both new checks get bands in `checks.yaml` with source notes. Changing the bar is a gate change, logged under "user, fix.md".

Tests: each filter removes exactly the designs it should on a fixture archive. `alienness` = 1 gives the knob values in the table. A knob you set yourself overrides `alienness`. `stowage` passes and fails on hand-made designs at the box edge. A `payload_view: down` box touches the body floor.

Gate (auto): tests pass. On the 80 m/s stand-in brief with `alienness: 1`, none of the 10 finalists has `conventional` as its nearest reference.

## Step F6: Mission → controls in the forge skill

Read: SPEC §13. `plugin/skills/forge/SKILL.md`, `data/controls.yaml`.

Build:
- Skill step 1 gains a "controls" pass:
  1. Read the `name`, `rank`, `cues` and `meaning` columns of `controls.yaml` (small YAML, allowed by the engine rules).
  2. From the mission text, set the required brief fields first (ranks 1–6).
  3. Then walk the table in rank order, and set a control only when a mission word matches its `cues` or you named it. Stop at 15 controls unless you gave more yourself.
  4. Prefer ranges to single values. Write a single value only when you stated one.
  5. Write each control with a comment: `# from: "<the mission words>"`.
  6. If the mission says "alien", "weird", "novel", "unlike anything" or similar and gives no `alienness`, ask once (0.5 bold, 0.8 strange, 1.0 alien). This replaces today's randomness question.
  7. Show the chosen controls as a short table in the same AskUserQuestion that asks for missing required fields (still at most one call, at most 4 questions).
- The skill stays at or under 80 lines (SPEC §13). Put the cue table in `controls.yaml`, not in the skill.
- Skill step 8 reply: add a line with the panel count, `alienness`, and the path to `panels/panels.svg`.

Tests: `claude plugin validate ./plugin` passes. Three headless runs of the worked examples below write briefs whose controls match the expected sets.

User gate: you run `/forge new I want the drone to …` with a mission of your own. Pass = the controls table makes sense to you before the search starts.

## Step F7: Proof run (extends plan.md step 13)

Run `forge new` on the 40 m/s and 80 m/s briefs at `alienness` 0, 0.5 and 1, with 3 seeds each and `body_style: faceted` (18 runs in the background). Also run the three worked examples below once each.

Gate (auto), numbers in the CHANGELOG step entry:
- All `plan.md` step 13 gates still hold.
- Every delivered body has exactly the requested panel count when `panel_count_exact` is set, and is within ±10 % otherwise.
- `panels.csv` exists and its face areas sum to the wetted area.
- At `alienness` = 1, none of the delivered designs has `conventional` as its nearest reference.
- The passable rate stays at least half of plan.md step 13's.

User gate: you look at the viewers. Pass = flat-panel bodies, different every run, and nothing that looks like a known aircraft at `alienness` ≥ 0.8.

---

## The 100 controls, ranked

Rank = how much the control changes the delivered design. Ranks 1–16 size the whole aircraft through physics. Ranks 17–30 decide how strange the search is allowed to be. After that, the controls shape parts. Claude reads this list top-down when it turns a mission into controls.

**Kind:** B = brief field, G = gene (a value pins it, a `[lo, hi]` range narrows it), K = search knob, F = finalist filter, C = check. **Default "auto"** = the search decides. **Status** = exists today, or the step that adds it.

### Mission: physics sizes the aircraft from these (1–16)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 1 | `payload_mass_kg` | > 0 kg | required | B | exists |
| 2 | `payload_box_m` | [l, w, h] m | required | B | exists |
| 3 | `span_max_m` | 0.6–4 m | required | B | exists |
| 4 | `cruise_speed_ms` | > 0 m/s | required | B | exists |
| 5 | `endurance_min` or `range_km` | one of the two | required | B | exists |
| 6 | `launch` | hand, catapult, runway | required | B | exists |
| 7 | `dash_speed_ms` | > cruise | 1.3 × cruise | B | exists |
| 8 | `stall_max_ms` | m/s | from launch | B | exists |
| 9 | `mass_max_kg` | kg | none | B | exists |
| 10 | `stowage_box_m` | [l, w, h] m | none | C | F5 |
| 11 | `payload_view` | none, down, forward, side | none | C | F5 |
| 12 | `recovery` | belly, skid, wheels, net | belly | B | exists |
| 13 | `n_limit` | load factor | 4 | B | exists |
| 14 | `altitude_m` | m | 0 | B | exists |
| 15 | `gust_ms` | m/s | 5 | B | exists |
| 16 | `efficiency_floor` | 0–1 × baseline L/D | 0.8 | B | exists |

### Alienness and search (17–30)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 17 | `alienness` | 0–1 | 0.5 | K | F5 |
| 18 | `randomness` | 0–1 | 0.5 | B | exists |
| 19 | `exclude_layouts` | list of reference families | from alienness | F | F5 |
| 20 | `min_conventional_distance` | 0–10 × d_min | from alienness | F | F5 |
| 21 | `novelty_band_bottom` | 0.5–5 × d_min | from r | K | exists (knob) |
| 22 | `novelty_band_top` | 1–20 × d_min | from r | K | exists (knob) |
| 23 | `bomb_cap` | 0.2–0.6 body_top_share | from alienness | F | step 11 |
| 24 | `critic_strangeness_weight` | 0–3 | from alienness | K | step 12 |
| 25 | `novelty_style_weight` | 0–1 | from r | K | exists (step 8) |
| 26 | `style_floor` | 0–1 | 0.6 | K | exists (`s_min`) |
| 27 | `finalist_spread` | 0.5–3 × d_min | 1 | K | exists (step 8) |
| 28 | `accept_other_failures_max` | 0 to all | from r | K | exists (knob) |
| 29 | `search_budget` | 1,000–50,000 | 5,000 | K | exists (`--budget`) |
| 30 | `seed` | integer | fresh each run | B | exists |

### Faceted body (31–55)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 31 | `body_style` | smooth, faceted | faceted for search, smooth for references | G | F1 |
| 32 | `body_panel_count` | 4–200 panels | auto | G | F1 |
| 33 | `panel_count_exact` | true, false | false | C | F1 |
| 34 | `body_section_sides` | 3–12 | auto | G | F1 |
| 35 | `body_stations` | 1–8 | auto | G | F1 |
| 36 | `chine_z_frac` | −1 to 1 | auto | G | F1 |
| 37 | `belly_flat_frac` | 0–1 | auto | G | F1 |
| 38 | `roof_pitch_deg` | 10–60° | auto | G | F1 |
| 39 | `side_tumblehome_deg` | 0–45° | auto | G | F1 |
| 40 | `lower_flare_deg` | 0–45° | auto | G | F1 |
| 41 | `nose_style` | point, chisel, blunt | auto | G | F1 |
| 42 | `tail_style` | point, knife, flat | auto | G | F1 |
| 43 | `tail_flatten` | 0–1 | auto | G | F1 |
| 44 | `tail_end_size_frac` | 0–0.5 | auto | G | F1 |
| 45 | `max_section_x_frac` | 0.2–0.7 | auto | G | F1 |
| 46 | `body_on` | off, on | auto | G | exists |
| 47 | `body_length_frac` | 0.3–1.0 | auto | G | exists |
| 48 | `fam_body_fineness` | 4–10 | auto | G | exists |
| 49 | `body_width_ratio` | 0.6–2.0 | auto | G | exists (step 10) |
| 50 | `body_nose_frac` | 0.10–0.35 | auto | G | exists (step 10) |
| 51 | `body_tail_frac` | 0.15–0.45 | auto | G | exists (step 10) |
| 52 | `body_tail_upsweep_deg` | 0–15° | auto | G | exists (step 10) |
| 53 | `box_layout` | inline, side_by_side, stacked, wing_root | auto | G | exists (step 10) |
| 54 | `edge_alignment` | off, soft, strict | auto | G | F3 |
| 55 | `fam_body_section_exp` | 2–5 (smooth bodies only) | auto | G | exists |

### Main wing (56–75)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 56 | `wing_area_scale` | 1–3 × stall minimum | auto | G | exists (step 10) |
| 57 | `s0_aspect_ratio` | 4–20 | auto | G | exists |
| 58 | `fam_sweep_deg` | 0–55° | auto | G | exists |
| 59 | `s0_sweep_sign` | aft, forward | auto | G | exists |
| 60 | `fam_taper` | 0.3–1.0 | auto | G | exists |
| 61 | `fam_dihedral_deg` | −15 to 25° | auto | G | exists |
| 62 | `s0_cant_deg` | −15 to 15° | auto | G | exists |
| 63 | `s0_break_frac` | 0.3–1.0 | auto | G | exists |
| 64 | `s0_outer_cant_delta_deg` | −45 to 80° | auto | G | exists |
| 65 | `s0_outer_sweep_delta_deg` | −15 to 25° | auto | G | exists |
| 66 | `s0_root_chord_scale` | 1–2 | auto | G | exists (step 10) |
| 67 | `s0_twist_deg` | −6 to 2° | auto | G | exists (step 10) |
| 68 | `s0_root_x_frac` | 0.3–0.6 | auto | G | exists |
| 69 | `s0_root_z_frac` | −1 to 1 (low to high wing) | auto | G | exists |
| 70 | `fam_tip_style` | square, rounded, raked | auto | G | exists |
| 71 | `fam_airfoil` | naca00, naca24, naca44, naca33 | auto | G | exists |
| 72 | `fam_tc` | 0.08–0.15 | auto | G | exists |
| 73 | `airfoil_facets` | 0 (smooth), 3–8 | auto | G | F3 |
| 74 | `s0_control_frac` | 0–0.35 | auto | G | exists |
| 75 | `s0_sweep_delta_deg` | −8 to 8° | auto | G | exists |

### Layout and extra surfaces (76–92)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 76 | `layout` | free, conventional, canard, tandem, three_surface, flying_wing, joined | auto | G | exists (step 10) |
| 77 | `surfaces_allowed` | [min, max], 1–4 | auto | F | F5 |
| 78 | `fin_policy` | any, no_vertical, v_tail_only, none | any | G | F5 |
| 79 | `tip_joins_allowed` | true, false | true | F | F5 |
| 80 | `s1_active` | off, on | auto | G | exists |
| 81 | `s1_root_x_frac` | 0.05–0.95 | auto | G | exists |
| 82 | `s1_cant_deg` | −90 to 90° | auto | G | exists |
| 83 | `s1_aspect_ratio` | 2–8 | auto | G | exists |
| 84 | `s1_area_weight` | 0.1–1.0 | auto | G | exists |
| 85 | `s2_active` | off, on | auto | G | exists |
| 86 | `s2_root_x_frac` | 0.05–0.95 | auto | G | exists |
| 87 | `s2_cant_deg` | −90 to 90° | auto | G | exists |
| 88 | `s2_aspect_ratio` | 2–8 | auto | G | exists |
| 89 | `s3_active` | off, on | auto | G | exists |
| 90 | `s3_root_x_frac` | 0.05–0.95 | auto | G | exists |
| 91 | `s3_cant_deg` | −90 to 90° | auto | G | exists |
| 92 | `s3_aspect_ratio` | 2–8 | auto | G | exists |

### Pods and propulsion (93–100)

| # | control | range / choices | default | kind | status |
|---|---|---|---|---|---|
| 93 | `pods_on` | off, on | auto | G | exists |
| 94 | `pod_y_frac` | 0.15–0.9 | auto | G | exists |
| 95 | `pod_x_frac` | 0.1–0.6 | auto | G | exists |
| 96 | `pod_length_frac` | 0.15–0.6 | auto | G | exists |
| 97 | `prop_pattern` | centre, pods, centre_pods | auto | G | exists |
| 98 | `prop_direction` | tractor, pusher | auto | G | exists |
| 99 | `prop_z_frac` | −0.8 to 0.8 | auto | G | exists |
| 100 | `prop_gap_frac` | 0–1 | auto | G | exists |

Of the 100: 73 exist today and only need exposing (F4). 27 are added by F1, F3 and F5, or by plan.md steps 11–12. The other genes (`sN_host`, `sN_sweep_sign` for s1–s3, `sN_tip_join`, `sN_control_frac`, the remaining `sN_*` shape deltas) stay search-only. They are too fine-grained to be worth asking about.

---

## Mission words → controls (the `cues` column, starting set)

| mission says | controls Claude sets |
|---|---|
| alien, weird, unlike anything, never seen | `alienness` 0.8–1.0 (asks once if unclear) |
| stealthy, low-signature, F-117 style, angular | `body_style: faceted`, `edge_alignment: strict`, `fin_policy: v_tail_only` or `none`, `airfoil_facets: [3, 5]`, `prop_direction: pusher` |
| N panels, "made of flat sheets", "cut from board" | `body_style: faceted`, `body_panel_count: N`, `panel_count_exact: true` |
| long endurance, loiter, hours | `wing_area_scale: [1.5, 3]`, `s0_aspect_ratio: [10, 20]` |
| fast, dash, sprint | `fam_sweep_deg: [25, 55]`, `wing_area_scale: [1, 1.3]` |
| camera, survey, mapping, inspect | `payload_view: down` (or `forward` for inspection) |
| fits in a car / backpack / case | `stowage_box_m` from the stated size |
| hand launch, throw | `launch: hand` |
| rough field, land in grass | `recovery: skid`, `belly_flat_frac: [0.5, 1]` |
| windy, gusty, coast | `gust_ms` raised, `n_limit` raised |
| quiet | `prop_direction: pusher`, `prop_z_frac: [0.3, 0.8]` (prop above the body, shielded from below) |
| flat, wide, lifting body | `body_width_ratio: [1.5, 2]`, `roof_pitch_deg: [10, 25]` |
| no tail, tailless | `layout: flying_wing`, `fin_policy: none` |

---

## Worked examples (for the F6 tests)

**1.** *"I want the drone to map farmland for 2 hours with a 0.5 kg camera looking down, fit in my car boot, made of about 40 flat panels, and look alien."*

```yaml
payload_mass_kg: 0.5
payload_box_m: [0.12, 0.08, 0.08]   # asked: camera size
span_max_m: 2.5                     # asked
cruise_speed_ms: 18                 # asked
endurance_min: 120
launch: hand                        # asked
controls:
  alienness: 0.9                    # from: "look alien"
  payload_view: down                # from: "camera looking down"
  stowage_box_m: [1.0, 0.9, 0.4]    # from: "fit in my car boot" (asked size)
  body_style: faceted               # from: "flat panels"
  body_panel_count: 40              # from: "about 40 flat panels"
  wing_area_scale: [1.5, 3]         # from: "2 hours"
  s0_aspect_ratio: [10, 20]         # from: "2 hours"
```

Note: with `stowage_box_m` shorter than `span_max_m`, the stowage check wins, and the search keeps only designs whose span fits the box. Claude says so before searching.

**2.** *"I want the drone to dash at 60 m/s, angular like a stealth jet, no vertical tail."*
→ `body_style: faceted`, `edge_alignment: strict`, `airfoil_facets: [3, 5]`, `fin_policy: none`, `fam_sweep_deg: [30, 55]`, `dash_speed_ms: 60`, `alienness` asked.

**3.** *"I want the drone to carry 2 kg 50 km and be the strangest thing that still flies."*
→ `range_km: 50`, `alienness: 1.0`, nothing else. The search keeps full freedom inside the passable bar.

---

## Risks

- **Drag.** Flat panels and sharp chines add drag, and the buildup does not see it (F2 Known issue). Since plan.md step 8, power is no longer in the ranking, so faceted designs are not pushed out. But the delivered power margin may be optimistic. The report says so.
- **Stall and lateral checks on faceted wings** depend on NeuralFoil handling sharp coordinates (checked in F3). If it doesn't, the fallback drag increment is a guess with a source note.
- **Search space.** About 15 new genes and 100 controls. Narrowing controls makes the search easier. Many free new genes make it harder. Measure the passable rate in F1 and F3 and raise `search_budget` only if fewer than 10 designs qualify.
- **Exact panel counts** limit (K, S) and can fight the box fit. `panel_count_exact` is a check, so a run that can't meet it fails loudly, never silently.
- **History and marks** must be archived again when F1 changes the genome length (same rule as step 10: archive, never delete).
- **File sizes.** `genome.py`, `layout.py` and `search.py` are near the 400-line limit. New logic goes into new modules (`facets.py`, `controls.py`).

## Not in this plan

Radar cross-section estimates, materials and coatings, folding wings for stowage, sawtooth panel edges, landing gear variety, and a motor/propeller model for the power check.
