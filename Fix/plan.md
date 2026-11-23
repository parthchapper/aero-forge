# aero-forge 1.0: the mission-driven, flat-panel drone upgrade

This is a full upgrade of aero-forge, from **0.3** (a novelty search with round bodies) to **1.0** (a plugin that turns a plain-language mission into one very specific, flat-panel, first-principles drone). It rewrites the spec first, then rebuilds the parts of the engine, the plugin skill and the deliverables that the new product needs.

When the upgrade is done, you install the plugin and type:

```
/aero-forge:forge new I want the drone to [your mission]
```

and get back one drone designed for that mission, built from flat panels, shaped by search, sized by physics, and pushed as far from known aircraft as you ask.

---

## 1. What changes

| | aero-forge 0.3 (today) | aero-forge 1.0 (after this upgrade) |
|---|---|---|
| Input | a brief with ~15 fields (payload, span, speeds …) | your mission sentence. Claude writes the brief **plus up to 15 controls** chosen from a ranked table of 100 |
| Your control over the shape | `randomness` only | 100 ranked controls: requirements, shape ranges, search knobs, filters, checks |
| Body | one smooth super-ellipse tube | **flat-panel (faceted) body** with the panel count you choose; smooth stays available |
| Wings | smooth airfoils | smooth or **flat-panel airfoils**; optional edge alignment |
| Strangeness | novelty band set by `randomness` | **`alienness` dial**: bans familiar layouts, sets the distance from a normal plane, caps body bulk, raises the critic's taste for strangeness |
| Novelty measure | 16 numbers, blind to the body | 21 numbers: sees body size, shadow areas and panel density |
| Very specific requests | can silently produce few or no qualifiers | a **funnel report** names the controls that removed the most designs; conflicts are caught before the search |
| Deliverables | STEP, GLB, renders, report, viewer | the same plus **`panels/` cut list** (every flat panel laid out, numbered, dimensioned) and a **Controls** section in the report |
| Skill | `/forge setup`, `new`, `status` | plus **`/forge controls`** (the 100 controls with plain explanations); mission-to-controls built in |
| Version | 0.3.0 | **1.0.0**; the engine updates itself when the plugin updates |

**What stays the same:** the physics (derive, tier-1 and tier-2 checks), the quality-diversity search, the CAD pipeline, the visual critic, and the rules in `RULES.md`.

---

## 2. Starting point (baseline)

The upgrade starts from the tree as it is today:
- 86 genes, including layout presets, wing area scale, body nose / tail / width / upsweep, box packing, twist and strakes
- the randomness scale `r` and fresh seeds
- finalist ranking by novelty and style, spread apart
- dash-power feasibility stop in `derive`
- shadow areas (`areas.py`)
- a 16-number descriptor

Two decisions about the baseline are still open, and U0 settles them.

---

## 3. How 1.0 works, end to end

```
mission sentence
  → /forge skill: brief.yaml + controls (≤ 15, each tagged with the words it came from)
  → you confirm (one question round)
  → validate: ranges, conflicts, exact panel count reachable?       (stops early, plain message)
  → derive: physics sizes MTOW, wing area, battery from the brief   (stops if impossible)
  → resolve controls: gene bounds, knobs, filters, extra checks
  → search: genome in [0,1]^D decoded inside your bounds;
            novelty (21-number descriptor) + style; funnel counts
  → select: passable bar, style floor, novelty band, alienness filters, bomb cap, spread
  → verify (tier 2) → build + render top 3 → visual critic picks
  → deliverables: STEP, GLB, views, report (+Controls, +funnel), viewer, panels/
```

---

## 4. Spec v4: the changes U0 writes into SPEC.md

U0 rewrites `SPEC.md` from v3 to **v4** before any code changes, so every later step builds against the new spec rather than logging deviations from the old one.

| SPEC § | v3 | v4 |
|---|---|---|
| Title / §1 | "one novel, good-looking, checked UAV concept" | "one mission-specific, novel, checked UAV concept". Principle becomes: *size by physics, shape by search, geometry by code. Claude never writes geometry; it may narrow search ranges from the mission text and writes exact values only when the user states them.* |
| §4 Brief | ~15 fields | adds `controls` (map of control → value, range or list) and `alienness`; table of control kinds (B, G, K, F, C) and the precedence order |
| §4 Genome | D ≈ 65 | D ≈ 101 (86 + 15 new); controls narrow gene ranges, never the genome length |
| §4 Design | bodies are super-ellipse | `Body.facets` (vertices + flat faces) for faceted bodies; airfoil sections may be polygons |
| §5 Invariants | 1–10 | adds 11: every face of a faceted part is planar; 12: faceted solids are closed (V − E + F = 2); 13: every gene lies inside its resolved control bounds |
| §6 Geometry | spline lofts, super-ellipse bodies | adds planar-face solids for faceted bodies and ruled lofts for faceted wings |
| §7 Style | ~12 features | adds span ratio, body_top_share, width ratio, layout, chine sharpness, edge alignment, panel count |
| §8 Checks | 11 checks | adds `stowage`, `payload_view` and `panel_count_exact`; a user-set check joins the passable bar for that run |
| §9 Novelty and search | R^16, top 10 by novelty | R^21; ranking by w·z(novelty) + (1 − w)·z(style); filters: exclude_layouts, min_conventional_distance, bomb_cap, surfaces_allowed, tip_joins_allowed; funnel report; alienness knob table |
| §11 Outputs | 14 files | adds `panels/panels.csv`, `panels/panels.svg`; report gains Controls and Funnel sections |
| §12 CLI | 11 commands | adds `controls [--group] [--brief]` |
| §13 Plugin | setup, new, status | adds `controls` mode and `controls-guide.md`; mission → controls procedure |
| §14 Acceptance | 5 items | replaced by the U10 gate list |

Writing spec v4 is a gate and interface change, so it needs your approval (RULES). **Asking for this upgrade is that approval** for everything in this table. U0 records it once in `CHANGELOG.md` under a new heading `## Upgrade 1.0` ("user, upgrade 1.0"). After that, later steps build to spec v4 and log deviations only from v4. Two baseline decisions in U0 are **not** pre-approved; you answer them.

---

## 5. How to run the upgrade

- Steps are named **U0 … U10**, to avoid confusion with the 0.3 step numbers in the CHANGELOG.
- One step per session (RULES): *"Do upgrade step U3 from Fix/plan.md."*
- Each step: ≤ 5-line plan in CHANGELOG → build → test → gate → commit (`upgrade U3: <summary>`) → stop.
- Steps marked **User gate** wait for you.
- After every step the plugin still runs end to end. Features appear as they land.

| step | what it builds | gate |
|---|---|---|
| U0 | baseline close-out, archive, spec v4, upgrade bookkeeping | user |
| U1 | flat-panel body geometry (pure) | auto |
| U2 | flat-panel body in CAD, physics, internals; panel cut list | user |
| U3 | flat-panel wings; edge alignment | user |
| U4 | descriptor v2 (body-aware novelty, bomb cap) | auto |
| U5 | controls engine (the 100 controls) | auto |
| U6 | alienness, filters, new checks, funnel, conflict check | auto |
| U7 | looks: style features, marks round, critic v2 | user |
| U8 | skill v2: mission → controls, `/forge controls` | user |
| U9 | release 1.0.0: packaging, migration, start-up, errors, self-update | auto |
| U10 | acceptance from a clean install | user |

---

## U0: Baseline close-out and upgrade setup

Read: `RULES.md`, `CHANGELOG.md` (Decisions, Known issues, the last step entry), `SPEC.md` (whole: it is being rewritten).

**Decision A (you choose).** `test_brief_exercises_its_box_host[stub]` fails. It assumes the stub brief never puts boxes in the wing, which stopped being true once wing area could grow.
- **A1 (recommended):** the assertion becomes "small_boxes must use the wing host; stub may". It still protects the wing-host path. Logged as user-approved.
- A2: forbid wing-held boxes for stub in the decoder.
- A3: leave it failing. This blocks every "tests pass" gate.

**Decision B (you choose).** `side_by_side` and `stacked` packing rarely shorten the body, because fineness applies to the wider packed body.
- **B1 (recommended):** apply fineness to the diameter that inline packing would need, so other packings give shorter bodies.
- B2: leave it. Faceted bodies size length from stations anyway.

Build:
- Apply decisions A and B.
- Archive everything tied to the current genome or descriptor size to `archive_<date>_pre-1.0/`: `history.jsonl`, marks files, winner fixtures, gallery baselines. Never delete.
- Freeze a 0.3 baseline for later comparison: decode all 44 references and 20 random genomes, and save them to `tests/fixtures/baseline_0.3/`. Add `test_baseline_0_3.py`: references must stay identical through U1–U9.
- Rewrite `SPEC.md` to v4 (table in section 4). Keep v3 as `SPEC_v3.md`.
- `PLAN.md`: add a top line saying the upgrade steps live in `Fix/plan.md`.
- `CLAUDE.md` Status: "aero-forge 1.0 upgrade in progress; steps U0–U10 in Fix/plan.md."
- `CHANGELOG.md`: new section `## Upgrade 1.0` with the approval line, the decisions and one entry per U step.

Tests: the fast suite passes (with decision A applied). The baseline test passes.

**User gate:** open `~/.aero-forge/step10_gate/gallery/gallery.html` (the 0.3 baseline gallery). Pass = no broken or floating parts, and you can see canards, tandems, flat bodies, long-span wings and short bodies. Also read spec v4 and say yes or no.

---

## How a flat-panel ("faceted") body is built

This applies to U1 and U2.

- The body has **S stations** along its length. At each station the cross-section is a **convex polygon with K sides**, mirror-symmetric left to right.
- Between two stations with the **same polygon shape**, the second polygon is the first one **scaled evenly and shifted up or down**. Every joining face is then a flat trapezoid by geometry, with no tolerance needed.
- At a **shape break** (the polygon changes proportions, for example a flatter tail), each joining face would twist. It is split into two flat triangles.
- The nose and tail close with flat faces.

**Panel count** (flat faces on one body, both sides, ends included):

$$N = K\,(S-1) + K\,B + n_{\text{nose}} + n_{\text{tail}}$$

- $B$ = shape breaks (0–2)
- $n_{\text{nose}}$ = $K$ for `point` or `chisel`, 1 for `blunt`
- $n_{\text{tail}}$ = $K$ for `point` or `knife`, 1 for `flat`

Example: 6 sides, 3 stations, 1 break, point nose, flat tail → 12 + 6 + 6 + 1 = **25 panels**.

When you ask for N panels, the decoder lists every combination that gives exactly N and takes the one closest to what the search's genes ask for. If none is exact, it takes the nearest and notes it. With `panel_count_exact: true`, an N that can't be made is rejected in `validate`, before the search. That depends on N alone.

**The cross-section.** One convex outline is built from the shape genes: belly width, lower flare, chine (the sharp widest edge), tumblehome (upper sides leaning in) and roof pitch. Then K corners are placed on it. The chine and the belly and roof corners are always corners. The rest are spread evenly along the outline. If K is too small for all forced corners, the least important ones drop first (roof ridge, then belly corners).

**Why convex.** Each slice between stations is convex, so "is this box inside?" is an exact flat-face test (invariant 5 stays exact).

## U1: Flat-panel body geometry (pure code)

Read: SPEC v4 §4, §5. `models.py`, `shapes.py`, `layout.py`, `areas.py`, `genes.yaml`, `genetable.py`, `reference.py`.

Build:
- New module `facets.py` (pure):
  - `section_polygon`
  - `body_polyhedron` (vertices and faces)
  - `panel_count` and `panel_plan(N, wishes)`
  - `inside_faceted` (flat-face containment)
- `models.py`: add `Polyhedron(vertices_m, faces)` and `Body.facets: Polyhedron | None` (None = smooth).
- New genes, each with a `genes.yaml` row and a `ref_default` so that the 44 references stay unchanged:

| gene | range | ref_default |
|---|---|---|
| `body_style` | smooth, faceted | smooth |
| `body_section_sides` | 3–12 | 8 |
| `body_stations` | 1–8 | 3 |
| `chine_z_frac` | −1 to 1 | 0 |
| `belly_flat_frac` | 0–1 | 0 |
| `roof_pitch_deg` | 10–60 | 45 |
| `side_tumblehome_deg` | 0–45 | 0 |
| `lower_flare_deg` | 0–45 | 0 |
| `nose_style` | point, chisel, blunt | point |
| `tail_style` | point, knife, flat | point |
| `tail_flatten` | 0–1 | 0 |
| `tail_end_size_frac` | 0–0.5 (was the constant `tail_radius_frac`) | today's constant |
| `max_section_x_frac` | 0.2–0.7 | today's value |

- Pods use the same `body_style` (invariant 9).
- `layout.py` `size_centre_body()`: for a faceted body, the smallest even scale at which every box corner is inside (closed form).
- `shapes.py` and `areas.py` dispatch to `facets.py` when `facets` is set. Areas become exact polygon areas.
- New code goes in `facets.py` (`genome.py` is near the 400-line limit).

Tests:
- Faces are planar (< 1e-9 × body length).
- Solids are closed (each edge shared by two faces, V − E + F = 2).
- Divergence-theorem volume equals the sum of slice volumes.
- Mirror symmetric.
- `panel_count` equals the real face count on 1,000 random genomes.
- `panel_plan` for every N from 4 to 200 is exact or nearest, and the exact flag rejects only unreachable N.
- Boxes are inside on 1,000 random genomes.
- Sections are convex.
- The 0.3 baseline test is green.

Gate (auto): tests pass. Facets plus areas cost < 10 ms per design.

## U2: Flat-panel body in CAD, physics, internals; panel cut list

Read: SPEC v4 §6, §8, §10, §11. `geom.py`, `mesh.py`, `mass.py`, `airplane.py`, `aero.py`, `internals.py`, `hardpoints.py`, `report.py`.

Build:
- `geom.py`: faceted bodies built from their flat faces (polygon wire → face → shell → solid), checked with the OCC shape checker. Verify the CadQuery calls on the installed version and record them in Decisions.
- `mesh.py`: compare against the exact polyhedron volume.
- `mass.py`: skin mass from the exact wetted area.
- `airplane.py` / `aero.py`: AeroBuildup gets an equivalent section (same width, height and area). Check the installed `FuselageXSec` arguments first. Known issue: the extra drag of sharp chines is not modelled. One line goes into the report's fidelity statement.
- `internals.py`: exact flat-face containment. Hatch outlines follow face edges.
- Writer for `designs/<id>/panels/`:
  - `panels.csv`: one row per face: id, part, area, edge lengths, bevel angle to each neighbour.
  - `panels.svg`: every face laid flat, numbered and dimensioned.

Tests:
- CAD builds at least 98 % of 100 random faceted genomes (`slow`).
- CAD volume within 1 % of the polyhedron volume.
- The equivalent aero section has the polygon's area.
- `panels.csv` has N rows per faceted body, and its areas sum to the wetted area.
- Internals are contained on the 5 winner fixtures re-decoded as faceted.

**User gate:** `python -m aeroforge gallery --n 50 --seed 1` with `body_style` pinned to faceted. Pass = clearly flat-panel bodies, nothing broken or floating, and one `panels.svg` that looks right to you.

## U3: Flat-panel wings and edge alignment

Read: SPEC v4 §4, §6, §8. `planform.py`, `airfoil.py`, `aero.py`, `geom.py`.

Build:
- Gene `airfoil_facets`: 0 (smooth) or 3–8 flat segments per side. The section becomes a polygon that follows the chosen airfoil's thickness. Section aero: NeuralFoil on the polygon if the installed version accepts sharp corners (record in Decisions). Otherwise, the parent airfoil's data plus a drag increment from a new constant with a source note (Known issue).
- Faceted wing panels are lofted with straight lines between sections. Faces that would twist are split into flat triangles.
- Gene `edge_alignment`: off, soft, strict.
  - **strict** snaps the edges seen from above (body nose and tail edges, wing leading and trailing edges, tail edges) to a few angles taken from the main wing's sweep.
  - **soft** rewards alignment in the style score.

  This is used for looks; no radar claim.

Tests:
- Faceted wing faces are flat.
- Area and span invariant (#3) holds.
- Under strict alignment, every edge seen from above is within 0.5° of an allowed angle.
- Check time grows < 20 %.

**User gate:** a gallery of 50 with `airfoil_facets` > 0 and `edge_alignment: strict`.

## U4: Descriptor v2 (body-aware novelty)

Read: SPEC v4 §9. `descriptor.py`, `search_cli.py` (history).

Build:
- Add 5 terms to `desc()`, each scaled to [0, 1] by a constant:
  1. `body_top_share`
  2. frontal area ÷ S
  3. total top area ÷ S
  4. centre body length ÷ √S
  5. `facet_density` (panel count ÷ (wetted area ÷ S); 0 for smooth bodies)

  `DESC_SIZE` 16 → 21.
- Recompute `d_min` from the references, and the CVT centroids from new random decodes. Start a new `history.jsonl` (the old one was archived in U0).
- Bomb-cap filter: finalists need `body_top_share` ≤ `bomb_cap` (default 0.45; U6 ties it to `alienness`).

Tests:
- Designs that differ only in body length get different descriptors.
- Values in [0, 1] on 1,000 random genomes.
- A design over the cap is never a finalist.

Gate (auto): tests pass. No **new** pair of references closer than 0.5 × d_min. Count the existing pairs under the gate brief first and record the count.

## U5: Controls engine

Read: SPEC v4 §4, §12. `models.py`, `brief.py`, `genetable.py`, `constants.py`, `search.py`, `hooks/validate_brief.py`.

Build:
- `data/controls.yaml`: one row per control (the 100 at the end of this file): `name`, `rank`, `group`, `kind`, `target`, `lo`/`hi` or `choices`, `unit`, `default`, `meaning`, `cues` (mission words), `effect` (one plain sentence).
- New module `controls.py` (pure): `resolve_controls(brief) → ResolvedControls` (gene bounds, knob values, filters, extra checks, and the source of each value).
- `Brief.controls: dict[str, value | [lo, hi] | list] | None` and `Brief.alienness`. Validation rejects:
  - unknown names
  - out-of-range values
  - lo > hi
  - a control that duplicates a brief field

  The hook uses the same model.
- **Bounded decode:** the genome stays [0,1]^D, and each gene maps into its resolved range: value = lo′ + g × (hi′ − lo′). A single value pins it. A list restricts choices. `decode(genome, req, bounds)` takes the bounds as data.
- References and the baseline always decode with the full ranges, so novelty and `d_min` stay comparable between runs.
- Precedence:
  1. your typed value
  2. Claude's reading of your mission
  3. `alienness`
  4. `randomness`
  5. the full range
- CLI `python -m aeroforge controls [--group G] [--brief B]`: the table, or a brief's resolved controls, as JSON.
- `report.md`: "Controls" section (value and source of each).

Tests:
- All 100 controls validate and round-trip.
- A pinned gene decodes to exactly its value on 100 genomes.
- A narrowed gene stays inside its range.
- References are unchanged.
- An unknown control fails the hook in 1 line.
- Precedence order holds.
- Invariant 13 holds on 1,000 random genomes under random bounds.

Gate (auto): tests pass. A brief pinning `body_style: faceted`, `body_panel_count: 40` and `layout: [canard, tandem]` gives 10 finalists that all obey them.

## U6: Alienness, filters, new checks, funnel, conflicts

Read: SPEC v4 §8, §9. `search.py`, `search_cli.py`, `checks.py`, `checks.yaml`, `layout.py`, `derive.py`, `brief.py`.

Build:
- **`alienness` (0–1)** moves several knobs together, as `knob_*` rows with `a0`/`a1` and linear interpolation. Anything you set yourself wins:

| knob | a = 0 | a = 0.5 | a = 1 |
|---|---|---|---|
| `randomness` (at least) | as set | 0.7 | 1.0 |
| `exclude_layouts` | none | conventional | conventional, canard, tandem, three_surface |
| `min_conventional_distance` (× d_min) | 0 | 3 | 6 |
| `bomb_cap` | 0.45 | 0.38 | 0.30 |
| `critic_strangeness_weight` | 1 | 2 | 3 |
| `novelty_style_weight` | 0.2 | 0.6 | 1.0 |

- **Filters** in `select()`: `exclude_layouts`, `min_conventional_distance`, `surfaces_allowed`, `tip_joins_allowed`, `bomb_cap`.
- **`fin_policy`** (gene policy on s1–s3 cant): any, no_vertical (|cant| ≤ 60°), v_tail_only (30–60°), none (none above 30°).
- **New checks** with bands in `checks.yaml`:
  - `stowage`: sorted bounding-box sizes fit the sorted box sizes. `derive` lowers the effective span limit to the box's longest side and says so.
  - `payload_view`: the payload box touches the chosen host face. `layout.py` places it there.
  - `panel_count_exact`.

  A check you set joins the passable bar for that run.
- **Funnel:** `search` counts the designs each check and filter removed, and writes the counts to `finalists.json` and `report.md`. Fewer than 10 qualifiers: double the budget once. Still short: deliver the best with FAIL rows shown, and **name the two controls that removed the most**.
- **Conflict check** in `validate` (exit 2, one plain line each), before any search:
  - stowage smaller than the payload
  - `surfaces_allowed` max 1 with a multi-surface layout
  - `fin_policy: none` with `layout: conventional`
  - unreachable exact panel count
  - `body_on: off` with `box_layout` other than `wing_root`
  - and similar

Tests:
- Each filter removes exactly the right designs from a fixture archive.
- `alienness` 0, 0.5 and 1 match the table.
- A typed knob beats `alienness`.
- `stowage` passes and fails at the box edge.
- `payload_view: down` touches the floor.
- The funnel counts add up.
- Each listed conflict is caught.

Gate (auto): tests pass. On the 80 m/s stand-in brief with `alienness: 1`, no finalist has `conventional` as its nearest reference.

## U7: Looks: style features, marks round, critic v2

Read: SPEC v4 §7. `style.py`, `style.yaml`, `tune.py`, `gallery.py`, `plugin/agents/visual-critic.md`.

Build:
- New style features: span ratio, `body_top_share`, body width ratio, layout, `chine_sharpness` (mean bevel at the chine), `edge_alignment_score`, `panel_count_norm`. Drop the positive push on `body_fineness`.
- Marks round (at most 3): a 60-tile gallery of 1.0 designs; you mark good or bad; `tune-style`. Your taste decides how flat panels score.
- Critic v2:
  - receives `alienness` and the mission sentence
  - pick score = coherence + proportion + c(alienness) × strangeness + defects + **fit to mission** (1–5: does it look like it does the job you asked for?)
  - must name the finalist that looks least like "a torpedo with wings"
  - a broken-looking design scores 1 on defects whatever else it does

Tests:
- `tune-style` k-fold accuracy above the baseline rate.
- The critic writes valid `critique.json` for three fixture renders.

**User gate:** the new gallery looks clearly better to you than the 0.3 baseline gallery.

## U8: Skill v2: mission → controls

Read: SPEC v4 §13. `plugin/skills/forge/SKILL.md`, `data/controls.yaml`.

Build:
- New `plugin/skills/forge/controls-guide.md` (the procedure). `SKILL.md` points to it and stays within its line limit:
  1. Fill the required brief fields from the mission (ranks 1–6). Ask only for what's missing.
  2. Run `PY -m aeroforge controls` and read `name`, `rank`, `cues`, `effect`.
  3. Walk the table in rank order. Set a control only when your words match its `cues`, or you named it. At most 15, unless you gave more yourself.
  4. Ranges by default. Exact values only when you stated them.
  5. Tag each control `# from: "<your words>"`.
  6. "alien / weird / never seen" with no number: ask for `alienness` (0.5 bold, 0.8 strange, 1.0 alien).
  7. Show the controls as a short table inside the one AskUserQuestion call (≤ 4 questions). One option is always "go".
  8. Run `validate` before `derive`. On a conflict, show the line and ask once.
- New mode `/forge controls [group]`: the 100 controls with their effect sentences.
- The final reply adds: panel count, `alienness`, the funnel's top filter, and the path to `panels/panels.svg`.

Tests:
- `claude plugin validate ./plugin` passes.
- Headless runs of the three worked examples write briefs whose controls match the expected sets.

**User gate:** you run `/forge new I want the drone to …` with your own mission. Pass = the controls table makes sense to you before the search starts.

## U9: Release 1.0.0

Read: SPEC v4 §2, §13. `plugin/.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `plugin/README.md`, `plugin/scripts/setup_env.py`, `hooks/hooks.json`, `engine/pyproject.toml`, `cli.py`.

Build:
- Version **1.0.0** in `plugin.json`, `marketplace.json` and `pyproject.toml`. New descriptions mention missions, flat-panel bodies, controls and alienness.
- `README.md` rewritten for 1.0: install, `/forge new` with two mission examples, `/forge controls`, outputs (with `panels/`), run time, limits.
- **Migration:** on the first 1.0 run, an old-size `history.jsonl` or marks file is moved to a dated copy and a new one is started. Never delete.
- **Self-containment:** `controls.yaml` and the guide ship as package data. Test from a `git archive` copy.
- **Start-up:** lazy-import CadQuery, render and aerosandbox in `cli.py`. `validate`, `controls` and `derive` must start in < 3 s.
- **Errors:** every exit 2 or 3 message names the field or control at fault and one fix. Exit 3 tells you to run `/forge setup`.
- **Self-update:** `setup_env.py` compares the engine version with the plugin version and reinstalls on a mismatch.

Tests:
- Plugin validate passes.
- A `git archive` copy runs `validate`, `controls` and `derive` on the sample brief.
- Migration keeps old files.
- A version mismatch triggers a reinstall (dry run).
- `validate` starts in < 3 s.

Gate (auto): tests pass and lint is clean.

## U10: Acceptance from a clean install

Read: SPEC v4 §14.

Run, from the **installed** 1.0.0 plugin, in a folder outside the repo, after `/aero-forge:forge setup`:
- 40 m/s and 80 m/s briefs × `alienness` 0, 0.5, 1 × 3 seeds, faceted (18 runs, background).
- The three worked examples below, once each, through `/forge new` with the mission sentence.

Gate (auto), with numbers in the CHANGELOG:
- The fast suite passes from a fresh clone after `setup_env.py`. `claude plugin validate ./plugin` passes.
- Finalists with one body, no joins and ≤ 2 surfaces: ≤ 30 % at `alienness` = 1 (0.3 baseline: 6/10 and 10/10).
- Median `body_top_share` of delivered designs ≤ 0.35 at `alienness` ≥ 0.5.
- ≥ 4 different `layout` values among the 12 delivered designs at `alienness` ≥ 0.5.
- Exact panel count delivered when `panel_count_exact` is set, ±10 % otherwise.
- `panels.csv` areas sum to the wetted area on every delivery.
- At `alienness` = 1, no delivered design has `conventional` as its nearest reference.
- Every delivered design passes the passable bar at tier 2, or the run names the blocking controls.
- Passable rate at least half of the 0.3 measurement (321 of 5,004 on the 80 m/s stand-in).
- Every run delivers all SPEC v4 §11 files.

**User gate:** you open the viewers. Pass = flat-panel bodies, a different design every run, each one clearly answering its mission, and nothing that looks like a known aircraft at `alienness` ≥ 0.8.

Then install for good:

```
/plugin marketplace add ./
/plugin install aero-forge@aero-forge-market
/aero-forge:forge setup
/aero-forge:forge new I want the drone to …
```

---

## Risks

- **Drag of flat panels** is not modelled (U2 Known issue). Power is not in the ranking, so faceted designs are not pushed out, but power margins may be optimistic. The report says so.
- **Faceted wings** depend on NeuralFoil accepting sharp corners (checked in U3). The fallback drag increment is an estimate with a source note.
- **Very narrow controls** can leave fewer than 10 qualifiers. The funnel names the culprit and the budget doubles once. A run never silently delivers a failing design.
- **Search time:** today about 18 min per 5,000 evaluations on 16 workers. ~15 more genes and new filters may need a bigger budget. Measure in U1, U3 and U6, and raise `search_budget` only when the funnel shows fewer than 10 qualifiers.
- **Fast test suite** is near its 90 s limit. New CAD and search tests go under `slow`. Never shrink tests.
- **File size limit** (400 lines). New logic goes in new modules (`facets.py`, `controls.py`).
- **History and marks** are reset by the new genome and descriptor. They are archived in U0, never deleted.

## Not in this upgrade

Radar cross-section estimates, materials, folding wings, sawtooth panel edges, landing-gear variety, VTOL, and a real motor/propeller model.

---

# The 100 controls: what each one does and how it works

## How controls work in general

**What a control is.** A named setting in your brief, under `controls:`. You (or Claude, from your mission words) can set any of the 100. Anything you don't set stays **auto**: the search is free to choose it.

**Five kinds**, by how the engine applies them:

| kind | letter | how it works | what happens if it can't be met |
|---|---|---|---|
| Brief requirement | **B** | Goes into `derive`. Physics sizes the whole aircraft from it: weight, wing area, battery, reference length. Every design in the search shares these numbers. | `derive` stops before searching and says which requirement and why (e.g. dash power). |
| Gene control | **G** | Narrows or pins one shape setting. The search still explores, but only inside your range. A single value fixes it. A list (for choices) keeps only those options. | Can't fail by itself. Too narrow a range shows up in the funnel. |
| Search knob | **K** | Changes how the search behaves: how strange, how spread out, how many tries. It never changes physics. | Can't fail. |
| Filter | **F** | A rule a design must meet to become a finalist. Applied after the checks, before the critic. | The funnel counts how many designs it removed. If too few pass, the run names it. |
| Check | **C** | A new pass/fail test, reported like stall or trim. When you set it, it joins the passable bar for that run. | The design fails that check; the report shows the margin. |

**Ranges vs values.** `fam_sweep_deg: [25, 55]` means "somewhere in 25–55°, the search decides". `fam_sweep_deg: 40` means "exactly 40°". Ranges keep the design alien and first-principles. Values make it specific. Claude uses ranges unless you give a number.

**Precedence** (highest wins):
1. a number you typed
2. Claude's reading of your mission words
3. `alienness`
4. `randomness`
5. the full range

The report lists the source of every value.

**Conflicts** are caught before the search (U6), for example a stowage box shorter than the payload, or "no tail" with a conventional layout. You get one plain line per conflict.

**Rank** = how much the control changes the delivered drone. Ranks 1–16 resize everything. Ranks 17–30 decide how strange it may be. Ranks 31–100 shape individual parts. Claude reads the table top-down when it turns a mission into controls.

**Status column:** "exists" = in the engine today (U5 only exposes it). A U-number = added by that upgrade step.

---

## Group 1: Mission. Physics sizes the drone from these (1–16)

| # | control | kind | range / default | what it does to the drone | how it works |
|---|---|---|---|---|---|
| 1 | `payload_mass_kg` | B | > 0 kg, required | The biggest driver of size: more payload means a heavier aircraft, a bigger wing and a bigger battery. | `derive` runs a weight closure: payload + structure + battery + motor … = MTOW, repeated until it settles. MTOW then sets wing area through the stall speed. |
| 2 | `payload_box_m` | B | [l, w, h] m, required | Sets how fat or long the body must be (or whether the payload can sit in the wing). | The layout packs this box with the battery and avionics. The body (or wing centre section) is sized so every box fits (invariant 5). |
| 3 | `span_max_m` | B | 0.6–4 m, required | Upper limit on wingspan. Bigger allows slender, efficient wings. | The decoder lowers the aspect ratio of any wing that would exceed it. `wing_area_scale` is capped by it too. |
| 4 | `cruise_speed_ms` | B | m/s, required | Sets the cruise flight point: drag, power and range are judged here. | Flight point P3. Energy and efficiency checks run at this speed. |
| 5 | `endurance_min` or `range_km` | B | one of them, required | Sets battery size: longer missions mean a bigger battery, a heavier aircraft and a bigger wing. | `derive` sizes battery Wh to meet it at loiter or cruise power. The energy check confirms it. |
| 6 | `launch` | B | hand, catapult, runway | Sets how slowly it must fly (hand launch means a slow stall, so a bigger wing) and the thrust needed to climb away. | Gives the default stall speed, the launch speed and the minimum thrust-to-weight (`checks.yaml`). |
| 7 | `dash_speed_ms` | B | default 1.3 × cruise | Top speed. High values need lots of power. Impossible values stop the run before searching. | Flight point P4. `derive` stops if the baseline needs more than 10 × installed power. |
| 8 | `stall_max_ms` | B | default from launch | Lower means a bigger wing (slow, floaty). Higher means a smaller wing. | Wing area minimum: S_min = 2 m g / (ρ V_s² C_Lmax). |
| 9 | `mass_max_kg` | B | none | A hard weight cap (e.g. a regulation class). | The mass check fails any design whose MTOW is above it. `derive` stops if even the baseline can't meet it. |
| 10 | `stowage_box_m` | C | none, U6 | The whole drone must fit in this box (car boot, case). Limits span and length. | Bounding-box check. `derive` also lowers the effective span limit to the box's longest side. |
| 11 | `payload_view` | C | none / down / forward / side, U6 | Puts the payload against a body face so a camera can look out that way. | The layout places the payload box against that face. The check confirms that it touches the face. |
| 12 | `recovery` | B | belly, skid, wheels, net | Changes the underside: skids or wheels add ground clearance and change where props may sit. | Sets the ground line for prop clearance and the recovery internals (SPEC §10). |
| 13 | `n_limit` | B | default 4 | How hard it may pull. Higher means a stronger, heavier spar. | Flight point P5. Spar stress × 1.5 must stay below allowable. |
| 14 | `altitude_m` | B | default 0 | Thinner air higher up means a bigger wing and more power. | Air density at that altitude in every flight point. |
| 15 | `gust_ms` | B | default 5 | Stronger gusts mean a stronger spar and more trim margin. | Flight point P6 adds the gust load. |
| 16 | `efficiency_floor` | B | default 0.8 | How close to a normal plane's lift-to-drag it must stay. Lower allows stranger, draggier shapes. | Efficiency check: L/D ≥ floor × baseline L/D. Not on the passable bar unless you set it. |

## Group 2: Alienness and search. How strange it may get (17–30)

| # | control | kind | range / default | what it does | how it works |
|---|---|---|---|---|---|
| 17 | `alienness` | K | 0–1, default 0.5, U6 | The master dial: 0 allows designs close to known aircraft, 1 throws out anything that looks familiar. | Sets controls 18–20, 23–25 together (table in U6). Anything you set yourself wins. |
| 18 | `randomness` | B | 0–1, default 0.5 | How wide the search wanders, and how far finalists may be from known aircraft. | Sets the novelty band, mutation size and reference seeding (existing knobs). |
| 19 | `exclude_layouts` | F | list, default from alienness, U6 | Bans whole families: a finalist whose nearest known aircraft is in the list is dropped. | Compares the design's descriptor to the 44 references and drops it if the closest one belongs to an excluded family. |
| 20 | `min_conventional_distance` | F | 0–10 × d_min, U6 | How far from a normal plane it must be. | Distance from the `conventional` reference in the descriptor space must be at least this. |
| 21 | `novelty_band_bottom` | K | 0.5–5 × d_min | The least novelty a finalist may have. | Overrides `knob_novelty_band_bottom`. |
| 22 | `novelty_band_top` | K | 1–20 × d_min | The most novelty allowed. Higher lets truly odd shapes through. | Overrides `knob_novelty_band_top`. |
| 23 | `bomb_cap` | F | 0.2–0.6, U4/U6 | Stops "torpedo with wings": limits how much of the top view may be body. | Filter on `body_top_share` (body top area ÷ total top area). |
| 24 | `critic_strangeness_weight` | K | 0–3, U7 | How much the visual critic rewards strangeness when picking the final design. | Multiplier c on the critic's strangeness score. |
| 25 | `novelty_style_weight` | K | 0–1 | Finalist ranking: 1 = most novel first, 0 = best-looking first. | Weight w in w·z(novelty) + (1 − w)·z(style) (existing). |
| 26 | `style_floor` | K | 0–1, default 0.6 | The minimum looks score to be considered. Lower allows uglier but odder designs. | Overrides `s_min`. |
| 27 | `finalist_spread` | K | 0.5–3 × d_min | How different the 10 finalists must be from each other. Higher gives the critic more distinct choices. | The minimum distance between chosen finalists in `select()`. |
| 28 | `accept_other_failures_max` | K | 0 to all | How many non-essential checks (power, energy, efficiency, mass) a delivered design may fail. 0 = strictest. | Overrides `knob_accept_other_failures_max` in verify. |
| 29 | `search_budget` | K | 1,000–50,000, default 5,000 | More tries means better, stranger results and a longer run (about 18 min per 5,000 on 16 cores). | Number of tier-1 evaluations (`--budget`). |
| 30 | `seed` | B | integer, fresh each run | Same seed + same brief = the same design again. Leave it out for a new one every time. | Seeds the search's random generator, and is recorded in the report. |

## Group 3: Flat-panel body (31–55)

| # | control | kind | range / default | what it does | how it works |
|---|---|---|---|---|---|
| 31 | `body_style` | G | smooth / faceted, U1 | Faceted = body built only from flat plates with sharp edges. Smooth = today's rounded body. | Chooses `facets.py` or the super-ellipse body. Pods follow the same style. |
| 32 | `body_panel_count` | G | 4–200, U1 | Total flat panels on the body. Few = bold and crystal-like. Many = finer and closer to round. | `panel_plan` picks sides, stations, breaks and end styles that give this count (formula above). |
| 33 | `panel_count_exact` | C | true / false, U1 | true = exactly that many panels or refuse to run. false = as close as possible. | Checked in `validate` from the count alone, before any search. |
| 34 | `body_section_sides` | G | 3–12, U1 | Sides of the body's cross-section: 3 = triangular, 4 = diamond or box, 6–8 = chined hexagon or octagon. | K in the polygon builder. |
| 35 | `body_stations` | G | 1–8, U1 | How many times the body changes size along its length. More = more panel rows. | S, the number of polygon stations. |
| 36 | `chine_z_frac` | G | −1 to 1, U1 | Height of the sharp widest edge: low (−1) gives a flat-bottomed wedge, high (+1) gives an inverted wedge. | Vertical position of the chine corner in the section outline. |
| 37 | `belly_flat_frac` | G | 0–1, U1 | How wide the flat underside is. 1 = a flat plate belly, good for skid landings. | Width of the flat bottom edge as a share of body width. |
| 38 | `roof_pitch_deg` | G | 10–60°, U1 | Steepness of the top panels. Low = flat top, high = sharp ridge. | Angle of the upper faces from horizontal. |
| 39 | `side_tumblehome_deg` | G | 0–45°, U1 | How much the upper sides lean inward above the chine (the classic faceted look). | Inward lean of the faces between the chine and the roof. |
| 40 | `lower_flare_deg` | G | 0–45°, U1 | How much the lower sides lean outward below the chine. | Outward lean of the faces between the belly and the chine. |
| 41 | `nose_style` | G | point / chisel / blunt, U1 | Point = pyramid nose, chisel = horizontal knife edge, blunt = one flat face. | Nose closure and its panel count (K or 1). |
| 42 | `tail_style` | G | point / knife / flat, U1 | Point = pyramid, knife = horizontal edge (flat exhaust look), flat = cut-off end. | Tail closure and its panel count. |
| 43 | `tail_flatten` | G | 0–1, U1 | Squashes the tail section into a flat wedge. Adds a shape break (one more panel ring). | The tail polygon is scaled in height by (1 − value), creating break B. |
| 44 | `tail_end_size_frac` | G | 0–0.5, U1 | How big the tail end is: 0 = sharp point, 0.5 = wide stub. | Was the constant `tail_radius_frac`, now a gene. |
| 45 | `max_section_x_frac` | G | 0.2–0.7, U1 | Where the body is widest: forward (0.2) gives a teardrop, aft (0.7) gives an arrow. | Station of the largest polygon along the length. |
| 46 | `body_on` | G | off / on | Off = no centre body (flying wing, boxes in the wing) when the boxes fit. | Body is forced on if the boxes don't fit the wing centre section. |
| 47 | `body_length_frac` | G | 0.3–1.0 | Minimum body length relative to the reference length. | Lower bound. The box fit may lengthen it. |
| 48 | `fam_body_fineness` | G | 4–10 | Length ÷ thickness: 4 = stubby, 10 = slender. | Sets body diameter from length (upgrade decision B in U0 decides which diameter). |
| 49 | `body_width_ratio` | G | 0.6–2.0 | Above 1 = wide, flat lifting body. Below 1 = tall and narrow. | Multiplies the width ÷ height set by the boxes. The body only grows, so boxes always fit. |
| 50 | `body_nose_frac` | G | 0.10–0.35 | Nose length as a share of the body. | Nose segment length. **Smooth bodies only: inert when `body_style` is faceted** (the widest station, `max_section_x_frac`, sets the nose and tail lengths; U1 audit F11). |
| 51 | `body_tail_frac` | G | 0.15–0.45 | Tail length as a share of the body. | Tail segment length. **Smooth bodies only: inert when `body_style` is faceted** (the widest station, `max_section_x_frac`, sets the nose and tail lengths; U1 audit F11). |
| 52 | `body_tail_upsweep_deg` | G | 0–15° | Tail bends upward (clears the ground, raises a pusher prop). | Tail axis rises at this angle. |
| 53 | `box_layout` | G | inline / side_by_side / stacked / wing_root | How payload, battery and avionics are packed: a row (long body), side by side (wide), stacked (tall), or in the wing roots (tiny body). | Layout packing rule. `wing_root` tries the wing first. |
| 54 | `edge_alignment` | G | off / soft / strict, U3 | Strict = all edges seen from above line up on a few shared angles (the classic angular look). Soft = encouraged. | Strict snaps edge angles in the decoder. Soft adds a style feature. |
| 55 | `fam_body_section_exp` | G | 2–5 | Smooth bodies only: 2 = oval, 5 = rounded box. | Super-ellipse exponent. Ignored when faceted. |

## Group 4: Main wing (56–75)

| # | control | kind | range / default | what it does | how it works |
|---|---|---|---|---|---|
| 56 | `wing_area_scale` | G | 1–3 × stall minimum | Bigger wing = slower landing, longer endurance and a more glider-like look. | S = scale × S_min, capped so the span stays within `span_max_m`. |
| 57 | `s0_aspect_ratio` | G | 4–20 | Wing slenderness: 4 = stubby and agile, 20 = long and thin, efficient. | Span b = √(AR × S). Lowered if b > span_max. |
| 58 | `fam_sweep_deg` | G | 0–55° | Wing sweep: 0 = straight, 55 = sharp arrow or delta. Shared by all surfaces. | The design family's sweep. Each surface adds a small delta. |
| 59 | `s0_sweep_sign` | G | aft / forward | Forward sweep gives a striking, unusual look. | Sign of the main wing's sweep. |
| 60 | `fam_taper` | G | 0.3–1.0 | Tip chord ÷ root chord: 1 = rectangular, 0.3 = pointed. | Family taper. |
| 61 | `fam_dihedral_deg` | G | −15 to 25° | Wing tips up (V) or down (anhedral). Affects roll stability. | Family dihedral. The lateral check judges the result. |
| 62 | `s0_cant_deg` | G | −15 to 15° | Extra tilt of the main wing's inner panel. | Inner-panel cant. |
| 63 | `s0_break_frac` | G | 0.3–1.0 | Where the wing kinks: 1 = no kink. | Break station as a share of the semi-span. |
| 64 | `s0_outer_cant_delta_deg` | G | −45 to 80° | Outer panels bent up (gull or winglet) or down. | Outer-panel cant added at the break, height-capped (existing). |
| 65 | `s0_outer_sweep_delta_deg` | G | −15 to 25° | Outer panels swept more or less than the inner ones (cranked wing). | Sweep added at the break. |
| 66 | `s0_root_chord_scale` | G | 1–2 | Strake or glove: a wide wing root that blends toward the body. | Root chord stretched, then the area re-normalised. |
| 67 | `s0_twist_deg` | G | −6 to 2° | Washout: tips at a lower angle so they stall last. | Linear twist from root to tip. |
| 68 | `s0_root_x_frac` | G | 0.3–0.6 | Wing position along the body: forward or aft. | Root station along the host. Balance and trim judge it. |
| 69 | `s0_root_z_frac` | G | −1 to 1 | Low, mid or high wing. | Root height within the body. |
| 70 | `fam_tip_style` | G | square / rounded / raked | Wing-tip shape. | Family tip treatment. |
| 71 | `fam_airfoil` | G | naca00 / 24 / 44 / 33 | Wing section: symmetric (aerobatic) to highly cambered (slow lift). | Airfoil family for every surface. |
| 72 | `fam_tc` | G | 0.08–0.15 | Wing thickness: thicker = stronger spar and room for boxes, slightly more drag. | Thickness ÷ chord. |
| 73 | `airfoil_facets` | G | 0 or 3–8, U3 | Flat-panel wing section to match the body: 3 = very angular, 8 = nearly smooth. | Polygon section, straight-line lofts, faces split if they twist. |
| 74 | `s0_control_frac` | G | 0–0.35 | Size of the ailerons or elevons. | Control chord share. The trim check needs enough of it. |
| 75 | `s0_sweep_delta_deg` | G | −8 to 8° | Fine sweep offset of the main wing from the family value. | Per-surface delta. |

## Group 5: Layout and extra surfaces (76–92)

| # | control | kind | range / default | what it does | how it works |
|---|---|---|---|---|---|
| 76 | `layout` | G | free / conventional / canard / tandem / three_surface / flying_wing / joined | The overall arrangement of wings and tails. A list allows several. `free` lets the slot genes invent one. | Applies the slot preset from `layouts.yaml`. The slot genes fine-tune it. |
| 77 | `surfaces_allowed` | F | [min, max] of 1–4, U6 | How many separate wings and tails. [1, 1] = a pure flying wing. [3, 4] = busy multi-surface designs. | Finalist filter on the count of active surfaces. |
| 78 | `fin_policy` | G | any / no_vertical / v_tail_only / none, U6 | Controls vertical tails: none at all, only V-tails, or no near-vertical fins. | Narrows the cant ranges of s1–s3. |
| 79 | `tip_joins_allowed` | F | true / false, U6 | Allows or bans joined and box wings (tips connected). | Filter on joined-tip count. Also pins `sN_tip_join` to none when false. |
| 80 | `s1_active` | G | off / on | Second surface present (tail, canard or second wing). | Slot 1 on or off. |
| 81 | `s1_root_x_frac` | G | 0.05–0.95 | Where it sits: front (canard) or back (tail). | Root station along its host. |
| 82 | `s1_cant_deg` | G | −90 to 90° | Its tilt: 0 = horizontal, 90 = vertical fin, 30–60 = V-tail. | Surface cant. |
| 83 | `s1_aspect_ratio` | G | 2–8 | Its slenderness. | Per-surface aspect ratio. |
| 84 | `s1_area_weight` | G | 0.1–1.0 | Its size relative to the main wing (1 = equal: tandem). | Share of lifting area before normalisation. |
| 85 | `s2_active` | G | off / on | Third surface present. | Slot 2 on or off. |
| 86 | `s2_root_x_frac` | G | 0.05–0.95 | Its position along the body. | Root station. |
| 87 | `s2_cant_deg` | G | −90 to 90° | Its tilt. | Surface cant. |
| 88 | `s2_aspect_ratio` | G | 2–8 | Its slenderness. | Aspect ratio. |
| 89 | `s3_active` | G | off / on | Fourth surface present. | Slot 3 on or off. |
| 90 | `s3_root_x_frac` | G | 0.05–0.95 | Its position. | Root station. |
| 91 | `s3_cant_deg` | G | −90 to 90° | Its tilt. | Surface cant. |
| 92 | `s3_aspect_ratio` | G | 2–8 | Its slenderness. | Aspect ratio. |

## Group 6: Pods and propulsion (93–100)

| # | control | kind | range / default | what it does | how it works |
|---|---|---|---|---|---|
| 93 | `pods_on` | G | off / on | A pair of pods on the wing (twin-boom look, motors or tails on pods). | Pod pair on surface 0. |
| 94 | `pod_y_frac` | G | 0.15–0.9 | How far out on the wing the pods sit. | Pod station along the inner panel. |
| 95 | `pod_x_frac` | G | 0.1–0.6 | How far the pods stick out ahead of the wing. | Pod length share ahead of the leading edge. |
| 96 | `pod_length_frac` | G | 0.15–0.6 | Pod length. | Pod length ÷ reference length. |
| 97 | `prop_pattern` | G | centre / pods / centre_pods | Number and position of motors: one in the body, two on pods, or three. | Propulsor hosts. Clearance invariant 7 is enforced. |
| 98 | `prop_direction` | G | tractor / pusher | Prop at the front (pulls) or back (pushes: clean nose for a camera, quieter look). | Hub at the host's nose or tail. |
| 99 | `prop_z_frac` | G | −0.8 to 0.8 | Prop raised or lowered (high = shielded from below, clear of the ground). | Hub height offset ÷ host end radius. |
| 100 | `prop_gap_frac` | G | 0–1 | Gap between the prop and its body. | Hub gap between the min and max from `constants.yaml`. |

**Totals:** 73 controls exist today and only need exposing (U5). 27 are added by U1, U3, U4, U6 and U7. The remaining genes (`sN_host`, `sN_sweep_sign` for s1–s3, `sN_tip_join`, `sN_control_frac`, the fine shape deltas) stay search-only. They are too fine to be worth asking about.

---

## Mission words → controls (starting `cues` column)

| your mission says | controls Claude sets |
|---|---|
| alien, weird, never seen, unlike anything | `alienness` 0.8–1.0 (asks once if unclear) |
| stealthy, angular, low-signature, F-117 style | `body_style: faceted`, `edge_alignment: strict`, `airfoil_facets: [3, 5]`, `fin_policy: v_tail_only` or `none`, `prop_direction: pusher` |
| N panels, flat sheets, cut from board | `body_style: faceted`, `body_panel_count: N`, `panel_count_exact: true` |
| long endurance, loiter, hours | `wing_area_scale: [1.5, 3]`, `s0_aspect_ratio: [10, 20]` |
| fast, dash, sprint | `fam_sweep_deg: [25, 55]`, `wing_area_scale: [1, 1.3]` |
| camera, survey, mapping | `payload_view: down` |
| inspect, look ahead | `payload_view: forward`, `prop_direction: pusher` |
| fits in a car / backpack / case | `stowage_box_m` (asks the size) |
| hand launch, throw | `launch: hand` |
| rough field, grass landing | `recovery: skid`, `belly_flat_frac: [0.5, 1]` |
| windy, coast, gusty | `gust_ms` raised, `n_limit` raised |
| quiet | `prop_direction: pusher`, `prop_z_frac: [0.3, 0.8]` |
| flat, wide, lifting body | `body_width_ratio: [1.5, 2]`, `roof_pitch_deg: [10, 25]` |
| no tail, tailless | `layout: flying_wing`, `fin_policy: none` |
| no body, all wing | `body_on: off`, `box_layout: wing_root` |
| many wings, stacked wings | `surfaces_allowed: [3, 4]` |

## Worked examples (used by the U8 tests)

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

Here the stowage box (1.0 m) is shorter than the span limit (2.5 m), so the box wins: the effective span limit becomes 1.0 m. Claude says so before searching, so you can change the box or accept it.

**2.** *"I want the drone to dash at 60 m/s, angular like a stealth jet, no vertical tail."*
→ `dash_speed_ms: 60`, `body_style: faceted`, `edge_alignment: strict`, `airfoil_facets: [3, 5]`, `fin_policy: none`, `fam_sweep_deg: [30, 55]`, `alienness` asked.

**3.** *"I want the drone to carry 2 kg 50 km and be the strangest thing that still flies."*
→ `payload_mass_kg: 2`, `range_km: 50`, `alienness: 1.0`, nothing else. The search keeps full freedom inside the checks.
