# SPEC — aero-forge v4

Constraints in → one mission-specific, novel, checked fixed-wing UAV concept out (STEP + GLB + panel cut list). Different every run.

v4 replaces v3 (kept as `SPEC_v3.md`) at upgrade U0 (Fix/plan.md; user approval "upgrade 1.0", CHANGELOG `## Upgrade 1.0`). The Deviations logged in CHANGELOG against v3 (steps 0–10) remain in force; later upgrade steps log Deviations only from v4. Sections marked **(Un)** are specified here and built by upgrade step Un; until then the 0.3 behaviour stands.

## 1 Scope
- Small electric fixed-wing UAVs: span 0.6–4 m, mass 0.3–25 kg.
- Principle: **size by physics, shape by search, geometry by code.** Claude never writes geometry. It may narrow search ranges from the mission text, and writes an exact value only when the user states one.
- Function is a pass/fail gate. The search maximises looks and novelty among designs that pass.
- Fidelity: concept level (buildup/VLM aero, beam-theory structure). Every report states this. Not flight-validated.
- Out of scope: VTOL, OpenVSP, CFD, FEA, flutter, autopilot, manufacturing drawings beyond the panel cut list, radar-signature claims, a `modify` command.

## 2 Stack
- Python 3.11, venv at `~/.aero-forge/venv`, engine installed into it from `plugin/engine` (`pip install`; `-e` for dev).
- aerosandbox (aero), cadquery (all CAD, one kernel), ribs (quality-diversity search), trimesh (STL → GLB), numpy, pydantic v2, pyyaml, pillow; dev: pytest, ruff.
- Cross-platform: setup and hook are Python scripts, no shell scripts.

## 3 Layout
```
CLAUDE.md RULES.md SPEC.md SPEC_v3.md PLAN.md CHANGELOG.md   Fix/plan.md (upgrade 1.0, U0–U10)
.claude-plugin/marketplace.json     # aero-forge-market → ./plugin
plugin/                             # self-contained: nothing it runs lives outside it
  .claude-plugin/plugin.json        # aero-forge (0.3.0; 1.0.0 at U9)
  skills/forge/SKILL.md  skills/forge/controls-guide.md (U8)
  agents/visual-critic.md
  hooks/hooks.json  hooks/validate_brief.py
  scripts/setup_env.py
  engine/pyproject.toml
plugin/engine/src/aeroforge/
  models.py      # Brief, Requirements, Design, CheckResult (pydantic, frozen)
  constants.py   # single loader for data/*.yaml
  genetable.py   # gene table, unpack, family_of      genome.py   # decode(genome, req[, bounds]) -> Design
  layout.py shapes.py planform.py assemble.py shift.py   # surfaces, bodies, box packing, anchoring
  facets.py      # (U1) faceted bodies: section polygons, polyhedron, panel plan, containment
  controls.py    # (U5) resolve_controls(brief) -> bounds, knobs, filters, extra checks
  derive.py      # Brief -> Requirements
  geom.py        # Design -> named CadQuery solids
  mesh.py        # solids -> trimesh Scene -> GLB / STL
  render.py      # z-buffer rasteriser -> PNG
  gallery.py     # contact sheet + marks HTML
  style.py       # features, score, tune
  mass.py        # component masses, CG range
  aero.py        # tier-1 / tier-2 wrappers around aerosandbox
  checks.py      # flight-point checks -> CheckResult
  descriptor.py  # desc(Design) -> R^D, novelty         areas.py   # top / frontal / body-top-share
  search.py      # pyribs loop, finalist selection, funnel
  internals.py hardpoints.py   # spars, mounts, trays, bays
  report.py  viewer.py  cli.py
  data/{genes,layouts,checks,style,constants,reference}.yaml  data/controls.yaml (U5)  data/airfoils/
tests/  tests/fixtures/brief_sample.yaml  tests/fixtures/baseline_0.3/
```

## 4 Data model
Frame: x aft from nose-most point, y right, z up. SI units. Angles in degrees. Field names carry units (`span_m`, `sweep_deg`).

**Brief** (user constraints, YAML)
| field | required | default |
|---|---|---|
| payload_mass_kg, payload_box_m [l,w,h] | yes | |
| span_max_m, cruise_speed_ms | yes | |
| endurance_min or range_km | one of | |
| launch: hand\|catapult\|runway | yes | |
| recovery: belly\|skid\|wheels\|net | | belly |
| stall_max_ms | | from launch (checks.yaml) |
| dash_speed_ms | | 1.3 × cruise |
| n_limit, gust_ms, altitude_m | | 4, 5, 0 |
| mass_max_kg | | none |
| efficiency_floor | | 0.8 |
| seed | | fresh 128-bit draw, recorded |
| randomness r | | 0.5 |
| alienness a (U6) | | 0 |
| controls (U5) | | none (all auto) |

**Controls (U5).** `controls` maps a control name to a value, a `[lo, hi]` range, or a list of choices. The table is `data/controls.yaml` (100 ranked rows: name, rank, group, kind, target, bounds, unit, default, meaning, cues, effect). Validation rejects unknown names, out-of-range values, lo > hi, and a control that duplicates a brief field. Five kinds:
| kind | applied by | effect |
|---|---|---|
| B brief requirement | `derive` | sizes the aircraft; `derive` stops with a plain message if impossible |
| G gene control | `decode` bounds | narrows or pins one gene (value = lo′ + g·(hi′ − lo′)); a list keeps only those choices |
| K search knob | `search` / `select` | behaviour of the search, never physics |
| F filter | `select` | rule a design must meet to be a finalist; counted in the funnel |
| C check | `checks` | extra pass/fail test; when set it joins the passable bar for that run |

Precedence, highest first: (1) a value the user typed, (2) Claude's reading of the mission, (3) `alienness`, (4) `randomness`, (5) the full range. The report lists the source of every value. References and the baseline always decode with the full ranges, so novelty and `d_min` stay comparable between runs.

**Requirements** (derived): MTOW, minimum lifting area (the stall area), battery Wh, reference length L, flight points P1–P6, baseline cruise L/D. `derive` stops when electrical power exceeds 10 × installed at the dash point.

**Genome**: float vector in [0,1]^D, D = 86 at 0.3 and ≈ 102 at 1.0 (86 + 16 new). Defined only by rows in `genes.yaml` (name, lo, hi, kind float|choice, meaning, `ref_default`). Controls narrow gene ranges; they never change the genome length. A gene added after 0.3 has a `ref_default` at which it is an exact no-op, so the 44 references decode as before.
| group | content |
|---|---|
| global | style family: sweep, taper, dihedral, tip style, airfoil family, t/c, body fineness and section exponent, body nose / tail / width ratio / upsweep, layout, box_layout, wing area scale |
| surface ×4 | active, root x/z, host (centre\|pod), area share, AR, sweep sign + delta, cant (−90…90°, fins emerge), break position, outer cant delta, outer sweep delta, tip-join target, control fraction, root chord scale, twist |
| bodies | centre body on/off + length; pod pair on/off, y, x, length |
| propulsion | count pattern, host, x offset, z offset, tractor\|pusher |
| faceted body (U1) | body_style, body_section_sides, body_stations, chine_z_frac, belly_flat_frac, roof_pitch_deg, side_tumblehome_deg, lower_flare_deg, nose_style, tail_style, tail_flatten, tail_end_size_frac, max_section_x_frac |
| faceted wing (U3, U3b) | airfoil_facets, edge_alignment, facet_coarseness |

Surface sweep/taper/tip/airfoil = global family value ± a bounded delta, so one design language holds by construction.

**Design**: decoded, physical, complete: surfaces (panels, sections, control surfaces), bodies, propulsors, component masses, hardpoints, its own `lifting_area_m2`. `Body.facets: Polyhedron | None` (U1; vertices and flat faces; None = smooth super-ellipse). Airfoil sections may be polygons (U3). Everything downstream consumes Design only.

## 5 Decoder invariants
Hold for every genome; property-tested on 1,000 seeded random genomes.
1. Symmetric about y = 0.
2. Every part attached: surface roots snap onto a body or host surface, pods onto surfaces. No floating parts.
3. Total lifting area = the design's own S (`lifting_area_m2`, ≥ the stall area). Span ≤ span_max (AR clipped to fit).
4. Surface 0 always active. Surfaces under 4 % of S are dropped.
5. Payload, battery and avionics boxes fit inside a body or thick centre section (body sized to fit).
6. Spar = largest tube in the constants.yaml table that fits local thickness minus 2 skins along the whole spar line. None fits → surface thickened, else dropped.
7. Prop discs clear all surfaces, bodies and the ground line by the margin in constants.yaml.
7a. (U2) A propeller's shaft starts at its host's end, except on a flat-panel host whose end is too slim for a motor plate of half the nominal radius: there the motor attach point (firewall, drive mass, start of the shaft) sits inside the host where the plate fits (at most `mount_slide_body_frac` of its length from the end), and the decoder says so in `Design.notes`. A mount that still has no plate of half the radius is a containment failure.
8. Tip joins close exactly (shared tip point and chord).
9. Style family respected, including: **body fineness (length / max diameter) is measured on the diameter that inline box packing would need (decision B1, U0)**. It is exact for inline packing, the wing_root fallback, plain bodies and pods, and at most the gene for side_by_side / stacked packing, whose section never drops below the box fit. Pods use the centre body's `body_style`. (U2 audit) A flat-panel body or pod that hosts a motor may be fatter than the gene (fineness below it, never above): its end is sized for a firewall plate (tail size, flat cap, blunt nose) and, if that is not enough, its section grows (a body by at most `prop_body_grow_max`, a pod down to fineness 1 / (2 `pod_radius_max_length_frac`)); the end shape of such a body therefore differs from the genes' (`Design.family.shape`: nose, tail, tail_end_size_frac only).
10. Deterministic: same genome + Requirements (+ bounds) → identical Design.
11. (U1) Every face of a faceted part is planar (deviation < 1e-9 × body length).
12. (U1) Faceted solids are closed (every edge shared by two faces; V − E + F = 2).
13. (U5) Every gene lies inside its resolved control bounds.

CAD build succeeds for ≥ 98 % of random genomes; a failure raises a typed error, never a bad solid.

## 6 Geometry and render
- `geom`: spline lofts through airfoil sections; super-ellipse body sections with tangent-continuous nose and tail; tip treatments. Fillets at junctions fail in the kernel, so parts stay separate overlapping solids (logged).
- (U2) Faceted bodies: planar-face solids, built from polygon wire → face → shell → solid and checked with the OCC shape checker. A face that would twist (a shape break between stations) is split into two flat triangles. Panel count N = K(S−1) + K·B + n_nose + n_tail. (U3) Faceted wings (`airfoil_facets` 3-8 flat segments per side): each section is a polygon whose vertices lie on the family NACA outline at x/c = (i / k)^p; the lofts between sections are ruled, every polygon edge sweeps a flat quad, or two flat triangles (shorter diagonal) where the quad would twist (cant change, washout), so every face is planar (invariant 11) and the solid closed (12). One polygon outline (`airfoil.wing_outline`) is read by CAD, spar and box room, containment and the cut list.
- (U3) Edge alignment (`edge_alignment`): the top-view outline edges (leading, trailing and tip-chord edges of each panel of a mirrored surface, on the planform; the outline hull of a flat-panel body) have a sweep angle from the y axis, measured on the finished design and listed in report.md for every design. **strict** snaps wing and tail edges to the set {0, 90, +-a, +-(90 - a)} (a = main wing quarter-chord sweep seen from above) by moving section leading edges and chords along x only (span, aspect ratio and projected area unchanged); tips become square, tip joins are dropped, a near-vertical panel loses its sweep; a flat-panel centre body gets a linear nose / tail law (point or blunt nose, flat tail) and its widest station and tail size are re-solved after each sizing pass onto the set plus the steep angles 90 - a / 2^k; pods and anything that cannot be snapped are named in the decoder notes. **soft** adds `edge_soft_weight` x the length-weighted alignment score to the style score; **off** changes nothing.
- (U3b) Facet coarseness (`facet_coarseness`: off, low, mid, high = c of 0, 0.35, 0.7, 1; **off** is the U3 geometry exactly; it acts only on faceted parts, so smooth designs and the 0.3 baseline are untouched). The F-117 look: few large flat panels, sharp creases, shared face directions. At coarseness c:
  - smallest crease `c x coarse_min_crease_deg` (15 deg at c = 1): a wing section keeps the largest flat-segment count at or under `coarse_wing_facets` (8 at c = 0, 4 at c = 1) whose interior corners all turn by at least that, and a faceted body section drops corners the same way; the radius law of a flat-panel body becomes linear (one straight nose edge and one straight tail edge, so each is one sheet per corner). A merge moves vertices by at most `coarse_move_frac` (3 %) of the part length (body: its length; wing: the chord); a merge over the cap is skipped and noted. Sheets across a crease of at most `coarse_cluster_deg` (3 deg: the two halves of a twisted strip) are counted as one panel.
  - budgets, measured and reported for every design (and a decoder note when exceeded): body 12-30 panels (`coarse_body_panels_max`, enforced by dropping section corners while moves stay in the cap), wing 4-12 panels per side (`coarse_wing_side_panels_max`), at most `coarse_directions_max` (8) face directions per side (3 deg clusters of face normals). The directions budget is measured, not enforced (no snapping of face normals).
  - wings: the washout becomes whole steps of `c x coarse_twist_step_deg` between sections (at least one step, one fewer than the strips at most, the tip twist kept), so strips between sections of one step stay flat quads and only the step strip is twisted (two flat triangles). The aero cost is the change of washout, which AeroBuildup sees, plus the corner-drag increment.
  - aero: the corner-drag constant stays; report.md lists the drag-area increment and the clmax change against the smooth parent section for every design (clmax change 0 as modelled: the parent's clmax is kept, optimistic, Known issue); no design is dropped for being coarse.
  - every override is a decoder note with its size; the finished part is measured (sheets, creases, face directions) in report.md section "Facet coarseness (measured)" and any shortfall is a note.
- (U2) Faceted sizing differs from smooth sizing: the boxes must fit the flat-panel envelope (closed-form box fit, then the B1 fineness lengthening of §5), so for the same genome a faceted body is longer than the smooth one: over 60 stub genomes 0.95 to 2.72 times its length (median 1.28), 0.63 to 3.46 times its wetted area and 0.62 to 10.2 times its frontal area (part of the frontal jump is the shadow of a rising tail). A search over faceted bodies therefore lands in a different mass and drag range from a smooth one; compare designs by checks, not by genome.
- One named solid per part, stable names: `surface0_L`, `body0`, `pod_R`, `prop1`.
- `mesh`: tessellate per part → trimesh Scene with node names → `assembly.glb`; STL per part; volume-checked against the exact volume.
- `render`: numpy z-buffer rasteriser, no GL. Views iso/top/front/side at 1024 px, Lambert shading + depth-edge outline. Gallery tile: iso + top at 320 px.
- `gallery`: static HTML, N tiles, click cycles good/bad/none, Export downloads `marks.json`. Opens from file://.

## 7 Style
- `features(Design)` → ~12 numbers at 0.3: body fineness, AR per surface, area balance fore/aft, sweep and taper spread, junction angles, part count, tip consistency, planform compactness, side-view height ratio. (U7) adds span ratio, body_top_share, body width ratio, layout, chine sharpness (mean bevel at the chine), edge alignment score, normalised panel count; the positive push on body fineness is dropped.
- `score = sigmoid(w·f + b)`, weights in style.yaml; `tune-style marks.json` fits them (numpy logistic regression, k-fold accuracy). `s_min` = 0.6.
- (U7) Critic v2 scores coherence, proportion, strangeness (weighted by alienness), defects and fit to the mission.

## 8 Physics
- (U3) A faceted wing enters AeroBuildup as its smooth parent NACA section plus an assumed profile-drag increment per polygon corner (NeuralFoil smooths polygons or loses confidence on them); clmax and stall are the parent's (optimistic). Stated in the report's fidelity paragraph.
**Flight points** (each at CG fore and aft; CG range from payload on/off + battery position tolerance)
| id | condition |
|---|---|
| P1 | launch / stall |
| P2 | loiter (min power) |
| P3 | cruise |
| P4 | dash |
| P5 | pull-up at n_limit |
| P6 | gust at cruise |

**Checks** (bands in checks.yaml; values below are defaults)
| id | at | pass |
|---|---|---|
| stall | P1 | Vs ≤ stall_max; launch speed ≥ 1.2 Vs |
| trim | all | Cm = 0 with control deflection ≤ 15°; CL ≤ 0.85 CLmax |
| static | all, aft CG | static margin 5–20 % MAC |
| lateral | P3 | Cnβ > 0, Clβ < 0 |
| stall order (tier 2 only) | P1 | foremost lifting surface reaches clmax first; peak local cl/clmax inboard of 70 % semi-span |
| mass | – | Σ component masses ≤ MTOW; MTOW ≤ mass_max_kg |
| power | all | P_required ≤ 0.8 P_available; T/W ≥ launch minimum |
| energy | P2, P3 | endurance or range ≥ brief |
| structure | P5, P6 | spar bending stress × 1.5 ≤ allowable; tip deflection ≤ 10 % semi-span |
| efficiency | P3 | L/D ≥ efficiency_floor × baseline L/D |
| fit | – | invariants 5–7 |
| stowage (U6, C) | – | sorted bounding-box sizes fit the sorted stowage box sizes |
| payload_view (U6, C) | – | the payload box touches the chosen host face |
| panel_count_exact (U6, C) | – | the faceted body has exactly the requested panel count |

- A check the user sets joins the passable bar for that run; the others are reported only.
- Each check returns a normalised margin m (m ≥ 0 passes). Feasible = all m ≥ 0. Violation = Σ max(0, −m). The passable bar and `accept_other_failures_max(r)` (2 at r = 0, every check at r = 1) decide what verify accepts; passable is always required.
- Tier 1 (search): `asb.AeroBuildup`, all points. Tier 2 (finalists): `asb.VortexLatticeMethod` with stability derivatives and trim.
- Profile drag always comes from the buildup; tier 2 replaces lift, moments and induced drag only. L/D and power comparisons use the same tier for candidate and baseline.
- (U2) A faceted body enters AeroBuildup as an equivalent section (same width, height and area); its friction area is the exact area of the flat faces (the equivalent sections alone undercount it by 2 to 20 %); the extra drag of sharp chines is not modelled and the report says so.
- CLmax: from section clmax at local Reynolds number, 3-D factor in checks.yaml.
- `mass`: structure (areal density × wetted area; exact for faceted bodies), spars, motor, ESC, prop, battery, avionics, servos, payload.
- `derive` fixes MTOW, S and battery once per run by fixed-point closure on the baseline (≤ 10 iterations, else typed infeasible). Candidates do not resize; they pass or fail the `mass` check.
- Baseline: the `conventional` entry of reference.yaml, decoded and checked by the same code; its cruise L/D is the reference.
- Every assumed constant lives in constants.yaml with unit and source note, and is listed in CHANGELOG "Assumed constants".

## 9 Novelty and search
- `desc(Design)` → R^D in [0,1], from geometry only. 0.3: R^16 (lifting-area histograms along x (5), y (4), z (3); vertical-area fraction; surface count; joined-tip count; body count). (U4) R^21: adds body_top_share, frontal area ÷ S, top area ÷ S, centre body length ÷ √S, facet density. Invariant to slot order and encoding.
- `novelty(D)` = min Euclidean distance from desc(D) to reference ∪ history.
- Reference set: `data/reference.yaml`, 44 known configurations; each entry = name + sparse gene overrides (`base: <name>` allowed), decoded by `genome.decode` with the full ranges.
- History: `~/.aero-forge/history.jsonl` (id, seed, brief hash, desc). Appended on delivery only. A new file starts at U4 (descriptor size changes); the 0.3 file is archived.
- `d_min` = median nearest-neighbour distance inside the reference set.
- Search (pyribs CVT archive): solution = genome; objective = style score if passable, else −violation; measures = desc. Default budget 5,000 tier-1 evaluations, spawn worker processes.
- Randomness r sets the novelty band [bottom, top] × d_min (`knob_*` rows with r0/r1). (U6) `alienness` moves several knobs together by linear interpolation (`a0`/`a1`): a floor on r, `exclude_layouts`, `min_conventional_distance` (× d_min), `bomb_cap`, `critic_strangeness_weight`, `novelty_style_weight`. Anything the user sets wins (precedence, §4).
- Selection (`select`): finalists = passable ∧ style ≥ s_min ∧ novelty inside the band ∧ filters; ranked by w·z(novelty) + (1 − w)·z(style); spread ≥ d_min apart. (U4/U6) Filters: `exclude_layouts`, `min_conventional_distance`, `surfaces_allowed`, `tip_joins_allowed`, `bomb_cap` (body_top_share ≤ cap, default 0.45).
- (U6) Funnel: `search` counts the designs each check and filter removed → `finalists.json`, `report.md`. Fewer than 10 qualifiers: double the budget once; still short: deliver the best with FAIL rows shown and name the two controls that removed the most. Never silent.
- Verify keeps score order and accepts a finalist with at most `accept_other_failures_max(r)` failures besides the passable bar; the first 3 accepted go to build + render; the critic picks one (no critic: highest score).
- (U6) Conflict check in `validate`, before any search (exit 2, one plain line each): stowage smaller than the payload, `surfaces_allowed` max 1 with a multi-surface layout, `fin_policy: none` with `layout: conventional`, unreachable exact panel count, `body_on: off` with a box_layout other than wing_root, and similar.
- Same seed + same history + same brief (+ controls) → same design.

## 10 Internals (final design only)
- Per surface: spar tube at 30 % chord following the panels; joiner at root and each break.
- Per control surface: hinge-line marker + servo tray.
- Per propulsor: mount plate with bolt circle, spinner, prop.
- Bays: battery, avionics, payload frames; one hatch outline per bay (U2: along face edges of a faceted body).
- Recovery provision per brief.
- Positions come from `hardpoints(design, recovery)` only, never measured from meshes. Internals stay separate solids.
- Completeness = every item above exists. Containment = each part's section fits the local outer mould (analytic; exact flat-face test for faceted bodies).

## 11 Outputs — `designs/<id>/`
`brief.yaml requirements.json design.yaml genome.json finalists.json check.json critique.json views/{iso,top,front,side}.png parts/*.step assembly.step assembly.glb report.md viewer.html`, and from U2 `panels/panels.csv` (flat-panel bodies and, U3, faceted wings; one row per physical sheet, neighbouring faces in one plane merged, so at most N rows for N panels: id, part, area, edge lengths, bevel angle to each neighbour) and `panels/panels.svg` (every sheet laid flat, numbered, dimensioned, paginated to the stock sheet size in `constants.yaml`; a sheet larger than the stock gets a page of its own, labelled oversize).
- `report.md`: brief vs achieved per check (PASS/FAIL + margin); (U3) a measured table of the top-view outline edges with their angles, gaps to the angle set and the angles shared by different parts; novelty (nearest reference by name + distance, nearest past design); seed and r (and alienness); shadow areas; assumed constants; fidelity statement; (U5) Controls section (value and source of each); (U6) Funnel section.
- `viewer.html`: self-contained; three.js + GLTFLoader + OrbitControls via importmap from cdn.jsdelivr.net; GLB as base64; click a part → name + Design fields; opens from file://.

## 12 CLI
`python -m aeroforge <cmd>`. One invocation per action. stdout = JSON summary ≤ 30 lines; artifacts on disk. Exit 0 ok, 1 check failed, 2 bad input, 3 environment.
| cmd | in → out |
|---|---|
| validate | brief.yaml → exit code (U5: controls; U6: conflicts) |
| derive | brief.yaml → requirements.json |
| search | design dir, --seed, --budget, --randomness → finalists.json |
| verify | design dir → check.json per finalist, verified.json |
| build | design dir [--full] → parts, assembly (U2: panels/) |
| render | design dir → views |
| report | design dir → report.md, viewer.html |
| check | design.yaml → check.json (dev) |
| gallery | --n, --seed [--top-by-style] → gallery.html |
| tune-style | marks.json → style.yaml |
| demo | brief.yaml, --runs N → gallery of winners |
| controls (U5) | [--group G] [--brief B] → the control table, or a brief's resolved controls, as JSON |

## 13 Plugin
- Skill `/forge`, modes: `setup` (run setup_env.py, verify imports), `new`, `status` (list designs), and from U8 `controls` (the 100 controls with plain explanations, from `controls-guide.md`).
- `new`: (1) write brief.yaml and up to 15 controls from the user's mission text, each tagged with the words it came from; at most one AskUserQuestion round, to confirm (U8) or for missing required fields; (2) validate; (3) derive; (4) search; (5) verify; (6) build + render top 3; (7) visual-critic picks; (8) build --full, report; (9) reply with the viewer path and a 5-line summary.
- The lead reads JSON summaries only, never STEP, GLB or meshes. Each engine call is one command: no cd, &&, or pipes. Search and verify run in the background with a long timeout.
- `visual-critic` agent frontmatter: name, description, `tools: Read, Write`, `model: sonnet`, `effort: medium`, `maxTurns: 6`. Reads 3 finalist tiles + finalists.json. Writes critique.json: pick, scores 1–5, one sentence each. Reply ≤ 40 words. (U7) Critic v2 adds fit to mission and the strangeness weight.
- Hook: PostToolUse `Write|Edit` → validate_brief.py: exit 0 unless the path is `designs/**/brief.yaml`; on invalid brief print ≤ 5 lines to stderr and exit 2; under 2 s. (U5) It uses the same Brief model as the engine, controls included.
- (U9) Version 1.0.0; the engine updates itself when the plugin updates.

## 14 Acceptance
Run from the installed 1.0.0 plugin, outside the repo, after `/forge setup` (U10, Fix/plan.md): 40 m/s and 80 m/s briefs × alienness 0, 0.5, 1 × 3 seeds, faceted (18 runs), and the three worked examples through `/forge new`. Auto gate, numbers in the CHANGELOG:
1. The fast suite passes from a fresh clone after `setup_env.py`; `claude plugin validate ./plugin` passes.
2. Finalists with one body, no joins and ≤ 2 surfaces: ≤ 30 % at alienness 1 (0.3 baseline: 6/10 and 10/10).
3. Median `body_top_share` of delivered designs ≤ 0.35 at alienness ≥ 0.5; ≥ 4 different layouts among the 12 delivered designs at alienness ≥ 0.5.
4. Exact panel count when `panel_count_exact` is set, ±10 % otherwise; the rows of `panels.csv` give the wetted area of every faceted body on every delivery as the sum of `count` × `area_m2` (`count` is 2 for a mirrored pod, 1 otherwise; the plain sum of the area column leaves out the second instance).
5. At alienness 1 no delivered design has `conventional` as its nearest reference.
6. Every delivered design passes the passable bar at tier 2, or the run names the blocking controls.
7. Passable rate ≥ half of the 0.3 measurement (321 of 5,004 on the 80 m/s stand-in).
8. Every run delivers all §11 files.
User gate: flat-panel bodies, a different design every run, each answering its mission, nothing that looks like a known aircraft at alienness ≥ 0.8. The v3 acceptance list is in `SPEC_v3.md`.
