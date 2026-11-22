# SPEC — aero-forge v3

Constraints in → one novel, good-looking, checked fixed-wing UAV concept out (STEP + GLB). Different every run.

## 1 Scope
- Small electric fixed-wing UAVs: span 0.6–4 m, mass 0.3–25 kg.
- Principle: **size by physics, shape by search, geometry by code.** Claude never writes geometry or genome values.
- Function is a pass/fail gate. The search maximises looks and novelty among designs that pass.
- Fidelity: concept level (buildup/VLM aero, beam-theory structure). Every report states this. Not flight-validated.
- Out of scope: VTOL, OpenVSP, CFD, FEA, flutter, autopilot, manufacturing drawings, a `modify` command.

## 2 Stack
- Python 3.11, venv at `~/.aero-forge/venv`, engine installed into it from `plugin/engine` (`pip install`; `-e` for dev).
- aerosandbox (aero), cadquery (all CAD, one kernel), ribs (quality-diversity search), trimesh (STL → GLB), numpy, pydantic v2, pyyaml, pillow; dev: pytest, ruff.
- Cross-platform: setup and hook are Python scripts, no shell scripts.

## 3 Layout
```
CLAUDE.md RULES.md SPEC.md PLAN.md CHANGELOG.md
.claude-plugin/marketplace.json     # aero-forge-market → ./plugin
plugin/                             # self-contained: nothing it runs lives outside it
  .claude-plugin/plugin.json        # aero-forge 0.3.0
  skills/forge/SKILL.md
  agents/visual-critic.md
  hooks/hooks.json  hooks/validate_brief.py
  scripts/setup_env.py
  engine/pyproject.toml
plugin/engine/src/aeroforge/
  models.py      # Brief, Requirements, Design, CheckResult (pydantic, frozen)
  constants.py   # single loader for data/*.yaml
  genome.py      # decode(genome, req) -> Design
  derive.py      # Brief -> Requirements
  geom.py        # Design -> named CadQuery solids
  mesh.py        # solids -> trimesh Scene -> GLB / STL
  render.py      # z-buffer rasteriser -> PNG
  gallery.py     # contact sheet + marks HTML
  style.py       # features, score, tune
  mass.py        # component masses, CG range
  aero.py        # tier-1 / tier-2 wrappers around aerosandbox
  checks.py      # flight-point checks -> CheckResult
  descriptor.py  # desc(Design) -> R^16, novelty
  search.py      # pyribs loop, finalist selection
  internals.py   # spars, mounts, trays, bays
  report.py  viewer.py  cli.py
  data/{genes,checks,style,constants,reference}.yaml  data/airfoils/
tests/  tests/fixtures/brief_sample.yaml
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
| seed | | OS entropy, recorded |

**Requirements** (derived): MTOW, lifting area S (from stall), battery Wh, reference length L, flight points P1–P6, baseline cruise L/D.

**Genome**: float vector in [0,1]^D, D ≈ 65. Defined only by rows in `genes.yaml` (name, lo, hi, kind float|choice, meaning).
| group | genes (each) | content |
|---|---|---|
| global ×1 | ~8 | style family: sweep, taper, dihedral, tip style, airfoil family, t/c, body fineness, body section exponent |
| surface ×4 | ~12 | active, root x/z, host (centre\|pod), area share, AR, sweep sign + delta, cant (−90…90°, fins emerge), break position, outer cant delta, outer sweep delta, tip-join target, control fraction |
| bodies | ~6 | centre body on/off + length; pod pair on/off, y, x, length |
| propulsion | ~5 | count pattern, host, x offset, z offset, tractor\|pusher |

Surface sweep/taper/tip/airfoil = global family value ± a bounded delta, so one design language holds by construction.

**Design**: decoded, physical, complete: surfaces (panels, sections, control surfaces), bodies, propulsors, component masses, hardpoints. Everything downstream consumes Design only.

## 5 Decoder invariants
Hold for every genome; property-tested on 1,000 seeded random genomes.
1. Symmetric about y = 0.
2. Every part attached: surface roots snap onto a body or host surface, pods onto surfaces. No floating parts.
3. Total lifting area = S. Span ≤ span_max (AR clipped to fit).
4. Surface 0 always active. Surfaces under 4 % of S are dropped.
5. Payload, battery and avionics boxes fit inside a body or thick centre section (body sized to fit).
6. Spar = largest tube in the constants.yaml table that fits local thickness minus 2 skins along the whole spar line. None fits → surface thickened, else dropped.
7. Prop discs clear all surfaces, bodies and the ground line by the margin in constants.yaml.
8. Tip joins close exactly (shared tip point and chord).
9. Style family respected.
10. Deterministic: same genome + Requirements → identical Design.

CAD build succeeds for ≥ 98 % of random genomes; a failure raises a typed error, never a bad solid.

## 6 Geometry and render
- `geom`: spline lofts through airfoil sections; super-ellipse body sections with tangent-continuous nose and tail; tip treatments; blend/fillet at junctions where the kernel allows, else plain union (logged). Start from `asb.Airplane.generate_cadquery_geometry` if step 0 finds it adequate; own loft where not.
- One named solid per part, stable names: `surface0_L`, `body0`, `pod_R`, `prop1`.
- `mesh`: tessellate per part → trimesh Scene with node names → `assembly.glb`; STL per part.
- `render`: numpy z-buffer rasteriser, no GL. Views iso/top/front/side at 1024 px, Lambert shading + depth-edge outline. Gallery tile: iso + top at 320 px.
- `gallery`: static HTML, N tiles, click cycles good/bad/none, Export downloads `marks.json`. Opens from file://.

## 7 Style
- `features(Design)` → ~12 numbers: body fineness, AR per surface, area balance fore/aft, sweep and taper spread, junction angles, part count, tip consistency, planform compactness, side-view height ratio.
- `score = sigmoid(w·f + b)`, weights in style.yaml. Initial weights by hand.
- `tune-style marks.json`: logistic regression in numpy, writes style.yaml, prints k-fold accuracy.
- `s_min` = 0.6.

## 8 Physics
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

- Each check returns a normalised margin m (m ≥ 0 passes). Feasible = all m ≥ 0. Violation = Σ max(0, −m).
- Tier 1 (search): `asb.AeroBuildup`, all points, target ≤ 0.1 s per candidate.
- Tier 2 (finalists): `asb.VortexLatticeMethod` with stability derivatives and trim.
- Profile drag always comes from the buildup; tier 2 replaces lift, moments and induced drag only. L/D and power comparisons use the same tier for candidate and baseline.
- CLmax: from section clmax at local Reynolds number (source per step-0 Decision), 3-D factor in checks.yaml.
- `mass`: structure (areal density × wetted area), spars, motor, ESC, prop, battery (energy density), avionics, servos, payload.
- `derive` fixes MTOW, S and battery once per run by fixed-point closure on the baseline (≤ 10 iterations, else typed infeasible). Candidates do not resize; they pass or fail the `mass` check.
- Baseline: the `conventional` entry of reference.yaml, decoded and checked by the same code; its cruise L/D is the reference.
- Every assumed constant lives in constants.yaml with unit and source note, and is listed in CHANGELOG "Assumed constants".

## 9 Novelty and search
- `desc(Design)` → R^16 in [0,1], from geometry only: lifting-area histograms along x (5), y (4), z (3); vertical-area fraction; surface count; joined-tip count; body count. Invariant to slot order and encoding.
- `novelty(D)` = min Euclidean distance from desc(D) to reference ∪ history.
- Reference set: `data/reference.yaml`, ≥ 40 known configurations (conventional, canard, flying wing, BWB, tandem, biplane, box, joined, twin-boom, three-surface, V-tail pusher, …). Each entry = name + sparse gene overrides, decoded by `genome.decode`. A configuration the genome cannot express goes under Known issues.
- History: `~/.aero-forge/history.jsonl` (id, seed, brief hash, desc). Appended on delivery only.
- `d_min` default = median nearest-neighbour distance inside the reference set.
- Search (pyribs): solution = genome; objective = style score if feasible, else −violation; measures = desc (archive type per step-0 Decision). Default budget 5,000 tier-1 evaluations.
- Selection: elites with feasible ∧ style ≥ s_min ∧ novelty ≥ d_min → top 10 by novelty → tier 2 → first 3 passing → build + render → critic picks one (no critic: highest novelty).
- Nothing passes: double the budget once, then deliver the best with FAIL rows shown. Never silently.
- Same seed + same history + same brief → same design.

## 10 Internals (final design only)
- Per surface: spar tube at 30 % chord following the panels; joiner at root and each break.
- Per control surface: hinge-line marker + servo tray.
- Per propulsor: mount plate with bolt circle, spinner, prop.
- Bays: battery, avionics, payload frames; one hatch outline per bay.
- Recovery provision per brief.
- Positions come from `Design.hardpoints` only, never measured from meshes. Internals stay separate solids.
- Completeness = every item above exists. Containment = each part's section fits the local outer-mould thickness (analytic).

## 11 Outputs — `designs/<id>/`
`brief.yaml requirements.json design.yaml genome.json finalists.json check.json critique.json views/{iso,top,front,side}.png parts/*.step assembly.step assembly.glb report.md viewer.html`
- `report.md`: brief vs achieved per check (PASS/FAIL + margin); novelty (nearest reference by name + distance, nearest past design); seed; assumed constants; fidelity statement.
- `viewer.html`: self-contained; three.js + GLTFLoader + OrbitControls via importmap from cdn.jsdelivr.net; GLB as base64; click a part → name + Design fields; opens from file://.

## 12 CLI
`python -m aeroforge <cmd>`. One invocation per action. stdout = JSON summary ≤ 30 lines; artifacts on disk. Exit 0 ok, 1 check failed, 2 bad input, 3 environment.
| cmd | in → out |
|---|---|
| validate | brief.yaml → exit code |
| derive | brief.yaml → requirements.json |
| search | design dir, --seed, --budget → finalists.json |
| verify | design dir → check.json per finalist |
| build | design dir [--full] → parts, assembly |
| render | design dir → views |
| report | design dir → report.md, viewer.html |
| check | design.yaml → check.json (dev) |
| gallery | --n, --seed [--top-by-style] → gallery.html |
| tune-style | marks.json → style.yaml |
| demo | brief.yaml, --runs N → gallery of winners |

## 13 Plugin
- Skill `/forge` (≤ 80 lines), modes: `setup` (run setup_env.py, verify imports), `new`, `status` (list designs).
- `new`: (1) write brief.yaml from the user's text; at most one AskUserQuestion call (≤ 4 questions), only for missing required fields; (2) derive; (3) search; (4) verify; (5) build + render top 3; (6) visual-critic picks; (7) build --full, report; (8) reply with the viewer path and a 5-line summary.
- The lead reads JSON summaries only, never STEP, GLB or meshes. Each engine call is one command: no cd, &&, or pipes.
- `visual-critic` agent frontmatter: name, description, `tools: Read, Write`, `model: sonnet`, `effort: medium`, `maxTurns: 6`. Reads 3 finalist tiles + finalists.json. Writes critique.json: pick, scores 1–5 (coherence, proportion, strangeness, defects), one sentence each. Reply ≤ 40 words.
- Hook: PostToolUse `Write|Edit` → validate_brief.py: exit 0 unless the path is `designs/**/brief.yaml`; on invalid brief print ≤ 5 lines to stderr and exit 2; under 2 s.

## 14 Acceptance
1. `pytest -q` green from a fresh clone after `python plugin/scripts/setup_env.py`.
2. `claude plugin validate ./plugin` passes.
3. `/forge new` on the sample brief, run from the installed plugin in a folder outside the repo, delivers every §11 file; all checks PASS at tier 2.
4. Ten runs of one brief: ten designs, pairwise and reference distances ≥ d_min, and the user rates them good-looking and different.
5. Final message: file tree, test result, `/plugin marketplace add ./`, `/plugin install aero-forge@aero-forge-market`.
