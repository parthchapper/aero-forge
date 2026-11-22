# plan.md: make aero-forge give novel designs, not bombs

This plan fixes the causes in `whatswrong.md`. It adds a randomness scale, new shape parameters (genes), and frontal and top-down area. It continues `bonjour/PLAN.md` (steps 0–6 are done), so the steps here are numbered 7 to 13. Follow `bonjour/RULES.md`: one step per session, test, gate, then stop.

This file sits next to `whatswrong.md`, not in `bonjour/`. macOS ignores letter case in file names, so `bonjour/plan.md` would overwrite `bonjour/PLAN.md`.

**Approval.** RULES.md says that a change to a gate, an interface or a check band needs the user's approval. This request is that approval for the changes listed below. Log each one under Deviations in `CHANGELOG.md`, with "user, novelty plan" as the reason.

---

## Audit: what exists today

| You asked for | Exists? | Where | What is wrong today |
|---|---|---|---|
| Wing sweep angle | Yes | `fam_sweep_deg` 0–35°, per surface `sN_sweep_sign`, `sN_sweep_delta_deg`, `sN_outer_sweep_delta_deg` | The range stops at 35°. All surfaces share one sweep family. |
| Wing dihedral angle | Yes | `fam_dihedral_deg` −4 to 8°, `sN_cant_deg`, `sN_outer_cant_delta_deg` | The main wing's own cant is only ±6°, so no strong gull or anhedral main wing. |
| Canards | Yes, indirectly | Slots s1–s3 with `sN_root_x_frac` down to 0.05. `reference.yaml` has `canard`, `canard_delta`, `canard_pusher`, `pod_canard`. | A canard needs several genes to change together. The drag ranking then removes it, because a second surface adds drag. |
| Wing chord size | No | Chord follows from area, aspect ratio and taper | Area is fixed by `derive`, so chord can only change by trading against span. |
| Wing span | No | Follows from $b = \sqrt{AR \cdot S}$ | The 3 m `span_max_m` is only a limit. The winners use about 1 m. |
| Wing area | No | `derive` fixes $S$ at the stall minimum | Every design has the same small wing area. |
| Twist / washout | No | `incidence_deg` is set by `prepare` for trim and is the same along the span | (none) |
| Body shape | Partly | `fam_body_fineness`, `fam_body_section_exp`, `body_length_frac` | Nose (0.22) and tail (0.40) lengths are fixed constants. The boxes always sit in one row. This forces a body of about 1.1 m. |
| Frontal area | No | Not computed anywhere | (none) |
| Top-down area | No | Only the lifting area is used. The body's top area is never measured. | (none) |
| Randomness control | No | The seed comes from the brief, else 0. The search sigmas are fixed in `constants.yaml`. | The same brief always gives the same design. SPEC §1 says "Different every run". |
| Novelty that sees the body | No | `descriptor.py` measures only where lifting area sits, plus part counts | A design that is mostly body still scores as novel. |
| Ranking by novelty | No | `search.py` `select()` sorts by `total_violation` | That sort minimises drag at dash speed, which produces the bomb. |

---

## Step 7: Randomness scale and seed

Read: SPEC §9, §12. `search.py`, `search_cli.py`, `brief.py`, `models.py`, `planform.py`, `plugin/skills/forge/SKILL.md`.

Build:
- Guards first, before any knob below widens the novelty band. The 8–20 × $d_{\min}$ picks that created the band were joined layouts and tall outer panels (Known issues). Add both guards from Known issues: cap the outer-panel length when $|\text{cant}| \ge 60°$ (in `build_planform`), and offset the shared tip station of joined surfaces for the tier-2 VLM.
- A brief field `randomness` (0 to 1, default 0.5) on the `Brief` model in `models.py`. `Brief` uses `extra="forbid"`, so a brief with the field fails today. The `validate_brief.py` hook checks briefs through the same model, so it needs no separate change. Add a CLI flag `--randomness` that overrides it. The value $r$ sets every search knob below. Each knob has an `r0` value and an `r1` value in `constants.yaml`, and the code interpolates linearly: $k(r) = k_0 + r\,(k_1 - k_0)$.

| knob | $r = 0$ (safe) | $r = 1$ (wild) |
|---|---|---|
| novelty band top (× $d_{\min}$) | 2 | 10 |
| novelty band bottom (× $d_{\min}$) | 1 | 2 |
| explore emitter sigma | 0.1 | 0.4 |
| share of the first batch seeded from reference genomes | 1.0 | 0.0 |
| ranking weight on novelty vs style (step 8) | 0.2 | 1.0 |
| critic weight on strangeness (step 12) | 1 | 3 |

- Seed: when the brief has no `seed`, draw one from `numpy.random.SeedSequence().entropy`, then write it to `finalists.json` and `report.md`. The same seed and the same $r$ still give byte-identical output, so the determinism rule holds. Only the default changes. Log this under Deviations (RULES "Determinism").
- The forge skill (`SKILL.md` step 3 says "S = brief `seed`, else 0") must stop passing 0. It passes the brief's seed only when one is set, and passes $r$ through, and asks for $r$ when the mission text says "novel", "weird" or similar.

Tests: $r$ outside [0, 1] fails validation. Two runs with the same seed and $r$ match byte for byte. Two runs with no seed get different seeds. Knob values at $r = 0$, 0.5 and 1 match the table. No decoded outer panel is taller than 0.3 × reference length.

Gate (auto): tests pass. `report.md` shows the seed and $r$.

## Step 8: Selection: stop ranking by drag

Read: SPEC §9, §14. `search.py`, `derive.py`, `checks.py`.

Build:
- `select()`: keep the qualifying filter (passable bar, style ≥ `s_min`, novelty inside the band from step 7). Rank qualifiers by
$$\text{score} = w(r)\,z(\text{novelty}) + \bigl(1 - w(r)\bigr)\,z(\text{style})$$
where $z$ is the rank-normalised value among the qualifiers. `total_violation` leaves the ranking. `power`, `energy`, `efficiency` and `mass` stay in the report as PASS/FAIL rows.
- Diverse finalists: choose the 10 finalists greedily. Take the best score first. After that, each pick must be at least $d_{\min}$ from every finalist already chosen. Then the critic gets three designs that differ from each other.
- Tier-2 verify (`search_cli.py` `accepted()`): today it sorts verified finalists by how many checks failed, and `checks.yaml` `accept_other_failures_max: 2` drops any design with more than 2 failures outside the bar. That brings the efficiency ranking back after `select()`. Change both: keep finalists that pass the bar at tier 2, in the step-8 score order. Replace the fixed cap of 2 with `accept_other_failures_max(r)` (2 at $r = 0$, no cap at $r = 1$). The `run_verify` fallback order (by `bar_violation`) stays.
- `derive`: compute the baseline's electrical power at the dash point. Compare it with the existing `installed_power_w(req)` in `mass.py` (`installed_power_w_per_kg` × MTOW). Do not add a new constant. When the baseline needs more than `derive_power_infeasible_ratio` × installed power (one new constant, for example 3), raise `InfeasibleError("power", ...)`. The skill then asks the user to rework the brief instead of searching. The 500 m/s brief (3.9 MW) stops here.

Tests: a qualifier with a large `power` failure can still rank first. With $r = 1$, the highest-novelty qualifier ranks first. All chosen finalists are at least $d_{\min}$ apart. At $r = 1$, a finalist that fails `power`, `energy` and `efficiency` but passes the bar is still delivered. The 500 m/s brief raises `InfeasibleError` with check `power`.

Gate (auto): tests pass. Rerun `designs/dash-80ms-3m-span` at $r = 0.5$. The 10 finalists must contain at least 3 different counts of (surfaces, bodies, joins). Today all 10 have one body and no joins.

## Step 9: Frontal area and top-down area

Read: SPEC §4, §9. `models.py`, `geom.py`, `style.py` (`panels`), `layout.py`.

Build: a new module `areas.py`, pure functions with no CAD:
- `top_area_m2(d)`: the area of the design's shadow on the $x$–$y$ plane (seen from above), for every body, pod and surface together. Overlaps count once.
- `frontal_area_m2(d)`: the area of the shadow on the $y$–$z$ plane (seen from the front). Surfaces add chord × thickness and their canted height. Bodies add their super-ellipse sections.
- `body_top_share(d)` = body top area ÷ total top area. This number measures "bomb-ness". By hand estimate, the 500 m/s winner is about 0.5. Step 9 measures the real value.
- Method: rasterise onto a numpy grid with cells of `area_grid_ref_frac` × reference length. Bodies: super-ellipse sections sampled along $x$. Surfaces: the panel quads from `panels()`, given thickness for the frontal view. Shortcut: the raster error is about one cell width times the outline length. Use a finer grid if the tests show more than 2 % error.
- Add all three to `report.md` and `finalists.json`.

Tests (against closed-form values): a super-ellipse body alone matches $4ab\,\Gamma(1 + 1/n)^2 / \Gamma(1 + 2/n)$ within 2 %. A flat rectangular wing gives top area = span × chord within 2 %. A wing canted to 90° gives top area near 0. Overlapping parts are not counted twice.

Gate (auto): tests pass. On 300 random genomes, `areas.py` adds less than 10 ms per design.

## Step 10: New shape genes

Read: SPEC §4, §5. `genes.yaml`, `genome.py`, `planform.py`, `layout.py`, `derive.py`, `reference.yaml`.

Build:
- First split `genome.py`. It is at 396 lines against the 400-line limit in RULES (Known issues already lists it). Move `BRIEFS` and `brief_requirements` to `brief.py`, as that entry suggests.
- Each new gene gets a `genes.yaml` row, decoder code and a `models.py` field.
- Reference designs: `reference.yaml` gives every gene it does not list the middle of the gene's range. The 44 references all come from two roots (`conventional`, `flying_wing`), and those roots list almost every gene. So a new gene, or a widened range on an unlisted gene, would silently change all 44 shapes. Add an optional `ref_default` field to `genes.yaml` (for example `wing_area_scale: 1.0`, `sN_twist_deg: 0`, `body_nose_frac: 0.22`, `body_tail_frac: 0.40`, `body_width_ratio: 1.0`, `body_tail_upsweep_deg: 0`, `sN_root_chord_scale: 1.0`). `reference.py` uses it instead of the range middle. Choice genes already take their first label, so `layout` lists `free` first and `box_layout` lists `inline` first.

| gene | range | effect |
|---|---|---|
| `layout` (choice) | free, conventional, canard, tandem, three_surface, flying_wing, joined | Sets default roles and stations for slots s1–s3 (for example, canard puts s1 at `root_x_frac` 0.1 with cant 0). One mutation then switches the layout. The slot genes still fine-tune it. `free` keeps today's behaviour: the slot genes alone decide. `free` is the default, and every `reference.yaml` entry uses it, so the references keep their shapes. |
| `wing_area_scale` | 1.0–3.0 | Lifting area $S = k\,S_{\min}$, where $S_{\min}$ is today's stall-speed area. Capped so that the span stays at or below `span_max_m`. |
| `s0_aspect_ratio` | widen 4–12 to 4–20 | Keep this gene. Do not replace it with a span fraction: 9 references set it, and a span fraction would make their shape change with each brief's area. The span follows as $b = \sqrt{AR \cdot S_0}$, now with $S_0$ grown by `wing_area_scale`. When $b > b_{\max}$, the decoder lowers $AR$ to $b_{\max}^2 / S_0$. Example: $S_0 = 0.36$ m² and $AR = 20$ give $b \approx 2.7$ m. |
| `sN_root_chord_scale` | 1.0–2.0 | Strake or glove: the inner-panel root chord is stretched by this factor, then the area is re-normalised. This gives chord control apart from span. |
| `sN_twist_deg` | −6 to +2 | Linear washout from root to tip (added to `incidence_deg` per section). |
| `fam_sweep_deg` | widen to 0–55 | More swept and delta shapes. |
| `fam_dihedral_deg` | widen to −15 to 25 | Strong anhedral and gull shapes. |
| `s0_cant_deg` | widen to ±15 | (none) |
| `body_nose_frac` | 0.10–0.35 | Moved from a constant to a gene. |
| `body_tail_frac` | 0.15–0.45 | Moved from a constant to a gene. |
| `body_width_ratio` | 0.6–2.0 | Multiplies today's width ÷ height, which comes from the boxes. 1.0 keeps today's body. Values above 1 give a flat lifting body. The body still has to hold every box. |
| `body_tail_upsweep_deg` | 0–15 | Tail cone bends upward. |
| `box_layout` (choice) | inline, side_by_side, stacked, wing_root | How the battery, payload and avionics boxes are packed. Anything but `inline` shortens the minimum body. `wing_root` uses the existing `boxes_in_wing` path. |

Interface changes (log under Deviations):
- `Requirements.lifting_area_m2` becomes `lifting_area_min_m2`. Each `Design` carries its own `lifting_area_m2`. Checks that use $S$ read it from the design. The consumers to update are `models.py`, `genome.py`, `derive.py`, `aero.py`, `airplane.py`, `assemble.py`, `physics_cli.py`, `data/requirements_stub.yaml` and `data/requirements_small_boxes.yaml`. Grep again at the start of the step.
- `Requirements.ref_length_m` keeps coming from $S_{\min}$. It sets body length, the descriptor scale and 9 other places in the code. Keeping it on $S_{\min}$ means a larger wing does not grow the body. It also makes the test "`wing_area_scale` = 1 and `box_layout` = inline gives today's body length" valid.
- The genome gets longer. Old genome files, `tests/fixtures/winners/`, `history.jsonl` and the marks files no longer decode. Keep the old ones in a dated folder. Do not delete them.
- Tests that hard-code the old genome or interface: `test_genome.py`, `test_invariants.py`, `test_models.py`, `test_descriptor.py` and the `requirements.json` fixture. Update them for the new genes and the new field name. Do not weaken them (RULES "Truth").
- The mass closure in `derive` uses the baseline at $S_{\min}$. A design with `wing_area_scale` = 3 carries more wing mass than the closure assumed. The `mass` check reports this, but `mass` is not on the passable bar. Leave it so, and record the decision in Decisions.

Tests: invariants 1–10 on 1,000 random genomes. CAD builds succeed on at least 98 % of 100 random genomes (slow). Every reference design decodes to the same `Design` as before, within float tolerance. `wing_area_scale` = 1 with `box_layout` = inline reproduces today's body length. For each `layout` value, at least one random genome decodes to it.

User gate: `python -m aeroforge gallery --n 50 --seed 1`. Pass = no broken or floating parts, and visible canards, tandems, flat bodies, long-span wings and short bodies.

## Step 11: Novelty that sees the body

Read: SPEC §9. `descriptor.py`, `search_cli.py` (`make_run`, history).

Build: add four terms to `desc()`, each scaled to [0, 1] by a constant in `constants.yaml`:
1. `body_top_share` (step 9)
2. frontal area ÷ $S$
3. total top area ÷ $S$
4. centre body length ÷ $\sqrt{S}$

`DESC_SIZE` goes from 16 to 20. Recompute $d_{\min}$ from the reference set, and the CVT centroids from new random decodes. Start a new `history.jsonl`, because old rows have 16 numbers. Old rows cannot be recomputed. History rows store only the descriptor, and the old `genome.json` files no longer decode after step 10. So past designs stop counting toward novelty. Archive the old file.

Add an anti-bomb filter: finalists must have `body_top_share` ≤ `bomb_cap(r)` (0.45 at $r = 0$, 0.35 at $r = 1$, both in `constants.yaml`).

Tests: two designs that differ only in body length now have different descriptors. Descriptor values stay in [0, 1] on 1,000 random genomes. A design over the bomb cap is not chosen as a finalist.

Gate (auto): tests pass. No new pair of reference designs falls below $0.5\,d_{\min}$. Today 3 of the 44 references already sit below that line under the 80 m/s requirements ($d_{\min} = 0.111$). The closest pair is `conventional` and `rounded_elliptic` at 0.030: they differ only in tip style, which the descriptor cannot see. So the gate is "at most 3 such pairs", not "none".

## Step 12: Retune looks and the critic

Read: SPEC §7. `style.py`, `style.yaml`, `plugin/agents/visual-critic.md`.

Build:
- Style features for the new genes: span ratio, `body_top_share`, body width ratio, layout. Remove the positive `body_fineness` push. Its weight comes from old marks only.
- One marks round (at most 3, as in step 2): 60-tile gallery, you mark it, `tune-style`.
- Critic prompt: add $r$. The pick score becomes coherence + proportion + $c(r)$ × strangeness + defects, with $c$ from the step-7 table. The critic must say which finalist looks least like a torpedo with wings.

Tests: `tune-style` k-fold accuracy is above the baseline rate. The critic writes valid `critique.json` for three fixture renders.

User gate: the gallery shows a clear look gain over round 3.

## Step 13: Proof run

Read: SPEC §14.

Run `forge new` on the 40 m/s and 80 m/s briefs at $r$ = 0, 0.5 and 1, 3 seeds each (18 runs, in the background).

Gate (auto), with numbers in the CHANGELOG step entry:
- Share of finalists with one body, no joined tips and at most two surfaces: at most 30 % at $r = 1$. Before: 6 of 10 in the 80 m/s run and 10 of 10 in the 500 m/s run.
- Median `body_top_share` of the delivered designs: at most 0.35 at $r = 1$.
- At least 4 different `layout` values among the 12 designs at $r \ge 0.5$ (2 briefs × 2 values of $r$ × 3 seeds).
- Passable rate stays at least half of today's (80 m/s run: 677 of 5004).
- Every delivered design passes the passable bar at tier 2.

User gate: you look at the 18 viewers. Pass = different every run, and no bombs at $r \ge 0.5$.

---

## Risks

- **Speed.** Tier-1 checks take about 1.7 s per candidate. Longer genomes and a wider band need more budget to fill the archive. Measure in step 10. Raise `search_budget` only if the funnel shows fewer than 10 qualifiers.
- **Physics on new shapes.** The VLM gives bad loads on joined tips (Known issues), and flat lifting bodies are outside AeroBuildup's fuselage model. Both may pass tier 1 and fail tier 2. Step 13 counts this.
- **The bar may get harder.** Long-span wings help the `power` check but hurt `structure`. Strong anhedral hurts `lateral`. Watch the passable rate in step 13.
- **Lost history.** The marks, fixtures and history all depend on the old genome length. Archive them. Do not delete them.
- **Test time.** The fast suite is already at about 60 s against a 90 s limit. New CAD tests go under `slow`.

## Not in this plan

Swing wings, morphing parts, landing gear variety, and a real motor/propeller model for the power check. Each of these can be its own plan later.

## Audit log (2026-11-17)

I checked every file, function, constant and number this plan names against the code. All names exist. The super-ellipse area formula matches the body sections in `shapes.py`. These problems were found and fixed in the steps above:

1. Step 7 widened the novelty band before the guards against broken designs, which sat in step 8. The guards now come first in step 7.
2. `Brief` forbids unknown fields, so the `randomness` field needs a `models.py` change. The brief hook uses the same model, so it needs nothing else.
3. The forge skill sends seed 0 when the brief has none. Step 7 now changes the skill.
4. New genes and widened ranges would have changed all 44 reference designs, because unlisted genes take the middle of their range. Step 10 now adds a `ref_default` field.
5. Replacing `s0_aspect_ratio` with a span fraction would have made 9 references change shape with each brief. Step 10 now keeps the gene and widens its range.
6. `genome.py` is at 396 of 400 allowed lines. Step 10 now splits it first.
7. Old history rows cannot be recomputed: they store only descriptors, and old genomes stop decoding after step 10. Step 11 now says so.
8. The step-11 gate could not pass: 3 reference pairs are already closer than $0.5\,d_{\min}$. The gate is now "no new pairs".
9. The step-13 baseline said 100 % single-body finalists. The real counts are 6 of 10 (80 m/s) and 10 of 10 (500 m/s). The design count at $r \ge 0.5$ is 12, not 9.
10. Steps 10 and 11 now list the tests that hard-code the old genome, and the wing-mass gap in the mass closure.

Not checked: the step-9 raster speed and accuracy, the share of new random genomes that build in CAD, and whether the passable rate holds with the new genes. Only the build steps can measure these.
