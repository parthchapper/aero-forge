# U2 audit (branch u2-wip, tip d8ee6c5)

Method: a `git archive` snapshot of d8ee6c5 (scratch dir, not the working tree) was run with `PYTHONPATH` pointing at it; scripts are throw-away. Status column is filled in at the end of the fix phase.

## What was checked and found clean

| # | Check | Result |
|---|---|---|
| 1 | Smooth-envelope reads on faceted bodies | Grep of `body_radius_frac`, `body_axis_z`, `inside_loft`, `section_exp`, `nose_frac`, `tail_frac`, `tail_radius_frac`, `half_width_m`, `half_height_m` over `src/`: every read that can see a flat-panel body is guarded by `facets is not None` (airplane, areas x2, assemble, hardpoints x2, layout.host_end, shapes.inside_body / body_samples, style x2) or reads the polygon's true extents (`genome.py:330`, `areas._scan_lines`, `pod_fit`). No unguarded read. |
| 2 | CAD of 270 random flat-panel bodies (all noses/tails, 3-12 sides, 1-8 stations, upsweep 0/5/15 deg, length 0.05-60 m log-uniform) | 270/270 built; valid, one shell, every edge shared by two faces, faces = polyhedron faces = `panel_count`, volume = polyhedron to 1.8e-15 relative (bar 1e-6), outward normals (OCC point classifier). One body at first looked wrong (4 tail triangles): a probe artefact (tail wedge thinner than the probe step), fine at 1e-8 m. |
| 3 | Hand check of 3 flat-panel bodies with upsweep 3-12 deg and a moved firewall | Friction area (asb) = hand sum of face areas to 7 digits; mass = area x 0.004 x 200 exactly; CG z follows the area centroid (0.0253, -0.0066, 0.0294 m); `prop0_drive` mass sits at `attach_m`, which follows the moved firewall. |
| 6 | Cut list on 75 designs (4,648 sheets from 4,875 faces) | rows = merged sheets; flat area = 3D area to 1e-9; no self-intersecting outline; every sheet in the svg, inside its page rectangle and the viewBox, no overlapping boxes; sum(count x area) = wetted area to 1e-9. |

## Findings

| ID | File:line | Severity | Issue | Failure scenario | Fix | Status |
|---|---|---|---|---|---|---|
| A1 | `rise.py`, `prop_end.py`, `pod_fit.py`, `genome.py`, `layout.py` (follow-ups 1, 1b, 2) | High | Three follow-up commits changed decode behaviour with no tests, no CHANGELOG entry and a syntactically broken intermediate commit (`dc9ff48`: `layout.py:103` IndentationError). | The fast suite at d8ee6c5 fails 14 tests (below); nothing guards the new behaviour. | Tests in `tests/test_u2_followups.py`; CHANGELOG entry. | Fixed (e368465, tests/test_u2_followups.py; CHANGELOG U2 audit 2) |
| A2 | `genome.py:_rise_start`, `rise.py` | High (frozen value) | Follow-up 2 keeps a tail upsweep on smooth bodies that 0.3 dropped: `test_baseline_0_3` fails (`random[0].bodies[0].tail_upsweep_deg` 8.85 != 0.0) and `test_a_tail_upsweep_is_dropped_...` fails. A frozen baseline value moved. | Every 0.3 baseline design with a root on the tail cone changes. | The rise start only moves on flat-panel bodies; smooth bodies drop as in 0.3. | Fixed (e368465): smooth bodies keep the 0.3 drop. Side effect and open user decision: smooth drop rate is 48.8 % again |
| A3 | `rise.py` (old), `layout._faceted_fit`, `facets.axis_rise_m` | High | A flat-panel body is straight only between stations, but the rise start was a free fraction (e.g. 0.838): the cell around it tilts from the previous station, so (a) the closed form for the start under-lifts nothing and loops (`no_progress`, 18 of 98 drops), (b) boxes sized with `axis_rise_m` stick out of the body (`test_decoded_faceted_bodies_hold_their_boxes...[sample]`: payload box 0.43 % outside the body), (c) the root reach used `axis_z_at_x` (includes the shape's own centre shift) in a formula that assumes a pure rise. | Boxes outside the skin on designs with a moved rise; 25 % of designs lose their upsweep (faceted 40 %, smooth 9.5 %) against a target of 5 %. | The start is chosen among the body's stations; reach is evaluated exactly per candidate station. | Fixed (e368465, c1f5c2b, ed83e0d): exact axis between stations; flat-panel drop 39.9 % -> 33.7 % |
| A4 | tests (7 strict xfails, 3 plain failures) | High | The 7 `xfail(strict)` misfit-bar tests XPASS since follow-up 1 (misfit 0.3 % stub / 0.6 % small_boxes on 4 x 400 genomes). Strict xfail then fails the suite. `test_the_tail_upsweep_reaches_cad_mass_and_the_aero_section[15]` used the law without the moved start; `...hatch_avionics-False` names a genome that now fits; `test_style_family_is_respected` compares the family the decoder changed. | 14 failures in the fast suite. | Remove the xfails (rate verified); update the three tests to the new contract (stricter or equal). | Fixed (11b7e8e) |
| A5 | `hardpoints.py:_mount`, `internals._tray_in_body`, `hardpoints._hatch_footprint_m` | Medium | Fit adjustments that are not reported: plate radius between 50 % and 100 % of nominal, hatch share of its bay and offset from the bay centre, servo tray slide (up to 1.5 root chords away from mid chord). Only the firewall slide is a decoder note. | A design can pass containment with a 60 % plate, a 30 % hatch off-centre by 20 %, a servo 1.4 chords from its hinge, and the report says nothing. | A "Fit of internals" section in report.md. | Fixed (fc131e2, report.md section + test) |
| A6 | `layout.py:size_centre_body` | Low | 51 lines (limit 50). | RULES breach. | Split. | Fixed (c1f5c2b) |
| A7 | `layout.py:host_end` | Low | Two consecutive `if body.facets is not None` blocks. | Readability. | Merge. | Fixed (c1f5c2b) |
| A8 | `panels.py:panels_csv` docstring | Low | Says "one row per flat face"; rows are physical sheets. | Misleading. | Docstring. | Fixed (c1f5c2b) |
| A9 | `facets.panel_plan`, `panel_count`, `mesh.write_stls`, `checks.run_checks` | Low | Only tests call them (no src caller). U1 / pre-U1 code. | Dead in the delivered product. | Deferred: they are the tested contract of SPEC §5; not removed. | Deferred (Known issues) |
| A10 | `test_geom.py::test_tessellated_volumes_match_the_analytic_shapes` | Medium (decision made) | Wing `surface0_L` meshes 3.15 % off its analytic volume (bar 2 %), same on main. | Slow suite fails. | Strict xfail with reason "wing tessellation, pre-U2, investigate U3" (user decision). Tolerance unchanged. | Done as decided (11b7e8e): strict xfail on the surface half only; body half runs and passes |
| A11 | `hardpoints._mount` | Info | Smooth pusher plates under half their nominal radius are accepted (smooth keeps the 0.3 rule). Measured and logged. | See Known issues. | Logged only (user decision). | Logged (Known issues): 67 % of smooth plates under half nominal, behaviour unchanged |
| A12 | `constants.yaml` | Low | `rise_fit_margin_frac` becomes unused with the station rule. | Dead constant. | Removed. | Fixed (e368465) |

## Before / after

| Measure | Before (d8ee6c5) | After |
|---|---|---|
| Fast suite | 353 passed, 14 failed, 1 skipped (1,455 s, loaded host) | 375 passed, 0 failed, 1 skipped, 165 deselected (875 s, idle) |
| Slow suite | not run on d8ee6c5 | 145 passed, 18 skipped (smoke variants), 2 xfailed (wing volume), 0 failed (5,687 s) |
| Strict xfails | 7 (misfit bars), all XPASS = failing | 2 (wing `surface0_L`, "wing tessellation, pre-U2, investigate U3") |
| Misfit, stub flat-panel (4 x 400) | 5 (0.31 %) | 0 |
| Misfit, small_boxes flat-panel | 9 (0.56 %) | 8 (0.50 %): hatch 5, mount 3 |
| Misfit, stub smooth / small_boxes smooth | 0 / 5 (0.31 %) | 0 / 5 (0.31 %) |
| #52 drop, flat-panel | 206 / 516 (39.9 %) | 174 / 516 (33.7 %) |
| #52 drop, smooth | 46 / 482 (9.5 %, frozen 0.3 value changed) | 235 / 482 (48.8 %, the 0.3 rule) |
| #52 drop, all (998 genomes) | 25.3 % | 41.0 % (target < 5 %: not met) |
| `areas()` median, idle host | n/a | 8.9 ms (flat-panel 11.6, smooth 6.6); gate 10 ms |
