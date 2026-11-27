# U3 audit (branch u3-wip, audited tip 07dee82, fixes to 93f4770)

Method: ~800 seeded designs (seeds 900000+i; faceted wings 3..8 in turn, coarseness off/low/mid/high in turn, strict alignment 4 of 5, two thirds forced faceted bodies, briefs stub / small_boxes). 100 got the full probe (CAD, cut list, containment), 400 more containment + cut list, 800 the CAD build of every faceted part. Probe scripts are throw-away (scratchpad).

## Checks

| # | Check | Result |
|---|---|---|
| 1 | CAD valid, watertight, volume = polyhedron, faces = panel_count | 350 parts (100 designs): valid, one shell, watertight, volume = polyhedron to 7e-13, OCC faces = polyhedron faces on every part. **Not clean (Medium, logged):** on merged (coarse) designs the CAD has more faces than the cut list has sheets (153 of 350 parts); SPEC 6 says merged faces are one CAD face. Deviation logged. Before the fixes 8 of 800 designs could not be built (F1, F2). |
| 2 | Misfit <= 1 % faceted; fallback rate | Misfits 3 of 488 faceted designs (0.61 %: mount_prop0 x2, servo_surface3). Coarse-body fallback (genes' body kept, noted): 45 of 319 coarse faceted-body designs (14.1 %). |
| 3 | Cut list | After F3: every sheet's area = its flat outline's area (<= 7e-10 relative, 800 designs), sum(count x area) = wetted area, 0 self-intersecting sheets (was 1: F2), 0 sheets outside their page. 80 of 96 designs have at least one sheet over the stock size (own oversize page, by design, as in U2). The first form of this check was circular (both sides read the same area function). |
| 4 | No silent dead controls | All notes reach report.md (12 reports checked: fallback x4, misfit x3, strict, soft, coarse on smooth parts). Gap: rejected strip merge was silent (F4, fixed); `facet_coarseness` on a design with no faceted part has no note (L1, logged). |
| 5 | 0.3 baseline and references | `git diff main` of reference.yaml, baseline_0.3, baseline_pre_step10 and their tests is empty; baseline tests pass in the fast suite. |
| 6 | Final commit's suites | Before the audit: no (e3ae5ec said slow suite not green, 07dee82 had no suite line, placeholders in the CHANGELOG). After: fast 437 passed and slow 144 passed / 3 failed oracle errors fixed and re-run, see the results table. The slow suite was not re-run in full after the oracle fixes. |

## Findings

| ID | Where | Severity | Issue | Fix | Status |
|---|---|---|---|---|---|
| F1 | facet_planar.py (cap tetras), facet_wing.py | High | Cap planarity used the first three ring corners (near-collinear after decimation) and an absolute determinant: a cap ended 13 um to 0.45 mm off its plane (3 of ~580 designs); `build` fails "wires not planar". | `planarisation_holds`: caps flat to facet_geom_tol_m, merged quads simple and facing as before; else next shrink, else unmoved wing. | Fixed (46de78a) |
| F2 | facet_planar.py | High | A merged quad could flip into a figure of eight; solid invalid (seed 1168). | Same gate (bow-tie test). | Fixed (46de78a) |
| F3 | facet_query.face_areas_m2, area_centroid_m | High | Fan triangles summed by absolute area: concave faces (wing caps of 5-16 corners) over-counted by up to 19 % in panels.csv, report, wetted area. Volume was right (signed). | Newell area vector per face; centroid uses signed areas. Body areas change by <= 2e-16. | Fixed (46de78a) |
| F4 | wing_geometry / coarse_metrics | High | The minimum-norm planarisation flattened 15 of 993 coarse wings (1.5 %) into one plane (volume 0, CAD "invalid or empty solid", every quad trivially flat). Pre-existing since U3b-2. | Gate: wing volume within coarse_volume_ratio_max (2, assumed) either way; rejection is a decoder note ("no strip merge was accepted"). 0 build failures in 800 designs after. | Fixed (93f4770) |
| M1 | geom.polyhedron_solid | Medium | CAD faces are the triangulated faces, not the merged sheets (SPEC 6 sentence). Area, volume, cut list correct. | Logged (Deviations / Known issues): unify same-domain faces and re-check validity. | Logged |
| M2 | facet_wing / coarse | Medium | Accepted merges move wing volume between 0.31 and 1.62 of the unmoved wing (1 to 5 % of wings under 0.9 or over 1.15); the spar / box / containment checks still read the section polygon. | Constant coarse_volume_ratio_max 2.0 is an assumption: user to confirm or tighten. | Logged |
| M3 | facet_planar `_tetras` | Medium | Root cause of F1 (cap basis) not changed: the gate rejects the merge, so those wings lose the merge (noted) instead of getting a good basis. | Use `_basis` for cap rings; shifts merge yield, re-measure the U3b-2 targets. | Logged |
| L1 | genome / coarse | Low | `facet_coarseness` > 0 on a design with smooth wings and a smooth body is a silent no-op (6 of 500). | Decoder note. | Logged |
| L2 | wing root caps | Low | A mirrored wing's root cap (internal) is a cut-list sheet. | Decide in U4 with the cut-list format. | Logged |
| L3 | panels | Info | 80 of 96 designs have an oversize sheet (longer than the stock). | None. | Info |

| T1 | tests/test_geom.py `_surface_volume` | Medium | A faceted wing's volume oracle used the unmoved section outlines; the planarised polyhedron differs by 2-4 % on coarse wings (U3b-2, red since e3ae5ec). | Oracle = polyhedron volume (as for bodies); the new ratio test keeps an independent check. | Fixed |
| T2 | tests/test_clearance.py `_lowest_z` | Low | A coarse body's lowest point went 8 mm below the smooth rule (nose z - half height); the test demanded too much (1 of 400 small_boxes designs). | Deeper of the rule and the polyhedron. | Fixed |
| T3 | search pool | High (process) | One candidate decode took hours (pod fit: `grow_to_fit` / `_plate_scan`, U2 code, unchanged against main), the slow suite waited on it. | `candidate_timeout_s` = 120 s (assumed): the candidate is dropped with a reason, stack and genome on stderr (fd345c2); the stuck thread keeps running until the worker ends. The pod-fit cost itself is logged. | Fixed (timeout), cost logged |

## Tests added (tests/test_planar_gate_u3.py, 18)
Candidate timeout (tests/test_candidate_timeout.py, 4). Volume ratio of coarse wings within coarse_volume_ratio_max of the genes' loft (1). Concave-prism area and centroid; every cut-list area = its flat outline; seeds 55, 1168, 6 and the seven flattened wings (flat, closed, build, volume > 0); rejected merge is a note; gate unit tests (bow-tie, cap off plane); random coarse wings build.

## Results

| Run | Commit | Result |
|---|---|---|
| Fast `-m "not slow"` | 93f4770 | 433 passed, 1 skipped, 165 deselected (4,671 s, host shared with nothing else) |
| Slow `-m slow`, serial, `--timeout=7200` | fd345c2 | 144 passed, 18 skipped, 3 failed (3 h 40 min). The test that hung (`test_chained_runs_give_tier2_passing_winners_beyond_d_min`) passes. The 3 failures were wrong test oracles on coarse designs (T1, T2), fixed in the next commit |
| The 3 failed tests re-run | after the oracle fixes | volume tests pass; clearance test passes (1 passed, 1 skipped) |
| Fast, `-m "not slow"` | fd345c2 | 437 passed, 1 skipped, 165 deselected (1,774 s) |
| New volume-ratio test | last test commit | passes (run alone) |
| Full slow suite after the oracle fixes | - | **not re-run in full**: only the 3 failing tests (source unchanged since fd345c2, only tests changed). Check 6 is met on the user's ruling (slow run clean apart from documented fixes) |
| Slow `-n 4 --timeout=900` | fd345c2 | discarded: 20 timeouts from CPU contention, 68 of 165 tests accounted for |
| Lint | 93f4770 | ruff check and format clean |
