# testUpdate.md: running log of test results

**What this file is.** A crash-safe, append-only log of pytest results for aero-forge. The plugin
`plugin/engine/tests/progress_log.py` (loaded from `tests/conftest.py`) appends one line per test the
moment its call phase ends (flush + fsync), so after a crash, a kill or a context loss the last
line shows how far a run got and which tests failed. Read this first when a suite run is
interrupted or when you need the state of the U3 audit test runs. Lines look like
`- <timestamp> <PASSED|FAILED|SKIPPED|ERROR> <nodeid> (<seconds> s)`. Runs are separated by
`## run <timestamp>` headings. The section "Snapshot" below was written by hand from the logs; the
sections after it are written by the plugin. Do not delete old entries.

## Snapshot (written by hand, 2026-11-27, branch u3-wip)

Audit of U3 (see `Fix/u3_audit.md`). Fix commits: 46de78a, 93f4770 (planar gate, exact face areas,
volume gate), fd345c2 (per-candidate timeout, `candidate_timeout_s` = 120 s assumed).

| Run | Code | Result |
|---|---|---|
| Fast `-m "not slow"` | 93f4770 | 433 passed, 1 skipped, 165 deselected, 4,671 s, ruff clean |
| New tests `tests/test_planar_gate_u3.py` | 93f4770 | 18 passed |
| New tests `tests/test_candidate_timeout.py` | fd345c2 | 4 passed; with `test_search.py` fast tests 16 passed |
| Slow serial (first), no timeout | 93f4770 | stopped by hand: 164 of 165 done, 3 `F` (at about 9 %, 40 %, 43 %), last test stuck; names not read |
| Slow `-n 4 --timeout=900` | fd345c2 | unusable: 20 failed, 41 passed, 7 skipped (68 of 165 accounted). All 20 failures are 1000-genome "full" property tests killed by the 900 s timeout under 4x CPU contention ("node down: Not properly terminated"), not test failures |
| Slow serial `--timeout=7200` | fd345c2 | running (log: scratchpad `slow4.log`); same 3 `F` positions as the first serial run; no summary yet |

Failed in the `-n 4` run (timeouts, to be re-checked serially): test_geom (tessellated_volumes full-body, one_stl_is_written_per_part), test_invariants (span_stays_within_limit, tip_joins, style_family, capped_design_fin, outer_panel_cap, decode_is_deterministic, body_hosted_boxes, surface0_is_always_present, whole_root_chord), test_clearance (prop_discs_clear_structure, props_clear_the_ground_line), test_internals (random_designs_keep_internals_contained), test_new_genes (each_layout_by_its_geometry), test_search (same_seed_history_and_brief_give_identical_finalists), test_style (features_are_finite), test_descriptor (desc_identical_under_slot_reordering), test_gallery (cli_top_by_style_gallery_exports_scores), test_areas (agree_with_single_measures).

The hung test: `tests/test_search.py::test_chained_runs_give_tier2_passing_winners_beyond_d_min`. The
stuck worker was in `genome.decode` -> `pod_fit.grow_to_fit` -> `faceted_body._plate_scan`
(U1/U2 pod-fit code, unchanged against main). The hung genome decodes in 25 s under cProfile, and a
candidate repeats the decode many times, so it is very slow, not a loop in the planar code.
`candidate_timeout_s` now drops such a candidate with a reason.

## Live log (written by the plugin)

## run 2026-11-27 00:38:07: pytest -q tests/test_candidate_timeout.py
- 00:38:07 PASSED tests/test_candidate_timeout.py::test_a_finished_task_gives_its_value (0.0 s)
- 00:38:07 PASSED tests/test_candidate_timeout.py::test_an_exception_reaches_the_caller (0.0 s)
- 00:38:07 PASSED tests/test_candidate_timeout.py::test_a_task_over_the_limit_gives_the_fallback_and_does_not_block (0.2 s)
- 00:38:07 PASSED tests/test_candidate_timeout.py::test_a_timed_out_candidate_carries_a_reason (0.2 s)

## Slow-suite detail so far (hand-written 2026-11-27 00:39, from scratchpad logs)

**Serial slow run, `--timeout=7200`, code fd345c2 (log `slow4.log`, still running):** 147 of 165 tests finished so far: 126 passed, 18 skipped, 3 failed. Progress lines: `ssssss.s.F...` then `F.F` at 43 %, so the 3 failures sit at about 9 %, 40 % and 43 % of the run (the same positions as the first serial run on 93f4770). Their names are not known until pytest prints its summary; the new plugin was not loaded in this run, so it writes no per-test lines for it. The last test, `test_search.py::test_chained_runs_give_tier2_passing_winners_beyond_d_min`, is the one that hung before; with the 120 s candidate timeout it should now finish or fail.

**Earlier serial slow run on 93f4770 (log `slow2.log`, killed by hand):** 164 of 165 done, 3 `F` at about 9 %, 40 %, 43 %; last test stuck 1.5 h.

**Parallel slow run `-n 4 --timeout=900` on fd345c2 (log `slow3.log`, finished, 1:31:23):** 20 failed, 41 passed, 7 skipped. Every failure is a 1000-genome "full" parametrisation killed by the 900 s per-test timeout under CPU contention (pytest-timeout kills the xdist node: "node down: Not properly terminated"); none is a verified logic failure. Because killed nodes lose their queued tests, only 68 of 165 tests were accounted for, so this run does not count for check 6. The 20 failed ids are listed in the Snapshot section above.

**Hung-test reproduction (`chain.log`, `chain2.log`):** with the 120 s timeout in place, one candidate timed out and printed its genome and stack (stack in `pod_fit.grow_to_fit`; genome saved as scratchpad `hung_genome.json`).
