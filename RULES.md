# RULES

## Session
1. Read CHANGELOG.md: Decisions, Known issues, last step entry.
2. Read the current PLAN.md step and only the SPEC.md sections it lists.
3. Write a ≤ 5-line plan in CHANGELOG before any code.
4. Build → test → gate → commit (`step N: <summary>`) → stop. Never start the next step.
5. Final message ≤ 15 lines: what was built, test result, what the user must do for the gate.

## Truth
- Do not trust memory for aerosandbox, cadquery, pyribs or Claude Code plugin APIs. Check the installed package (`help()`, source in site-packages) or current docs, then record the answer in Decisions. Check once; reuse the Decision.
- Never weaken, skip or delete a test to make it pass. Never hardcode an expected output. No fake backends: tests run the real libraries on small inputs.
- Fix budget per failing test: 3 tries (5 for geom.py, internals.py, aero.py). Then log under Known issues, mark the step failed, stop.
- A deviation from SPEC is allowed only when logged under Deviations with the reason. If it changes an interface, a gate or a check band: stop and ask.
- Report failures plainly. A step is "done" only when its tests pass and its gate is met.

## Code
- Python 3.11, `src/` layout, type hints on every function, `ruff check` and `ruff format` clean.
- Core is pure: decode, desc, features, checks take data and return data. No I/O, no globals, no hidden state. File and process I/O live in cli.py and the writers only.
- Determinism: randomness only through a `numpy.random.Generator` passed as an argument. No global seeds, no wall-clock in outputs, sorted iteration. Same inputs → byte-identical outputs.
- Units: SI, degrees, one frame (SPEC §4). Unit suffix in every field and argument name.
- No magic numbers. Every tunable is in `data/*.yaml` with unit and source note, read through constants.py.
- Models: pydantic v2, frozen. Validate at boundaries (file load, CLI), not inside loops.
- Errors: specific exception types; no bare `except`; no silent fallback. In the search loop a failed candidate becomes an infeasible result with a reason; it never crashes the run.
- Size: functions ≤ 50 lines, files ≤ 400 lines. Split when over.
- No dead code, commented-out code, or TODOs without a Known issues entry.
- Comments explain why, never what. Public functions get a one-line docstring with units.
- `logging`, not `print` (except the CLI JSON summary). Quiet by default.
- Vectorise with numpy. Measure before optimising.
- Write only what the current step needs. No speculative options, flags or abstractions.

## Tests
- pytest. Each module gets its tests in the same step.
- Property tests: seeded loops over random genomes; assert invariants, not snapshots.
- Fast suite (`pytest -q -m "not slow"`) under 90 s (user, step 4 r2). Mark CAD-heavy and search tests `slow`.
- One assertion idea per test; test names state the behaviour.

## Tokens
- Do not re-read a file written this session. Edit with targeted edits, not rewrites.
- Run tests with `-q -x --tb=short`. Send long output to a file and read only the tail.
- Never open STEP, STL, GLB or large JSON. Inspect them with a summary script (counts, names, bounds).
- View at most 4 images per step; use a contact sheet.
- No subagents unless the step says so.
- No docs beyond CHANGELOG until step 6. No restating SPEC in code or comments.
- Keep CHANGELOG entries to the template; one line per item.

## Boundaries
- Write only inside the repo and `~/.aero-forge/`.
- Network: package installs and the doc lookups above only.
- New dependency: needs a Decisions line with the reason.
- Never delete past CHANGELOG entries.

## CHANGELOG.md template
```
# CHANGELOG — aero-forge
## Environment        (OS, Python, package versions)
## Decisions          (question → answer, source)
## Assumed constants  (name = value unit, source)
## Deviations         (SPEC § → what changed, why)
## Known issues       (issue — impact — suggested fix)
## Steps
### [step N] <name> — planned | done | failed
- Plan: (≤ 5 lines)
- Result: (1 line)
- Tests: (command → PASS/FAIL)
- Gate: (auto: met/not met | user: waiting/pass)
## Rebuild            (exact commands from a fresh clone)
```
