---
name: forge
description: Design a novel, physics-checked small fixed-wing UAV from a mission brief and deliver CAD, report and a 3D viewer. Use for "/forge setup", "/forge new <mission>", "/forge status".
argument-hint: "setup | new <mission in plain words> | status"
allowed-tools: Bash, Read, Write, Edit, Glob, AskUserQuestion, Agent
---

# aero-forge

Mode = first word of `$ARGUMENTS` (`setup`, `new`, `status`); the rest is the mission text.

## Engine rules
- PY = `~/.aero-forge/venv/Scripts/python.exe` on Windows, `~/.aero-forge/venv/bin/python` elsewhere.
- Engine call = `PY -m aeroforge <cmd> ...`: one command per Bash call, no `cd`, `&&` or pipes.
- stdout is a JSON summary; exit 0 ok, 1 check failed, 2 bad input, 3 environment (→ run setup).
- Read only JSON summaries and small JSON files. Never open STEP, GLB, STL or meshes.
- `search` and `verify` take minutes to tens of minutes: run them with `run_in_background` and
  `timeout: 7200000` (the background default stops them at 10 min), then wait for the
  completion notice. Give every other engine call `timeout: 600000`.

## setup
1. `python "${CLAUDE_PLUGIN_ROOT}/scripts/setup_env.py"` (`python3` if `python` is missing;
   any Python ≥ 3.8). It creates the venv on Python 3.11, installs the engine, verifies imports.
2. Report `venv_python` from its JSON, or the error and that exit 3 means the environment.

## new
1. Brief. Id = short kebab-case slug of the mission; D = `designs/<id>` under the current
   folder (add `-2`, `-3` … if it exists). Write `D/brief.yaml` from the mission text:
   - required: `payload_mass_kg`, `payload_box_m: [l, w, h]`, `span_max_m`, `cruise_speed_ms`,
     exactly one of `endurance_min` / `range_km`, `launch` (hand | catapult | runway).
   - optional (only if stated): `recovery` (belly | skid | wheels | net), `stall_max_ms`,
     `dash_speed_ms`, `n_limit`, `gust_ms`, `altitude_m`, `mass_max_kg`, `efficiency_floor`,
     `seed`, `randomness` (0 safe … 1 wild, default 0.5). SI units, numbers only, one comment
     line with the mission text. Never invent a `seed`: without one every run is different.
   - If the mission says "novel", "weird", "alien", "unusual", "wild" or similar and gives no
     randomness: one AskUserQuestion (how wild: 0.5 balanced, 0.8 bold, 1.0 wildest), then write
     the answer as `randomness`.
   - If required fields are missing: one AskUserQuestion call (≤ 4 questions), only for those.
   - A hook validates the file on write; if it reports problems, fix the brief and rewrite.
2. `PY -m aeroforge derive D/brief.yaml` → exit 1: the brief is infeasible; give the user the
   check and error from the JSON and ask them to rework the brief (check `power`: a lower
   dash speed or a higher mass limit), then stop. Do not search.
3. `PY -m aeroforge search D`, in the background. It reads the brief's `seed` and `randomness`;
   with no `seed` it draws a fresh one. Add `--seed S` or `--randomness R` only when the user
   asked for that value in this request. The JSON reports the `seed` and `randomness` used.
4. `PY -m aeroforge verify D`, in the background.
5. Read `D/verified.json`: top 3 = `passing` (accepted finalists, best first), or the first 3
   of `order` when `fallback` is true.
   For each id F: `PY -m aeroforge build D --finalist F`, then `PY -m aeroforge render D
   --finalist F`. If a build exits 1, replace F with the next id in `order`.
6. Agent `aero-forge:visual-critic` with: the design dir D, the 3 ids, and their views
   `D/finalists/<F>/views/iso.png`. It writes `D/critique.json` with its `pick`.
7. `PY -m aeroforge build D --full` (adopts the pick), `PY -m aeroforge render D`,
   `PY -m aeroforge report D`. A build exit 1 lists missing or uncontained internals: report it.
8. Reply: the absolute path of `D/viewer.html`, then a 5-line summary from the JSON you have:
   (1) picked finalist + the critic's reason; (2) MTOW and battery (derive JSON);
   (3) failed checks of the pick (verify JSON `failed`, "all PASS" if none);
   (4) nearest reference and nearest past design with distances (report JSON);
   (5) warnings: fallback, CAD errors, missing internals — or "no warnings".
   Then one line: the search `seed` and `randomness` (search JSON), so the run can be replayed.

## status
Glob `designs/*/brief.yaml` under the current folder. For each D list one line:
`<id> — <stage>` where stage is the last present of brief.yaml, requirements.json,
finalists.json, verified.json, critique.json, assembly.step, viewer.html; add the viewer path
when it exists. No engine calls.
