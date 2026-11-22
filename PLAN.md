> The aero-forge 1.0 upgrade steps (U0–U10) live in `Fix/plan.md`; this file keeps the 0.3 steps 0–6.

# PLAN

One step per session, new session each step. Model: Opus. Effort: high for steps 1–4, medium for 0, 5, 6.
- Start a step: `Do PLAN.md step N.`
- Resume a stopped step: `Continue step N.`
- After a user gate: `Step N gate: pass.` or `Step N gate: <tile numbers + what is wrong>.`

CLI commands a step adds follow SPEC §12.

## Step 0 — Environment and spike
Read: SPEC §2, §3.
Build: git repo, `plugin/engine/pyproject.toml` (src layout, ruff, pytest), `plugin/scripts/setup_env.py`, CHANGELOG.md from the RULES template, `constants.py`.
Spike in a throwaway `spike/` dir; record each answer in Decisions, then delete the dir:
- a. Versions of aerosandbox, cadquery, ribs, trimesh that install together on Python 3.11.
- b. aerosandbox: an Airplane with 3 wings + 1 fuselage; AeroBuildup and VortexLatticeMethod each return CL, CD, Cm, neutral point, Cnβ, Clβ; seconds per call; a control deflection changes Cm; AeroBuildup gives profile drag; a source for section clmax at Re 2e5.
- c. aerosandbox → CadQuery STEP export works; render one view; list visible defects.
- d. CadQuery: spline loft through 5 airfoil sections, union with a body, one fillet attempt, STEP + STL export, solid count.
- e. trimesh: STLs → Scene with named nodes → GLB → reload, names intact.
- f. pyribs: current ask/tell API, an archive that accepts 16-D measures, seeding.
- g. Claude Code: plugin.json, marketplace.json and hooks.json schemas; validate command; subagent frontmatter fields; how a hook receives the file path; the plugin-root variable; whether installed plugins are copied to a cache; venv python path on this OS.

Tests: setup_env.py exits 0; one smoke test imports every dependency.
Gate (auto): Decisions answers a–g. If b or d fails: stop and report options.

## Step 1 — Shape
Read: SPEC §4, §5, §6.
Build: `models.py` (Design), `genes.yaml`, `constants.yaml` (geometry entries), `genome.py`, `geom.py`, `mesh.py`, `render.py`, `gallery.py`, CLI `gallery`. Use a fixed dummy Requirements; derive arrives in step 3.
Tests:
- Invariants 1–10 on 1,000 random genomes (no CAD).
- CAD build on 100 random genomes ≥ 98 % (slow).
- Same input → identical PNG bytes.
- GLB node names match part names.

**User gate:** `python -m aeroforge gallery --n 50 --seed 1`, open gallery.html. Pass = no floating, broken or tangled parts, and clear variety.

## Step 2 — Looks
Read: SPEC §7.
Build: `style.py`, `style.yaml`, CLI `tune-style`, `gallery --top-by-style`.
Loop, at most 3 rounds: gallery of 60 (30 top by score, 30 random) → user marks and exports marks.json → `tune-style` → next round. Fix a systematic defect at its source (gene range or decoder rule) and log it.
Tests:
- Features finite on 1,000 genomes.
- `tune-style` recovers known weights from synthetic marks.

**User gate:** at least 14 of the top 20 by score marked good.

## Step 3 — Physics
Read: SPEC §8.
Build: `derive.py`, `mass.py`, `aero.py`, `checks.py`, `checks.yaml`, remaining `constants.yaml` entries, the `conventional` entry in `reference.yaml`, `tests/fixtures/brief_sample.yaml`, CLI `validate`, `derive`, `check`.
Tests:
- Conventional reference passes every check at tier 2.
- Battery moved aft until static margin < 5 % → `static` fails, other checks unchanged.
- Brief "10 km in 1 h" → infeasible, reason names stall.
- Mass closure converges, or raises the typed error on an impossible brief.
- Tier 1 vs tier 2 neutral point within 5 % MAC on 5 varied designs; log larger gaps under Known issues.
- Tier-1 time per candidate measured and logged against the 0.1 s target.

Gate (auto), then print for user review: the conventional reference's check table and the Assumed constants list.

## Step 4 — Search
Read: SPEC §9.
Build: `descriptor.py`, `reference.yaml` (≥ 40 entries), `search.py`, history file, CLI `search`, `verify`, `demo`. Save 5 winners to `tests/fixtures/winners/` for step 5.
Tests:
- desc is identical under slot reordering; conventional, canard and flying wing are pairwise > d_min apart.
- Same seed + history + brief → identical finalists.
- Injected infeasible and CAD-failing candidates never crash the loop.
- 5 runs at reduced budget (slow): each winner passes tier 2 and is ≥ d_min from every reference and from the other winners.

**User gate:** `python -m aeroforge demo tests/fixtures/brief_sample.yaml --runs 10`, open the gallery. Pass = all 10 look good and different. If not: return to the step 2 loop, or adjust d_min / efficiency_floor and log it.

## Step 5 — Internals and deliverables
Read: SPEC §10, §11.
Build: `internals.py`, `report.py`, `viewer.py`, CLI `build`, `render`, `report`.
Tests:
- Completeness list satisfied on the 5 saved step-4 winners.
- Containment holds; a deliberately oversized spar fails it.
- assembly.step reloads with the expected solid count and names.
- viewer.html embeds the GLB and the part table.
- report.md has every check row and the fidelity statement.

Gate (auto), then the user opens one viewer.html.

## Step 6 — Plugin
Read: SPEC §12, §13, §14.
Build: marketplace.json, plugin.json, SKILL.md, visual-critic.md, hooks.json, validate_brief.py, README.md, CHANGELOG Rebuild section.
Tests:
- `claude plugin validate ./plugin`.
- Hook: invalid brief under designs/ → exit 2; any other path → exit 0.
- Fresh venv from the Rebuild commands → `pytest -q` green.
- One full `/forge new` on the sample brief, from the installed plugin in a folder outside the repo, delivers every §11 file.

Gate: SPEC §14, all five items.
