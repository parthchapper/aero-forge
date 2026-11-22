# aero-forge

Claude Code plugin that designs novel small fixed-wing UAVs from a mission brief. It derives
requirements from the brief, runs a quality-diversity search over aircraft layouts (style score,
novelty against 40+ reference configurations and your past designs), checks the finalists at
tier 2 (vortex-lattice aerodynamics, trim, stability, stall, structure, fit), and delivers CAD
with internals, rendered views, a report and a self-contained 3D viewer.

## Install
```
/plugin marketplace add ./          # from the repository root
/plugin install aero-forge@aero-forge-market
/aero-forge:forge setup             # one time: Python 3.11 venv in ~/.aero-forge/venv
```
Setup needs any Python ≥ 3.8 on PATH; it finds Python 3.11 or installs it with `uv`.
The engine (aerosandbox, CadQuery, pyribs, trimesh) is a large download on first setup.

## Use
- `/aero-forge:forge new <mission>`: e.g. "hand-launched survey plane, 0.5 kg camera
  18×10×7 cm, 2.2 m span max, 18 m/s cruise, one hour". Writes `designs/<id>/` in the
  current folder. A full run takes roughly 20–60 min, mostly the search.
- `/aero-forge:forge status`: lists the design folders and how far each got.

## Outputs (`designs/<id>/`)
`brief.yaml requirements.json design.yaml genome.json finalists.json check.json critique.json
views/{iso,top,front,side}.png parts/*.step assembly.step assembly.glb report.md viewer.html`
- `report.md`: each check PASS/FAIL with margin, novelty, seed, assumed constants, fidelity.
- `viewer.html`: opens from disk; click a part for its name and design fields.
- Delivered designs are appended to `~/.aero-forge/history.jsonl`, so the next design of the
  same brief must differ from them.

## Parts
- `skills/forge/SKILL.md`: the `/forge` workflow (setup, new, status).
- `agents/visual-critic.md`: scores three finalist renders and picks one.
- `hooks/`: validates `designs/**/brief.yaml` after every write (exit 2 with the problems).
- `engine/`: the Python package (`python -m aeroforge <cmd>`); `scripts/setup_env.py`.

## Limits
Conceptual design only: low-order aerodynamics, analytic structure and mass estimates.
Not flight-cleared; check every design before building or flying it.
