"""report.md (SPEC §11): brief vs achieved per check, novelty, seed, assumed constants, fidelity."""

from dataclasses import dataclass

from aeroforge.areas import Areas, areas
from aeroforge.checks import CheckReport
from aeroforge.constants import constant, load_table
from aeroforge.models import Brief, Design, Requirements

ASSUMED_TABLES = ("constants", "checks")


@dataclass(frozen=True)
class Novelty:
    """Distances in descriptor space: nearest reference and nearest past delivered design."""

    d_min: float
    reference: str
    reference_distance: float
    past: str | None
    past_distance: float | None


def _fmt(v: object) -> str:
    """Compact value text (4 significant digits for floats)."""
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _brief_rows(b: Brief, req: Requirements) -> list[str]:
    """Brief fields as given, then the derived requirements."""
    rows = ["| field | value |", "|---|---|"]
    rows += [f"| {k} | {_fmt(v)} |" for k, v in b.model_dump().items() if v is not None]
    derived = req.model_dump(exclude={"points", "payload_box_m", "span_max_m", "recovery"})
    rows += [f"| derived {k} | {_fmt(v)} |" for k, v in derived.items() if v is not None]
    rows += [
        f"| point {p.id} | {_fmt(p.speed_ms)} m/s, n {_fmt(p.load_factor)} |" for p in req.points
    ]
    return rows


def _check_rows(report: CheckReport) -> list[str]:
    """One row per check: PASS/FAIL, margin and the achieved values behind it."""
    rows = ["| check | result | margin | achieved |", "|---|---|---|---|"]
    for c in report.checks:
        detail = ", ".join(f"{k} {_fmt(v)}" for k, v in c.detail.items())
        rows.append(f"| {c.id} | {'PASS' if c.passed else 'FAIL'} | {c.margin:+.4f} | {detail} |")
    return rows


def assumed_constants() -> list[str]:
    """Every data-table entry whose source note says it is assumed: `name = value unit (source)`."""
    out = []
    for table in ASSUMED_TABLES:
        for name, entry in load_table(table).items():
            source = str(entry.get("source", "")) if isinstance(entry, dict) else ""
            if source.lower().startswith("assumed"):
                out.append(f"- {name} = {_fmt(entry['value'])} {entry.get('unit', '')} ({source})")
    return out


def _novelty_rows(n: Novelty) -> list[str]:
    """Nearest reference by name and nearest past design, in d_min units too."""
    rows = [
        f"- Nearest reference: {n.reference}, distance {n.reference_distance:.4f}"
        f" ({n.reference_distance / n.d_min:.2f} x d_min, d_min {n.d_min:.4f}).",
    ]
    if n.past is None or n.past_distance is None:
        rows.append("- Nearest past design: none in the history.")
    else:
        rows.append(
            f"- Nearest past design: {n.past}, distance {n.past_distance:.4f}"
            f" ({n.past_distance / n.d_min:.2f} x d_min)."
        )
    return rows


def area_rows(shadow: Areas) -> list[str]:
    """Report lines for the shadow areas (top-down, frontal, body share of the top-down one)."""
    return [
        f"- Top-down shadow area {shadow.top_m2:.4f} m2;"
        f" the body is {shadow.body_top_share:.0%} of it.",
        f"- Frontal shadow area {shadow.frontal_m2:.4f} m2.",
    ]


def _faceted_fidelity(design: Design) -> list[str]:
    """The faceted-body sentence of the fidelity statement, when a body is flat-panel."""
    if any(b.facets is not None for b in design.bodies):
        return [str(constant("constants", "report_fidelity_faceted"))]
    return []


def report_md(
    b: Brief,
    req: Requirements,
    design: Design,
    report: CheckReport,
    novelty: Novelty,
    seed: int,
    randomness: float,
    finalist: str,
) -> str:
    """Markdown report of one delivered design."""
    verdict = "feasible" if report.feasible else "NOT feasible (FAIL rows below)"
    lines = [
        f"# aero-forge design report — {finalist}",
        "",
        f"Tier-{report.tier} checks: {verdict}; total violation {report.violation:.4f}."
        f" Search seed {seed}, randomness {randomness:g}.",
        "",
        "## Brief",
        *_brief_rows(b, req),
        "",
        "## Checks (brief vs achieved)",
        *_check_rows(report),
        "",
        "## Balance and adjustments",
        f"- body shift {report.body_shift_m:+.4f} m ({report.shift_outcome}), battery shift"
        f" {report.battery_shift_m:+.4f} m, stack shift {report.stack_shift_m:+.4f} m"
        f" ({report.balance}); fin scale {report.fin_scale:.3f}; incidence"
        f" {report.incidence_deg:+.2f} deg ({report.incidence_rule}).",
        *[f"- note: {n}" for n in report.notes],
        "",
        "## Shadow areas",
        *area_rows(areas(design)),
        "",
        "## Novelty",
        *_novelty_rows(novelty),
        "",
        "## Assumed constants",
        *assumed_constants(),
        "",
        "## Fidelity",
        str(constant("constants", "report_fidelity")),
        *_faceted_fidelity(design),
        "",
    ]
    return "\n".join(lines)
