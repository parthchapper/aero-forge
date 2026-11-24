"""One search candidate: decode → descriptor → tier-1 prepare and checks → style (SPEC §9).

A failure becomes a result with a reason; it never raises into the search loop (RULES)."""

from dataclasses import dataclass

import numpy as np

from aeroforge.aero import AeroError
from aeroforge.checks import CheckError, CheckReport, evaluate
from aeroforge.constants import constant
from aeroforge.descriptor import DescriptorError, desc
from aeroforge.edges import soft_style_bonus
from aeroforge.finish import LayoutError
from aeroforge.genome import DecodeError, decode
from aeroforge.models import Adjust, Brief, Design, Requirements
from aeroforge.style import features, packaged_model, score

CANDIDATE_ERRORS = (
    AeroError,
    CheckError,
    DecodeError,
    DescriptorError,
    LayoutError,
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
)


def passable_bar() -> frozenset[str]:
    """Check ids of the step-4 bar (checks.yaml passable_checks)."""
    bar = constant("checks", "passable_checks")
    if not isinstance(bar, list):
        raise CheckError("checks.yaml passable_checks must be a list")
    return frozenset(str(b) for b in bar)


@dataclass(frozen=True)
class Candidate:
    """Search result of one genome; report and adjust are None when evaluation failed."""

    genome: tuple[float, ...]
    desc: tuple[float, ...] | None
    style: float
    report: CheckReport | None
    adjust: Adjust | None
    reason: str | None

    @property
    def passable(self) -> bool:
        """Every check of the step-4 bar passes (user Deviation §14.3)."""
        return self.report is not None and self.report.passable

    @property
    def bar_violation(self) -> float:
        """Sum of negative margins over the passable-bar checks (inf when evaluation failed)."""
        if self.report is None:
            return float("inf")
        bar = passable_bar()
        return float(sum(max(0.0, -c.margin) for c in self.report.checks if c.id in bar))


def decoded_desc(genome: np.ndarray, req: Requirements) -> tuple[float, ...] | None:
    """Descriptor of the decoded genome, None when it does not decode."""
    try:
        return tuple(float(v) for v in desc(decode(genome, req)))
    except CANDIDATE_ERRORS:
        return None


def _checked(
    genome: np.ndarray, brief: Brief, req: Requirements, tier: int
) -> tuple[Candidate, Design | None]:
    """Candidate and the delivered (prepared) design at the tier."""
    g = tuple(float(v) for v in genome)
    x = decoded_desc(genome, req)
    if x is None:
        return Candidate(g, None, 0.0, None, None, "genome does not decode"), None
    try:
        design, report = evaluate(np.asarray(genome, float), brief, req, tier)
    except CANDIDATE_ERRORS as exc:
        return Candidate(g, x, 0.0, None, None, f"{type(exc).__name__}: {exc}"), None
    adjust = Adjust(
        body_shift_m=report.body_shift_m,
        fin_scale=report.fin_scale,
        add_fin=report.added_fin,
        incidence_deg=report.incidence_deg,
    )
    style = min(1.0, score(features(design), packaged_model()) + soft_style_bonus(design))
    return Candidate(g, x, style, report, adjust, None), design


def evaluate_candidate(genome: np.ndarray, brief: Brief, req: Requirements, tier: int) -> Candidate:
    """Descriptor of the decoded genome, then the prepared design's checks and style score."""
    return _checked(genome, brief, req, tier)[0]


def verify_candidate(
    genome: np.ndarray, brief: Brief, req: Requirements
) -> tuple[Candidate, Design | None]:
    """Tier-2 candidate and its delivered design (None when evaluation failed)."""
    return _checked(genome, brief, req, 2)
