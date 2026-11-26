"""One search candidate: decode → descriptor → tier-1 prepare and checks → style (SPEC §9).

A failure becomes a result with a reason; it never raises into the search loop (RULES)."""

import faulthandler
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from aeroforge.aero import AeroError
from aeroforge.checks import CheckError, CheckReport, evaluate
from aeroforge.constants import constant, number
from aeroforge.descriptor import DescriptorError, desc
from aeroforge.edges import soft_style_bonus
from aeroforge.finish import LayoutError
from aeroforge.genome import DecodeError, decode
from aeroforge.models import Adjust, Brief, Design, Requirements
from aeroforge.style import features, packaged_model, score

_T = TypeVar("_T")

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


def run_limited(work: Callable[[], _T], limit_s: float, on_timeout: Callable[[], _T]) -> _T:
    """work() in a daemon thread; on_timeout() when it has not finished after limit_s seconds.
    The stuck thread cannot be killed: it is left running (the worker's other tasks go on, at
    a lower speed) and its stack goes to stderr."""
    box: list[tuple[bool, _T | BaseException]] = []

    def target() -> None:
        try:
            box.append((True, work()))
        except BaseException as exc:  # handed to the caller's thread
            box.append((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(limit_s)
    if not box:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        return on_timeout()
    ok, value = box[0]
    if not ok:
        assert isinstance(value, BaseException)
        raise value
    return value  # type: ignore[return-value]


def _timed_out(genome: np.ndarray) -> Candidate:
    """The candidate of a genome that took over candidate_timeout_s (dropped, reason says so)."""
    limit = number("candidate_timeout_s")
    g = tuple(float(v) for v in genome)
    sys.stderr.write(f"candidate timed out after {limit:g} s, genome {list(g)}\n")
    return Candidate(g, None, 0.0, None, None, f"timed out after {limit:g} s")


def evaluate_candidate(genome: np.ndarray, brief: Brief, req: Requirements, tier: int) -> Candidate:
    """Descriptor of the decoded genome, then the prepared design's checks and style score
    (a candidate over candidate_timeout_s is dropped with a reason)."""
    return run_limited(
        lambda: _checked(genome, brief, req, tier)[0],
        number("candidate_timeout_s"),
        lambda: _timed_out(genome),
    )


def pool_desc(genome: np.ndarray, req: Requirements) -> tuple[float, ...] | None:
    """decoded_desc under candidate_timeout_s (None when it does not decode or takes too long)."""
    return run_limited(
        lambda: decoded_desc(genome, req), number("candidate_timeout_s"), lambda: None
    )


def verify_candidate(
    genome: np.ndarray, brief: Brief, req: Requirements
) -> tuple[Candidate, Design | None]:
    """Tier-2 candidate and its delivered design (None when evaluation failed or timed out)."""
    return run_limited(
        lambda: _checked(genome, brief, req, 2),
        number("candidate_timeout_s"),
        lambda: (_timed_out(genome), None),
    )
