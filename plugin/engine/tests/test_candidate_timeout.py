"""A candidate that does not finish is dropped with a reason; it never hangs a search worker
(U3 audit: one slow decode kept the slow suite waiting for hours)."""

import threading

import numpy as np
import pytest

from aeroforge.candidate import run_limited


def test_a_finished_task_gives_its_value() -> None:
    assert run_limited(lambda: 7, 5.0, lambda: -1) == 7


def test_an_exception_reaches_the_caller() -> None:
    def boom() -> int:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        run_limited(boom, 5.0, lambda: -1)


def test_a_task_over_the_limit_gives_the_fallback_and_does_not_block() -> None:
    stop = threading.Event()
    assert run_limited(lambda: 1 if stop.wait(30) else 2, 0.2, lambda: -1) == -1
    stop.set()


def test_a_timed_out_candidate_carries_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from aeroforge import candidate

    monkeypatch.setattr(candidate, "number", lambda *_a: 0.2)
    stop = threading.Event()
    monkeypatch.setattr(candidate, "_checked", lambda *_a: stop.wait(30) or None)
    c = candidate.evaluate_candidate(np.zeros(3), None, None, 1)  # type: ignore[arg-type]
    stop.set()
    assert c.report is None and c.reason is not None and "timed out" in c.reason
