"""pytest plugin: append every test result to <repo>/testUpdate.md as it happens (crash-safe).

One line per test, written and fsynced when its call phase ends, so an interrupted run leaves a
record of how far it got (see the description at the top of testUpdate.md)."""

import os
import time
from pathlib import Path

import pytest

LOG = Path(__file__).resolve().parents[3] / "testUpdate.md"
HEADER = (
    "# testUpdate.md: running log of test results\n\n"
    "Crash-safe, append-only log written by plugin/engine/tests/progress_log.py: one line per "
    "test as it finishes. Read it first when a suite run is interrupted.\n\n## Live log\n"
)


def _append(text: str) -> None:
    if not LOG.exists():
        LOG.write_text(HEADER, encoding="utf-8")
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def pytest_sessionstart(session: pytest.Session) -> None:
    if getattr(session.config, "workerinput", None) is None:  # not an xdist worker
        args = " ".join(session.config.invocation_params.args)
        _append(f"\n## run {time.strftime('%Y-%m-%d %H:%M:%S')}: pytest {args}\n")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    done_call = report.when == "call"
    broken = report.when in ("setup", "teardown") and report.outcome == "failed"
    skipped = report.when == "setup" and report.outcome == "skipped"
    if done_call or broken or skipped:
        name = report.outcome.upper() if not broken else "ERROR"
        stamp = time.strftime("%H:%M:%S")
        _append(f"- {stamp} {name} {report.nodeid} ({report.duration:.1f} s)\n")
