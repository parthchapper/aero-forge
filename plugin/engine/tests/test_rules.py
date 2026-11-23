"""RULES.md size limit: no source file over 400 lines."""

from pathlib import Path

MAX_LINES = 400
SOURCE = Path(__file__).parents[1] / "src" / "aeroforge"


def test_no_source_file_is_longer_than_400_lines() -> None:
    long = {
        p.name: n
        for p in SOURCE.glob("*.py")
        if (n := len(p.read_text("utf-8").splitlines())) > MAX_LINES
    }
    assert not long
