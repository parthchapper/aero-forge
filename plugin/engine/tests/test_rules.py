"""RULES.md size limits: no source file over 400 lines and no function over 50 lines."""

import ast
from pathlib import Path

MAX_LINES = 400
MAX_FUNC_LINES = 50
SOURCE = Path(__file__).parents[1] / "src" / "aeroforge"


def test_no_source_file_is_longer_than_400_lines() -> None:
    long = {
        p.name: n
        for p in SOURCE.glob("*.py")
        if (n := len(p.read_text("utf-8").splitlines())) > MAX_LINES
    }
    assert not long


def test_no_source_function_is_longer_than_50_lines() -> None:
    long = {}
    for p in sorted(SOURCE.glob("*.py")):
        tree = ast.parse(p.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                if length > MAX_FUNC_LINES:
                    long[f"{p.name}:{node.name}"] = length
    assert not long
