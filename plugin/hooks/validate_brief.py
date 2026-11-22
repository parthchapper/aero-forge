"""PostToolUse hook: validate designs/**/brief.yaml against the engine's Brief model.

Reads the hook JSON on stdin. Exit 0 for any other path or a valid brief; exit 2 with at most
5 lines on stderr for an invalid brief (Claude Code feeds them back to the model); exit 1 when
the engine cannot be imported (setup not run). Runs on the engine venv Python.
"""

import json
import sys
from pathlib import PureWindowsPath

DESIGNS_DIR, BRIEF_NAME = "designs", "brief.yaml"
MAX_LINES = 5
EXIT_OK, EXIT_ENV, EXIT_BLOCK = 0, 1, 2


def brief_path(event: dict) -> str | None:
    """The written file when it is designs/**/brief.yaml, else None."""
    raw = (event.get("tool_input") or {}).get("file_path")
    if not isinstance(raw, str):
        return None
    # PureWindowsPath splits on both separators, so one rule covers every OS.
    parts = PureWindowsPath(raw).parts
    if parts[-1:] != (BRIEF_NAME,) or DESIGNS_DIR not in parts[:-1]:
        return None
    return raw


def errors(path: str) -> list[str]:
    """Problems in the brief file; empty when it validates."""
    import yaml
    from pydantic import ValidationError

    from aeroforge.models import Brief

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        Brief.model_validate(raw)
    except (OSError, yaml.YAMLError) as exc:
        return [f"{type(exc).__name__}: {exc}"]
    except ValidationError as exc:
        return [f"{'.'.join(map(str, e['loc'])) or 'brief'}: {e['msg']}" for e in exc.errors()]
    return []


def main() -> int:
    """Hook entry point."""
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return EXIT_OK
    path = brief_path(event) if isinstance(event, dict) else None
    if path is None:
        return EXIT_OK
    try:
        found = errors(path)
    except ImportError as exc:
        print(f"aero-forge engine missing ({exc}); run /forge setup", file=sys.stderr)
        return EXIT_ENV
    if not found:
        return EXIT_OK
    head = f"invalid brief {path} ({len(found)} problem(s)); fix and rewrite it:"
    print("\n".join([head, *found][:MAX_LINES]), file=sys.stderr)
    return EXIT_BLOCK


if __name__ == "__main__":
    sys.exit(main())
