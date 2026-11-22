"""Create ~/.aero-forge/venv on Python 3.11 and install the engine into it.

Runs on any Python 3.8+ using only the standard library. Exit 0 ok, 3 environment error.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON_VERSION = "3.11"
HOME_DIR = Path.home() / ".aero-forge"
VENV_DIR = HOME_DIR / "venv"
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
IMPORTS = ["aerosandbox", "cadquery", "ribs", "trimesh", "numpy", "pydantic", "yaml", "PIL"]


class EnvironmentSetupError(Exception):
    """Setup cannot proceed on this machine."""


def venv_python(venv_dir: Path) -> Path:
    """Return the interpreter path inside a venv for this OS."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command quietly; raise EnvironmentSetupError with its output tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
        raise EnvironmentSetupError(f"{' '.join(cmd[:4])} failed: " + " | ".join(tail))
    return proc


def is_target_python(exe: str) -> bool:
    """True when exe runs and reports the target major.minor version."""
    try:
        proc = subprocess.run(
            [exe, "-c", "import sys;print(sys.version[:4])"], capture_output=True, text=True
        )
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip().startswith(PYTHON_VERSION)


def find_python() -> str:
    """Locate a Python 3.11 interpreter, installing one with uv if none is found."""
    if sys.version.startswith(PYTHON_VERSION):
        return sys.executable
    candidates = [shutil.which(f"python{PYTHON_VERSION}")]
    uv = shutil.which("uv")
    if uv:
        found = subprocess.run(
            [uv, "python", "find", PYTHON_VERSION], capture_output=True, text=True
        )
        candidates.append(found.stdout.strip() if found.returncode == 0 else None)
    for exe in filter(None, candidates):
        if is_target_python(exe):
            return exe
    if sys.platform == "win32" and shutil.which("py") and is_target_python_py():
        return run(
            ["py", f"-{PYTHON_VERSION}", "-c", "import sys;print(sys.executable)"]
        ).stdout.strip()
    if uv:
        run([uv, "python", "install", PYTHON_VERSION])
        return run([uv, "python", "find", PYTHON_VERSION]).stdout.strip()
    raise EnvironmentSetupError(f"Python {PYTHON_VERSION} not found; install it or install uv")


def is_target_python_py() -> bool:
    """True when the Windows py launcher can start the target version."""
    proc = subprocess.run(["py", f"-{PYTHON_VERSION}", "-c", "pass"], capture_output=True)
    return proc.returncode == 0


def setup(dev: bool) -> dict:
    """Create the venv if needed, install the engine, verify imports; return a summary."""
    py = venv_python(VENV_DIR)
    if not py.exists() or not is_target_python(str(py)):
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        run([find_python(), "-m", "venv", str(VENV_DIR)])
    run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    target = f"{ENGINE_DIR}[dev]" if dev else str(ENGINE_DIR)
    run([str(py), "-m", "pip", "install", "-q", *(["-e"] if dev else []), target])
    run([str(py), "-c", "import " + ", ".join([*IMPORTS, "aeroforge"])])
    return {"ok": True, "venv_python": str(py), "engine": str(ENGINE_DIR), "dev": dev}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="editable install with pytest and ruff")
    args = parser.parse_args()
    try:
        summary = setup(args.dev)
    except EnvironmentSetupError as err:
        print(json.dumps({"ok": False, "error": str(err)}, indent=2))
        return 3
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
