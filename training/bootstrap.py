"""Interpreter bootstrap for the one-click scripts.

``start_training.py`` may be launched with the system ``python3`` (which has no
torch).  :func:`ensure_environment` detects that situation and re-executes the
current process with the project virtual environment, so that
``python3 start_training.py`` "just works" once the environment is installed.

Candidate interpreters, in order:

1. ``$JEV_PYTHON`` (explicit override)
2. ``<project>/.venv/bin/python``

Only the standard library is used here; the module must import before torch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
REQUIRED_MODULES: tuple[str, ...] = ("torch", "transformers", "peft", "yaml", "numpy")
_GUARD_ENV = "JEV_BOOTSTRAPPED"


def missing_modules(modules: tuple[str, ...] = REQUIRED_MODULES) -> list[str]:
    """Names of the required modules that the *current* interpreter cannot import."""
    out: list[str] = []
    for name in modules:
        try:
            if importlib.util.find_spec(name) is None:
                out.append(name)
        except (ImportError, ValueError):  # broken namespace / stale finder
            out.append(name)
    return out


def candidate_interpreters() -> list[str]:
    """Interpreters to try, most specific first (duplicates and self removed)."""
    cands = [
        os.environ.get("JEV_PYTHON"),
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
    ]
    # NOTE: do not resolve symlinks - a venv's bin/python is usually a symlink to
    # the base interpreter, and it must be executed through the venv path to
    # pick up the venv's site-packages.
    current = os.path.abspath(sys.executable)
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if not c:
            continue
        path = os.path.abspath(os.path.expanduser(c))
        if path in seen or path == current:
            continue
        if os.path.isfile(path) and os.access(path, os.X_OK):
            seen.add(path)
            out.append(path)
    return out


def ensure_environment(argv: list[str] | None = None) -> None:
    """Re-exec with the project venv when the current interpreter lacks the ML stack.

    Does nothing when everything imports.  Prints a clear message and exits
    with code 3 when no usable interpreter is found.  A guard variable prevents
    exec loops.
    """
    missing = missing_modules()
    if not missing:
        return
    if os.environ.get(_GUARD_ENV) == "1":
        sys.stderr.write(
            f"[jev] interpreter {sys.executable} still lacks {missing} after bootstrap; "
            "install the requirements (see README.md) to (re)create the environment.\n"
        )
        sys.exit(3)
    for python in candidate_interpreters():
        os.environ[_GUARD_ENV] = "1"
        args = [python, *(argv if argv is not None else sys.argv)]
        sys.stderr.write(f"[jev] {sys.executable} lacks {missing}; re-executing with {python}\n")
        sys.stderr.flush()
        try:
            os.execv(python, args)
        except OSError as exc:  # pragma: no cover - try the next candidate
            sys.stderr.write(f"[jev] could not exec {python}: {exc}\n")
            continue
    sys.stderr.write(
        f"[jev] no Python environment with {missing} found. Install the requirements (see README.md) "
        "into ./.venv, or set JEV_PYTHON=/path/to/venv/bin/python.\n"
    )
    sys.exit(3)


__all__ = ["ensure_environment", "candidate_interpreters", "missing_modules"]
