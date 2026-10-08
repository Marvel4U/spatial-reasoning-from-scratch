"""Pytest hooks shared across the repo."""
from __future__ import annotations

import shutil
from pathlib import Path


def pytest_sessionfinish(session, exitstatus) -> None:
    """Remove stray diagnostics run dirs from tests that wrote under runs/diagnostics/."""
    diag = Path(__file__).resolve().parents[1] / "runs" / "diagnostics"
    if not diag.is_dir():
        return
    for d in diag.iterdir():
        if d.is_dir() and d.name.endswith("_pytest"):
            shutil.rmtree(d, ignore_errors=True)
