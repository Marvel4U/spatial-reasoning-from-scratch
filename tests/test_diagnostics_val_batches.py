"""Tier-2 val batch cap (c2 task-major grid)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import config
import eval_spatial


def test_c2_diagnostics_use_full_val_grid():
    saved = config.task_rung
    config.task_rung = "c2"
    try:
        assert eval_spatial.diagnostics_val_max_batches(20) is None
    finally:
        config.task_rung = saved


def test_c0_diagnostics_respect_ce_eval_iters():
    saved = config.task_rung
    config.task_rung = "c0"
    try:
        assert eval_spatial.diagnostics_val_max_batches(20) == 20
    finally:
        config.task_rung = saved
