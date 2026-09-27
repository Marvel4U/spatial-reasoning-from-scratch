"""c4 route loader smoke (Phase B gate) — checks live in harness/experiments.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from spatial_data.c4_tasks import TASK_KEYS, get_c4_task

try:
    import torch
except ImportError:
    torch = None

pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuda required", allow_module_level=True)

from experiments import c4_loader_smoke_checks

ROUTES = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4" / "c4"
pytestmark = pytest.mark.skipif(
    not (ROUTES / "c4_routes_train.npz").is_file(),
    reason="c4 smoke store missing",
)


def test_c4_cond_ids_distinct():
    pads = [get_c4_task(k).padded_cond_ids() for k in TASK_KEYS]
    assert len(set(pads)) == 2


def test_c4_loader_smoke_checks():
    stats = c4_loader_smoke_checks(device=torch.device("cuda"))
    assert stats["train_fg_cells"] > 0
    assert stats["n_train_routes"] <= 64
