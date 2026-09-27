"""N5 m1: food & drink within 100 m — conjunctive targets."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_data.c3_markers import marker_table_for_split
from spatial_data.c3_tasks import (
    ESTAB_FOOD_DRINK_ATOM,
    M1_TASK_KEY,
    REL_WITHIN_100M,
    cell_target_food_drink_within_100m_numpy,
    get_c3_task,
    within_train_markers,
)
from spatial_data.c3_targets import c3_targets_for_spec
from spatial_data.dataset_c3_gpu import load_crop_planes_v4

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DIST = ROOT / "data" / "amsterdam" / "de_pijp"
DEV = torch.device("cuda") if torch.cuda.is_available() else pytest.skip("cuda required")


def test_m1_spec_cond_tokens():
    spec = get_c3_task(M1_TASK_KEY)
    assert spec.radius_m == 100
    assert spec.conjunction_atom_id == ESTAB_FOOD_DRINK_ATOM
    assert spec.padded_cond_ids() == (ESTAB_FOOD_DRINK_ATOM, REL_WITHIN_100M)
    assert within_train_markers(100) == (1, 0.0)


def test_m1_targets_match_numpy():
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
    planes = load_crop_planes_v4(CROPS, "val", DEV, max_items=8)
    cols, rows = marker_table_for_split(CROPS, DIST, "val")
    estab_np = planes[:, 2].cpu().numpy()
    spec = get_c3_task(M1_TASK_KEY)
    mc = torch.from_numpy(cols[:8]).to(DEV)
    mr = torch.from_numpy(rows[:8]).to(DEV)
    gpu = c3_targets_for_spec(planes, spec, marker_col=mc, marker_row=mr).cpu().numpy()
    for i in range(8):
        ref = cell_target_food_drink_within_100m_numpy(estab_np[i], int(cols[i]), int(rows[i]))
        assert (gpu[i] == ref.astype(np.int64)).all()
