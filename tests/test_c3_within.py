"""within_20m targets vs numpy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_data.c3_markers import marker_table_for_split
from spatial_data.c3_tasks import cell_target_within_numpy, get_c3_task, within_train_markers
from spatial_data.c3_targets import c3_targets_for_spec
from spatial_data.dataset_c3_gpu import load_crop_planes_v4

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DIST = ROOT / "data" / "amsterdam" / "de_pijp"
DEV = torch.device("cuda") if torch.cuda.is_available() else pytest.skip("cuda required")


@pytest.mark.parametrize("radius_m", [20, 50, 75, 100])
def test_within_radius_targets(radius_m: int):
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
    key = f"within_{radius_m}m"
    planes = load_crop_planes_v4(CROPS, "val", DEV, max_items=8)
    cols, rows = marker_table_for_split(CROPS, DIST, "val")
    spec = get_c3_task(key)
    assert spec.relative_cond_id == {20: 14, 50: 15, 75: 17, 100: 16}[radius_m]
    mc = torch.from_numpy(cols[:8]).to(DEV)
    mr = torch.from_numpy(rows[:8]).to(DEV)
    gpu = c3_targets_for_spec(planes, spec, marker_col=mc, marker_row=mr).cpu().numpy()
    for i in range(8):
        ref = cell_target_within_numpy(int(cols[i]), int(rows[i]), radius_m=radius_m)
        assert (gpu[i] == ref.astype(np.int64)).all()


def test_within_train_marker_counts():
    assert within_train_markers(20) == (4, 40.0)
    assert within_train_markers(50) == (2, 100.0)
    assert within_train_markers(75) == (1, 0.0)
    assert within_train_markers(100) == (1, 0.0)
