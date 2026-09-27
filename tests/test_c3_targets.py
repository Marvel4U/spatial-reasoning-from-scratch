"""c3 GPU targets vs numpy reference (rule sweep logic)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_data.c3_tasks import cell_target_numpy, get_c3_task
from spatial_data.c3_targets import c3_targets_from_planes
from spatial_data.dataset_c3_gpu import load_crop_planes_v4

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DEV = torch.device("cuda") if torch.cuda.is_available() else pytest.param(
    "cuda", marks=pytest.mark.skip(reason="c3 GPU targets need cuda")
)


@pytest.mark.parametrize("rule", ["median_strict", "quietest_tertile"])
def test_c3_targets_match_numpy(rule: str):
    if not CROPS.is_dir():
        pytest.skip("v4 crops not on disk")
    planes = load_crop_planes_v4(CROPS, "val", DEV, max_items=12)
    spec = get_c3_task(rule)
    rel = torch.full((planes.shape[0],), spec.relative_cond_id, device=DEV)
    gpu = c3_targets_from_planes(planes, rel).cpu().numpy()
    n7 = planes[:, 0].cpu().numpy()
    surf = planes[:, 1].cpu().numpy()
    for i in range(planes.shape[0]):
        ref = cell_target_numpy(n7[i], surf[i], rule)
        assert (gpu[i] == ref.astype(np.int64)).all()
