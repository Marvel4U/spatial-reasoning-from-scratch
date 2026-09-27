"""v4 cropset: RGB planes match v3b; c2 loaders use legacy 3-layer stack only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

from spatial_data.c2_tasks import get_c2_task
from spatial_data.dataset_c2_gpu import c2_targets_from_planes, load_crop_planes

CROPS_V3B = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v3b"
CROPS_V4 = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.skipif(not CROPS_V4.is_dir(), reason="v4 crops not on disk")
def test_v4_rgb_planes_match_v3b_train_stack():
    v3 = load_crop_planes(CROPS_V3B, "train", DEV, max_items=32)
    v4 = load_crop_planes(CROPS_V4, "train", DEV, max_items=32)
    assert v3.shape == v4.shape
    assert torch.equal(v3, v4)


@pytest.mark.skipif(not CROPS_V4.is_dir(), reason="v4 crops not on disk")
def test_v4_c2_targets_match_v3b_for_core_task():
    v3 = load_crop_planes(CROPS_V3B, "train", DEV, max_items=8)
    v4 = load_crop_planes(CROPS_V4, "train", DEV, max_items=8)
    spec = get_c2_task("noise_0+surface_2")
    cond = torch.tensor([spec.padded_cond_ids()] * 8, dtype=torch.long, device=DEV)
    t3 = c2_targets_from_planes(v3, cond)
    t4 = c2_targets_from_planes(v4, cond)
    assert torch.equal(t3, t4)


@pytest.mark.skipif(not CROPS_V4.is_dir(), reason="v4 crops not on disk")
def test_v4_npz_has_extra_planes():
    f = next((CROPS_V4 / "train").glob("*_labels.npz"))
    with np.load(f) as z:
        assert "noise7_256" in z.files and "sun_256" in z.files
        assert int(z["noise7_256"].max()) <= 6
        assert int(z["sun_256"].max()) <= 7  # classes 0-6 ground, 7 = building interior
