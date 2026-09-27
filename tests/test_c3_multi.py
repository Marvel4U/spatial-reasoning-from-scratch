"""N5 c3 multi-task mix loader."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_data.c3_tasks import M1_TASK_KEY, N5_MIX_TASK_KEYS, get_c3_task
from spatial_data.dataset_c3_gpu import C3MultiTaskGpuTrainLoader, load_crop_planes_v4

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DIST = ROOT / "data" / "amsterdam" / "de_pijp"
DEV = torch.device("cuda") if torch.cuda.is_available() else pytest.skip("cuda required")


def test_n5_mix_cond_ids_distinct():
    specs = [get_c3_task(k) for k in N5_MIX_TASK_KEYS]
    pads = [s.padded_cond_ids() for s in specs]
    assert len(set(pads)) == 3
    assert get_c3_task("food_drink").padded_cond_ids()[1] == -1
    assert get_c3_task(M1_TASK_KEY).padded_cond_ids()[0] == get_c3_task("food_drink").padded_cond_ids()[0]


def test_n5_mix_loader_batch():
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
    from spatial_data.c3_markers import marker_tensors

    planes = load_crop_planes_v4(CROPS, "train", DEV, max_items=32)
    mc, mr = marker_tensors(CROPS, DIST, "train", DEV)
    mc, mr = mc[:32], mr[:32]
    loader = C3MultiTaskGpuTrainLoader(
        list(N5_MIX_TASK_KEYS), "onehot_v4", 64, 256, 4, 42,
        crops=planes, marker_col=mc, marker_row=mr,
    )
    img, tgt, task_id, cond_ids = loader.next_batch()
    assert img.shape[0] == 4 and tgt.shape == (4, 64, 64)
    assert task_id.shape == (4,)
    assert cond_ids.shape == (4, 2)
    assert cond_ids.dtype == torch.long


def test_multi_val_snap_batches_hit_m1():
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
    from spatial_data.c3_markers import marker_tensors
    from spatial_data.dataset_c3_gpu import C3MultiTaskGpuValLoader

    planes = load_crop_planes_v4(CROPS, "val", DEV, max_items=32)
    mc, mr = marker_tensors(CROPS, DIST, "val", DEV)
    mc, mr = mc[:32], mr[:32]
    loader = C3MultiTaskGpuValLoader(
        planes, list(N5_MIX_TASK_KEYS), "onehot_v4", 64, 256, 8,
        marker_col=mc, marker_row=mr,
    )
    m1_ix = list(N5_MIX_TASK_KEYS).index(M1_TASK_KEY)
    batches = list(loader.iter_batches_for_task(m1_ix, max_batches=3))
    assert len(batches) == 3
    assert all(tid.unique().numel() == 1 and int(tid[0]) == m1_ix for _, _, tid, _ in batches)


def test_weighted_mix_sampling():
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
    from spatial_data.c3_markers import marker_tensors

    planes = load_crop_planes_v4(CROPS, "train", DEV, max_items=64)
    mc, mr = marker_tensors(CROPS, DIST, "train", DEV)
    mc, mr = mc[:64], mr[:64]
    loader = C3MultiTaskGpuTrainLoader(
        list(N5_MIX_TASK_KEYS), "onehot_v4", 64, 256, 64, 99,
        crops=planes, marker_col=mc, marker_row=mr,
        train_task_weights=[0.1, 0.1, 0.8],
    )
    counts = torch.zeros(3, device=DEV)
    for _ in range(200):
        _, _, tid, _ = loader.next_batch()
        for t in tid:
            counts[int(t)] += 1
    fr = (counts / counts.sum()).cpu().tolist()
    assert fr[2] > 0.65
    assert fr[0] + fr[1] < 0.45
