import numpy as np
import torch

from spatial_data.c1_tasks import (
    DEFAULT_HELD_OUT_KEYS,
    FACTORISED_TWELVE_KEYS,
    compositional_wrong_task_index,
    get_c1_task,
    per_task_fg_ce_weight,
    train_task_indices,
    train_task_keys,
)
from spatial_data.dataset_c1_gpu import C1GpuBatchLoader, C1GpuResidentStore
from spatial_data.dataset_c1 import C1CropIndex, C1CropItem, load_c1_item_tensor
from spatial_data.targets import c1_mask_grid_from_plane, c1_surface_pure_cell_mask


def test_factorised_twelve_train_holdout():
    keys = list(FACTORISED_TWELVE_KEYS)
    assert len(keys) == 12
    train = train_task_keys("factorised_twelve", DEFAULT_HELD_OUT_KEYS)
    assert len(train) == 10
    assert set(DEFAULT_HELD_OUT_KEYS).isdisjoint(train)
    tix = train_task_indices("factorised_twelve", DEFAULT_HELD_OUT_KEYS, keys)
    assert tix is not None and len(tix) == 10


def test_compositional_wrong_task_same_layer():
    keys = list(FACTORISED_TWELVE_KEYS)
    tid = keys.index("surface_0")
    wrong = compositional_wrong_task_index(tid, keys, cropset="v2")
    assert keys[wrong] == "surface_1"


def test_v3b_estab_majority_and_food_weight():
    fd = get_c1_task("food_drink", "v3b")
    assert fd.downsample == "majority"
    w = per_task_fg_ce_weight("food_drink", cropset="v3b", clamp=100.0)
    assert 38.0 < w < 44.0
    assert get_c1_task("food_drink", "v2").downsample == "any"


def test_building_majority_matches_block_rule():
    task = get_c1_task("building")
    plane = np.zeros((256, 256), dtype=np.uint8)
    plane[0:4, 0:4] = 3
    tgt = c1_mask_grid_from_plane(torch.from_numpy(plane.astype(np.int64)), task)
    assert tgt[0, 0].item() == 1
    assert tgt[0, 1].item() == 0


def test_cpu_and_gpu_target_agree(tmp_path):
    plane = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    np.savez(tmp_path / "x_labels.npz", noise_256=plane, surface_256=plane, estab_256=plane)
    item = C1CropItem(tmp_path / "x_labels.npz")
    task = get_c1_task("building")
    img_cpu, tgt_cpu = load_c1_item_tensor(item, task, encoding="scalar", grid_size=64, img_size=256)
    if not torch.cuda.is_available():
        return
    from spatial_data.layer_cache import LayerCache
    dev = torch.device("cuda")
    lc = LayerCache()
    lc.warm({item.labels_path})
    store = C1GpuResidentStore.from_items([item], lc, dev)
    crop_ix = torch.zeros(1, dtype=torch.long, device=dev)
    index = C1CropIndex([item], task, "scalar", 64, 256, layer_cache=lc)
    loader = C1GpuBatchLoader(store, crop_ix, index, 1, 0)
    img_gpu, tgt_gpu = loader.batch_from_crop_indices(crop_ix)
    assert torch.allclose(img_cpu, img_gpu[0].cpu(), atol=1e-5)
    assert torch.equal(tgt_cpu, tgt_gpu[0].cpu())
