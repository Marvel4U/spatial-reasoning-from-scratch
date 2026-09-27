"""CPU vs GPU c0 loader equivalence (worldsnap layers)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import config
from spatial_data.dataset_c0 import C0JsonlIndex, default_task_paths, load_c0_items, load_c0_item_tensor
from spatial_data.dataset_c0_gpu import C0GpuBatchLoader, C0GpuResidentStore
from spatial_data.layer_cache import LayerCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fixture_loaders(encoding: str = "scalar", n_items: int = 32):
    config.encoding_mode = encoding
    config.sync_in_chans_from_encoding()
    district = config.district_data_dir()
    crops_dir = district / "crops" / config.cropset
    paths = default_task_paths(district, config.taskset)
    items = load_c0_items(paths["train"], crops_dir, max_items=n_items)
    g, h = config.grid_out_size, config.img_size
    cache = LayerCache()
    cache.warm({it.labels_path for it in items})
    index = C0JsonlIndex(items, encoding, g, h, layer_cache=cache)
    device = torch.device("cuda")
    store = C0GpuResidentStore.from_items(items, cache, device)
    gpu = C0GpuBatchLoader(store, index, batch_size=8, seed=1337)
    return items, index, gpu, cache, g, h


def test_gpu_batch_matches_cpu_per_item_scalar():
    items, index, gpu, cache, g, h = _fixture_loaders("scalar")
    enc = "scalar"
    for ii in [0, 1, 7, len(items) - 1]:
        cpu_img, cpu_tgt = load_c0_item_tensor(
            items[ii], encoding=enc, grid_size=g, img_size=h, layer_cache=cache,
        )
        g_img, g_tgt = gpu.batch_from_item_indices([ii])
        assert g_img.shape == (1, 4, h, h)
        assert torch.allclose(g_img[0].cpu(), cpu_img, atol=0, rtol=0)
        assert torch.equal(g_tgt[0].cpu(), cpu_tgt)


def test_gpu_batch_matches_cpu_random_indices_scalar():
    items, index, gpu, cache, g, h = _fixture_loaders("scalar", n_items=64)
    rng = random.Random(42)
    for _ in range(5):
        idxs = [rng.randrange(len(items)) for _ in range(8)]
        cpu_imgs, cpu_tgts = [], []
        for ii in idxs:
            img, tgt = load_c0_item_tensor(
                items[ii], encoding="scalar", grid_size=g, img_size=h, layer_cache=cache,
            )
            cpu_imgs.append(img)
            cpu_tgts.append(tgt)
        cpu_imgs = torch.stack(cpu_imgs)
        cpu_tgts = torch.stack(cpu_tgts)
        g_img, g_tgt = gpu.batch_from_item_indices(idxs)
        assert torch.allclose(g_img.cpu(), cpu_imgs, atol=0, rtol=0)
        assert torch.equal(g_tgt.cpu(), cpu_tgts)


def test_gpu_batch_matches_cpu_onehot():
    items, index, gpu, cache, g, h = _fixture_loaders("onehot", n_items=16)
    for ii in range(min(8, len(items))):
        cpu_img, cpu_tgt = load_c0_item_tensor(
            items[ii], encoding="onehot", grid_size=g, img_size=h, layer_cache=cache,
        )
        g_img, g_tgt = gpu.batch_from_item_indices([ii])
        assert g_img.shape[1] == 13
        assert torch.allclose(g_img[0].cpu(), cpu_img, atol=0, rtol=0)
        assert torch.equal(g_tgt[0].cpu(), cpu_tgt)
