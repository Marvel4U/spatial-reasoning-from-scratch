"""GPU-resident c0 crop store + batch loader (worldsnap, Step 2 perf path)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .channels import Encoding, NUM_CLASSES_PER_LAYER
from .dataset_c0 import C0Item, C0JsonlIndex
from .layer_cache import LayerCache
from .targets import FOREGROUND_CLASS

_SCALAR_SCALE = float(NUM_CLASSES_PER_LAYER - 1)


class C0GpuResidentStore:
    """Unique crop label planes (uint8) and per-item tables, all on one CUDA device."""

    def __init__(
        self,
        crops: torch.Tensor,
        crop_ix: torch.Tensor,
        marker_row: torch.Tensor,
        marker_col: torch.Tensor,
    ):
        if crops.device != crop_ix.device:
            raise ValueError("crops and index tensors must share device")
        self.device = crops.device
        self.crops = crops
        self.crop_ix = crop_ix
        self.marker_row = marker_row
        self.marker_col = marker_col

    @property
    def n_items(self) -> int:
        return int(self.crop_ix.shape[0])

    @property
    def n_crops(self) -> int:
        return int(self.crops.shape[0])

    @classmethod
    def from_items(
        cls,
        items: list[C0Item],
        layer_cache: LayerCache,
        device: torch.device,
        *,
        log=None,
    ) -> C0GpuResidentStore:
        if not items:
            raise ValueError("empty item list")
        if device.type != "cuda":
            raise ValueError(f"GPU store requires cuda device, got {device}")
        t0 = time.perf_counter()
        paths = sorted({it.labels_path.resolve() for it in items})
        path_id = {p: i for i, p in enumerate(paths)}
        planes_list = [np.stack(layer_cache.get(p), axis=0) for p in paths]
        stack = np.stack(planes_list, axis=0).astype(np.uint8, copy=False)
        crops = torch.from_numpy(stack).to(device, non_blocking=True)
        crop_ix = torch.tensor(
            [path_id[it.labels_path.resolve()] for it in items],
            dtype=torch.long,
            device=device,
        )
        marker_row = torch.tensor(
            [it.marker_row for it in items], dtype=torch.long, device=device,
        )
        marker_col = torch.tensor(
            [it.marker_col for it in items], dtype=torch.long, device=device,
        )
        if log:
            mb = crops.numel() / 1e6
            log(
                f"  gpu_store: {len(paths)} crops, {len(items)} items, "
                f"{mb:.0f} M uint8 on {device} ({time.perf_counter() - t0:.1f}s)"
            )
        return cls(crops, crop_ix, marker_row, marker_col)


class C0GpuBatchLoader:
    """Harness-compatible: .B, .index, .next_batch(); batches live on CUDA."""

    def __init__(
        self,
        store: C0GpuResidentStore,
        index: C0JsonlIndex,
        batch_size: int,
        seed: int,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.store = store
        self.index = index
        self.B = batch_size
        self._gen = torch.Generator(device=store.device)
        self._gen.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.index)

    def _sample_item_ix(self, batch: int) -> torch.Tensor:
        n = self.store.n_items
        return torch.randint(0, n, (batch,), device=self.store.device, generator=self._gen)

    def batch_from_item_indices(self, item_ix: torch.Tensor | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic batch for tests (same item order as indices)."""
        if isinstance(item_ix, list):
            item_ix = torch.tensor(item_ix, dtype=torch.long, device=self.store.device)
        else:
            item_ix = item_ix.to(device=self.store.device, dtype=torch.long)
        return self._build_batch(item_ix)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._build_batch(self._sample_item_ix(self.B))

    def _build_batch(self, item_ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self.store
        enc: Encoding = self.index.encoding
        g, h = self.index.grid_size, self.index.img_size
        b = int(item_ix.shape[0])
        crop_ids = st.crop_ix[item_ix]
        layers_u8 = st.crops[crop_ids]
        mr = st.marker_row[item_ix]
        mc = st.marker_col[item_ix]
        ar = torch.arange(b, device=st.device)
        if enc == "scalar":
            in_ch = 4
            img = torch.zeros(b, in_ch, h, h, device=st.device, dtype=torch.float32)
            img[:, :3] = layers_u8.float() / _SCALAR_SCALE
            img[ar, 3, mr, mc] = 1.0
        elif enc == "onehot":
            in_ch = 13
            img = torch.zeros(b, in_ch, h, h, device=st.device, dtype=torch.float32)
            chunks = []
            for li in range(3):
                plane = layers_u8[:, li].long()
                chunks.append(F.one_hot(plane, NUM_CLASSES_PER_LAYER).permute(0, 3, 1, 2).float())
            img[:, :12] = torch.cat(chunks, dim=1)
            img[ar, 12, mr, mc] = 1.0
        else:
            raise ValueError(f"unknown encoding: {enc!r}")
        tgt = torch.zeros(b, g, g, dtype=torch.long, device=st.device)
        gr = mr * g // h
        gc = mc * g // h
        tgt[ar, gr, gc] = FOREGROUND_CLASS
        return img, tgt
