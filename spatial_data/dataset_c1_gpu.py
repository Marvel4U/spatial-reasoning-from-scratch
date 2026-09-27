"""GPU-resident c1 crop store + batch loader."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .c1_tasks import C1TaskSpec
from . import channels as _ch
from .channels import Encoding, NUM_CLASSES_PER_LAYER, NUM_NOISE7_CLASSES, marker_plane
from .dataset_c1 import C1CropIndex, C1CropItem
from .layer_cache import LayerCache
from .targets import FOREGROUND_CLASS, c1_positive_pixels

_SCALAR_SCALE = float(NUM_CLASSES_PER_LAYER - 1)


class C1GpuResidentStore:
    def __init__(self, crops: torch.Tensor):
        self.device = crops.device
        self.crops = crops

    @property
    def n_crops(self) -> int:
        return int(self.crops.shape[0])

    @classmethod
    def from_items(
        cls,
        items: list[C1CropItem],
        layer_cache: LayerCache,
        device: torch.device,
        *,
        log=None,
    ) -> C1GpuResidentStore:
        if device.type != "cuda":
            raise ValueError(f"GPU store requires cuda, got {device}")
        t0 = time.perf_counter()
        paths = sorted({it.labels_path.resolve() for it in items})
        planes_list = [np.stack(layer_cache.get(p), axis=0) for p in paths]
        stack = np.stack(planes_list, axis=0).astype(np.uint8, copy=False)
        crops = torch.from_numpy(stack).to(device, non_blocking=True)
        if log:
            mb = crops.numel() / 1e6
            log(
                f"  c1 gpu_store: {len(paths)} crops "
                f"({mb:.0f} M uint8 on {device}, {time.perf_counter() - t0:.1f}s)"
            )
        return cls(crops)


def _cell_pos_from_plane(plane: torch.Tensor, task: C1TaskSpec, g: int, h: int) -> torch.Tensor:
    s = h // g
    pos = c1_positive_pixels(plane, task)
    blk = pos.reshape(-1, g, s, g, s).permute(0, 1, 3, 2, 4).reshape(-1, g, g, s * s)
    cnt = blk.sum(dim=-1)
    if task.downsample == "any":
        return cnt > 0
    if task.downsample == "majority":
        return cnt >= (s * s + 1) // 2
    if task.downsample == "centre":
        return plane[:, s // 2 :: s, s // 2 :: s][:, :g, :g]
    raise ValueError(task.downsample)


def build_image_from_crops(
    layers_u8: torch.Tensor,
    encoding: Encoding,
    *,
    marker_col: torch.Tensor | None = None,
    marker_row: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B,3,H,W) uint8 class planes -> model input (B,C,H,W) float.

    Split out of build_batch_from_crops so the c2 loader builds the identical input
    (the extra plane stays zero: c1/c2 have no marker, but in_chans must not change).
    """
    st_dev = layers_u8.device
    b, _, h, w = layers_u8.shape
    if encoding == "scalar":
        img = torch.zeros(b, 4, h, w, device=st_dev, dtype=torch.float32)
        img[:, :3] = layers_u8.float() / _SCALAR_SCALE
    elif encoding == "onehot":
        img = torch.zeros(b, 13, h, w, device=st_dev, dtype=torch.float32)
        for li in range(3):
            pl = layers_u8[:, li].long()
            img[:, li * 4 : (li + 1) * 4] = F.one_hot(pl, NUM_CLASSES_PER_LAYER).permute(0, 3, 1, 2).float()
    elif encoding == "onehot_v4":
        img = torch.zeros(b, 16, h, w, device=st_dev, dtype=torch.float32)
        img[:, :7] = F.one_hot(layers_u8[:, 0].long(), NUM_NOISE7_CLASSES).permute(0, 3, 1, 2).float()
        img[:, 7:11] = F.one_hot(layers_u8[:, 1].long(), NUM_CLASSES_PER_LAYER).permute(0, 3, 1, 2).float()
        img[:, 11:15] = F.one_hot(layers_u8[:, 2].long(), NUM_CLASSES_PER_LAYER).permute(0, 3, 1, 2).float()
    else:
        raise ValueError(encoding)
    if marker_col is not None and marker_row is not None:
        mc = marker_col.view(b, -1)  # (B,) or (B,K); -1 = unused slot
        mr = marker_row.view(b, -1)
        # Vectorised paint: no per-marker .item() syncs, no CPU planes, no H2D copies
        # (the loop version cost ~17 ms per batch of 16, more than the model step).
        mc = mc.to(img.device); mr = mr.to(img.device)
        ok = (mc >= 0) & (mr >= 0)
        radius = int(_ch.MARKER_RADIUS_PX)  # read at call time: harness/data.py overrides it
        if radius <= 0:
            bi = torch.arange(b, device=img.device).view(b, 1).expand_as(mc)
            img[bi[ok], -1, mr[ok], mc[ok]] = 1.0
        else:
            yy = torch.arange(h, device=img.device).view(1, 1, h, 1)
            xx = torch.arange(w, device=img.device).view(1, 1, 1, w)
            cy = mr.view(b, -1, 1, 1); cx = mc.view(b, -1, 1, 1)
            disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius   # (B,K,H,W)
            disc = disc & ok.view(b, -1, 1, 1)
            img[:, -1] = torch.maximum(img[:, -1], disc.any(dim=1).float())
    return img


def build_batch_from_crops(
    layers_u8: torch.Tensor,
    task_ids: torch.Tensor | None,
    task: C1TaskSpec | list[C1TaskSpec],
    *,
    encoding: Encoding,
    grid_size: int,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """layers_u8 (B,3,H,W); task single spec or list + task_ids (B,) for multi."""
    st_dev = layers_u8.device
    b, _, h, h2 = layers_u8.shape
    g = grid_size
    if h != img_size or h2 != img_size:
        raise ValueError("layer spatial size mismatch")
    specs = task if isinstance(task, list) else None
    single = task if isinstance(task, C1TaskSpec) else None
    tgt = torch.zeros(b, g, g, dtype=torch.long, device=st_dev)
    img = build_image_from_crops(layers_u8, encoding)
    if single is not None:
        plane = layers_u8[:, single.layer_index]
        cell_pos = _cell_pos_from_plane(plane, single, g, h)
        tgt[cell_pos] = FOREGROUND_CLASS
        return img, tgt
    assert specs is not None and task_ids is not None
    for i in range(b):
        spec = specs[int(task_ids[i].item())]
        plane = layers_u8[i, spec.layer_index]
        cell_pos = _cell_pos_from_plane(plane.unsqueeze(0), spec, g, h)[0]
        tgt[i][cell_pos] = FOREGROUND_CLASS
    return img, tgt


class C1GpuBatchLoader:
    """Single-task L5a: returns (img, tgt)."""

    def __init__(
        self,
        store: C1GpuResidentStore,
        crop_ix: torch.Tensor,
        index: C1CropIndex,
        batch_size: int,
        seed: int,
    ):
        self.store = store
        self.crop_ix = crop_ix
        self.index = index
        self.B = batch_size
        self._gen = torch.Generator(device=store.device)
        self._gen.manual_seed(seed)

    def __len__(self) -> int:
        return int(self.crop_ix.shape[0])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.crop_ix.shape[0]
        ix = torch.randint(0, n, (self.B,), device=self.store.device, generator=self._gen)
        return self._build_batch(self.crop_ix[ix])

    def batch_from_crop_indices(self, crop_ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._build_batch(crop_ix.to(device=self.store.device, dtype=torch.long))

    def _build_batch(self, crop_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layers_u8 = self.store.crops[crop_ids]
        return build_batch_from_crops(
            layers_u8,
            None,
            self.index.task,
            encoding=self.index.encoding,
            grid_size=self.index.grid_size,
            img_size=self.index.img_size,
        )


class C1MultiTaskGpuBatchLoader:
    """L5b+: returns (img, tgt, task_id)."""

    def __init__(
        self,
        store: C1GpuResidentStore,
        crop_ix: torch.Tensor,
        task_specs: list[C1TaskSpec],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        seed: int,
        *,
        fixed_pairs: list[tuple[int, int]] | None = None,
        train_task_indices: list[int] | None = None,
    ):
        self.store = store
        self.crop_ix = crop_ix
        self.task_specs = task_specs
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self._gen = torch.Generator(device=store.device)
        self._gen.manual_seed(seed)
        self._fixed_pairs = fixed_pairs
        self._train_task_indices = train_task_indices
        if train_task_indices is not None:
            self._train_task_ix_t = torch.tensor(train_task_indices, dtype=torch.long, device=store.device)
        else:
            self._train_task_ix_t = None
        if fixed_pairs is not None:
            self._pair_crop = torch.tensor([p[0] for p in fixed_pairs], dtype=torch.long, device=store.device)
            self._pair_task = torch.tensor([p[1] for p in fixed_pairs], dtype=torch.long, device=store.device)

    def __len__(self) -> int:
        if self._fixed_pairs is not None:
            return len(self._fixed_pairs)
        return int(self.crop_ix.shape[0])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._fixed_pairs is not None:
            n = self._pair_crop.shape[0]
            ix = torch.randint(0, n, (self.B,), device=self.store.device, generator=self._gen)
            item_ix = self._pair_crop[ix]
            task_id = self._pair_task[ix]
        else:
            n = self.crop_ix.shape[0]
            item_ix = torch.randint(0, n, (self.B,), device=self.store.device, generator=self._gen)
            if self._train_task_ix_t is not None:
                pick = torch.randint(0, self._train_task_ix_t.shape[0], (self.B,), device=self.store.device, generator=self._gen)
                task_id = self._train_task_ix_t[pick]
            else:
                task_id = torch.randint(0, len(self.task_specs), (self.B,), device=self.store.device, generator=self._gen)
        crop_ids = self.crop_ix[item_ix]
        layers_u8 = self.store.crops[crop_ids]
        img, tgt = build_batch_from_crops(
            layers_u8,
            task_id,
            self.task_specs,
            encoding=self.encoding,
            grid_size=self.grid_size,
            img_size=self.img_size,
        )
        return img, tgt, task_id
