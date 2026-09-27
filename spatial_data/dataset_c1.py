"""c1: worldsnap crops + task-defined masks (L5a: one task, no task embedding)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .c1_tasks import C1TaskSpec, get_c1_task
from .channels import Encoding, build_input_from_layers
from .layer_cache import LayerCache
from .targets import c1_mask_grid_from_plane


@dataclass(frozen=True)
class C1CropItem:
    labels_path: Path


def list_crop_items(crops_dir: Path, split: str) -> list[C1CropItem]:
    split_dir = crops_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    paths = sorted(split_dir.glob("*_labels.npz"))
    if not paths:
        raise FileNotFoundError(f"no *_labels.npz under {split_dir}")
    return [C1CropItem(p) for p in paths]


def load_c1_item_tensor(
    item: C1CropItem,
    task: C1TaskSpec,
    *,
    encoding: Encoding,
    grid_size: int,
    img_size: int,
    layer_cache: LayerCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_cache is not None:
        planes = layer_cache.get(item.labels_path)
    else:
        with np.load(item.labels_path) as npz:
            from .channels import LAYER_KEYS
            planes = tuple(np.asarray(npz[k]) for k in LAYER_KEYS)
    li = task.layer_index
    tgt = c1_mask_grid_from_plane(
        torch.from_numpy(planes[li].astype(np.int64)),
        task,
        grid_size=grid_size,
        img_size=img_size,
    )
    h, w = planes[0].shape
    img = build_input_from_layers(
        planes,
        encoding=encoding,
        marker_row=h // 2,
        marker_col=w // 2,
    )
    img[-1].zero_()
    return img, tgt


class C1CropIndex:
    def __init__(
        self,
        items: list[C1CropItem],
        task: C1TaskSpec,
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        *,
        layer_cache: LayerCache | None = None,
        materialized: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        if not items:
            raise ValueError("empty crop list")
        self.items = items
        self.task = task
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.layer_cache = layer_cache
        self._materialized = materialized

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        if self._materialized is not None:
            return self._materialized[rng.randrange(len(self._materialized))]
        item = self.items[rng.randrange(len(self.items))]
        return load_c1_item_tensor(
            item,
            self.task,
            encoding=self.encoding,
            grid_size=self.grid_size,
            img_size=self.img_size,
            layer_cache=self.layer_cache,
        )


class C1BatchLoader:
    """Harness-compatible: .B and .next_batch()."""

    def __init__(self, index: C1CropIndex, batch_size: int, seed: int):
        self.index = index
        self.B = batch_size
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, tgts = [], []
        for _ in range(self.B):
            img, tgt = self.index.sample(self._rng)
            imgs.append(img)
            tgts.append(tgt)
        return torch.stack(imgs), torch.stack(tgts)


def resolve_c1_task(task_key: str, *, cropset: str = "v2") -> C1TaskSpec:
    return get_c1_task(task_key, cropset)


class C1MultiTaskBatchLoader:
    """L5b CPU path: (img, tgt, task_id)."""

    def __init__(
        self,
        items: list[C1CropItem],
        task_specs: list[C1TaskSpec],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        seed: int,
        *,
        layer_cache: LayerCache | None = None,
        fixed_pairs: list[tuple[int, int]] | None = None,
        train_task_indices: list[int] | None = None,
    ):
        self.items = items
        self.task_specs = task_specs
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.layer_cache = layer_cache
        self.B = batch_size
        self._rng = random.Random(seed)
        self._fixed_pairs = fixed_pairs
        self._train_task_indices = train_task_indices

    def __len__(self) -> int:
        return len(self._fixed_pairs) if self._fixed_pairs is not None else len(self.items)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, tgts, tids = [], [], []
        for _ in range(self.B):
            if self._fixed_pairs is not None:
                crop_i, tid = self._fixed_pairs[self._rng.randrange(len(self._fixed_pairs))]
            else:
                crop_i = self._rng.randrange(len(self.items))
                if self._train_task_indices is not None:
                    tid = self._train_task_indices[self._rng.randrange(len(self._train_task_indices))]
                else:
                    tid = self._rng.randrange(len(self.task_specs))
            img, tgt = load_c1_item_tensor(
                self.items[crop_i],
                self.task_specs[tid],
                encoding=self.encoding,
                grid_size=self.grid_size,
                img_size=self.img_size,
                layer_cache=self.layer_cache,
            )
            imgs.append(img)
            tgts.append(tgt)
            tids.append(tid)
        return torch.stack(imgs), torch.stack(tgts), torch.tensor(tids, dtype=torch.long)
