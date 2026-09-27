"""Load one c2 worldsnap val crop + conjunction / single-atom target."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from spatial_data.c2_tasks import C2TaskSpec, get_c2_task
from spatial_data.channels import Encoding
from spatial_data.dataset_c1 import list_crop_items
from spatial_data.dataset_c2_gpu import build_image_from_crops, c2_targets_from_planes
from spatial_data.layer_cache import LayerCache

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class C2SampleMeta:
    split: Split
    crop_index: int
    crop_id: str
    task_key: str
    encoding: Encoding
    img_size: int
    grid_size: int


@dataclass
class C2LoadedSample:
    img: torch.Tensor
    target: torch.Tensor
    meta: C2SampleMeta
    cond_ids: tuple[int, int]
    label_planes: tuple  # noise, surface, estab planes for RGB panel A


def load_c2_worldsnap_sample(
    *,
    split: Split,
    crop_index: int,
    task_key: str,
    encoding: Encoding = "onehot",
    img_size: int = 256,
    grid_size: int = 64,
    district_dir: Path | None = None,
    cropset: str = "v3b",
) -> C2LoadedSample:
    from analysis.load_sample import default_district_dir

    district = district_dir or default_district_dir()
    crops_dir = district / "crops" / cropset
    spec: C2TaskSpec = get_c2_task(task_key)
    items = list_crop_items(crops_dir, split)
    if crop_index < 0 or crop_index >= len(items):
        raise IndexError(f"crop_index {crop_index} out of range (0..{len(items) - 1})")
    item = items[crop_index]
    crop_id = item.labels_path.stem.replace("_labels", "")
    cache = LayerCache()
    cache.warm({item.labels_path})
    planes = cache.get(item.labels_path)  # (noise, surface, estab) uint8 planes, same as c1 loader
    stack = np.stack(planes).astype(np.uint8)
    layers = torch.from_numpy(stack).unsqueeze(0)
    cond = torch.tensor([spec.padded_cond_ids()], dtype=torch.long)
    img = build_image_from_crops(layers, encoding)[0]
    tgt = c2_targets_from_planes(layers, cond, grid_size=grid_size)[0]
    meta = C2SampleMeta(
        split=split,
        crop_index=crop_index,
        crop_id=crop_id,
        task_key=task_key,
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
    )
    return C2LoadedSample(
        img=img, target=tgt, meta=meta, cond_ids=spec.padded_cond_ids(), label_planes=planes,
    )
