"""Load one c1 worldsnap crop + task mask."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from spatial_data.c1_tasks import get_c1_task
from spatial_data.channels import Encoding, build_input_from_layers
from spatial_data.dataset_c1 import list_crop_items
from spatial_data.layer_cache import LayerCache
from spatial_data.targets import c1_mask_grid_from_plane

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class C1SampleMeta:
    split: Split
    crop_index: int
    crop_id: str
    task_key: str
    encoding: Encoding
    img_size: int
    grid_size: int


@dataclass
class C1LoadedSample:
    img: torch.Tensor
    target: torch.Tensor
    meta: C1SampleMeta
    label_planes: tuple[np.ndarray, np.ndarray, np.ndarray]


def load_c1_worldsnap_sample(
    *,
    split: Split,
    crop_index: int,
    task_key: str = "building",
    encoding: Encoding = "scalar",
    img_size: int = 256,
    grid_size: int = 64,
    district_dir: Path | None = None,
    cropset: str = "v2",
) -> C1LoadedSample:
    from analysis.load_sample import default_district_dir, repo_root

    district = district_dir or default_district_dir()
    crops_dir = district / "crops" / cropset
    task = get_c1_task(task_key, cropset)
    items = list_crop_items(crops_dir, split)
    if crop_index < 0 or crop_index >= len(items):
        raise IndexError(f"crop_index {crop_index} out of range (0..{len(items) - 1})")
    item = items[crop_index]
    crop_id = item.labels_path.stem.replace("_labels", "")
    cache = LayerCache()
    cache.warm({item.labels_path})
    planes = cache.get(item.labels_path)
    li = task.layer_index
    tgt = c1_mask_grid_from_plane(
        torch.from_numpy(planes[li].astype(np.int64)),
        task,
        grid_size=grid_size,
        img_size=img_size,
    )
    h, w = planes[0].shape
    img = build_input_from_layers(planes, encoding=encoding, marker_row=h // 2, marker_col=w // 2)
    img[-1].zero_()
    meta = C1SampleMeta(
        split=split,
        crop_index=crop_index,
        crop_id=crop_id,
        task_key=task_key,
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
    )
    return C1LoadedSample(img=img, target=tgt, meta=meta, label_planes=planes)
