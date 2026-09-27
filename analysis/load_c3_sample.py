"""Load one c3 worldsnap val crop + target."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from spatial_data.c3_markers import marker_table_for_split
from spatial_data.c3_targets import c3_targets_for_spec
from spatial_data.c3_tasks import C3TaskSpec, get_c3_task
from spatial_data.channels import Encoding
from spatial_data.dataset_c1 import list_crop_items
from spatial_data.dataset_c1_gpu import build_image_from_crops

Split = Literal["train", "val", "test"]
C3_NPZ = ("noise7_256", "surface_256", "estab_256")


@dataclass(frozen=True)
class C3SampleMeta:
    split: Split
    crop_index: int
    crop_id: str
    task_key: str
    encoding: Encoding
    img_size: int
    grid_size: int
    marker_row: int = 0
    marker_col: int = 0


@dataclass
class C3LoadedSample:
    img: torch.Tensor
    target: torch.Tensor
    meta: C3SampleMeta
    cond_ids: tuple[int, int]
    label_planes: tuple


def load_c3_worldsnap_sample(
    *,
    split: Split,
    crop_index: int,
    task_key: str,
    encoding: Encoding = "onehot_v4",
    img_size: int = 256,
    grid_size: int = 64,
    district_dir: Path | None = None,
    cropset: str = "v4",
) -> C3LoadedSample:
    from analysis.load_sample import default_district_dir

    district = district_dir or default_district_dir()
    crops_dir = district / "crops" / cropset
    spec: C3TaskSpec = get_c3_task(task_key)
    items = list_crop_items(crops_dir, split)
    if crop_index < 0 or crop_index >= len(items):
        raise IndexError(f"crop_index {crop_index} out of range (0..{len(items) - 1})")
    item = items[crop_index]
    crop_id = item.labels_path.stem.replace("_labels", "")
    z = np.load(item.labels_path)
    stack = np.stack([z[k] for k in C3_NPZ]).astype(np.uint8)
    rgb_planes = (z["noise_256"], z["surface_256"], z["estab_256"])
    layers = torch.from_numpy(stack).unsqueeze(0)
    mc = mr = None
    mcol, mrow = 0, 0
    if spec.uses_marker:
        cols, rows = marker_table_for_split(crops_dir, district, split)
        mcol, mrow = int(cols[crop_index]), int(rows[crop_index])
        mc = torch.tensor([mcol], dtype=torch.long)
        mr = torch.tensor([mrow], dtype=torch.long)
    img = build_image_from_crops(layers, encoding, marker_col=mc, marker_row=mr)[0]
    tgt = c3_targets_for_spec(layers, spec, grid_size=grid_size, marker_col=mc, marker_row=mr)[0]
    meta = C3SampleMeta(
        split=split,
        crop_index=crop_index,
        crop_id=crop_id,
        task_key=task_key,
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
        marker_row=mrow,
        marker_col=mcol,
    )
    return C3LoadedSample(
        img=img, target=tgt, meta=meta, cond_ids=spec.padded_cond_ids(), label_planes=rgb_planes,
    )
