"""Load one c0 sample (worldsnap jsonl index or synthetic pool)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from spatial_data.channels import Encoding, build_input_from_layers
from spatial_data.dataset_c0 import C0Item, default_task_paths
from spatial_data.layer_cache import LayerCache
from spatial_data.targets import c0_target_grid

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SampleMeta:
    split: Split
    item_index: int
    crop_id: str | None
    marker_row: int
    marker_col: int
    encoding: Encoding
    img_size: int
    grid_size: int
    source: Literal["worldsnap", "synthetic"]
    item_id: str | None = None


@dataclass
class LoadedSample:
    img: torch.Tensor
    target: torch.Tensor
    meta: SampleMeta
    label_planes: tuple[np.ndarray, np.ndarray, np.ndarray] | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_district_dir(data_root: Path | None = None) -> Path:
    root = data_root or repo_root() / "data"
    return root / "amsterdam" / "de_pijp"


def _read_jsonl_line(path: Path, index: int) -> dict:
    if index < 0:
        raise IndexError(f"item index must be >= 0, got {index}")
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if n == index:
            return json.loads(line)
        n += 1
    raise IndexError(f"item {index} out of range for {path} ({n} items)")


def load_worldsnap_sample(
    *,
    split: Split,
    item_index: int,
    encoding: Encoding = "scalar",
    img_size: int = 256,
    grid_size: int = 64,
    district_dir: Path | None = None,
    cropset: str = "v2",
    taskset: str = "t0_point_v0",
) -> LoadedSample:
    district = district_dir or default_district_dir()
    crops_dir = district / "crops" / cropset
    paths = default_task_paths(district, taskset)
    jsonl = paths[split]
    rec = _read_jsonl_line(jsonl, item_index)
    mc, mr = rec["marker_256"]
    crop_id = rec.get("crop_id")
    item_id = rec.get("item_id")
    labels_path = crops_dir / split / f"{crop_id}_labels.npz"
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    item = C0Item(labels_path, int(mc), int(mr))
    cache = LayerCache()
    cache.warm({labels_path})
    planes = cache.get(labels_path)
    img = build_input_from_layers(
        planes, encoding=encoding, marker_row=int(mr), marker_col=int(mc),
    )
    tgt = c0_target_grid(grid_size, img_size, int(mr), int(mc))
    meta = SampleMeta(
        split=split,
        item_index=item_index,
        crop_id=crop_id,
        marker_row=int(mr),
        marker_col=int(mc),
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
        source="worldsnap",
        item_id=item_id,
    )
    return LoadedSample(img=img, target=tgt, meta=meta, label_planes=planes)


def _synthetic_one(rng: random.Random, h: int, g: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    layers = torch.tensor(rng.choices([0.0, 1 / 3, 2 / 3, 1.0], k=3 * h * h)).view(3, h, h)
    mr, mc = rng.randrange(h), rng.randrange(h)
    marker = torch.zeros(1, h, h)
    marker[0, mr, mc] = 1.0
    img = torch.cat([layers, marker], dim=0)
    tgt = torch.zeros(g, g, dtype=torch.long)
    tgt[mr * g // h, mc * g // h] = 1
    return img, tgt, mr, mc


def load_synthetic_sample(
    *,
    split: Split,
    item_index: int,
    seed: int = 1337,
    img_size: int = 256,
    grid_size: int = 64,
) -> LoadedSample:
    split_seed = seed if split == "train" else seed + 1
    rng = random.Random(split_seed)
    for _ in range(item_index):
        _synthetic_one(rng, img_size, grid_size)
    img, tgt, mr, mc = _synthetic_one(rng, img_size, grid_size)
    planes = tuple(
        (img[i].numpy() * 3).round().astype(np.int64).clip(0, 3) for i in range(3)
    )
    meta = SampleMeta(
        split=split,
        item_index=item_index,
        crop_id=None,
        marker_row=mr,
        marker_col=mc,
        encoding="scalar",
        img_size=img_size,
        grid_size=grid_size,
        source="synthetic",
    )
    return LoadedSample(img=img, target=tgt, meta=meta, label_planes=planes)
