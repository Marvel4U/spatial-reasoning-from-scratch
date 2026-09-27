"""c0 samples from worldsnap crops + task items (marker echo on real layers)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .channels import Encoding, build_input_from_layers, build_input_from_npz
from .layer_cache import LayerCache
from .targets import c0_target_grid


@dataclass(frozen=True)
class C0Item:
    labels_path: Path
    marker_col: int
    marker_row: int


def load_c0_items(
    jsonl_path: Path,
    crops_dir: Path,
    *,
    max_items: int | None = None,
) -> list[C0Item]:
    items: list[C0Item] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        split = rec["split"]
        crop_id = rec["crop_id"]
        col, row = rec["marker_256"]
        path = crops_dir / split / f"{crop_id}_labels.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append(C0Item(path, int(col), int(row)))
        if max_items is not None and len(items) >= max_items:
            break
    return items


def load_c0_item_tensor(
    item: C0Item,
    *,
    encoding: Encoding,
    grid_size: int,
    img_size: int,
    layer_cache: LayerCache | None = None,
):
    if layer_cache is not None:
        planes = layer_cache.get(item.labels_path)
        img = build_input_from_layers(
            planes,
            encoding=encoding,
            marker_row=item.marker_row,
            marker_col=item.marker_col,
        )
    else:
        with np.load(item.labels_path) as npz:
            img = build_input_from_npz(
                npz,
                encoding=encoding,
                marker_row=item.marker_row,
                marker_col=item.marker_col,
            )
    tgt = c0_target_grid(
        grid_size, img_size, item.marker_row, item.marker_col,
    )
    return img, tgt


class C0JsonlIndex:
    """c0 loader: jsonl items; optional in-RAM layer cache or full tensor materialize."""

    def __init__(
        self,
        items: list[C0Item],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        *,
        layer_cache: LayerCache | None = None,
        materialized: list[tuple] | None = None,
    ):
        if not items:
            raise ValueError("empty item list")
        self.items = items
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.layer_cache = layer_cache
        self._materialized = materialized

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, rng: random.Random):
        if self._materialized is not None:
            return self._materialized[rng.randrange(len(self._materialized))]
        item = self.items[rng.randrange(len(self.items))]
        return load_c0_item_tensor(
            item,
            encoding=self.encoding,
            grid_size=self.grid_size,
            img_size=self.img_size,
            layer_cache=self.layer_cache,
        )


def materialize_c0_tensors(
    items: list[C0Item],
    encoding: Encoding,
    grid_size: int,
    img_size: int,
    layer_cache: LayerCache,
    *,
    log=None,
) -> list[tuple]:
    import time
    t0 = time.perf_counter()
    out: list[tuple] = []
    step = max(1, len(items) // 10)
    for i, item in enumerate(items):
        out.append(
            load_c0_item_tensor(
                item,
                encoding=encoding,
                grid_size=grid_size,
                img_size=img_size,
                layer_cache=layer_cache,
            )
        )
        if log and (i + 1) % step == 0:
            log(f"  materialize: {i + 1}/{len(items)}")
    if log:
        log(f"  materialize: done {len(items)} tensors in {time.perf_counter() - t0:.1f}s")
    return out


class C0BatchLoader:
    """Harness-compatible: .B and .next_batch()."""

    def __init__(self, index: C0JsonlIndex, batch_size: int, seed: int):
        self.index = index
        self.B = batch_size
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def next_batch(self):
        import torch
        imgs, tgts = [], []
        for _ in range(self.B):
            img, tgt = self.index.sample(self._rng)
            imgs.append(img)
            tgts.append(tgt)
        return torch.stack(imgs), torch.stack(tgts)


def default_task_paths(district_dir: Path, taskset: str) -> dict[str, Path]:
    tdir = district_dir / "tasks" / taskset
    return {
        "train": tdir / "train.jsonl",
        "val": tdir / "val.jsonl",
        "test": tdir / "test.jsonl",
    }
