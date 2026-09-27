"""Marker positions per crop (t0_point_v0), aligned with sorted *_labels.npz order."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _markers_by_crop_id(jsonl_path: Path) -> dict[str, list[tuple[int, int]]]:
    by_crop: dict[str, list[tuple[int, int]]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec["crop_id"]
        col, row = rec["marker_256"]
        by_crop.setdefault(cid, []).append((int(col), int(row)))
    return by_crop


def marker_table_for_split(
    crops_dir: Path,
    district_dir: Path,
    split: str,
    *,
    taskset: str = "t0_point_v0",
    pick: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    """(cols, rows) uint16 length N, one marker per crop file in sorted glob order."""
    files = sorted((crops_dir / split).glob("*_labels.npz"))
    if not files:
        raise FileNotFoundError(f"no crops under {crops_dir}/{split}")
    by_crop = _markers_by_crop_id(district_dir / "tasks" / taskset / f"{split}.jsonl")
    cols, rows = [], []
    for f in files:
        cid = f.stem.replace("_labels", "")
        opts = by_crop.get(cid)
        if not opts:
            raise KeyError(f"no t0 marker for crop {cid!r} in {taskset}")
        if pick == "first":
            c, r = opts[0]
        else:
            raise ValueError(pick)
        cols.append(c)
        rows.append(r)
    return np.array(cols, dtype=np.int64), np.array(rows, dtype=np.int64)


def marker_tensors(
    crops_dir: Path,
    district_dir: Path,
    split: str,
    device: torch.device,
    **kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, r = marker_table_for_split(crops_dir, district_dir, split, **kw)
    return torch.from_numpy(c).to(device), torch.from_numpy(r).to(device)
