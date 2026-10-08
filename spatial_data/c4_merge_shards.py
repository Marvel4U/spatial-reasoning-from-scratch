"""Merge c4 route store shards (c4_routes_{split}_shard{k}of{n}.npz) into c4_routes_{split}.npz.
Usage: python -m spatial_data.c4_merge_shards --store DIR --split train"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--store", required=True); ap.add_argument("--split", default="train")
    a = ap.parse_args(); d = Path(a.store)
    parts = sorted(d.glob(f"c4_routes_{a.split}_shard*of*.npz"))
    assert parts, "no shards"
    zs = [np.load(p) for p in parts]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    np.savez_compressed(d / f"c4_routes_{a.split}.npz", **out)
    stats = [json.loads(p.with_name(p.stem + "_stats.json").read_text()) for p in parts]
    merged = {"n": int(len(out["origin"])), "shards": [p.name for p in parts],
              "seconds_max": max(s["seconds"] for s in stats),
              **{k: sum(s[k] for s in stats) for k in ("windows", "blocked", "gave_up", "path_hits_building", "stratum_full", "too_long")},
              "strata": list(np.bincount(out["stratum"], minlength=3).astype(int).tolist()),
              "mean_detour": float(out["detour_ratio"].mean()), "mean_path_m": float(out["path_px"].mean()),
              **{k: stats[0][k] for k in ("crop_px", "d_min_px", "d_max_px", "max_expansions")}}
    (d / f"c4_routes_{a.split}_stats.json").write_text(json.dumps(merged, indent=1))
    print(merged)


if __name__ == "__main__":
    main()
