"""Compare c3 relative target rules on v4 noise7 (C3_PLAN §2, step 2).

Uses train crop origins + district 1 m planes (same pool as c3_relative_stats.py).
Reports empty-target rate, sidewalk target share, and local-only ceiling per rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MIN_SW_PX = 50
GRID = 64
MAJ = 8  # pixel-AND then 4x4 majority (match c2 cell rule on 256 px crops)


def _cell_target(pix: np.ndarray) -> np.ndarray:
    """bool (256,256) -> bool (64,64) majority."""
    blk = pix.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3).reshape(64, 64, 16)
    return blk.sum(-1) >= MAJ


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = a | b
    if not u.any():
        return 1.0
    return float((a & b).sum() / u.sum())


def _rank_tertile(v: np.ndarray) -> np.ndarray:
    """Per-crop ranks 0..n-1 on sidewalk pixels; return bool mask for lowest third."""
    order = np.argsort(v, kind="stable")
    n = len(v)
    k = max(1, n // 3)
    low = np.zeros(n, dtype=bool)
    low[order[:k]] = True
    return low


def build_target(sw: np.ndarray, v: np.ndarray, rule: str) -> np.ndarray | None:
    """Pixel bool target on full crop, or None if empty/degenerate."""
    if sw.sum() < MIN_SW_PX:
        return None
    vv = v[sw]
    med = int(np.median(vv))
    if rule == "median_strict":
        pix = sw & (v < med)
    elif rule == "at_or_below_median":
        pix = sw & (v <= med)
    elif rule == "quietest_tertile":
        low = np.zeros(sw.shape, dtype=bool)
        low[sw] = _rank_tertile(vv)
        pix = low
    elif rule == "below_mean_level":
        m = int(np.floor(vv.mean()))
        pix = sw & (v < m)
    elif rule == "quietest_level_present":
        pix = sw & (v == vv.min())
    else:
        raise ValueError(rule)
    if not pix.any():
        return None
    return pix


def global_local_ceiling(crops: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Best single threshold t for all crops: mean cell-IoU of (sidewalk & v < t) vs target."""
    if not crops:
        return 0.0
    top = max(int(v.max()) for _, v, _ in crops)
    best_mean = 0.0
    for t in range(1, top + 2):
        ious = [_iou(_cell_target(sw & (v < t)), tgt_cell) for sw, v, tgt_cell in crops]
        best_mean = max(best_mean, float(np.mean(ious)))
    return best_mean


def sweep(cropset: str, split: str = "train") -> None:
    ddir = REPO / "data" / "amsterdam" / "de_pijp"
    st = np.load(ddir / "crops" / cropset / "district_labels_1m.npz")
    surf, nz4, nz7 = st["surface"], st["noise"], st["noise7"]
    idx = [x for x in map(json.loads, open(ddir / f"crops/{cropset}/index.jsonl")) if x["split"] == split]
    rules = (
        "median_strict",
        "at_or_below_median",
        "quietest_tertile",
        "below_mean_level",
        "quietest_level_present",
    )
    print(f"c3 relative rule sweep  cropset={cropset}  split={split}  crops={len(idx)}")
    print(f"noise planes: 4-band merged vs noise7 (0..{int(nz7.max())})\n")
    for nz_name, nz in (("noise7", nz7), ("noise_4band", nz4)):
        rows: list[dict] = []
        for rule in rules:
            sw_shares: list[float] = []
            n_used = n_empty = 0
            eval_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for x in idx:
                dx, dy = x["origin_m_from_district_nw"]
                sl = (slice(dy, dy + 256), slice(dx, dx + 256))
                sw = surf[sl] == 2
                v = nz[sl]
                pix = build_target(sw, v, rule)
                if pix is None:
                    n_empty += 1
                    continue
                n_used += 1
                sw_shares.append(pix[sw].mean())
                eval_triples.append((sw, v, _cell_target(pix)))
            sh = np.array(sw_shares) if sw_shares else np.array([0.0])
            rows.append({
                "rule": rule,
                "n_used": n_used,
                "empty_frac": n_empty / max(len(idx), 1),
                "sw_p10": float(np.percentile(sh, 10)),
                "sw_p50": float(np.percentile(sh, 50)),
                "sw_p90": float(np.percentile(sh, 90)),
                "local_ceiling": global_local_ceiling(eval_triples),
            })
        print(f"=== {nz_name} ===")
        print(f"{'rule':26s} {'empty%':>7s} {'sw p50':>7s} {'local ceil':>10s}  n_used")
        for r in rows:
            print(f"{r['rule']:26s} {100*r['empty_frac']:6.1f}% {100*r['sw_p50']:6.1f}% {r['local_ceiling']:10.3f}  {r['n_used']}")
        best = max(rows, key=lambda r: (1 - r["local_ceiling"], -r["empty_frac"]))
        print(f"  → largest headroom vs local-only: {best['rule']} (ceiling {best['local_ceiling']:.3f})\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cropset", default="v4")
    p.add_argument("--split", default="train")
    args = p.parse_args()
    sweep(args.cropset, args.split)


if __name__ == "__main__":
    main()
