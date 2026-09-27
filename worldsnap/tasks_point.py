"""T0 "point read" task items: "what is the {surface|noise} class at this point?".

One item = one crop, one layer, one pixel, one class word. Points are sampled
CLASS-BALANCED (answer class uniform among the classes present in that crop's
1024 px layer, then a uniform random pixel of that class), because the raw
pixel distribution is dominated by buildings and by "below 55 dB".

Every item carries both model forms of the same question -- Track A (legend
text + natural-language prompt, coordinates normalized to 0-1000) and Track B
(task token + marker pixel in the 256 px frame + class token) -- plus the
distance to the nearest differently-classed pixel, so accuracy can be
stratified by how close to a boundary the point sits instead of dropping hard
points, and the answer read off the 256 px frame, which near a boundary may
disagree with the 1024 px answer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rig
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .config import DISTRICTS, District
from .crops import CROPSET, MEANINGS

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TASKSET = "t0_point_v0"
#: sub-task -> (layer key in the crop npz, phrase used in the prompt)
TASKS = {"surface_at": ("surface", "surface class"),
         "noise_at": ("noise", "road-traffic noise band")}
QUESTION = ("What is the {phrase} at the point (x={x}, y={y})? Coordinates are normalized "
            "to 0-1000, origin top-left, x to the right, y down. "
            "Answer with exactly one of: {options}.")


def _boundary_dist(arr: np.ndarray, cls: int, cache: dict) -> np.ndarray:
    """Per-pixel distance to the nearest pixel of a *different* class (class ``cls`` only)."""
    if cls not in cache:
        cache[cls] = distance_transform_edt(arr == cls)
    return cache[cls]


def _items_for_crop(meta: dict, labels, per_crop: int, rng: random.Random,
                    legend_text: str) -> list[dict]:
    out = []
    for task, (layer, phrase) in TASKS.items():
        fine, coarse = labels[f"{layer}_1024"], labels[f"{layer}_256"]
        options = ", ".join(f'"{w}"' for w in MEANINGS[layer].values())
        present = sorted(int(c) for c in np.unique(fine))
        pix = {c: np.flatnonzero(fine == c) for c in present}
        dist_cache: dict[int, np.ndarray] = {}
        for j in range(per_crop):
            cls = present[rng.randrange(len(present))]
            flat = int(pix[cls][rng.randrange(pix[cls].size)])
            row, col = divmod(flat, 1024)
            c256, r256 = col // 4, row // 4
            a256 = int(coarse[r256, c256])
            x, y = round(col / 1023 * 1000), round(row / 1023 * 1000)
            out.append({
                "item_id": f"{meta['crop_id']}_{task}_{j:02d}", "crop_id": meta["crop_id"],
                "split": meta["split"], "task": task, "layer": layer,
                "point_1024": [col, row], "point_256": [c256, r256], "x": x, "y": y,
                "answer_class": cls, "answer": MEANINGS[layer][cls],
                "answer_256_class": a256, "answer_256": MEANINGS[layer][a256],
                "agree_1024_256": a256 == cls,
                "boundary_dist_px": round(float(_boundary_dist(fine, cls, dist_cache)[row, col]), 3),
                "prompt": legend_text + QUESTION.format(phrase=phrase, x=x, y=y, options=options),
                "target": MEANINGS[layer][cls],
                "task_token": f"<{task}>", "marker_256": [c256, r256],
                "target_token": f"<{layer}:{cls}>",
            })
    return out


def build_tasks(district: District, per_crop: int, seed: int,
                data_root: Path = DATA_ROOT) -> None:
    t0 = time.time()
    ddir = data_root / district.city / district.name
    cdir = ddir / "crops" / CROPSET
    out = ddir / "tasks" / TASKSET
    out.mkdir(parents=True, exist_ok=True)
    legend_text = json.loads((cdir / "legend.json").read_text())["legend_text"]

    crops = [json.loads(l) for l in (cdir / "index.jsonl").read_text().splitlines() if l]
    by_split: dict[str, list[dict]] = {}
    for rec in crops:
        by_split.setdefault(rec["split"], []).append(rec)

    stats: dict[tuple, int] = {}
    disagree = {s: [0, 0] for s in by_split}
    dists = {s: [] for s in by_split}
    for split, recs in by_split.items():
        rng = random.Random(f"{seed}:{split}")
        lines = []
        for rec in recs:
            cid = rec["crop_id"]
            meta = {"crop_id": cid, "split": split}
            with np.load(cdir / split / f"{cid}_labels.npz") as labels:
                items = _items_for_crop(meta, labels, per_crop, rng, legend_text)
            for it in items:
                stats[(split, it["task"], it["answer_class"])] = \
                    stats.get((split, it["task"], it["answer_class"]), 0) + 1
                disagree[split][0] += not it["agree_1024_256"]
                disagree[split][1] += 1
                dists[split].append(it["boundary_dist_px"])
                lines.append(json.dumps(it, ensure_ascii=False))
        (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(lines)} items from {len(recs)} crops "
              f"-> {(out / f'{split}.jsonl').stat().st_size / 1e6:.1f} MB")

    for split in by_split:
        for task, (layer, _) in TASKS.items():
            row = "  ".join(f"{c}:{MEANINGS[layer][c]}={stats.get((split, task, c), 0)}"
                            for c in range(4))
            print(f"  {split:5s} {task:10s} {row}")
        n_bad, n_tot = disagree[split]
        p10, p50, p90 = np.percentile(dists[split], [10, 50, 90])
        print(f"  {split:5s} agree_1024_256 false: {n_bad}/{n_tot} = {n_bad / n_tot:.4f}; "
              f"boundary_dist_px p10/p50/p90 = {p10:.1f}/{p50:.1f}/{p90:.1f}")
    _examples(ddir, cdir, out, seed)
    print(f"{time.time() - t0:.1f} s")


def _examples(ddir: Path, cdir: Path, out: Path, seed: int, n: int = 8) -> None:
    """One inspection figure: n random test items, ring + crosshair drawn HERE only."""
    items = [json.loads(l) for l in (out / "test.jsonl").read_text().splitlines() if l]
    picks = random.Random(seed).sample(items, n)
    # Height: square axes (4.2 in) plus room for the 3-line titles, or row 2's title
    # would be drawn over row 1's image now that the axes are clamped to the crop.
    fig, axes = plt.subplots(2, n // 2, figsize=(4.2 * (n // 2), 10.4), dpi=100)
    for it, ax in zip(picks, axes.ravel()):
        rgb = np.asarray(Image.open(cdir / "test" / f"{it['crop_id']}_rgb.png"))
        col, row = it["point_1024"]
        ax.imshow(rgb, interpolation="nearest")
        ax.add_patch(plt.Circle((col, row), 34, ec="white", fc="none", lw=2.2))
        ax.add_patch(plt.Circle((col, row), 34, ec="black", fc="none", lw=0.9))
        ax.plot([col - 70, col - 44], [row, row], color="white", lw=2.0)
        ax.plot([col + 44, col + 70], [row, row], color="white", lw=2.0)
        ax.plot([col, col], [row - 70, row - 44], color="white", lw=2.0)
        ax.plot([col, col], [row + 44, row + 70], color="white", lw=2.0)
        ax.set_title(f"{it['crop_id']}  {it['task']}\nx={it['x']} y={it['y']} "
                     f"(px {col},{row})\nanswer: {it['answer']}   "
                     f"boundary {it['boundary_dist_px']:.1f} px", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        # Clamp to the image: a marker near an edge would otherwise widen the axes
        # and leave white bands around the crop.
        ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
    fig.suptitle(f"tasks/{TASKSET} - 8 random test items (marker drawn for inspection only; "
                 "R noise, G surface, B establishments)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    (ddir / "overlays").mkdir(exist_ok=True)
    fig.savefig(ddir / "overlays" / f"{TASKSET}_examples.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"inspection figure: overlays/{TASKSET}_examples.png")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="worldsnap.tasks_point", description=__doc__)
    p.add_argument("--district", required=True, choices=sorted(DISTRICTS))
    p.add_argument("--per-crop", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args(argv)
    build_tasks(DISTRICTS[args.district], args.per_crop, args.seed, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
