"""Plot harness diagnostics for c3 within (or any c3 run with eval.jsonl).

Example:
  python -m analysis.c3_within_diagnostics_plot --run-id within_50m_6k_rand2
  python -m analysis.c3_within_diagnostics_plot --run-id c3_within_20m_2k --compare c3_quieter_sidewalk_median_strict_2k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from diagnostics.paths import resolve_eval, resolve_tier1, summary_path


def _load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    for r in rows:
        if key in r:
            steps.append(r["step"])
            vals.append(float(r[key]))
    return np.array(steps, dtype=np.int64), np.array(vals, dtype=np.float64)


def _mean_keys(rows: list[dict], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([]), np.array([])
    keys = [k for k in rows[0] if k.startswith(prefix)]
    if not keys:
        return np.array([]), np.array([])
    steps = np.array([r["step"] for r in rows], dtype=np.int64)
    mat = np.stack([[float(r[k]) for k in keys] for r in rows])
    return steps, mat.mean(axis=1)


def _min_layer_entropy(rows: list[dict], layer: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"attn_entropy/{layer}/"
    keys = [k for k in rows[0] if k.startswith(prefix)] if rows else []
    if not keys:
        return np.array([]), np.array([])
    steps = np.array([r["step"] for r in rows], dtype=np.int64)
    mat = np.stack([[float(r[k]) for k in keys] for r in rows])
    return steps, mat.min(axis=1)


def plot_run(
    run_id: str,
    *,
    compare_id: str | None,
    out_path: Path,
) -> None:
    eval_p = resolve_eval(run_id)
    eval_rows = _load_jsonl(eval_p)
    tier_rows = _load_jsonl(resolve_tier1(run_id))
    if not eval_rows:
        raise FileNotFoundError(f"no eval log for {run_id!r} (looked at {eval_p})")

    compare_eval: list[dict] = []
    if compare_id:
        compare_eval = _load_jsonl(resolve_eval(compare_id))

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(f"c3 diagnostics: {run_id}" + (f" vs {compare_id}" if compare_id else ""), fontsize=12)

    ax = axes[0, 0]
    s, v = _series(eval_rows, "IoU_fg")
    ax.plot(s, v, "o-", color="C0", label=run_id.split("_")[0] + "…", ms=4)
    ax.set_ylabel("val IoU_fg (within)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_xlabel("step")
    yhi = max(0.05, float(v.max()) * 1.25 + 0.02) if len(v) else 0.1
    ax.set_ylim(-0.005, yhi)
    ax.axhline(0, color="gray", lw=0.5)
    if compare_eval:
        s2, v2 = _series(compare_eval, "IoU_fg")
        ax2 = ax.twinx()
        ax2.plot(s2, v2, "o-", color="C2", alpha=0.85, label="median_strict", ms=4)
        ax2.set_ylabel("val IoU_fg (compare)", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")
        ax2.set_ylim(0, min(1.0, float(v2.max()) * 1.05 + 0.02) if len(v2) else 1.0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    else:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    s, v = _series(eval_rows, "val_loss_eval")
    ax.plot(s, v, "o-", color="C0", label="val (CE eval)", ms=4)
    s, v = _series(eval_rows, "train_loss_eval")
    ax.plot(s, v, "s--", color="C1", label="train (CE eval)", ms=3)
    ax.set_ylabel("loss @ eval")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for layer, col in ((0, "C3"), (1, "C4")):
        s, v = _series(eval_rows, f"resid_rms/{layer}")
        ax.plot(s, v, "o-", color=col, label=f"L{layer} resid_rms", ms=4)
    ax.set_ylabel("resid_rms (probe)")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    s, v = _series(eval_rows, "gelu_off/0")
    ax.plot(s, v, "o-", color="C5", label="gelu_off L0", ms=4)
    s, ent_min = _min_layer_entropy(eval_rows, 0)
    if len(s):
        ax2 = ax.twinx()
        ax2.plot(s, ent_min, "s--", color="C6", label="min attn_entropy L0", ms=3)
        ax2.set_ylabel("min head entropy L0", color="C6")
        ax2.tick_params(axis="y", labelcolor="C6")
    ax.set_ylabel("gelu_off L0")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    s, v = _mean_keys(eval_rows, "attn_to_marker/0/")
    ax.plot(s, v, "o-", color="C0", label="mean attn→marker L0", ms=4)
    s, v = _mean_keys(eval_rows, "attn_to_marker/1/")
    ax.plot(s, v, "s--", color="C1", label="mean attn→marker L1", ms=3)
    ax.axhline(1 / 256, color="gray", ls=":", lw=1, label="uniform (1/256)")
    ax.set_ylabel("attention mass on marker patch")
    ax.set_xlabel("step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if tier_rows:
        s, v = _series(tier_rows, "loss")
        ax.plot(s, v, color="C7", alpha=0.35, lw=0.8, label="train loss (tier1)")
        s, g = _series(tier_rows, "grad_norm/patch_embed/marker")
        ax2 = ax.twinx()
        ax2.plot(s, g, color="C8", lw=1.2, label="‖grad‖ marker embed")
        ax2.set_ylabel("marker grad norm", color="C8")
        ax2.tick_params(axis="y", labelcolor="C8")
        ax.set_ylabel("train loss")
        ax.set_xlabel("step")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no tier1.jsonl", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="c3_within_20m_2k")
    p.add_argument("--compare", default="c3_quieter_sidewalk_median_strict_2k")
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or summary_path(args.run_id)
    compare = None if args.no_compare else args.compare
    plot_run(args.run_id, compare_id=compare, out_path=out)


if __name__ == "__main__":
    main()
