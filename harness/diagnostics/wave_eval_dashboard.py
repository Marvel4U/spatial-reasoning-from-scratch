#!/usr/bin/env python3
"""Per-task wave metrics over training from diagnostics eval.jsonl (mix runs, plateau check)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_HARNESS = _REPO / "harness"
for p in (str(_REPO), str(_HARNESS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from diagnostics.plot_run import load_jsonl, _series, _steps

WAVE_TASKS = (
    "wave_first",
    "wave_step",
    "wave_reach",
    "wave_back",
    "wave_origin",
    "wave_draw",
)
FORWARD = ("wave_first", "wave_step", "wave_reach")
BACKWARD = ("wave_back", "wave_origin", "wave_draw")

TASK_COLORS = {
    "wave_first": "#1f77b4",
    "wave_step": "#ff7f0e",
    "wave_reach": "#2ca02c",
    "wave_back": "#d62728",
    "wave_origin": "#9467bd",
    "wave_draw": "#8c564b",
}


def _tasks_in_rows(rows: list[dict]) -> list[str]:
    found: set[str] = set()
    for r in rows:
        for t in WAVE_TASKS:
            if f"iou_dilated_{t}" in r:
                found.add(t)
    order = [t for t in WAVE_TASKS if t in found]
    if "wave_path" in {k.removeprefix("iou_dilated_") for r in rows for k in r if k.startswith("iou_dilated_wave")}:
        order.append("wave_path")
    return order


def _delta_window(rows: list[dict], key_fn, tasks: list[str], lo: int, hi: int) -> dict[str, float | None]:
    sub = [r for r in rows if lo <= int(r["step"]) <= hi]
    if len(sub) < 2:
        return {t: None for t in tasks}
    a, b = sub[0], sub[-1]
    out: dict[str, float | None] = {}
    for t in tasks:
        k = key_fn(t)
        va, vb = a.get(k), b.get(k)
        out[t] = (float(vb) - float(va)) if va is not None and vb is not None else None
    return out


def _write_summary(rows: list[dict], run_id: str, out_md: Path, tasks: list[str]) -> None:
    if not rows:
        return
    last = rows[-1]
    step = int(last["step"])
    lines = [
        f"# Wave eval summary — `{run_id}`",
        "",
        f"Last logged CE eval: **step {step}** ({len(rows)} eval rows, every 1000 steps).",
        "",
        "## Latest per-task (val, full grid in eval.jsonl)",
        "",
        "| task | dil IoU | plain IoU | new recall | copied recall | false FG | tok acc |",
        "|------|---------|-----------|------------|---------------|----------|---------|",
    ]
    for t in tasks:
        lines.append(
            f"| `{t}` "
            f"| {_fmt(last.get(f'iou_dilated_{t}'))} "
            f"| {_fmt(last.get(f'iou_{t}'))} "
            f"| {_fmt(last.get(f'c5_wave_new_recall_{t}'))} "
            f"| {_fmt(last.get(f'c5_wave_copied_recall_{t}'))} "
            f"| {_fmt(last.get(f'c5_wave_false_fg_{t}'))} "
            f"| {_fmt(last.get(f'c5_tok_acc_{t}'))} |"
        )
    has_rule = any(f"c5_wave_new_recall_rule_{t}" in last for t in tasks)
    if has_rule:
        lines.extend(
            [
                "",
                "## Decision rule (eval only, P(fg) > w/(1+w); strips still argmax)",
                "",
                "| task | τ | new (rule) | copied (rule) | false FG (rule) |",
                "|------|---|------------|---------------|-----------------|",
            ]
        )
        for t in tasks:
            lines.append(
                f"| `{t}` "
                f"| {_fmt(last.get(f'c5_wave_decision_tau_{t}'))} "
                f"| {_fmt(last.get(f'c5_wave_new_recall_rule_{t}'))} "
                f"| {_fmt(last.get(f'c5_wave_copied_recall_rule_{t}'))} "
                f"| {_fmt(last.get(f'c5_wave_false_fg_rule_{t}'))} |"
            )
    lines.extend(["", "## Change over late training (plateau check)", ""])
    if step >= 8000:
        for label, lo, hi in (
            ("steps 7k→10k", 7000, 10000),
            ("steps 10k→last", 10000, step),
            ("last 3k window", max(0, step - 3000), step),
        ):
            d_dil = _delta_window(rows, lambda t: f"iou_dilated_{t}", tasks, lo, hi)
            d_new = _delta_window(rows, lambda t: f"c5_wave_new_recall_{t}", tasks, lo, hi)
            lines.append(f"### Δ {label}")
            lines.append("")
            lines.append("| task | Δ dil IoU | Δ new recall |")
            lines.append("|------|-----------|--------------|")
            for t in tasks:
                lines.append(f"| `{t}` | {_fmt(d_dil.get(t), signed=True)} | {_fmt(d_new.get(t), signed=True)} |")
            lines.append("")
    lines.extend([
        "## How to read this",
        "",
        "- **new recall** = fraction of *new* target cells (not on input scratchpad) predicted fg — "
        "this is what iterative rollout needs; dil IoU can look fine while new recall stays ~0.",
        "- **copied recall** ≈ 1 means the model preserves scratchpad; extension failures show up in **new recall**.",
        "- **(rule)** columns appear when `c5_wave_eval_decision_rule=True`; they do not change pred strips or rollout.",
        "",
    ])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v, *, signed: bool = False) -> str:
    if v is None:
        return "—"
    if signed:
        return f"{v:+.4f}"
    return f"{float(v):.4f}"


def plot_wave_dashboard(rows: list[dict], run_id: str, out: Path) -> None:
    tasks = _tasks_in_rows(rows)
    if not tasks:
        raise SystemExit(f"no wave task metrics in eval rows for {run_id}")
    steps = _steps(rows)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=120)
    ax_loss, ax_dil = axes[0]
    ax_new, ax_copy = axes[1]
    ax_tok, ax_plain = axes[2]

    ax_loss.plot(steps, _series(rows, "train_loss_eval"), label="train CE", lw=1.2)
    ax_loss.plot(steps, _series(rows, "val_loss_eval"), label="val CE", lw=1.2)
    ax_loss.plot(steps, _series(rows, "mean_loss"), label="val CE (eval pass)", lw=1.0, ls="--", alpha=0.8)
    ax_loss.set_title("loss @ CE eval")
    ax_loss.legend(fontsize=7)
    ax_loss.set_xlabel("step")

    for t in tasks:
        c = TASK_COLORS.get(t, None)
        ax_dil.plot(steps, _series(rows, f"iou_dilated_{t}"), label=t.replace("wave_", ""), lw=1.4, color=c)
    ax_dil.set_ylim(-0.02, 1.02)
    ax_dil.set_title("dilated IoU by task")
    ax_dil.legend(fontsize=6, ncol=2)
    ax_dil.set_xlabel("step")

    for t in tasks:
        k = f"c5_wave_new_recall_{t}"
        if any(k in r for r in rows):
            ax_new.plot(steps, _series(rows, k), label=t.replace("wave_", ""), lw=1.4, color=TASK_COLORS.get(t))
    ax_new.set_ylim(-0.02, 1.02)
    ax_new.set_title("new-cell recall (extension)")
    ax_new.legend(fontsize=6, ncol=2)
    ax_new.set_xlabel("step")

    for t in tasks:
        k = f"c5_wave_copied_recall_{t}"
        if any(k in r for r in rows):
            ax_copy.plot(steps, _series(rows, k), label=t.replace("wave_", ""), lw=1.2, color=TASK_COLORS.get(t))
    ax_copy.set_ylim(-0.02, 1.02)
    ax_copy.set_title("copied scratchpad recall")
    ax_copy.legend(fontsize=6, ncol=2)
    ax_copy.set_xlabel("step")

    for t in tasks:
        k = f"c5_tok_acc_{t}"
        if any(k in r for r in rows):
            ax_tok.plot(steps, _series(rows, k), label=t.replace("wave_", ""), lw=1.2, color=TASK_COLORS.get(t))
    ax_tok.set_ylim(-0.02, 1.02)
    ax_tok.set_title("state token accuracy")
    ax_tok.legend(fontsize=6, ncol=2)
    ax_tok.set_xlabel("step")

    for t in tasks:
        ax_plain.plot(steps, _series(rows, f"iou_{t}"), label=t.replace("wave_", ""), lw=1.0, alpha=0.85, color=TASK_COLORS.get(t))
    ax_plain.set_ylim(-0.02, 1.02)
    ax_plain.set_title("plain IoU by task")
    ax_plain.legend(fontsize=6, ncol=2)
    ax_plain.set_xlabel("step")

    fwd_new = []
    for r in rows:
        vals = [r.get(f"c5_wave_new_recall_{t}") for t in FORWARD if t in tasks]
        vals = [float(v) for v in vals if v is not None]
        fwd_new.append(sum(vals) / len(vals) if vals else None)
    ax_new.plot(steps, fwd_new, color="k", lw=2.2, ls="--", label="forward mean")
    ax_new.legend(fontsize=6, ncol=2)

    fig.suptitle(f"{run_id}\nwave subtask val metrics (eval.jsonl)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_compare_forward(rows_a: list[dict], id_a: str, rows_b: list[dict], id_b: str, out: Path) -> None:
    """Overlay forward tasks: mix4 vs mix6 at same steps."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=110)
    metrics = (
        ("iou_dilated_", "dilated IoU"),
        ("c5_wave_new_recall_", "new recall"),
        ("c5_wave_copied_recall_", "copied recall"),
        ("iou_", "plain IoU"),
    )
    for ax, (prefix, title) in zip(axes.flat, metrics):
        for t in FORWARD:
            for rows, rid, ls in ((rows_a, id_a, "-"), (rows_b, id_b, "--")):
                key = f"{prefix}{t}" if prefix.startswith("c5") else f"{prefix}{t}"
                if not any(key in r for r in rows):
                    continue
                ax.plot(_steps(rows), _series(rows, key), ls=ls, lw=1.3, label=f"{t.replace('wave_','')} ({rid.split('_')[3]})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend(fontsize=5)
    fig.suptitle("Forward wave tasks: mix4 vs mix6", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Wave mix per-task eval dashboard from eval.jsonl")
    p.add_argument("run_id")
    p.add_argument("--compare", default=None, help="second run_id for forward-task overlay (e.g. mix4)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    run_dir = Path(config.diagnostics_dir) / args.run_id
    eval_path = run_dir / "eval.jsonl"
    rows = load_jsonl(eval_path)
    if not rows:
        raise SystemExit(f"no eval.jsonl at {eval_path}")
    out_dir = args.out_dir or run_dir
    png = out_dir / "wave_eval_dashboard.png"
    md = out_dir / "wave_eval_summary.md"
    tasks = _tasks_in_rows(rows)
    plot_wave_dashboard(rows, args.run_id, png)
    _write_summary(rows, args.run_id, md, tasks)
    print(f"wrote {png}")
    print(f"wrote {md}")
    if args.compare:
        rows_b = load_jsonl(Path(config.diagnostics_dir) / args.compare / "eval.jsonl")
        if rows_b:
            cmp = out_dir / "wave_forward_mix_compare.png"
            plot_compare_forward(rows_b, args.compare, rows, args.run_id, cmp)
            print(f"wrote {cmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
