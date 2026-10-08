#!/usr/bin/env python3
"""A/B eval on track_a val: task-only vs oracle endpoint coordinates.

(i)  two-pass — model finds markers, then path (no GT coords in prompt)
(ii) oracle — GT start/goal on 0–1000 grid given in the prompt

Usage:
  python -m qwen_reference.eval_ab --limit 200
  python -m qwen_reference.eval_ab --limit 200 --modes two-pass,oracle --device-map cuda:0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from qwen_reference.infer import infer_route
from qwen_reference.load_model import load, release
from spatial_data.track_a_export import load_predictions, load_routes, score, summary_line

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/qwen_vanilla"
DEFAULT_EXPORT = ROOT / "data/track_a/de_pijp_val_plain"


def _model_tag(model_path: Path, tag: str | None) -> str:
    if tag:
        return tag
    name = model_path.name
    if "4B" in name or "4b" in name:
        return "4b"
    return ""  # default 2B: legacy pred_{mode}_n*.jsonl names


def _run_mode(
    proc,
    model,
    export: Path,
    routes: list[dict],
    mode: str,
    max_new_tokens: int,
    quiet: bool,
    tag: str,
) -> Path:
    stem = f"pred_{tag}_{mode}" if tag else f"pred_{mode}"
    pred_path = RUN_DIR / f"{stem}_n{len(routes)}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w") as f:
        for i, rec in enumerate(routes):
            img = Image.open(export / rec["image"]).convert("RGB")
            _, points, _ = infer_route(
                proc,
                model,
                img,
                mode=mode,
                max_new_tokens=max_new_tokens,
                verbose=not quiet,
                record=rec if mode == "oracle" else None,
            )
            f.write(json.dumps({"route": rec["route"], "points": points}) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == len(routes):
                print(f"  [{mode}] {i + 1}/{len(routes)} route {rec['route']}")
    return pred_path


def main() -> None:
    p = argparse.ArgumentParser(description="A/B: two-pass vs oracle endpoints")
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--modes", default="two-pass,oracle", help="comma-separated infer modes")
    p.add_argument("--model", default=None)
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--quiet", action="store_true", help="no per-request prompt dumps")
    p.add_argument(
        "--tag",
        default=None,
        help="filename tag for preds/report (default: infer from model dir, e.g. 2b / 4b)",
    )
    args = p.parse_args()

    export = Path(args.export)
    routes = load_routes(export / "routes.jsonl")[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"Eval {len(routes)} routes from {export}  modes={modes}")
    proc, model, model_path = load(args.model, device_map=args.device_map)
    tag = _model_tag(model_path, args.tag)
    print(f"Model: {model_path}" + (f"  tag={tag}" if tag else ""))

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "tag": tag,
        "export": str(export),
        "n_routes": len(routes),
        "modes": {},
    }

    all_routes = load_routes(export / "routes.jsonl")
    for mode in modes:
        print(f"\n=== {mode} ===")
        pred_path = _run_mode(
            proc, model, export, routes, mode, args.max_new_tokens, args.quiet, tag
        )
        rep = score(all_routes[: len(routes)], load_predictions(pred_path))
        line = summary_line(rep)
        print(line)
        report["modes"][mode] = {
            "pred": str(pred_path),
            "summary": line,
            "report": rep,
        }

    report_stem = f"eval_ab_{tag}" if tag else "eval_ab"
    out_report = RUN_DIR / f"{report_stem}_n{len(routes)}.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_report}")
    release(proc, model)


if __name__ == "__main__":
    main()
