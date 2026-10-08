#!/usr/bin/env python3
"""Generate route_v1 preds with a LoRA adapter and score with track_a polyline rule."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from qwen_fine_tuning.load_lora import load_lora
from qwen_reference.infer import infer_route
from qwen_reference.load_model import release
from spatial_data.track_a_export import load_routes, load_predictions, score, summary_line

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL = ROOT / "data/track_a/de_pijp_val_plain"


def main() -> None:
    p = argparse.ArgumentParser(description="Val/test eval for a LoRA adapter (single-shot v1)")
    p.add_argument("--adapter", required=True, help="directory from train_lora (adapter/)")
    p.add_argument("--export", default=str(DEFAULT_VAL))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--model", default=None, help="base HF model if not default 2B")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--out", default=None, help="pred jsonl; default next to adapter parent")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    adapter = Path(args.adapter)
    export = Path(args.export)
    all_routes = load_routes(export / "routes.jsonl")
    routes = all_routes[: args.limit]
    run_dir = adapter.parent if adapter.name == "adapter" else adapter
    pred_path = Path(args.out) if args.out else run_dir / f"pred_val_v1_n{len(routes)}.jsonl"

    proc, model, base, _ = load_lora(adapter, args.model, device_map=args.device_map)
    print(f"base={base}  adapter={adapter}  routes={len(routes)} from {export}")

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w") as f:
        for i, rec in enumerate(routes):
            img = Image.open(export / rec["image"]).convert("RGB")
            _, points, _ = infer_route(
                proc, model, img, mode="v1", max_new_tokens=args.max_new_tokens, verbose=not args.quiet
            )
            f.write(json.dumps({"route": rec["route"], "points": points}) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == len(routes):
                print(f"  [{i + 1}/{len(routes)}] route {rec['route']}")

    rep = score(all_routes[: len(routes)], load_predictions(pred_path))
    line = summary_line(rep)
    print(line)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": str(base),
        "adapter": str(adapter),
        "export": str(export),
        "n_routes": len(routes),
        "pred": str(pred_path),
        "summary": line,
        "report": rep,
    }
    report_path = run_dir / f"eval_val_v1_n{len(routes)}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {pred_path}\nWrote {report_path}")
    release(proc, model)


if __name__ == "__main__":
    main()
