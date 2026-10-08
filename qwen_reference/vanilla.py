#!/usr/bin/env python3
"""§2.3 vanilla route on track_a plain exports.

Usage (repo root):
  python -m qwen_reference.vanilla one --route 0 --overlay
  python -m qwen_reference.vanilla one --route 0 --prompt two-pass --overlay
  python -m qwen_reference.vanilla run --limit 20 --out runs/qwen_vanilla/val_20.jsonl --score
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_reference.coords import parse_route_failure_hint
from qwen_reference.infer import infer_route
from qwen_reference.load_model import load, release
from qwen_reference.overlay import save_route_overlay
from spatial_data.track_a_export import load_predictions, load_routes, score, summary_line

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/qwen_vanilla"
DEFAULT_EXPORT = ROOT / "data/track_a/de_pijp_val_plain"


def _load_all_routes(export_dir: Path) -> list[dict]:
    return load_routes(export_dir / "routes.jsonl")


def _write_pred_line(f, route_id: int, points: list | None) -> None:
    f.write(json.dumps({"route": route_id, "points": points}) + "\n")


def _overlay_path(route_id: int, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return RUN_DIR / f"route_{route_id:03d}_overlay.png"


def cmd_one(args) -> None:
    export = Path(args.export)
    routes = {r["route"]: r for r in _load_all_routes(export)}
    if args.route not in routes:
        raise SystemExit(f"unknown route {args.route}")
    rec = routes[args.route]
    proc, model, path = load(args.model, device_map=args.device_map)
    print(f"model: {path}  prompt: {args.prompt}  device_map: {args.device_map}")
    from PIL import Image
    img = Image.open(export / rec["image"]).convert("RGB")
    raw, points, meta = infer_route(
        proc,
        model,
        img,
        mode=args.prompt,
        max_new_tokens=args.max_new_tokens,
        verbose=not args.quiet,
        record=rec if args.prompt == "oracle" else None,
    )
    if args.quiet:
        print(raw)
    if meta.get("markers"):
        g, r = meta["markers"]["green"], meta["markers"]["red"]
        w, h = meta["w"], meta["h"]
        from qwen_reference.coords import qwen1000_to_image_px
        print(
            "markers -> px:",
            "green", [round(v, 1) for v in qwen1000_to_image_px(g, w, h)],
            "GT", rec["start"],
            "| red", [round(v, 1) for v in qwen1000_to_image_px(r, w, h)],
            "GT", rec["goal"],
        )
    if points is None:
        print("points (image px): None —", parse_route_failure_hint(raw))
    else:
        print(f"points (image px): {len(points)} vertices")
        print(points)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"route_{args.route:03d}_{args.prompt}.txt"
    log_path.write_text(raw + "\n\n" + json.dumps(meta, indent=2))
    print("log:", log_path)
    pred_path = Path(args.out) if args.out else RUN_DIR / f"route_{args.route:03d}_{args.prompt}.jsonl"
    with pred_path.open("w") as f:
        _write_pred_line(f, args.route, points)
    print("pred:", pred_path)
    if args.overlay is not None:
        op = _overlay_path(args.route, args.overlay or None)
        save_route_overlay(export / rec["image"], op, points, rec, show_teacher=not args.no_teacher)
    release(proc, model)


def cmd_run(args) -> None:
    export = Path(args.export)
    routes = _load_all_routes(export)
    if args.limit is not None:
        routes = routes[: args.limit]
    proc, model, path = load(args.model, device_map=args.device_map)
    print(f"model: {path}  prompt: {args.prompt}  routes: {len(routes)}")
    from PIL import Image
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, rec in enumerate(routes):
            img = Image.open(export / rec["image"]).convert("RGB")
            _, points, _ = infer_route(
                proc,
                model,
                img,
                mode=args.prompt,
                max_new_tokens=args.max_new_tokens,
                verbose=not args.quiet,
                record=rec if args.prompt == "oracle" else None,
            )
            _write_pred_line(f, rec["route"], points)
            if (i + 1) % args.log_every == 0 or i + 1 == len(routes):
                print(f"  {i + 1}/{len(routes)} route {rec['route']} points={'ok' if points else 'fail'}")
    print("wrote", out)
    if args.score:
        rep = score(_load_all_routes(export), load_predictions(out))
        print(summary_line(rep))
        if args.report:
            Path(args.report).write_text(json.dumps(rep, indent=2))
    release(proc, model)


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None)
    common.add_argument("--export", default=str(DEFAULT_EXPORT))
    common.add_argument("--prompt", choices=("v1", "v2", "two-pass", "oracle"), default="two-pass")
    common.add_argument("--max-new-tokens", type=int, default=384)
    common.add_argument("--quiet", action="store_true", help="hide full prompt/reply blocks (batch runs)")
    common.add_argument(
        "--device-map",
        default="auto",
        help='HF device_map (default auto). Use cuda:0 if VRAM is free (~5 GB for 2B bf16).',
    )
    p = argparse.ArgumentParser(description="Vanilla Qwen route inference")
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one", parents=[common], help="single route")
    one.add_argument("--route", type=int, default=0)
    one.add_argument("--out", default=None, help=f"default {RUN_DIR}/route_NNN_<prompt>.jsonl")
    one.add_argument(
        "--overlay",
        nargs="?",
        const="",
        default=None,
        help=f"write PNG (default {RUN_DIR}/route_NNN_overlay.png)",
    )
    one.add_argument("--no-teacher", action="store_true")
    one.set_defaults(func=cmd_one)
    run = sub.add_parser("run", parents=[common], help="batch -> jsonl")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--out", default=str(RUN_DIR / "predictions.jsonl"))
    run.add_argument("--score", action="store_true")
    run.add_argument("--report", default=None)
    run.add_argument("--log-every", type=int, default=10)
    run.set_defaults(func=cmd_run)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
