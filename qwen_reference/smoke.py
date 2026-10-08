#!/usr/bin/env python3
"""§2.1–2.2 smoke: load Qwen3-VL, describe a map, probe marker coordinates.

Usage (repo root, sprint venv):
  python -m qwen_reference.smoke describe --route 0
  python -m qwen_reference.smoke markers --route 0
  python -m qwen_reference.smoke inspect-tokens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from qwen_reference.coords import parse_json_from_model, qwen1000_to_image_px
from qwen_reference.load_model import generate, load
from qwen_reference.prompts import describe_brief, marker_probe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data/track_a/de_pijp_val_plain"


def _route_record(export_dir: Path, route_id: int) -> dict:
    with (export_dir / "routes.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["route"] == route_id:
                return r
    raise SystemExit(f"route {route_id} not in {export_dir / 'routes.jsonl'}")


def _image(export_dir: Path, record: dict) -> Image.Image:
    return Image.open(export_dir / record["image"]).convert("RGB")


def cmd_describe(args) -> None:
    export = Path(args.export)
    rec = _route_record(export, args.route)
    img = _image(export, rec)
    proc, model, path = load(args.model)
    print(f"model: {path}")
    print(generate(proc, model, img, describe_brief(), max_new_tokens=args.max_new_tokens, verbose=not args.quiet, label="describe"))


def cmd_markers(args) -> None:
    export = Path(args.export)
    rec = _route_record(export, args.route)
    img = _image(export, rec)
    w, h = img.size
    proc, model, path = load(args.model)
    print(f"model: {path}")
    ans = generate(proc, model, img, marker_probe(w, h), max_new_tokens=args.max_new_tokens, verbose=not args.quiet, label="markers")
    if args.quiet:
        print("model raw:", ans)
    obj = parse_json_from_model(ans)
    if not isinstance(obj, dict):
        print("parse: failed")
        return
    for key, gt_key, label in (("green", "start", "start"), ("red", "goal", "goal")):
        if key not in obj:
            continue
        px = qwen1000_to_image_px(obj[key], w, h)
        print(f"  {label}: qwen1000 {obj[key]} -> px {[round(v, 1) for v in px]}  GT {rec[gt_key]}")


def cmd_inspect_tokens(_args) -> None:
    import subprocess
    import sys
    script = ROOT / "models/inspect_qwen.py"
    name = Path(_args.model).name if _args.model else "Qwen3-VL-2B-Instruct"
    subprocess.run([sys.executable, str(script), name], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-VL track_a smoke tests")
    p.add_argument("--model", default=None, help="path under repo or absolute (default: models/hf/Qwen3-VL-2B-Instruct)")
    p.add_argument("--export", default=str(DEFAULT_EXPORT), help="track_a export dir")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="brief caption")
    d.add_argument("--route", type=int, default=0)
    d.set_defaults(func=cmd_describe)
    m = sub.add_parser("markers", help="green/red JSON probe + GT")
    m.add_argument("--route", type=int, default=0)
    m.set_defaults(func=cmd_markers)
    t = sub.add_parser("inspect-tokens", help="CPU/meta token geometry (models/inspect_qwen.py)")
    t.set_defaults(func=cmd_inspect_tokens)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
