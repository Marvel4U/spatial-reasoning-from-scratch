"""Draw predicted (and optional teacher) routes on track_a map PNGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/qwen_vanilla"
DEFAULT_EXPORT = ROOT / "data/track_a/de_pijp_val_plain"

PRED_COLOR = "#E040FB"
TEACHER_COLOR = "#1565C0"
VERTEX_R = 4


def _as_tuples(points: list) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in points]


def draw_route_overlay(
    base: Image.Image,
    pred_points: list | None,
    *,
    teacher_points: list | None = None,
) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    if teacher_points and len(teacher_points) >= 2:
        t = _as_tuples(teacher_points)
        draw.line(t, fill=TEACHER_COLOR, width=2)
        for x, y in t:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=TEACHER_COLOR)
    if pred_points and len(pred_points) >= 2:
        p = _as_tuples(pred_points)
        draw.line(p, fill=PRED_COLOR, width=3)
        for x, y in p:
            draw.ellipse((x - VERTEX_R, y - VERTEX_R, x + VERTEX_R, y + VERTEX_R), outline=PRED_COLOR, width=2)
    return img


def save_route_overlay(
    image_path: Path,
    out_path: Path,
    pred_points: list | None,
    record: dict | None = None,
    *,
    show_teacher: bool = True,
) -> None:
    img = Image.open(image_path).convert("RGB")
    teacher = record.get("teacher") if (record and show_teacher) else None
    out = draw_route_overlay(img, pred_points, teacher_points=teacher)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print("overlay:", out_path, "(magenta=pred, blue=teacher)" if teacher else "(magenta=pred)")


def _route_record(export_dir: Path, route_id: int) -> dict:
    with (export_dir / "routes.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["route"] == route_id:
                return r
    raise SystemExit(f"route {route_id} not found")


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay route polylines on a track_a image")
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--route", type=int, required=True)
    p.add_argument("--pred", default=None, help="predictions.jsonl line or file with points for this route")
    p.add_argument("--points", default=None, help='JSON list of [x,y] image pixels, e.g. "[[1,2],[3,4]]"')
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("-o", "--out", default=None, help=f"default {RUN_DIR}/route_NNN_overlay.png")
    args = p.parse_args()
    export = Path(args.export)
    rec = _route_record(export, args.route)
    points = None
    if args.points:
        points = json.loads(args.points)
    elif args.pred:
        pred_path = Path(args.pred)
        if pred_path.is_file():
            from spatial_data.track_a_export import load_predictions
            points = load_predictions(pred_path).get(args.route)
        else:
            points = json.loads(pred_path.read_text()).get("points")
    out = Path(args.out) if args.out else RUN_DIR / f"route_{args.route:03d}_overlay.png"
    save_route_overlay(export / rec["image"], out, points, rec, show_teacher=not args.no_teacher)


if __name__ == "__main__":
    main()
