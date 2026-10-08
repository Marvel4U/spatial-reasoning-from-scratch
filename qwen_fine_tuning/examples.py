"""Track A SFT examples: teacher polyline in Qwen 0–1000 JSON (matches route_v1 eval)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from qwen_reference.coords import image_px_to_qwen1000
from qwen_reference.prompts import route_v1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data/track_a/de_pijp_train20k_plain"


def load_route(export: Path, route_id: int) -> dict:
    with (export / "routes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["route"] == route_id:
                return rec
    raise SystemExit(f"route {route_id} not in {export / 'routes.jsonl'}")


def first_train_route(export: Path, *, skip_leaves_window: bool = True) -> dict:
    rows = load_route_list(export, limit=1, offset=0, skip_leaves_window=skip_leaves_window)
    return rows[0]


def load_route_list(
    export: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    skip_leaves_window: bool = True,
) -> list[dict]:
    out: list[dict] = []
    with (export / "routes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if skip_leaves_window and rec.get("leaves_window"):
                continue
            if offset > 0:
                offset -= 1
                continue
            out.append(rec)
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise SystemExit(f"no routes (limit={limit} offset={offset}) in {export / 'routes.jsonl'}")
    return out


def teacher_answer_qwen1000(record: dict) -> str:
    w, h = record["image_px"]
    pts = []
    for x, y in record["teacher"]:
        q = image_px_to_qwen1000([x, y], w, h)
        pts.append([int(round(q[0])), int(round(q[1]))])
    return json.dumps(pts, separators=(",", ":"))


def sft_user_prompt(record: dict) -> str:
    w, h = record["image_px"]
    return route_v1(w, h)


def load_example_image(export: Path, record: dict) -> Image.Image:
    return Image.open(export / record["image"]).convert("RGB")
