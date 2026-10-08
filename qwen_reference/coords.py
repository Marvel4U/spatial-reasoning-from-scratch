"""Qwen3-VL grounding grid (0–1000) ↔ track_a image pixels (top-left origin)."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def qwen1000_to_image_px(xy, w: int, h: int) -> list[float]:
    x, y = float(xy[0]), float(xy[1])
    return [x / 1000.0 * w, y / 1000.0 * h]


def image_px_to_qwen1000(xy, w: int, h: int) -> list[float]:
    x, y = float(xy[0]), float(xy[1])
    return [x / w * 1000.0, y / h * 1000.0]


def _first_json_text(text: str) -> str | None:
    m = JSON_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return text[i : j + 1]
    i, j = text.find("["), text.rfind("]")
    if i >= 0 and j > i:
        return text[i : j + 1]
    return None


def parse_json_from_model(text: str) -> Any | None:
    raw = _first_json_text(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _pairs_from_text(text: str) -> list[list[float]] | None:
    """Salvage [x,y] pairs when JSON is truncated or wrapped in markdown."""
    pairs = [[float(a), float(b)] for a, b in PAIR.findall(text)]
    return pairs if len(pairs) >= 2 else None


def _list_to_px(obj: list, w: int, h: int) -> np.ndarray | None:
    pts = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        pts.append(qwen1000_to_image_px(item, w, h))
    p = np.array(pts, np.float64)
    return p if len(p) >= 2 and np.isfinite(p).all() else None


def parse_route_points_qwen1000(text: str, w: int, h: int, *, max_points: int = 12) -> np.ndarray | None:
    """Model answer -> (n, 2) float64 image pixels, or None."""
    obj = parse_json_from_model(text)
    p = None
    if obj is not None:
        if isinstance(obj, dict):
            if "path" in obj and isinstance(obj["path"], list):
                obj = obj["path"]
            else:
                for key in ("route", "points", "polyline"):
                    if key in obj:
                        obj = obj[key]
                        break
        if isinstance(obj, list):
            p = _list_to_px(obj, w, h)
    if p is None:
        raw_pairs = _pairs_from_text(text)
        if raw_pairs is None:
            return None
        p = _list_to_px(raw_pairs[:max_points], w, h)
    if p is None:
        return None
    return p[:max_points] if len(p) > max_points else p


def parse_markers_qwen1000(text: str) -> dict[str, list[float]] | None:
    obj = parse_json_from_model(text)
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in ("green", "red"):
        if k not in obj or not isinstance(obj[k], (list, tuple)) or len(obj[k]) != 2:
            return None
        out[k] = [float(obj[k][0]), float(obj[k][1])]
    return out


def parse_route_failure_hint(text: str) -> str:
    if not text.strip():
        return "empty model output"
    if "```" in text and text.rstrip().endswith(("```", "[", ",", " ")):
        return "JSON likely truncated (increase --max-new-tokens or shorten the path in the prompt)"
    if PAIR.findall(text):
        return "could not parse as JSON; try salvage or fix format"
    return "no coordinate pairs found in output"


def round_points_for_export(p: np.ndarray) -> list[list[float | int]]:
    out = []
    for x, y in p:
        rx, ry = round(float(x), 3), round(float(y), 3)
        out.append([int(rx) if rx == int(rx) else rx, int(ry) if ry == int(ry) else ry])
    return out
