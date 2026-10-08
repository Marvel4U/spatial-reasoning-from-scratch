"""Route inference strategies (prompt variants)."""
from __future__ import annotations

from PIL import Image

from qwen_reference.coords import (
    image_px_to_qwen1000,
    parse_markers_qwen1000,
    parse_route_points_qwen1000,
    round_points_for_export,
)
from qwen_reference.load_model import generate
from qwen_reference.prompts import (
    marker_probe,
    route_after_markers,
    route_oracle_endpoints,
    route_v1,
    route_v2_structured,
)


def infer_route(
    proc,
    model,
    img: Image.Image,
    *,
    mode: str = "two-pass",
    max_new_tokens: int = 384,
    verbose: bool = True,
    record: dict | None = None,
) -> tuple[str, list | None, dict]:
    """Returns (combined log text, points in image px or None, meta)."""
    w, h = img.size
    meta: dict = {"mode": mode, "w": w, "h": h}
    gen_kw = dict(max_new_tokens=max_new_tokens, verbose=verbose)
    if mode == "oracle":
        if record is None:
            raise ValueError("oracle mode requires record with start/goal image pixels")
        start_q = image_px_to_qwen1000(record["start"], w, h)
        goal_q = image_px_to_qwen1000(record["goal"], w, h)
        meta["oracle_start_q"] = start_q
        meta["oracle_goal_q"] = goal_q
        raw = generate(
            proc,
            model,
            img,
            route_oracle_endpoints(w, h, start_q, goal_q),
            label="route oracle",
            **gen_kw,
        )
        pts = parse_route_points_qwen1000(raw, w, h)
        return raw, round_points_for_export(pts) if pts is not None else None, meta
    if mode == "v1":
        raw = generate(proc, model, img, route_v1(w, h), label="route v1", **gen_kw)
        pts = parse_route_points_qwen1000(raw, w, h)
        return raw, round_points_for_export(pts) if pts is not None else None, meta
    if mode == "v2":
        raw = generate(proc, model, img, route_v2_structured(w, h), label="route v2", **gen_kw)
        pts = parse_route_points_qwen1000(raw, w, h)
        return raw, round_points_for_export(pts) if pts is not None else None, meta
    if mode == "two-pass":
        mraw = generate(proc, model, img, marker_probe(w, h), max_new_tokens=128, verbose=verbose, label="markers")
        markers = parse_markers_qwen1000(mraw)
        meta["markers_raw"] = mraw
        meta["markers"] = markers
        if not markers:
            return f"--- markers ---\n{mraw}\n--- route ---\n(failed: bad markers)", None, meta
        green, red = markers["green"], markers["red"]
        rprompt = route_after_markers(w, h, green, red)
        rraw = generate(proc, model, img, rprompt, label="route anchored", **gen_kw)
        pts = parse_route_points_qwen1000(rraw, w, h)
        log = f"--- markers ---\n{mraw}\n--- route (anchored) ---\n{rraw}"
        meta["route_prompt"] = rprompt
        return log, round_points_for_export(pts) if pts is not None else None, meta
    raise ValueError(f"unknown mode {mode!r}; use v1, v2, two-pass, oracle")
