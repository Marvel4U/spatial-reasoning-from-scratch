#!/usr/bin/env python3
"""Minimal interactive chat with Qwen3-VL (stdin REPL).

Loads the model once; optional map/image for the whole session. Multi-turn history
is kept in memory until /clear.

Usage (repo root):
  python -m qwen_reference.chat
  python -m qwen_reference.chat --image data/track_a/de_pijp_val_plain/images/route_000000.png
  python -m qwen_reference.chat --device-map cuda:0

Commands: /quit  /clear  /verbose  /show  (last templated prompt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from qwen_reference.load_model import load, release

ROOT = Path(__file__).resolve().parents[1]


def _user_content(text: str, image: Image.Image | None) -> list | str:
    if image is None:
        return text
    return [{"type": "image", "image": image}, {"type": "text", "text": text}]


@torch.inference_mode()
def _reply(proc, model, messages: list, image: Image.Image | None, max_new_tokens: int) -> tuple[str, str]:
    chat_prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    inputs = proc(text=[chat_prompt], images=imgs, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    return chat_prompt, text


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive Qwen3-VL chat")
    p.add_argument("--model", default=None)
    p.add_argument("--image", default=None, help="optional PNG/JPEG for the session")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--verbose", action="store_true", help="print full chat template each turn")
    args = p.parse_args()

    image = None
    if args.image:
        path = Path(args.image)
        if not path.is_absolute():
            path = ROOT / path
        image = Image.open(path).convert("RGB")
        print(f"Image: {path} ({image.size[0]}×{image.size[1]})")

    print("Loading model…")
    proc, model, model_path = load(args.model, device_map=args.device_map)
    print(f"Model: {model_path}  device_map={args.device_map}")
    print("Type a message and Enter. Commands: /quit /clear /verbose /show\n")

    messages: list[dict] = []
    last_prompt = ""

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "/q"):
            break
        if low == "/clear":
            messages.clear()
            print("(history cleared)\n")
            continue
        if low == "/verbose":
            args.verbose = not args.verbose
            print(f"verbose={args.verbose}\n")
            continue
        if low == "/show":
            print(last_prompt or "(no turn yet)\n")
            continue

        use_img = image if not any(m["role"] == "user" for m in messages) else None
        messages.append({"role": "user", "content": _user_content(line, use_img)})

        chat_prompt, ans = _reply(proc, model, messages, image, args.max_new_tokens)
        last_prompt = chat_prompt
        messages.append({"role": "assistant", "content": ans})

        if args.verbose:
            print("\n--- chat template ---\n", chat_prompt, "\n--- end ---\n", sep="")
        print(f"\nAssistant> {ans}\n")

    release(proc, model)
    print("Bye.")


if __name__ == "__main__":
    main()
