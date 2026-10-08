#!/usr/bin/env python3
"""SFT pipeline smoke test: one (image, prompt, teacher JSON) example, masked CE loss.

Not training — verifies processor + chat template + label mask before LoRA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_fine_tuning.examples import (
    DEFAULT_EXPORT,
    first_train_route,
    load_example_image,
    load_route,
    sft_user_prompt,
    teacher_answer_qwen1000,
)
from qwen_reference.load_model import release, resolve_model


def _mask_labels(input_ids: torch.Tensor, prefix_len: int, pad_id: int | None) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    if pad_id is not None:
        labels[input_ids == pad_id] = -100
    return labels


def main() -> None:
    p = argparse.ArgumentParser(description="One-example SFT forward (loss smoke test)")
    p.add_argument("--export", default=str(DEFAULT_EXPORT))
    p.add_argument("--route", type=int, default=None, help="default: first train row without leaves_window")
    p.add_argument("--model", default=None)
    p.add_argument("--device-map", default="cuda:0")
    args = p.parse_args()

    export = Path(args.export)
    record = load_route(export, args.route) if args.route is not None else first_train_route(export)
    img = load_example_image(export, record)
    user_text = sft_user_prompt(record)
    answer = teacher_answer_qwen1000(record)

    path = resolve_model(args.model)
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map=args.device_map
    )
    model.train()

    user_msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}],
        }
    ]
    full_msgs = user_msgs + [{"role": "assistant", "content": answer}]
    prefix_text = proc.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    full_text = proc.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prefix_in = proc(text=[prefix_text], images=[img], return_tensors="pt")
    full_in = proc(text=[full_text], images=[img], return_tensors="pt")
    prefix_len = prefix_in["input_ids"].shape[1]
    batch = {k: v.to(model.device) for k, v in full_in.items()}
    pad_id = proc.tokenizer.pad_token_id
    labels = _mask_labels(batch["input_ids"], prefix_len, pad_id).to(model.device)

    n_train = int((labels != -100).sum().item())
    print(f"route {record['route']}  export={export}")
    print(f"image {record['image']}  n_teacher_pts={record['n_points']}  leaves_window={record.get('leaves_window')}")
    print(f"user prompt ({len(user_text)} chars), assistant ({len(answer)} chars): {answer[:120]}...")
    print(f"input_ids {batch['input_ids'].shape[1]}  trainable label tokens {n_train}  prefix_len {prefix_len}")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**batch, labels=labels)
    loss = out.loss.item()
    print(f"loss {loss:.4f}  (expect finite, ~few nats for ~{n_train} answer tokens)")

    trainable = labels[0][labels[0] != -100]
    decoded = proc.tokenizer.decode(trainable, skip_special_tokens=False)
    print("\nDecoded label span (should match assistant JSON):\n", decoded[:500])

    release(proc, model, batch, prefix_in, full_in, out)


if __name__ == "__main__":
    main()
