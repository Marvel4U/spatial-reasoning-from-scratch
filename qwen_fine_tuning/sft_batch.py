"""Encode track_a routes into model inputs + masked labels for SFT."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from qwen_fine_tuning.examples import (
    load_example_image,
    load_route_list,
    sft_user_prompt,
    teacher_answer_qwen1000,
)


def encode_sft_example(proc, export: Path, record: dict) -> dict[str, torch.Tensor | int]:
    img = load_example_image(export, record)
    user_text = sft_user_prompt(record)
    answer = teacher_answer_qwen1000(record)
    user_msgs = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]},
    ]
    full_msgs = user_msgs + [{"role": "assistant", "content": answer}]
    prefix_text = proc.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    full_text = proc.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prefix_in = proc(text=[prefix_text], images=[img], return_tensors="pt")
    full_in = proc(text=[full_text], images=[img], return_tensors="pt")
    prefix_len = int(prefix_in["input_ids"].shape[1])
    labels = full_in["input_ids"].clone()
    labels[:, :prefix_len] = -100
    pad_id = proc.tokenizer.pad_token_id
    if pad_id is not None:
        labels[full_in["input_ids"] == pad_id] = -100
    out = {k: v.squeeze(0) for k, v in full_in.items()}
    out["labels"] = labels.squeeze(0)
    out["route"] = int(record["route"])
    return out


def collate_sft_batch(items: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in items)
    keys = [k for k in items[0].keys() if k not in ("route",)]
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        if key == "pixel_values":
            batch[key] = torch.cat([x[key] for x in items], dim=0)
            continue
        if key == "image_grid_thw":
            batch[key] = torch.stack([x[key] for x in items], dim=0)
            continue
        rows = []
        for x in items:
            t = x[key]
            pad_len = max_len - t.shape[0]
            if pad_len:
                if key == "labels":
                    fill = -100
                elif key == "attention_mask":
                    fill = 0
                elif key == "mm_token_type_ids":
                    fill = 0
                else:
                    fill = pad_id
                t = torch.cat([t, torch.full((pad_len,), fill, dtype=t.dtype)])
            rows.append(t)
        batch[key] = torch.stack(rows, dim=0)
    batch["routes"] = torch.tensor([x["route"] for x in items], dtype=torch.long)
    return batch


class RouteSFTDataset(Dataset):
    def __init__(self, proc, export: Path, records: list[dict]):
        self.proc = proc
        self.export = export
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return encode_sft_example(self.proc, self.export, self.records[i])


def train_dev_split(
    export: Path,
    train_limit: int,
    dev_limit: int,
    *,
    skip_leaves_window: bool = True,
) -> tuple[list[dict], list[dict]]:
    need = train_limit + dev_limit
    pool = load_route_list(export, limit=need, offset=0, skip_leaves_window=skip_leaves_window)
    if len(pool) < dev_limit + 1:
        raise SystemExit(f"need at least {dev_limit + 1} routes, got {len(pool)} (after leaves filter)")
    if len(pool) < need:
        train_n = len(pool) - dev_limit
        print(
            f"train_dev_split: requested train={train_limit}+dev={dev_limit}, "
            f"only {len(pool)} usable → train={train_n} dev={dev_limit}",
            flush=True,
        )
    else:
        train_n = train_limit
    return pool[:train_n], pool[train_n : train_n + dev_limit]
