"""Load Qwen3-VL base + saved LoRA adapter."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_reference.load_model import DEFAULT_MODEL, resolve_model


def load_lora(
    adapter_dir: str | Path,
    base_model: str | Path | None = None,
    *,
    dtype=torch.bfloat16,
    device_map: str | None = "cuda:0",
):
    adapter_dir = Path(adapter_dir)
    base = resolve_model(base_model) if base_model else DEFAULT_MODEL
    proc = AutoProcessor.from_pretrained(adapter_dir)
    base_model_obj = AutoModelForImageTextToText.from_pretrained(
        base, dtype=dtype, device_map=device_map
    )
    model = PeftModel.from_pretrained(base_model_obj, adapter_dir)
    model.eval()
    return proc, model, base, adapter_dir
