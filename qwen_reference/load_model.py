"""Load Qwen3-VL processor + model from models/hf/."""
from __future__ import annotations

import gc
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/hf/Qwen3-VL-2B-Instruct"


def resolve_model(path: str | Path | None) -> Path:
    p = Path(path) if path else DEFAULT_MODEL
    return p if p.is_absolute() else ROOT / p


def load(
    model_path: str | Path | None = None,
    dtype=torch.bfloat16,
    device_map: str | None = "auto",
):
    path = resolve_model(model_path)
    proc = AutoProcessor.from_pretrained(path)
    kw = dict(dtype=dtype)
    if device_map is not None:
        kw["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(path, **kw)
    model.eval()
    return proc, model, path


def release(*objects) -> None:
    """Drop model references and return cached CUDA memory to the driver (this process)."""
    for o in objects:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _print_llm_exchange(label: str, user_text: str, chat_prompt: str, image: Image.Image, n_tokens: int, reply: str) -> None:
    tag = f" ({label})" if label else ""
    bar = "=" * 72
    w, h = image.size
    print(f"\n{bar}\nLLM request{tag}\n{bar}")
    print("User message (your task text):\n", user_text, sep="")
    print("\nFull prompt after apply_chat_template (what the decoder conditions on; "
          f"<|image_pad|> × N replaced by vision tokens at runtime):\n")
    print(chat_prompt)
    print(f"\nAttached image: {w}×{h} px  |  input_ids length: {n_tokens}")
    print(f"\n{bar}\nLLM reply{tag} (raw decode, assistant turn only)\n{bar}")
    print(reply)
    print(bar)


@torch.inference_mode()
def generate(
    proc,
    model,
    image: Image.Image,
    user_text: str,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    verbose: bool = True,
    label: str = "",
) -> str:
    img = image.convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_text}]}]
    chat_prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[chat_prompt], images=[img], return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    reply = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    if verbose:
        _print_llm_exchange(label, user_text, chat_prompt, img, n_in, reply)
    return reply
