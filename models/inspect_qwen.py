"""Print the module tree, parameter counts and image-token geometry of the Track A backbones.
Runs on the meta device (no weights loaded), CPU only. Usage: python models/inspect_qwen.py Qwen3-VL-2B-Instruct Qwen3.5-0.8B"""
import sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def count(m):
    return sum(p.numel() for p in m.parameters())


def describe(mod, indent):
    for n, c in mod.named_children():
        if n in ("layers", "blocks"):
            kinds = {}
            for i, blk in enumerate(c):
                sig = ", ".join(f"{k}:{type(v).__name__}" for k, v in blk.named_children())
                kinds.setdefault(sig, []).append(i)
            print(f"{indent}{n}: {len(c)} blocks, {count(c[0]) / 1e6:.2f}M each")
            for sig, ids in kinds.items():
                print(f"{indent}  layers {ids[0]}..{ids[-1]} ({len(ids)}): {sig}")
        else:
            print(f"{indent}{n}: {type(c).__name__} {count(c) / 1e6:.2f}M")
            if n in ("model", "visual", "language_model"):
                describe(c, indent + "  ")


for name in sys.argv[1:]:
    path = "models/hf/" + name
    print("=" * 100)
    print(name)
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    print("class", type(model).__name__, "total", f"{count(model) / 1e6:.1f}M")
    describe(model, "  ")
    for n, p in model.named_parameters():
        if "patch_embed" in n or "lm_head" in n or "embed_tokens" in n or "merger.linear_fc2" in n or "mtp" in n and "weight" in n and p.dim() == 2 and "layers" not in n:
            print("  param", n, tuple(p.shape))
    proc = AutoProcessor.from_pretrained(path)
    for px in (256, 512):
        img = Image.fromarray(np.zeros((px, px, 3), np.uint8))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt")
        n_img = int((enc["input_ids"] == cfg.image_token_id).sum())
        print(f"  {px}px image -> {n_img} image tokens, grid_thw {enc['image_grid_thw'].tolist()}, "
              f"pixel_values {tuple(enc['pixel_values'].shape)}, seq len {enc['input_ids'].shape[1]}")
    ids = enc["input_ids"][0].tolist()
    toks = proc.tokenizer.convert_ids_to_tokens(ids)
    print("  prompt:", toks[:10], "... [image tokens] ...", toks[-10:])
