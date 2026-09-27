#!/usr/bin/env python3
"""Write pred_step{step}_routes_{ref}.png using the reference c4 val store (default: 20k ``c4``)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config
import data
import grid_vit_adapter
from checkpoints import checkpoint_model_config
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT

from .paths import run_dir
from .tier2_eval import Tier2EvalDiagnostics


def _overrides_for_run(run_id: str) -> dict:
    path = Path(config.RESULTS_PATH)
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    for r in doc.get("runs", []):
        if r.get("id") == run_id:
            return dict(r.get("overrides") or {})
    return {}


def _apply_meta_config(meta: dict) -> dict:
    saved: dict = {}
    top = meta.get("config") or {}
    for k, v in top.items():
        if hasattr(config, k):
            saved[k] = getattr(config, k)
            if k in ("data_root", "harness_checkpoint_dir", "diagnostics_dir") and v is not None:
                v = Path(v)
            setattr(config, k, v)
    return saved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C4 reference-val pred strip for alt route stores")
    p.add_argument("run_id")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: from experiments summary)")
    p.add_argument(
        "--ref-subdir",
        default=None,
        help=f"reference val store under crops/{{cropset}}/ (default: {config.diagnostics_pred_c4_reference_subdir})",
    )
    args = p.parse_args(argv)
    ref = args.ref_subdir or config.diagnostics_pred_c4_reference_subdir
    overrides = _overrides_for_run(args.run_id)
    meta_path = run_dir(args.run_id) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    saved_meta = _apply_meta_config(meta) if meta else {}
    saved_ov: dict = {}
    for k, v in overrides.items():
        if hasattr(config, k):
            saved_ov[k] = getattr(config, k)
            setattr(config, k, v)
    step = args.step
    if step is None:
        path = Path(config.RESULTS_PATH)
        if path.is_file():
            for r in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
                if r.get("id") == args.run_id:
                    step = int((r.get("summary") or {}).get("steps", 0))
                    break
    if step is None:
        raise SystemExit("could not resolve step; pass --step")
    ckpt = Path(config.harness_checkpoint_dir) / f"{args.run_id}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    try:
        config.sync_in_chans_from_encoding()
        data.setup_device()
        data.load_data()
        ckpt_doc = read_checkpoint(ckpt, map_location="cpu")
        cfg_dict = checkpoint_model_config(ckpt_doc) or dict(meta.get("model") or {})
        model = LocalGridViT(LocalGridViTConfig(**cfg_dict))
        model.to(data.device)
        grid_vit_adapter.load_checkpoint(ckpt, model=model)
        model.eval()
        t2 = Tier2EvalDiagnostics(args.run_id)
        if ref:
            config.diagnostics_pred_c4_reference_subdir = ref
        out = t2.maybe_save_c4_reference_pred_strip(model, step)
        if out is None:
            print("skipped (not c4, same store as reference, or pred disabled)")
            return 1
        print(f"saved {out}")
        return 0
    finally:
        for k, v in saved_ov.items():
            setattr(config, k, v)
        for k, v in saved_meta.items():
            setattr(config, k, v)


if __name__ == "__main__":
    raise SystemExit(main())
