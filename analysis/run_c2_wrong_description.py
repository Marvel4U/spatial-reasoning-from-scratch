"""Full val-grid c2 eval including held-out wrong-description control (L6c)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

import config
import data
import eval_spatial
import grid_vit_adapter
from experiments import c2_l6c_c2_full_2k


def _apply_overrides(overrides: dict) -> dict:
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(config, k)
        setattr(config, k, v)
    return saved


def _restore_overrides(saved: dict) -> None:
    for k, v in saved.items():
        setattr(config, k, v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=Path,
        default=REPO / "runs/checkpoints/c2_L6c_c2_full_2k.pt",
    )
    args = p.parse_args()
    spec = c2_l6c_c2_full_2k()
    saved = _apply_overrides(spec["overrides"])
    saved["use_gpu"] = config.use_gpu
    config.use_gpu = True
    try:
        data.setup_device()
        config.ensure_c2_encoding()
        model = grid_vit_adapter.build_model(device=data.device)
        grid_vit_adapter.load_checkpoint(args.ckpt, model=model)
        data.load_data()
        result = eval_spatial.eval_model(model, max_batches=None)
        eval_spatial.print_report(result)
    finally:
        _restore_overrides(saved)
        if "use_gpu" in saved:
            config.use_gpu = saved["use_gpu"]


if __name__ == "__main__":
    main()
