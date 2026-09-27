"""Load c3 checkpoint + val data for offline analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _ensure_harness():
    h = REPO / "harness"
    for p in (str(REPO), str(h)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_c3_run(
    run_id: str,
    *,
    task_key: str | None = None,
    checkpoint: Path | None = None,
) -> tuple[torch.nn.Module, object, object]:
    """Returns (model, cfg, val_loader) on cuda."""
    _ensure_harness()
    import config
    import data
    from plain_gpt_module.checkpoint import read_checkpoint
    from plain_gpt_module.local_config import LocalGridViTConfig
    from plain_gpt_module.local_grid_vit import LocalGridViT

    if task_key is None:
        if "within" in run_id:
            import re
            m = re.search(r"within_(\d+)m", run_id)
            task_key = f"within_{m.group(1)}m" if m else "within_20m"
        elif "tertile" in run_id:
            task_key = "quietest_tertile"
        else:
            task_key = "median_strict"
    config.task_rung = "c3"
    config.c3_task_key = task_key
    config.cropset = "v4"
    config.data_source = "worldsnap"
    config.batch_size = 16
    config.diagnostics_probe_batch_size = 64
    config.ensure_c3_encoding()
    data.setup_device()
    data.load_data()
    ckpt = checkpoint or REPO / "runs/checkpoints" / f"{run_id}.pt"
    doc = read_checkpoint(ckpt)
    cfg = LocalGridViTConfig(**doc["model_config"])
    model = LocalGridViT(cfg).to(data.device)
    model.load_state_dict(doc["model_state_dict"])
    model.eval()
    return model, cfg, data.val_loader
