"""Full-grid c2 evaluation of saved checkpoints (all 250 val crops x every task), per task.
Run from the harness dir:  ../.venv/bin/python <this> <run_id> [<run_id> ...]"""
import json, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch
import config
import data, grid_vit_adapter as A, eval_c2

ledger = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for run in sys.argv[1:]:
    rec = ledger[run]
    for k, v in rec["overrides"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.verbose = False
    config.use_gpu = True
    data.setup_device() if hasattr(data, 'setup_device') else None
    data.load_data()
    model, _ = A.load_checkpoint(REPO / "runs/checkpoints" / f"{run}.pt")
    keys = config.c2_active_task_keys or config.c2_task_keys_active()
    res = eval_c2.evaluate_c2(model, data.val_loader, task_keys=keys, weights=config.c2_fg_weights_active(),
                              encoding=config.encoding_mode, held_out=tuple(config.c2_held_out_keys_active() if hasattr(config, "c2_held_out_keys_active") and config.c2_held_out else ()),
                              autocast_dtype=data.dtype, max_batches=None)
    print(f"\n##### {run}  ({len(keys)} tasks, full grid)")
    eval_c2.print_table(res, prefix="  ")
