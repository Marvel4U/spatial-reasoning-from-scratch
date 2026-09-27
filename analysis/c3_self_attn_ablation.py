"""Full val IoU: normal encoder vs diagonal self-attention only (C3_PLAN §3 point 4).

Example:
  python -m analysis.c3_self_attn_ablation --run-id c3_quieter_sidewalk_median_strict_2k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

from analysis.c3_load import load_c3_run
from spatial_data.c3_tasks import LOCAL_CEILING


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--task-key", default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()
    import config
    model, _cfg, val_loader = load_c3_run(
        args.run_id, task_key=args.task_key, checkpoint=args.checkpoint,
    )
    import eval_spatial

    full = eval_spatial._c3_iou_accum(model, val_loader, max_batches=None)
    patch_local = eval_spatial._c3_iou_accum(
        model, val_loader, max_batches=None, encoder_patch_local_only=True,
    )
    diagonal = eval_spatial._c3_iou_accum(
        model, val_loader, max_batches=None, encoder_self_attn_only=True,
    )
    key = config.c3_task_key
    ceil = LOCAL_CEILING.get(key, float("nan"))
    pl_iou = patch_local["mean_iou_fg"]
    print(f"c3 attention ablation  run={args.run_id}  task={key}")
    print(f"  full val IoU (nonempty):     {full['mean_iou_fg']:.4f}")
    print(f"  patch-local IoU:           {pl_iou:.4f}  (no patch↔patch; cond prefix visible)")
    print(f"  drop (full - patch-local):  {full['mean_iou_fg'] - pl_iou:+.4f}")
    print(f"  local-only rule ceiling:    {ceil:.4f}")
    print(f"  patch-local vs ceiling:     {pl_iou - ceil:+.4f}")
    print(f"  strict diagonal IoU:        {diagonal['mean_iou_fg']:.4f}  (cond isolated; not comparable to ceiling)")


if __name__ == "__main__":
    main()
