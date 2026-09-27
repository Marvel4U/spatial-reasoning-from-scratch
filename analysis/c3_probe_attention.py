"""Offline c3 attention probe: sidewalk mass + entropy on fixed val batch (C3_PLAN §3).

Example:
  python -m analysis.c3_probe_attention --run-id c3_quieter_sidewalk_median_strict_2k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

from analysis.c3_load import load_c3_run
from diagnostics.attn_probe import attention_matrix, full_seq_sidewalk_mask, sidewalk_attention_scalars
from diagnostics.probe_batch import ProbeBatch
from spatial_batch import forward as model_forward


@torch.no_grad()
def probe_stats(model, cfg, probe: ProbeBatch) -> dict[str, float]:
    acts: dict[str, torch.Tensor] = {}
    handles = []
    core = model

    def save_in(i):
        def f(_m, inp):
            acts[i] = inp[0].detach()
        return f

    for i, blk in enumerate(core.encoder):
        handles.append(blk.attn.register_forward_pre_hook(save_in(i)))
    model_forward(model, probe.img, None, probe.task_ids, probe.cond_ids)
    for h in handles:
        h.remove()
    sw_full, sw_frac = full_seq_sidewalk_mask(probe.img, cfg)
    rows: dict[str, float] = {"probe_sw_patch_frac": sw_frac.mean().item()}
    for i, blk in enumerate(core.encoder):
        att = attention_matrix(acts[i].float(), blk.attn)
        s = sidewalk_attention_scalars(att, sw_full, sw_frac)
        rows[f"L{i}_mass_sidewalk"] = s["mass_sidewalk"]
        rows[f"L{i}_mass_vs_uniform"] = s["mass_vs_uniform"]
        rows[f"L{i}_entropy_mean"] = s["entropy_mean"]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--task-key", default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()
    model, cfg, _loader = load_c3_run(args.run_id, task_key=args.task_key, checkpoint=args.checkpoint)
    import config as harness_config
    probe = ProbeBatch.build()
    stats = probe_stats(model, cfg, probe)
    print(f"c3 probe attention  run={args.run_id}  task={harness_config.c3_task_key}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]:.4f}")


if __name__ == "__main__":
    main()
