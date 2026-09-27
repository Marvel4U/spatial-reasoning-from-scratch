"""Forward probe batch with hooks; scalar metrics for eval.jsonl."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import config
import data
from plain_gpt_module import unwrap_compiled
from spatial_batch import forward as model_forward

from .attn_probe import attention_matrix, full_seq_sidewalk_mask, layer_probe_scalars
from .probe_batch import ProbeBatch


@torch.no_grad()
def probe_forward_metrics(model, probe: ProbeBatch) -> dict[str, float]:
    core = unwrap_compiled(model)
    was_training = model.training
    model.eval()
    device = probe.img.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    acts: dict[str, torch.Tensor] = {}
    handles = []

    def save_out(name):
        def f(_mod, _inp, out):
            acts[name] = out.detach()
        return f

    def save_in(name):
        def f(_mod, inp):
            acts[name] = inp[0].detach()
        return f

    for i, blk in enumerate(core.encoder):
        handles.append(blk.register_forward_hook(save_out(f"resid/{i}")))
        handles.append(blk.mlp.c_fc.register_forward_hook(save_out(f"mlp_pre/{i}")))
        handles.append(blk.attn.register_forward_pre_hook(save_in(f"attn_in/{i}")))

    cfg = core.config
    marker_tok = probe.marker_token_indices(cfg)
    sw_full, sw_frac = None, None
    c3_marker_task = False
    if config.task_rung == "c3":
        from spatial_data.c3_tasks import get_c3_task
        c3_marker_task = get_c3_task(config.c3_eval_key()).show_marker_plane
        if not c3_marker_task:
            sw_full, sw_frac = full_seq_sidewalk_mask(probe.img, cfg)
    rec: dict[str, float] = {}
    try:
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, _ = model_forward(model, probe.img, None, probe.task_ids, probe.cond_ids)
        fg = cfg.foreground_class
        if cfg.num_classes > 1:
            score = logits[..., fg] - logits[..., 0]
        else:
            score = logits.reshape(logits.shape[0], -1)
        rec["head_logit_std"] = score.float().std().item()

        for i, blk in enumerate(core.encoder):
            x = acts[f"attn_in/{i}"].float()
            b, t, c = x.shape
            nh = blk.attn.n_head
            att = attention_matrix(x, blk.attn)
            ent = -(att * att.clamp_min(1e-12).log()).sum(-1).mean(dim=(0, 2))
            for h in range(nh):
                rec[f"attn_entropy/{i}/{h}"] = ent[h].item()
            if config.task_rung == "c3" and sw_full is not None:
                rec.update(layer_probe_scalars(att, sw_full, sw_frac, i))
            elif config.task_rung == "c3" and c3_marker_task:
                tok = marker_tok.to(att.device).view(b, 1, 1, 1).expand(b, nh, t, 1)
                to_marker = att.gather(dim=-1, index=tok).squeeze(-1)
                mass = to_marker.mean(dim=(0, 2))
                for h in range(nh):
                    rec[f"attn_to_marker/{i}/{h}"] = mass[h].item()
            else:
                tok = marker_tok.to(att.device).view(b, 1, 1, 1).expand(b, nh, t, 1)
                to_marker = att.gather(dim=-1, index=tok).squeeze(-1)
                mass = to_marker.mean(dim=(0, 2))
                for h in range(nh):
                    rec[f"attn_to_marker/{i}/{h}"] = mass[h].item()
            rec[f"resid_rms/{i}"] = acts[f"resid/{i}"].float().pow(2).mean().sqrt().item()
            rec[f"gelu_off/{i}"] = (acts[f"mlp_pre/{i}"] < -3).float().mean().item()
        if config.task_rung == "c3" and sw_frac is not None:
            rec["probe_sw_patch_frac"] = sw_frac.mean().item()
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in rec.items()}
