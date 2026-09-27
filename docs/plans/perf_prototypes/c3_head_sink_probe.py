import sys, os, json, math
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, torch.nn.functional as F, config
run = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 else f"runs/checkpoints/{run}.pt"
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[run]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data, grid_vit_adapter as A
data.setup_device(); data.load_data()
model, _ = A.load_checkpoint(REPO / ck); model.train(False)
core = model._orig_mod if hasattr(model, "_orig_mod") else model
img, tgt, tix, cond = data.val_loader.next_batch()
n_prefix = core.config.n_cond_token_slots
acts = {}
hs = [blk.attn.register_forward_pre_hook((lambda name: (lambda m, i: acts.__setitem__(name, i[0].detach())))(f"in/{k}")) for k, blk in enumerate(core.encoder)]
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    core(img, task_id=tix, cond_ids=cond)
for h in hs: h.remove()
print(f"== {run} @ {ck}")
for k, blk in enumerate(core.encoder):
    x = acts[f"in/{k}"].float(); B, T, C = x.shape; nh = blk.attn.n_head
    q, kk, _ = blk.attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T, nh, C // nh).transpose(1, 2); kk = kk.view(B, T, nh, C // nh).transpose(1, 2)
    logits = q @ kk.transpose(-2, -1) / math.sqrt(C // nh)
    if blk.rel_pos_bias is not None:
        logits = logits + blk.rel_pos_bias(core.rel_pos_bucket_idx).permute(2, 0, 1).unsqueeze(0)
    att = F.softmax(logits, dim=-1)
    prefix_mass = att[..., :n_prefix].sum(-1).mean(dim=(0, 2))            # per head
    self_mass = att.diagonal(dim1=-2, dim2=-1).mean(dim=(0, 2))
    top = att.argmax(-1)                                                   # (B, nh, T)
    frac_top_prefix = (top < n_prefix).float().mean(dim=(0, 2))
    lstd = logits.std(dim=(-1)).mean(dim=(0, 2))
    print(f"block {k}: mass on prefix tokens per head {[round(v,2) for v in prefix_mass.tolist()]}")
    print(f"         mass on self per head          {[round(v,2) for v in self_mass.tolist()]}")
    print(f"         frac rows whose top token is a prefix token {[round(v,2) for v in frac_top_prefix.tolist()]}")
    print(f"         logit std over keys per head   {[round(v,1) for v in lstd.tolist()]}")
    xn = x.norm(dim=-1); print(f"         token norm: prefix {xn[:, :n_prefix].mean():.1f}  patches {xn[:, n_prefix:].mean():.1f}")
