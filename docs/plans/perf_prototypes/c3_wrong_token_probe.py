"""Wrong-token test on a mixed-task checkpoint: feed m1 val crops with each task's condition token.
If the tokens steer, the output under the food_drink token is the dense mask, under within_100m the disc,
under m1 the conjunction. Also: attention-to-marker of the strongest heads under each token."""
import sys, os, json, math
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, torch.nn.functional as F, config
RUN = sys.argv[1] if len(sys.argv) > 1 else "food_drink_within_100m_mix33_L4_16k_b32_disc3_relV2"
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data, grid_vit_adapter as A
from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
data.setup_device(); data.load_data()
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt"); model.train(False)
core = model._orig_mod if hasattr(model, "_orig_mod") else model
vl = data.val_loader
keys = [s.key for s in vl.task_specs]; m1 = keys.index("food_drink_within_100m")
cond_tab = vl._cond
print("tasks:", keys, "cond table:", cond_tab.tolist())
def iou(a, b):
    i = (a & b).flatten(1).sum(1).float(); u = (a | b).flatten(1).sum(1).float(); ok = u > 0
    return (i[ok] / u[ok]).mean().item()
acc = {k: {"tgt": [], "estab": [], "disc": []} for k in keys}; atm = {k: {} for k in keys}
n_prefix = core.config.n_cond_token_slots
for img, tgt, tix, cond in vl.iter_batches_for_task(m1, max_batches=16):
    B = img.shape[0]
    estab = img[:, 14] > 0.5                                    # onehot_v4: estab classes at 11..14, class 3 = channel 14
    mk = img[:, 15].flatten(1).argmax(1); mr, mc = mk // 256, mk % 256
    disc = _within_radius_pix(mc, mr, 100.0, h=256, w=256)
    est_c = _cells_from_pix(estab).bool(); disc_c = _cells_from_pix(disc).bool(); tgt_c = tgt.bool()
    P = core.config.patch_size; g = 256 // P; mtok = (mr // P) * g + (mc // P) + n_prefix
    for ti, key in enumerate(keys):
        c = cond_tab[ti].view(1, -1).expand(B, -1)
        acts = {}
        hs = [blk.attn.register_forward_pre_hook((lambda name: (lambda m, i: acts.__setitem__(name, i[0].detach())))(f"in/{k}")) for k, blk in enumerate(core.encoder)]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
            logits, _ = A.model_forward(model, img, None, torch.full_like(tix, ti), c)
        for h in hs: h.remove()
        pred = torch.softmax(logits.float(), -1)[..., 1] > 0.5
        acc[key]["tgt"].append(iou(pred, tgt_c)); acc[key]["estab"].append(iou(pred, est_c)); acc[key]["disc"].append(iou(pred, disc_c))
        for bi, blk in enumerate(core.encoder):
            x = acts[f"in/{bi}"].float(); _, T, C = x.shape; nh = blk.attn.n_head
            q, kk, _ = blk.attn.c_attn(x).split(C, dim=2)
            q = q.view(B, T, nh, C // nh).transpose(1, 2); kk = kk.view(B, T, nh, C // nh).transpose(1, 2)
            lg = q @ kk.transpose(-2, -1) / math.sqrt(C // nh)
            if blk.rel_pos_bias is not None:
                lg = lg + blk.rel_pos_bias(core.rel_pos_bucket_idx).permute(2, 0, 1).unsqueeze(0)
            att = F.softmax(lg, -1)
            tok = mtok.view(B, 1, 1, 1).expand(B, nh, T, 1)
            mass = att.gather(-1, tok).squeeze(-1).mean(dim=(0, 2))
            for h in range(nh): atm[key].setdefault(f"{bi}/{h}", []).append(mass[h].item())
print(f"\nm1 val crops, prediction under each token (IoU against: m1 target | all food&drink | disc)")
for key in keys:
    a = acc[key]; print(f"  token {key:24s}: target {sum(a['tgt'])/len(a['tgt']):.3f} | food&drink {sum(a['estab'])/len(a['estab']):.3f} | disc {sum(a['disc'])/len(a['disc']):.3f}")
print("\nattention to marker token, top heads, per token:")
allheads = sorted({h for k in keys for h in atm[k]}, key=lambda h: -max(sum(atm[k][h])/len(atm[k][h]) for k in keys))[:4]
for h in allheads:
    print(f"  head {h}: " + "  ".join(f"{k}={sum(atm[k][h])/len(atm[k][h]):.2f}" for k in keys))
