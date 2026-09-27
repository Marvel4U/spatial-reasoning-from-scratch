"""Mix checkpoints: under the m1 token, share of food&drink cells marked inside vs outside the 100 m disc."""
import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
RUN = sys.argv[1]
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data, grid_vit_adapter as A
from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
data.setup_device(); data.load_data()
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt"); model.train(False)
vl = data.val_loader
keys = [s.key for s in vl.task_specs]; m1 = keys.index("food_drink_within_100m")
inside = outside = n_in = n_out = 0.0; ious = []
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    for img, tgt, tix, cond in vl.iter_batches_for_task(m1, max_batches=16):
        estab = img[:, 14] > 0.5
        mk = img[:, 15].flatten(1).argmax(1); mr, mc = mk // 256, mk % 256
        disc = _within_radius_pix(mc, mr, 100.0, h=256, w=256)
        est_c = _cells_from_pix(estab).bool(); disc_c = _cells_from_pix(disc).bool()
        logits, _ = A.model_forward(model, img, None, tix, cond)
        pred = torch.softmax(logits.float(), -1)[..., 1] > 0.5
        inside += (pred & est_c & disc_c).sum().item(); n_in += (est_c & disc_c).sum().item()
        outside += (pred & est_c & ~disc_c).sum().item(); n_out += (est_c & ~disc_c).sum().item()
        t = tgt.bool(); i = (pred & t).flatten(1).sum(1).float(); u = (pred | t).flatten(1).sum(1).float(); ok = u > 0
        ious.append((i[ok] / u[ok]).mean().item())
print(f"{RUN}: m1 IoU {sum(ious)/len(ious):.3f} | food&drink cells marked: inside disc {inside/n_in:.3f}, outside disc {outside/n_out:.3f}")
