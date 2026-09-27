"""m1 val: IoU of shortcuts vs true target, and how close the trained model is to each shortcut."""
import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
RUN = sys.argv[1] if len(sys.argv) > 1 else "food_drink_within_100m_rand1"
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data, grid_vit_adapter as A
data.setup_device(); data.load_data()
from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
vl = data.val_loader
crops = vl.crops; N = crops.shape[0]
mc, mr = vl._marker_col, vl._marker_row
print("val crops", tuple(crops.shape), crops.dtype, "markers", tuple(mc.shape))
estab = crops[:, 2] == 3                                   # (N,256,256) food & drink pixels
disc = _within_radius_pix(mc, mr, 100.0, h=256, w=256)     # (N,256,256)
est_c = _cells_from_pix(estab).bool(); disc_c = _cells_from_pix(disc).bool(); tgt_c = _cells_from_pix(estab & disc).bool()
def iou(a, b):
    i = (a & b).flatten(1).sum(1).float(); u = (a | b).flatten(1).sum(1).float()
    ok = u > 0; return (i[ok] / u[ok]).mean().item(), int(ok.sum())
print("cells fg share: target %.4f  estab %.4f  disc %.4f" % (tgt_c.float().mean(), est_c.float().mean(), disc_c.float().mean()))
print("crops with empty target: %.2f" % (tgt_c.flatten(1).sum(1) == 0).float().mean())
print("IoU(estab-everywhere, target) = %.3f over %d crops" % iou(est_c, tgt_c))
print("IoU(disc-only, target)        = %.3f over %d crops" % iou(disc_c, tgt_c))
# model
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt"); model.train(False)
preds = []
vl.reset()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    for img, tgt, tix, cond in vl.iter_batches():
        logits, _ = A.model_forward(model, img, None, tix, cond)
        p = torch.softmax(logits.float(), -1)[..., 1]
        preds.append(p > 0.5)
pred = torch.cat(preds)[:N]
print("model:", RUN)
print("  IoU(pred, target)          = %.3f" % iou(pred, tgt_c)[0])
print("  IoU(pred, estab-everywhere)= %.3f" % iou(pred, est_c)[0])
print("  IoU(pred, disc-only)       = %.3f" % iou(pred, disc_c)[0])
fp_out = (pred & est_c & ~disc_c).flatten(1).sum(1).float(); est_out = (est_c & ~disc_c).flatten(1).sum(1).float()
ok = est_out > 0
print("  share of food&drink cells OUTSIDE the disc that the model marks: %.3f" % (fp_out[ok] / est_out[ok]).mean())
inn = (pred & tgt_c).flatten(1).sum(1).float(); tin = tgt_c.flatten(1).sum(1).float(); ok = tin > 0
print("  share of target cells (inside disc) the model marks: %.3f" % (inn[ok] / tin[ok]).mean())
# distance profile: marked share of food&drink cells by distance band from marker
g = 64; cy = (mr.float() / 4).view(N, 1, 1); cx = (mc.float() / 4).view(N, 1, 1)
yy = torch.arange(g, device=crops.device).view(1, g, 1).float(); xx = torch.arange(g, device=crops.device).view(1, 1, g).float()
d_m = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() * 4  # cell distance in metres (4 m per cell? 256 m / 64)
for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 400)):
    band = est_c & (d_m >= lo) & (d_m < hi)
    n = band.sum().item(); m = (pred & band).sum().item()
    print(f"  food&drink cells at {lo:3d}-{hi:3d} m from marker: marked {m/n if n else float('nan'):.3f}  (n={n})")
