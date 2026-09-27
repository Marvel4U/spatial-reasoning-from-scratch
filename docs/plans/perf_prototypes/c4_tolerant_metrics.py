"""c4 val: plain IoU vs 1-cell-tolerant metrics for thin-line targets."""
import sys, os, json
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, torch.nn.functional as F, config
RUN = sys.argv[1]
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data, grid_vit_adapter as A
data.setup_device(); data.load_data()
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt"); model.train(False)
vl = data.val_loader; vl.reset()
def dil(x): return F.max_pool2d(x.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0
tot = dict(iou=[], iou_d=[], prec1=[], rec1=[]); n = 0
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    for batch in vl.iter_batches():
        img, tgt, tix, cond = batch[:4]
        logits, _ = A.model_forward(model, img, None, tix, cond)
        pred = torch.softmax(logits.float(), -1)[..., 1] > 0.5; t = tgt.bool()
        i = (pred & t).flatten(1).sum(1).float(); u = (pred | t).flatten(1).sum(1).float(); ok = u > 0
        tot["iou"] += (i[ok] / u[ok]).tolist()
        pd, td = dil(pred), dil(t)
        i2 = (pd & td).flatten(1).sum(1).float(); u2 = (pd | td).flatten(1).sum(1).float()
        tot["iou_d"] += (i2[ok] / u2[ok]).tolist()
        pp = pred.flatten(1).sum(1).float(); tt = t.flatten(1).sum(1).float()
        prec = (pred & td).flatten(1).sum(1).float() / pp.clamp(min=1); rec = (t & pd).flatten(1).sum(1).float() / tt.clamp(min=1)
        tot["prec1"] += prec[tt > 0].tolist(); tot["rec1"] += rec[tt > 0].tolist(); n += int(ok.sum())
m = {k: sum(v) / len(v) for k, v in tot.items()}
print(f"{RUN}: n={n} | plain IoU {m['iou']:.3f} | IoU after 1-cell dilation of both {m['iou_d']:.3f} | precision within 1 cell {m['prec1']:.3f} | recall within 1 cell {m['rec1']:.3f}")
