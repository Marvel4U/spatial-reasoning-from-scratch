"""Read-only probe of the within_20m batch the harness feeds the model."""
import sys, os, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
for k, v in led["c3_within_20m_2k"]["overrides"].items():
    if hasattr(config, k): setattr(config, k, v)
config.verbose = False
import data
data.setup_device(); data.load_data()
L = data.train_loader
img, tgt, tix, cond = L.next_batch()
print("img", tuple(img.shape), "in_chans", config.in_chans, "| tgt", tuple(tgt.shape), tgt.dtype, "| cond_ids", cond[0].tolist(), "| task_index", tix[:4].tolist())
mk = img[:, -1]
print("marker plane: ones per sample", mk.flatten(1).sum(1)[:8].tolist(), "| any other channel touched by marker?", [float(img[:, c].max()) for c in range(img.shape[1])][-3:])
print("target: fg cells per sample", (tgt == 1).flatten(1).sum(1)[:8].tolist(), "(a 20 m disc at 1 m/px ~ 1257 px ~ 79 cells at 4x4 majority)")
# does the target disc sit where the marker is?
for b in range(3):
    r, c = torch.nonzero(mk[b])[0].tolist()
    fr, fc = torch.nonzero(tgt[b] == 1).float().mean(0).tolist()
    print(f"  sample {b}: marker px (row {r}, col {c}) -> cell ({r//4},{c//4}); target centroid cell ({fr:.1f},{fc:.1f})")
print("marker stored cols/rows range:", int(L._marker_col.min()), int(L._marker_col.max()), int(L._marker_row.min()), int(L._marker_row.max()), "| n markers", len(L._marker_col), "| n crops", L.crops.shape[0])
# is the same marker always used for a crop? sample the same crop twice
ix = torch.tensor([5, 5, 5], device=L.device)
a = L._build(L.crops[ix], ix)[0][:, -1].flatten(1).argmax(1).tolist()
print("crop 5 sampled 3x -> marker position:", a)
# forward pass of the trained checkpoint: how many fg cells does it predict?
import grid_vit_adapter as A
model, _ = A.load_checkpoint(REPO / "runs/checkpoints/c3_within_20m_2k.pt")
model.train(False)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    logits, _ = model(img, task_id=tix, cond_ids=cond)
p = torch.softmax(logits.float(), -1)[..., 1]
print("pred P(fg): max per sample", [round(x, 3) for x in p.flatten(1).max(1).values[:6].tolist()], "| mean", round(float(p.mean()), 4))
print("pred fg cells at 0.5:", (p > 0.5).flatten(1).sum(1)[:6].tolist())
