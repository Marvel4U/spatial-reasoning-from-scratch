"""c4 val: IoU(pred, target) vs IoU(pred, segment shortcut) on detour task.

Usage (project venv, from repo root):
  python docs/plans/perf_prototypes/c4_route_shortcut_probe.py [run_id]

Default run_id: c4_loader_smoke (Phase B/C checkpoint).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO / "harness")
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import torch
import config

RUN = sys.argv[1] if len(sys.argv) > 1 else "c4_loader_smoke"
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
if RUN not in led:
    raise SystemExit(f"unknown run {RUN!r}")
for k, v in led[RUN]["overrides"].items():
    if hasattr(config, k):
        setattr(config, k, v)
config.verbose = False
import data
import grid_vit_adapter as A
from eval_c4 import segment_grid_from_markers, _iou_np, pred_connects_markers
from spatial_data.c4_tasks import TASK_KEYS

data.setup_device()
data.load_data()
vl = data.val_loader
if not hasattr(vl, "next_batch_with_meta"):
    raise SystemExit("val loader is not c4")
model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{RUN}.pt")
model.eval()
keys = list(config.c4_keys_active())
detour_ix = keys.index("detour")
preds, tgts, seg_masks, conns = [], [], [], []
vl.reset()
n_batches = (len(vl) + vl.B - 1) // vl.B
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    for _ in range(n_batches):
        img, tgt, tix, cond, meta = vl.next_batch_with_meta()
        logits, _ = A.model_forward(model, img, None, tix, cond)
        p = (logits.argmax(-1) == 1).cpu().numpy()
        t = (tgt.cpu().numpy() == 1)
        markers = meta["markers"].cpu().numpy()
        tix_np = meta["task_ix"].cpu().numpy()
        for i in range(p.shape[0]):
            if int(tix_np[i]) != detour_ix:
                continue
            preds.append(p[i])
            tgts.append(t[i])
            seg_masks.append(segment_grid_from_markers(markers[i]))
            conns.append(pred_connects_markers(p[i], markers[i]))
if not preds:
    raise SystemExit("no detour val items in loader")
import numpy as np

def mean_iou(a_list, b_list):
    xs = [_iou_np(a, b) for a, b in zip(a_list, b_list)]
    return float(np.mean(xs)), len(xs)

iou_tgt, n = mean_iou(preds, tgts)
iou_seg, _ = mean_iou(preds, seg_masks)
print("run:", RUN, "detour val n =", n)
print("  IoU(pred, detour target) = %.4f" % iou_tgt)
print("  IoU(pred, segment mask)  = %.4f  (shortcut: higher => more segment-like)" % iou_seg)
print("  connectivity             = %.4f" % float(np.mean(conns)))
print("  shortcut gap (seg - tgt) = %+.4f" % (iou_seg - iou_tgt))
