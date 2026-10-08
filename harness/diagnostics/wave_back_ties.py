"""Diagnostic (2 Oct 2026): where do the false cells of the backward hop come from?

For val samples of wave_back / wave_origin the teacher's field is recomputed (same code as spatial_data/c5_wave.py)
and the checkpoint's prediction is compared with:
  - the hop of the target (new cells = target lit, pad unlit),
  - the tie family: every pixel on SOME cheapest path start -> tip (priced metric), within the hop's range; the
    target hop is one member picked by Dijkstra's parent links. Near-tie families allow +0.5 / +2 priced metres.
  - the cheapest connection from the tip to ANY point of the next front (fronts are rings of walking length, the
    wave is priced, so the teacher's hop need not be the cheapest way to the front).
False cells are attributed in this order: consumed front not erased / inside the exact tie family / inside the
+2 m family / within 4 m of the hop / within 4 m of any target cell / elsewhere.

Usage (from harness/): PYTHONPATH=. ../.venv/bin/python diagnostics/wave_back_ties.py RUN_ID [--n 300] [--task wave_back]
Writes runs/diagnostics/RUN_ID/wave_back_ties_{task}.json and .png."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra

import config, data, eval_c4, experiments, grid_vit_adapter
import harness as H
from spatial_data import c5_trajectories as T
from spatial_data.c4_tasks import get_c4_task
from spatial_data.c5_wave import BUILDING_CLASS, MARGIN_M
from spatial_data.dataset_c4_gpu import C4GpuValLoader, load_c4_route_index, load_district_planes_v4


def find_spec(name: str) -> dict:
    """A spec builder name or a run id from any c5_ queue builder (as in the ground check)."""
    fn = getattr(experiments, name, None)
    if fn is not None:
        return fn()
    for attr in dir(experiments):
        f = getattr(experiments, attr)
        if attr.startswith("c5_") and callable(f):
            try:
                out = f()
            except TypeError:
                continue
            for sp in (out if isinstance(out, list) else [out]):
                if isinstance(sp, dict) and sp.get("id") == name:
                    return sp
    raise SystemExit(f"no spec builder or run id {name!r}")


ap = argparse.ArgumentParser()
ap.add_argument("run_id"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--task", default="wave_back")
ap.add_argument("--examples", type=int, default=8)
ap.add_argument("--spec", default=None, help="spec builder expression if the run id is not found, e.g. a builder name")
args = ap.parse_args()

data.setup_device()
spec = H._resolve_spec(find_spec(args.spec or args.run_id)); spec["id"] = args.run_id
H._apply_overrides(dict(spec["overrides"])); config.sync_in_chans_from_encoding()
import spatial_data.channels as _ch
_ch.MARKER_RADIUS_PX = int(config.marker_radius_px)
ckpt_path = Path(config.harness_checkpoint_dir) / f"{spec['id']}.pt"
from plain_gpt_module.checkpoint import read_checkpoint
n_cond = int(read_checkpoint(ckpt_path, map_location="cpu")["model_state_dict"]["cond_emb.weight"].shape[0])
_orig = config.c4_model_extra_config
config.c4_model_extra_config = lambda: {**_orig(), "n_cond_emb": n_cond}
model = grid_vit_adapter.build_model(); grid_vit_adapter.load_checkpoint(ckpt_path, model)
config.c4_model_extra_config = _orig
model.train(False)

crops, store = config.c3_crops_dir(), config.c4_routes_dir()
loader = C4GpuValLoader(get_c4_task(args.task), config.encoding_mode, config.grid_out_size, 64,
                        district=load_district_planes_v4(crops, data.device),
                        route_index=load_c4_route_index(store, "val", data.device, max_items=args.n))
rz = np.load(Path(store) / "c4_routes_val.npz"); origin, markers = rz["origin"], rz["markers"]
bld = np.load(Path(store).parent / "district_labels_1m.npz")["surface"] == BUILDING_CLASS
crop = config.img_size; g = config.grid_out_size; s = crop // g
K3 = np.ones((3, 3), bool)


def cells(px_xy, thick=True):
    out = np.zeros((g, g), bool)
    if len(px_xy):
        ok = (px_xy[:, 0] >= 0) & (px_xy[:, 0] < crop) & (px_xy[:, 1] >= 0) & (px_xy[:, 1] < crop)
        out[px_xy[ok, 1] // s, px_xy[ok, 0] // s] = True
    return ndimage.binary_dilation(out, K3) if thick else out


def teacher(i, new):
    """Recompute the field of val route i; return the hop's tie families and the front-connection gap."""
    dx, dy = int(origin[i][0]), int(origin[i][1]); M = int(MARGIN_M)
    x0, y0 = max(dx - M, 0), max(dy - M, 0)
    mask = bld[y0:min(dy + crop + M, bld.shape[0]), x0:min(dx + crop + M, bld.shape[1])]
    off = np.array([dx - x0, dy - y0])
    start, goal = markers[i][0] + off, markers[i][1] + off
    f = T._Field(mask)
    src, gl = f.idx[int(start[1]), int(start[0])], f.idx[int(goal[1]), int(goal[0])]
    D, pred = dijkstra(f.G, indices=src, return_predecessors=True)
    L = T._walk_length(pred, np.isfinite(D), f.fy, f.fx, src)
    nodes = [gl]
    while nodes[-1] != src:
        nodes.append(int(pred[nodes[-1]]))
    nodes = np.array(nodes)
    wx, wy = f.fx[nodes] - off[0], f.fy[nodes] - off[1]
    inwin = (wx >= 0) & (wx < crop) & (wy >= 0) & (wy < crop)
    core = np.zeros(len(nodes), bool)
    core[inwin] = new[wy[inwin] // s, wx[inwin] // s]
    if not core.any():
        return None
    idx = np.nonzero(core)[0]
    hop = nodes[idx[0]:idx[-1] + 1]                       # goal side first (the tip), start side last (on the next front)
    tip, end = int(hop[0]), int(hop[-1])
    Dt = dijkstra(f.G, indices=tip)
    hop_cost = D[tip] - D[end]
    slack = D + Dt - D[tip]                                # 0 on every cheapest path start -> tip
    rng = (D >= D[end] - 1e-6) & (D <= D[tip] + 1e-6)
    fam = {}
    for name, eps in (("tie", 1e-6), ("tie05", 0.5), ("tie2", 2.0)):
        m = rng & (slack <= eps)
        fam[name] = cells(np.stack([f.fx[m] - off[0], f.fy[m] - off[1]], 1))
    ring = np.isfinite(L) & (np.abs(L - L[end]) <= 1.0)
    ring &= (f.fx - off[0] >= 0) & (f.fx - off[0] < crop) & (f.fy - off[1] >= 0) & (f.fy - off[1] < crop)
    best = float(Dt[ring].min()) if ring.any() else hop_cost
    k = int(np.argmin(np.where(ring, Dt, np.inf)))
    return dict(fam=fam, hop_cost=float(hop_cost), front_best_cost=best, first_hop=float(idx[0] <= 2),
                j=float(round(L[end] / 40.0) + 1), n_goal=float(np.floor(L[gl] / 40.0) + 1),
                front_best_shift_m=float(np.hypot(f.fx[k] - f.fx[end], f.fy[k] - f.fy[end])),
                hop_walk_m=float(L[tip] - L[end]), hop_chord_m=float(np.hypot(f.fx[tip] - f.fx[end], f.fy[tip] - f.fy[end])),
                clr_mean=float(f.clr[f.fy[hop], f.fx[hop]].mean()), flat_share=float((f.clr[f.fy[hop], f.fx[hop]] >= T.CAP).mean()))


rows, examples = [], []
TAUS = (0.3, 0.5, 0.7, 0.8, 0.88, 0.95)
sweep = {tau: [] for tau in TAUS}
dev_type = "cuda" if str(data.device).startswith("cuda") else "cpu"
for img, tgt, task_id, cond_ids, meta in eval_c4._iter_c4_val(loader, None):
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=data.dtype):
        logits, _ = eval_c4.model_forward(model, img, tgt, task_id, cond_ids)
    prob = torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()
    tgt_np = tgt.cpu().numpy(); pad = meta["pad_cells"].cpu().numpy().astype(bool); rix = meta["route_ix"].cpu().numpy()
    for b in range(len(rix)):
        t = tgt_np[b] == 1; p = prob[b] > 0.5; new = t & ~pad[b]
        if not new.any():
            continue
        te = teacher(int(rix[b]), new)
        if te is None:
            continue
        fp = p & ~t
        near_hop = ndimage.binary_dilation(new, K3, iterations=2); near_any = ndimage.binary_dilation(t, K3, iterations=2)
        cat = {}
        rest = fp.copy()
        for name, m in (("unerased_front", pad[b] & ~t), ("tie", te["fam"]["tie"]), ("tie2", te["fam"]["tie2"]),
                        ("near_hop_4m", near_hop), ("near_target_4m", near_any)):
            cat[name] = int((rest & m).sum()); rest &= ~m
        cat["elsewhere"] = int(rest.sum())
        erase = pad[b] & ~t
        for tau in TAUS:                                   # the same sample read at other decision thresholds
            q = prob[b] > tau
            sweep[tau].append((float(q[new].mean()), int((q & ~t).sum()), float((~q[erase]).mean()) if erase.any() else 1.0,
                               float(q[pad[b] & t].mean()) if (pad[b] & t).any() else 1.0))
        pnew = p & ~pad[b] & near_hop
        rows.append(dict(route=int(rix[b]), n_new=int(new.sum()), recall_new=float(p[new].mean()), n_false=int(fp.sum()),
                         n_erase=int(erase.sum()), erased=float((~p[erase]).mean()) if erase.any() else 1.0,
                         fam_tie=int((te["fam"]["tie"] & ~pad[b]).sum()), fam_tie05=int((te["fam"]["tie05"] & ~pad[b]).sum()),
                         fam_tie2=int((te["fam"]["tie2"] & ~pad[b]).sum()),
                         pred_in_tie2=float((pnew & te["fam"]["tie2"]).sum() / max(pnew.sum(), 1)),
                         **{k: te[k] for k in te if k != "fam"}, **{"fp_" + k: v for k, v in cat.items()}))
        if len(examples) < args.examples:
            examples.append((int(rix[b]), pad[b], t, prob[b], te["fam"]["tie"], te["fam"]["tie2"]))

R = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
n = len(rows)
ratio = R["fam_tie"] / R["n_new"]; ratio2 = R["fam_tie2"] / R["n_new"]
gap = (R["hop_cost"] - R["front_best_cost"]) / np.maximum(R["hop_cost"], 1e-6)
out = dict(run=spec["id"], task=args.task, n=n,
           new_cells_mean=R["n_new"].mean(), recall_new=R["recall_new"].mean(), false_cells_mean=R["n_false"].mean(),
           erased_share=R["erased"].mean(),
           false_by_cause={k[3:]: round(float(R[k].sum() / max(R["n_false"].sum(), 1)), 3) for k in R if k.startswith("fp_")},
           family_cells_over_hop_cells=dict(exact_median=float(np.median(ratio)), exact_p90=float(np.percentile(ratio, 90)),
                                            plus2m_median=float(np.median(ratio2)), plus2m_p90=float(np.percentile(ratio2, 90)),
                                            share_hops_exact_family_ge_1p5x=float((ratio >= 1.5).mean())),
           pred_new_cells_inside_plus2m_family=R["pred_in_tie2"].mean(),
           hop=dict(walk_m_mean=R["hop_walk_m"].mean(), chord_m_mean=R["hop_chord_m"].mean(), flat_share_mean=R["flat_share"].mean()),
           front_connection=dict(share_hops_with_cheaper_front_point_gt5pct=float((gap > 0.05).mean()),
                                 share_gt20pct=float((gap > 0.20).mean()), gap_median=float(np.median(gap)),
                                 shift_m_median=float(np.median(R["front_best_shift_m"])),
                                 shift_m_p90=float(np.percentile(R["front_best_shift_m"], 90))))
bins = {}
for name, x in (("family_ratio_plus2m", ratio2), ("flat_share", R["flat_share"]), ("front_gap", gap)):
    q = np.quantile(x, [1 / 3, 2 / 3]); which = np.digitize(x, q)
    bins[name] = [dict(tercile=int(k), lo=float(x[which == k].min()), hi=float(x[which == k].max()), n=int((which == k).sum()),
                       recall_new=float(R["recall_new"][which == k].mean()), false_cells=float(R["n_false"][which == k].mean()))
                  for k in range(3) if (which == k).any()]
    bins[name + "_corr_false"] = float(np.corrcoef(x, R["n_false"])[0, 1]); bins[name + "_corr_recall"] = float(np.corrcoef(x, R["recall_new"])[0, 1])
out["by_tercile"] = bins
out["threshold_sweep"] = [dict(tau=tau, recall_new=float(np.mean([v[0] for v in sweep[tau]])), false_cells=float(np.mean([v[1] for v in sweep[tau]])),
                               erased=float(np.mean([v[2] for v in sweep[tau]])), copied=float(np.mean([v[3] for v in sweep[tau]]))) for tau in TAUS]
blob = R["n_false"] - R["fp_unerased_front"]
out["first_hop_vs_later"] = {name: dict(n=int(m.sum()), recall_new=float(R["recall_new"][m].mean()), false_cells=float(R["n_false"][m].mean()),
                                        blob_cells=float(blob[m].mean()), new_cells=float(R["n_new"][m].mean()), erased=float(R["erased"][m].mean()))
                               for name, m in (("first hop (tip = goal marker, no path on the pad)", R["first_hop"] > 0), ("later hops (tip = end of the drawn path)", R["first_hop"] == 0)) if m.any()}
out["by_fronts_on_pad"] = [dict(j=int(jj), n=int(m.sum()), recall_new=float(R["recall_new"][m].mean()), false_cells=float(R["n_false"][m].mean()),
                                erased=float(R["erased"][m].mean()), blob_cells=float(blob[m].mean()), first_pass_share=float((R["j"][m] == R["n_goal"][m]).mean()))
                           for jj in sorted(set(R["j"].astype(int))) for m in [R["j"].astype(int) == jj] if m.sum() >= 5]
out["by_hop_length"] = [dict(cells=f"{lo}-{hi}", n=int(m.sum()), recall_new=float(R["recall_new"][m].mean()), false_cells=float(R["n_false"][m].mean()))
                        for lo, hi in ((0, 20), (20, 50), (50, 80), (80, 10 ** 6)) for m in [(R["n_new"] >= lo) & (R["n_new"] < hi)] if m.any()]
odir = Path(config.REPO_ROOT) / "runs/diagnostics" / spec["id"]
(odir / f"wave_back_ties_{args.task}.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(len(examples), 4, figsize=(13, 3.2 * len(examples)))
for r, (rid, pd_, t, pr, fam, fam2) in enumerate(examples):
    new = t & ~pd_
    ys, xs = np.nonzero(ndimage.binary_dilation(new | fam2, K3, iterations=6)); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sl = (slice(y0, y1), slice(x0, x1))
    a = np.zeros((g, g, 3)); a[pd_] = (0.25, 0.45, 1.0)
    b_ = np.zeros((g, g, 3)); b_[t] = (0.5, 0.5, 0.5); b_[new] = (1, 0.85, 0)
    c = np.zeros((g, g, 3)); c[fam2] = (0.0, 0.35, 0.0); c[fam] = (0.0, 0.8, 0.0); c[new] = (1, 0.85, 0)
    for k, (im, title) in enumerate(((a, f"route {rid}: pad in"), (b_, "target (hop yellow)"), (None, "P(fg)"),
                                     (c, "tie family: exact green, +2 m dark green, teacher's pick yellow"))):
        ax[r, k].imshow(pr[sl], cmap="magma", vmin=0, vmax=1) if im is None else ax[r, k].imshow(im[sl])
        ax[r, k].set_title(title, fontsize=8); ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
fig.suptitle(f"{spec['id']} · {args.task} · zoom on the hop", fontsize=10); fig.tight_layout()
fig.savefig(odir / f"wave_back_ties_{args.task}.png", dpi=110)
print("wrote", odir / f"wave_back_ties_{args.task}.png")
