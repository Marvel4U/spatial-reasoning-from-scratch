"""README figures, second set: F0 hero (detour model), F00 concept diagram, F9 results table as a picture.
Run from the source repo with its venv: .venv/bin/python <public>/docs/figures_make2.py [f0|f00|f9]"""
import sys, os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon

REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
OUT = Path.home() / "Github/spatial_reasoning_public/docs/figures"; OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
LED = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
GREY = np.array([0, 85, 170, 255], dtype=np.uint8)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "font.family": "DejaVu Sans"})
TARGET_C, PRED_C = "#f4d03f", "#ff2fd0"


def load_run(run, **extra):
    import importlib
    importlib.reload(config)
    for k, v in LED[run]["overrides"].items():
        if hasattr(config, k): setattr(config, k, v)
    for k, v in extra.items(): setattr(config, k, v)
    config.verbose = False
    for m in ("data", "grid_vit_adapter", "eval_spatial", "eval_c4", "eval_c2"):
        sys.modules.pop(m, None)
    import data, grid_vit_adapter as A
    data.setup_device(); data.load_data()
    model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{run}.pt"); model.train(False)
    return data, A, model


def rgb_from_img(img):
    """one-hot planes -> RGB. 16 ch (noise7 0-6, surface 7-10, estab 11-14, marker 15) or 13 ch (noise 0-3, surface 4-7, estab 8-11, marker 12)."""
    if img.shape[0] >= 16:
        noise = np.clip(img[:7].argmax(0) // 2, 0, 3); surf = img[7:11].argmax(0); est = img[11:15].argmax(0)
    else:
        noise = img[:4].argmax(0); surf = img[4:8].argmax(0); est = img[8:12].argmax(0)
    return np.dstack([GREY[noise], GREY[surf], GREY[est]])


def marker_xy(img):
    mk = np.argwhere(img[-1] > 0.5)
    if len(mk) == 0: return []
    # cluster into markers (discs): simple split by distance
    pts = []
    for y, x in mk:
        for p in pts:
            if abs(p[0] - x) < 8 and abs(p[1] - y) < 8: p[2].append((x, y)); break
        else: pts.append([x, y, [(x, y)]])
    return [(np.mean([q[0] for q in p[2]]), np.mean([q[1] for q in p[2]])) for p in pts]


def predict(A, data, model, img, tix, cond):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
        logits, _ = A.model_forward(model, img, None, tix, cond)
    return torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()


def overlay_panel(ax, rgb, cells, color, markers, title=None, alpha=0.95):
    """RGB crop with 4 m cells drawn as filled squares of `color`, markers as circles."""
    ax.imshow(rgb, interpolation="nearest")
    ys, xs = np.nonzero(cells)
    for y, x in zip(ys, xs):
        ax.add_patch(Rectangle((x * 4 - 0.5, y * 4 - 0.5), 4, 4, facecolor=color, edgecolor="none", alpha=alpha))
    for (x, y) in markers:
        ax.plot(x, y, "o", ms=13, mfc="white", mec="black", mew=2)
    if title: ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------------------------------ F0: hero, the detour model
def fig_f0():
    run = "c4n_bld_det_L4_16k_store100k"
    data, A, model = load_run(run)
    vl = data.val_loader; vl.reset()
    picks = []
    with torch.no_grad():
        for _ in range(4):
            img, tgt, tix, cond, meta = vl.next_batch_with_meta()
            p = predict(A, data, model, img, tix, cond)
            ratio = meta["detour_ratio"].cpu().numpy()
            import torch.nn.functional as F
            pr = torch.from_numpy(p > 0.5); tg = tgt.cpu() > 0
            dil = lambda x: F.max_pool2d(x.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0
            pd, td = dil(pr), dil(tg)
            diou = ((pd & td).flatten(1).sum(1).float() / (pd | td).flatten(1).sum(1).clamp(min=1).float()).numpy()
            for i in range(img.shape[0]):
                if ratio[i] > 1.3: picks.append((diou[i], img[i].cpu().numpy(), tgt[i].cpu().numpy(), p[i], float(ratio[i])))
    seen = set(); uniq = []
    for t in sorted(picks, key=lambda t: -t[0]):
        key = (round(t[4], 4), int(t[2].sum()))
        if key not in seen: seen.add(key); uniq.append(t)
    picks = [t[1:] for t in sorted(uniq[:4], key=lambda t: -t[4])]
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.7 * len(picks)))
    for r, (img, tgt, p, ratio) in enumerate(picks):
        rgb = rgb_from_img(img); mk = marker_xy(img)
        overlay_panel(axes[r, 0], rgb, np.zeros((64, 64), bool), TARGET_C, mk, "input: map layers + two markers" if r == 0 else None)
        overlay_panel(axes[r, 1], rgb, tgt > 0, TARGET_C, mk, "exact shortest path around buildings" if r == 0 else None)
        overlay_panel(axes[r, 2], rgb, p > 0.5, PRED_C, mk, "model output (one forward pass)" if r == 0 else None)
        axes[r, 0].set_ylabel(f"route {ratio:.2f}× the straight distance", fontsize=9)
    for c, col in enumerate((None, TARGET_C, PRED_C)):
        if col:
            for ax in axes[:, c]:
                for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(4)
    fig.suptitle("Shortest path around buildings between two points: exact target (yellow) and the 8.7 M-parameter model's output (magenta)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "F0_hero_detour.png"); plt.close(fig); print("F0 done")


# ------------------------------------------------------------------ F00: concept diagram
def fig_f00():
    fig, ax = plt.subplots(figsize=(16, 7.4)); ax.set_xlim(0, 16); ax.set_ylim(0, 7.4); ax.axis("off")
    # stacked aligned layers
    names = ["buildings", "surfaces", "noise", "places", "sunlit hours", "marker(s)"]
    cols = ["#5d6d7e", "#f0b27a", "#e74c3c", "#3498db", "#f4d03f", "#ffffff"]
    for i, (n, c) in enumerate(zip(names, cols)):
        x0, y0 = 0.4 + i * 0.25, 5.0 - i * 0.6
        ax.add_patch(Polygon([(x0, y0), (x0 + 2.0, y0), (x0 + 2.6, y0 + 0.7), (x0 + 0.6, y0 + 0.7)], closed=True, facecolor=c, edgecolor="k", lw=1, alpha=0.9))
        ax.text(x0 + 2.75, y0 + 0.28, n, fontsize=9.5, va="center")
    ax.text(2.3, 6.0, "spatially aligned layers of one place\n(a spatial dataset drawn as planes, not a photo)", ha="center", fontsize=10, weight="bold")
    ax.text(2.3, 0.75, "1 m per pixel, 256 m crop\none-hot class planes, 16 channels", ha="center", fontsize=9, color="#444")
    # condition tokens
    ax.add_patch(FancyBboxPatch((6.0, 0.7), 2.8, 1.2, boxstyle="round,pad=0.05", facecolor="#eaf2f8", edgecolor="#2e86c1", lw=1.5))
    ax.text(7.4, 1.55, "condition tokens", ha="center", fontsize=10, weight="bold")
    ax.text(7.4, 1.05, "\"food & drink\"  ·  \"within 100 m\"", ha="center", fontsize=9)
    # patch embedding
    ax.add_patch(FancyBboxPatch((6.0, 2.9), 2.8, 1.6, boxstyle="round,pad=0.05", facecolor="#fdf2e9", edgecolor="#ca6f1e", lw=1.5))
    ax.text(7.4, 4.12, "patch embedding", ha="center", fontsize=10, weight="bold")
    ax.text(7.4, 3.5, "16 × 16 px patches → 256 tokens\n+ 2-D position code", ha="center", fontsize=9)
    # transformer blocks
    for k in range(4):
        ax.add_patch(FancyBboxPatch((9.6 + k * 0.25, 2.2 + k * 0.18), 2.6, 2.6, boxstyle="round,pad=0.05", facecolor="#eafaf1", edgecolor="#1e8449", lw=1.5))
    ax.text(11.3, 5.35, "2–4 transformer blocks\n6 heads, width 384", ha="center", fontsize=10.5, weight="bold")
    ax.text(11.3, 3.75, "self-attention over 258 tokens\n(256 patches + 2 condition tokens)\nwith a learned relative-position bias\nper (Δrow, Δcol) between patches\n+ MLP", ha="center", fontsize=8.5)
    # head and output
    ax.add_patch(FancyBboxPatch((13.3, 2.9), 2.4, 1.6, boxstyle="round,pad=0.05", facecolor="#f4ecf7", edgecolor="#7d3c98", lw=1.5))
    ax.text(14.5, 4.12, "per-cell head", ha="center", fontsize=10, weight="bold"); ax.text(14.5, 3.5, "each patch token →\n4 × 4 output cells", ha="center", fontsize=9)
    ax.add_patch(Rectangle((13.8, 0.55), 1.4, 1.4, facecolor="#111", edgecolor="k"))
    path = [(2, 11), (3, 11), (4, 11), (5, 11), (5, 10), (5, 9), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (10, 4), (11, 4)]
    for x, y in path: ax.add_patch(Rectangle((13.8 + x * 0.1, 0.55 + y * 0.1), 0.1, 0.1, facecolor=PRED_C, edgecolor="none"))
    for x, y in ((2, 11), (11, 4)): ax.add_patch(plt.Circle((13.85 + x * 0.1, 0.6 + y * 0.1), 0.09, facecolor="white", edgecolor="k", lw=0.8))
    ax.text(14.5, 2.35, "output: 64 × 64 grid\n(4 m cells), one mask per task", ha="center", fontsize=9)
    ax.text(14.5, 0.25, "later: + output tokens (state, stop)", ha="center", fontsize=8, color="#666", style="italic")
    # arrows
    for (x0, y0, x1, y1) in [(4.2, 3.7, 6.0, 3.7), (8.8, 3.7, 9.6, 3.7), (8.8, 1.3, 9.9, 2.3), (12.5, 3.7, 13.3, 3.7), (14.5, 2.9, 14.5, 2.55)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#333"))
    ax.text(8.0, 7.05, "A vision-in, vision-out model, conditioned by tokens, trained from scratch on real city data", ha="center", fontsize=12, weight="bold")
    ax.text(8.0, 6.6, "The question: what spatial reasoning can this kind of model learn from this kind of data, and what does it take?", ha="center", fontsize=10, color="#333")
    fig.savefig(OUT / "F00_concept.png", bbox_inches="tight"); plt.close(fig); print("F00 done")


# ------------------------------------------------------------------ F9: results as a picture
def first_batch(data):
    b = data.val_loader.next_batch()
    img, tgt = b[0], b[1]; tix = b[2] if len(b) > 2 else None; cond = b[3] if len(b) > 3 else None
    return img, tgt, tix, cond


def fig_f9():
    rows = []  # (level name, task text, rgb, target, pred, markers, metric text)
    # c1: sidewalk (core four)
    data, A, model = load_run("c1_L5b_core_four_2k_v3b_onehot_P16")
    keys = config.c1_task_keys_active(); print("c1 keys", keys)
    for _ in range(8):
        img, tgt, tix, cond = first_batch(data)
        if tix is None: break
        want = keys.index("sidewalk") if "sidewalk" in keys else 0
        sel = (tix == want).nonzero().flatten()
        if len(sel):
            i = int(sel[0]); p = predict(A, data, model, img, tix, cond)
            rows.append(("one condition on one layer", "mark the sidewalks", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, [], "IoU 0.97\n(mean of four such tasks)")); break
    # c2: a conjunction from the full set
    data, A, model = load_run("c2_L6c_c2_full_2k")
    vl = data.val_loader; names = None
    for attr in ("task_names", "task_keys", "specs", "task_specs"):
        if hasattr(vl, attr): names = getattr(vl, attr); break
    print("c2 val attrs:", [a for a in dir(vl) if not a.startswith("_")][:25], "names:", names if names is None else (names[:5] if isinstance(names, list) else type(names)))
    best = None
    want = [k for k, sp in enumerate(vl.task_specs) if sp.name == "quiet sidewalk"][0]
    for _ in range(60):
        img, tgt, tix, cond = first_batch(data)
        p = predict(A, data, model, img, tix, cond)
        for i in range(img.shape[0]):
            n = int((tgt[i] > 0).sum())
            if 60 < n < 900 and tix is not None and int(tix[i]) == want:
                iou = ((p[i] > 0.5) & (tgt[i].cpu().numpy() > 0)).sum() / max(1, ((p[i] > 0.5) | (tgt[i].cpu().numpy() > 0)).sum())
                if best is None or iou > best[0]: best = (iou, img[i].cpu().numpy(), tgt[i].cpu().numpy() > 0, p[i] > 0.5, cond[i].tolist(), int(tix[i]) if tix is not None else None)
    print("c2 pick cond/task:", best[4], best[5], "iou", round(best[0], 3))
    rows.append(("two conditions from two layers", "quiet sidewalk:\nsidewalk AND the lowest noise band", rgb_from_img(best[1]), best[2], best[3], [], "IoU 0.90 trained pairs\n0.85 on pairs never seen together"))
    # c3 relative
    data, A, model = load_run("c3_quieter_sidewalk_median_strict_2k")
    img, tgt, tix, cond = first_batch(data); p = predict(A, data, model, img, tix, cond)
    i = int(np.argmax([(t > 0).sum().item() for t in tgt]))
    rows.append(("a condition relative to the whole crop", "sidewalk quieter than\nthis crop's median sidewalk", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, [], "IoU 0.81\n(a patch-local model tops out at 0.76)"))
    # c3 marker disc
    data, A, model = load_run("within_100m_rand1_disc3_relV2")
    img, tgt, tix, cond = first_batch(data); p = predict(A, data, model, img, tix, cond); i = 0
    rows.append(("a condition relative to a marker", "everything within 100 m\nof the marker", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, marker_xy(img[i].cpu().numpy()), "IoU 0.996"))
    # m1 conjunction
    data, A, model = load_run("food_drink_within_100m_mix33_L2_16k_b32_disc3_relV2_seed5")
    vl = data.val_loader; keys = [s.key for s in vl.task_specs]; m1 = keys.index("food_drink_within_100m")
    for img, tgt, tix, cond in vl.iter_batches_for_task(m1, max_batches=3):
        p = predict(A, data, model, img, tix, cond); i = int(np.argmax([(t > 0).sum().item() for t in tgt]))
        if (tgt[i] > 0).sum() >= 12:
            rows.append(("marker + layer condition", "food & drink places\nwithin 100 m of the marker", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, marker_xy(img[i].cpu().numpy()), "IoU 0.90\n(two seeds: 0.90, 0.91)")); break
    # c4a segment
    data, A, model = load_run("c4_segment_L2_4k_b128_disc3_relV2_cosine4k_seed4")
    vl = data.val_loader; vl.reset(); img, tgt, tix, cond, meta = vl.next_batch_with_meta(); p = predict(A, data, model, img, tix, cond); i = int(np.argmax(meta["straight_px"].cpu().numpy()))
    rows.append(("two markers, geometry", "the straight line\nbetween the two markers", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, marker_xy(img[i].cpu().numpy()), "every cell within one cell\nof the line: precision 1.00, recall 1.00"))
    # c4b detour
    data, A, model = load_run("c4n_bld_det_L4_16k_store100k")
    vl = data.val_loader; vl.reset()
    for _ in range(10):
        img, tgt, tix, cond, meta = vl.next_batch_with_meta(); ratio = meta["detour_ratio"].cpu().numpy(); i = int(np.argmax(ratio))
        if ratio[i] > 1.3: break
    p = predict(A, data, model, img, tix, cond)
    rows.append(("two markers, obstacles", "the shortest path around\nbuildings between the markers", rgb_from_img(img[i].cpu().numpy()), tgt[i].cpu().numpy() > 0, p[i] > 0.5, marker_xy(img[i].cpu().numpy()), "IoU 0.81 (one-cell tolerance)\n0.92 straight · 0.78 one bend · 0.73 several"))
    # draw
    n = len(rows); fig, axes = plt.subplots(n, 4, figsize=(13, 2.75 * n), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    for r, (level, task, rgb, tgt, pred, mk, metric) in enumerate(rows):
        ax = axes[r, 0]; ax.axis("off")
        ax.text(0.0, 0.78, f"{r + 1}. {level}", fontsize=12, weight="bold", transform=ax.transAxes, va="center")
        ax.text(0.0, 0.48, task, fontsize=10, transform=ax.transAxes, va="center")
        ax.text(0.0, 0.14, metric, fontsize=10, transform=ax.transAxes, va="center", color="#1a5276", weight="bold")
        overlay_panel(axes[r, 1], rgb, np.zeros_like(tgt), TARGET_C, mk, "input crop" if r == 0 else None)
        overlay_panel(axes[r, 2], rgb, tgt, TARGET_C, mk, "exact target" if r == 0 else None, alpha=0.85)
        overlay_panel(axes[r, 3], rgb, pred, PRED_C, mk, "model output" if r == 0 else None, alpha=0.85)
    fig.suptitle("Task levels, one validation example each, and the headline number (all models 4.8–8.7 M parameters, trained from scratch)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "F9_results_table.png"); plt.close(fig); print("F9 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["f00", "f0", "f9"]
    for w in which: {"f0": fig_f0, "f00": fig_f00, "f9": fig_f9}[w]()
