"""README figures F1-F8 for the public repository. Run from the source repo with its venv:
  .venv/bin/python docs/figures_make.py  (writes to ~/Github/spatial_reasoning_public/docs/figures/)"""
import sys, os, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
OUT = Path.home() / "Github/spatial_reasoning_public/docs/figures"; OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
LED = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
GREY = np.array([0, 85, 170, 255], dtype=np.uint8)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})


def load_run(run, **extra):
    import importlib
    importlib.reload(config)
    for k, v in LED[run]["overrides"].items():
        if hasattr(config, k): setattr(config, k, v)
    for k, v in extra.items(): setattr(config, k, v)
    config.verbose = False
    for m in ("data", "grid_vit_adapter", "eval_spatial", "eval_c4"):
        sys.modules.pop(m, None)
    import data, grid_vit_adapter as A
    data.setup_device(); data.load_data()
    model, _ = A.load_checkpoint(REPO / f"runs/checkpoints/{run}.pt"); model.train(False)
    return data, A, model


def rgb_from_img(img):
    """img (16,256,256) one-hot v4: noise7 0-6, surface 7-10, estab 11-14, marker 15 -> uint8 RGB."""
    noise7 = img[:7].argmax(0); surf = img[7:11].argmax(0); est = img[11:15].argmax(0)
    r = GREY[np.clip(noise7 // 2, 0, 3)]; g = GREY[surf]; b = GREY[est]
    return np.dstack([r, g, b])


# ------------------------------------------------------------------ F1: one crop, three tokens
def fig_f1():
    run = "food_drink_within_100m_mix33_L2_16k_b32_disc3_relV2_seed5"
    data, A, model = load_run(run)
    vl = data.val_loader; keys = [s.key for s in vl.task_specs]; m1 = keys.index("food_drink_within_100m")
    cond = vl._cond
    picks = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
        for img, tgt, tix, cnd in vl.iter_batches_for_task(m1, max_batches=6):
            outs = []
            for ti in range(len(keys)):
                c = cond[ti].view(1, -1).expand(img.shape[0], -1)
                logits, _ = A.model_forward(model, img, None, torch.full_like(tix, ti), c)
                outs.append(torch.softmax(logits.float(), -1)[..., 1].cpu().numpy())
            best = max(range(img.shape[0]), key=lambda i: tgt[i].sum().item())
            crop_key = img[best, 7:11].argmax(0).cpu().numpy().tobytes()
            if tgt[best].sum().item() >= 12 and crop_key not in [k for k, *_ in picks]:
                picks.append((crop_key, img[best].cpu().numpy(), tgt[best].cpu().numpy(), [o[best] for o in outs]))
            if len(picks) >= 4: break
    names = {"food_drink": "token: food & drink", "within_100m": "token: within 100 m", "food_drink_within_100m": "token: food & drink within 100 m"}
    order = [keys.index("food_drink"), keys.index("within_100m"), m1]
    fig, axes = plt.subplots(len(picks), 5, figsize=(12.5, 2.6 * len(picks)))
    for r, (_, img, tgt, outs) in enumerate(picks):
        rgb = rgb_from_img(img); mk = np.argwhere(img[15] > 0.5); ry, rx = mk.mean(0)
        ax = axes[r, 0]; ax.imshow(rgb, interpolation="nearest"); ax.plot(rx, ry, "y+", ms=14, mew=2); ax.set_title("crop with marker" if r == 0 else "")
        ax = axes[r, 1]; ax.imshow(tgt, cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest"); ax.set_title("target: food & drink within 100 m" if r == 0 else "")
        for c, ti in enumerate(order):
            ax = axes[r, 2 + c]; ax.imshow(outs[ti], cmap="magma", vmin=0, vmax=1, interpolation="nearest"); ax.set_title(names[keys[ti]] if r == 0 else "")
        for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("One model, one crop, three condition tokens: the token selects the computation (P(foreground) per 4 m cell)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "F1_three_tokens.png"); plt.close(fig); print("F1 done")


# ------------------------------------------------------------------ F2: shortcut vs solution (distance profile)
def fig_f2():
    from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
    bands = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 400)); series = {}
    for label, run in (("shortcut model (4 blocks, pixel marker, run 4)", "food_drink_within_100m_L4_8k_b32"),
                       ("solved model (4 blocks, disc marker + bias, run 12)", "food_drink_within_100m_L4_16k_b32_disc3_relV2")):
        data, A, model = load_run(run)
        vl = data.val_loader; crops = vl.crops; N = crops.shape[0]; mc, mr = vl._marker_col, vl._marker_row
        est_c = _cells_from_pix(crops[:, 2] == 3).bool()
        preds = []; vl.reset()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
            for img, tgt, tix, cond in vl.iter_batches():
                logits, _ = A.model_forward(model, img, None, tix, cond); preds.append(torch.softmax(logits.float(), -1)[..., 1] > 0.5)
        pred = torch.cat(preds)[:N]
        g = 64; cy = (mr.float() / 4).view(N, 1, 1); cx = (mc.float() / 4).view(N, 1, 1)
        yy = torch.arange(g, device=crops.device).view(1, g, 1).float(); xx = torch.arange(g, device=crops.device).view(1, 1, g).float()
        d = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() * 4
        vals = []
        for lo, hi in bands:
            band = est_c & (d >= lo) & (d < hi); vals.append((pred & band).sum().item() / max(1, band.sum().item()))
        series[label] = vals
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(len(bands)); w = 0.38
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=["#c0392b", "#2e86c1"][i])
    ax.axvline(1.5, color="k", ls=":", lw=1); ax.text(1.55, 0.97, "100 m", va="top", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{lo}–{hi} m" for lo, hi in bands]); ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of food & drink cells marked"); ax.set_xlabel("distance of the cell from the marker")
    ax.set_title("Shortcut vs solution: the same task, two models"); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "F2_shortcut_vs_solution.png"); plt.close(fig); print("F2 done")


# ------------------------------------------------------------------ F3: escape steps by recipe
def fig_f3():
    NONE = None
    groups = [
        ("single task\npixel marker\n2 blocks", [NONE, NONE]),                                  # run 1 (L2), run 7 disc-only L2 excluded; runs 1,9 -> pixel L2 & disc L2 none
        ("single task\npixel marker\n4 blocks", [NONE, NONE]),                                  # runs 4, 5
        ("single task\ndisc + bias\n2 blocks", [NONE, NONE, 7000]),                             # run 9, L2 b32 seed4 none, seed5 7000
        ("single task\ndisc + bias\n4 blocks", [6500, 2500, NONE, NONE, 5000, 7000, 3000]),     # runs 11, 12, seeds 1-3, C1, C2
        ("mix with the\nbare disc task\n2 or 4 blocks", [3000, 2000, 3000, 1000, 2000, 2000, NONE, 5000, 2000, 1000, 1000, 2000]),  # M1-M4, 8k seeds, 80/10/10 seeds, flat, L2 seeds
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); rng = np.random.default_rng(0)
    for gi, (label, steps) in enumerate(groups):
        for s in steps:
            jitter = rng.uniform(-0.18, 0.18)
            if s is None:
                ax.plot(gi + jitter, 8800, "x", color="#c0392b", ms=8, mew=1.8)
            else:
                ax.plot(gi + jitter, s, "o", color="#2e86c1", ms=7, alpha=0.85)
    ax.axhline(8000, color="grey", ls=":", lw=1); ax.text(4.45, 8100, "8k = run length", fontsize=8, ha="right", color="grey")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("step at which the model left the shortcut"); ax.set_ylim(0, 9400)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 8800]); ax.set_yticklabels(["0", "2k", "4k", "6k", "8k", "never"])
    ax.set_title("Food & drink within 100 m: when does training escape the shortcut?")
    ax.plot([], [], "o", color="#2e86c1", label="escaped"); ax.plot([], [], "x", color="#c0392b", mew=1.8, label="no escape within the run"); ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(OUT / "F3_escape_steps.png"); plt.close(fig); print("F3 done")


# ------------------------------------------------------------------ F4: the marker head
def fig_f4():
    rows = [json.loads(l) for l in open(REPO / "runs/diagnostics/food_drink_within_100m_L4_16k_b32_disc3_relV2/eval.jsonl")]
    by = {r["step"]: r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, step, title in zip(axes, (2000, 3000, 8000), ("step 2000: shortcut (IoU 0.40)", "step 3000: just escaped (IoU 0.78)", "step 8000: solved (IoU 0.85)")):
        M = np.array([[by[step].get(f"attn_to_marker/{b}/{h}", 0.0) for h in range(6)] for b in range(4)])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1); ax.set_title(title)
        ax.set_xlabel("head"); ax.set_ylabel("block"); ax.set_xticks(range(6)); ax.set_yticks(range(4))
        for b in range(4):
            for h in range(6): ax.text(h, b, f"{M[b, h]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[b, h] < 0.6 else "k")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="attention on the marker token")
    fig.suptitle("Where the marker circuit forms: attention paid to the marker's patch token, per head", fontsize=11)
    fig.savefig(OUT / "F4_marker_head.png", bbox_inches="tight"); plt.close(fig); print("F4 done")


# ------------------------------------------------------------------ F5: pipeline (layers -> tile -> targets)
def fig_f5():
    ov = REPO / "data/amsterdam/de_pijp/overlays"
    layers = [("buildings_height.png", "3DBAG buildings"), ("streets_bgt_functie.png", "BGT surfaces"), ("noise_lden.png", "noise map 2021"),
              ("pois.png", "OSM + Overture places"), ("sunlit_hours_0621.png", "sunlit hours (derived)"), ("tiles_rgb_v1_district.png", "→ RGB class tile")]
    run = "food_drink_within_100m_mix33_L2_16k_b32_disc3_relV2_seed5"
    data, A, model = load_run(run)
    vl = data.val_loader; keys = [s.key for s in vl.task_specs]; m1 = keys.index("food_drink_within_100m")
    img, tgt, tix, cond = next(iter(vl.iter_batches_for_task(m1, max_batches=1)))
    i = int((tgt.flatten(1).sum(1) > 12).nonzero()[0]); im = img[i].cpu().numpy(); rgb = rgb_from_img(im)
    from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
    mk = np.argwhere(im[15] > 0.5); ry, rx = mk.mean(0)
    surf = im[7:11].argmax(0); sidewalk = torch.from_numpy((surf == 2)).view(1, 256, 256)
    disc = _within_radius_pix(torch.tensor([int(rx)]), torch.tensor([int(ry)]), 100.0, h=256, w=256)
    t_side = _cells_from_pix(sidewalk)[0].numpy(); t_disc = _cells_from_pix(disc)[0].numpy(); t_m1 = tgt[i].cpu().numpy()
    fig = plt.figure(figsize=(13, 6.2)); gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1])
    for c, (f, name) in enumerate(layers):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(Image.open(ov / f).convert("RGB")); ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    panels = [(rgb, "one 256 m crop\n(16 input planes, marker +)"), (t_side, "c1 target:\nsidewalk"), (t_disc, "c3 target:\nwithin 100 m of the marker"), (t_m1, "c3 target:\nfood & drink within 100 m")]
    for c, (arr, name) in enumerate(panels):
        ax = fig.add_subplot(gs[1, c + 1]);
        if arr.ndim == 3: ax.imshow(arr, interpolation="nearest"); ax.plot(rx, ry, "y+", ms=14, mew=2)
        else: ax.imshow(arr, cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest")
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Real layers of De Pijp, Amsterdam → aligned class planes → exact targets on a 64 × 64 grid", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "F5_pipeline.png"); plt.close(fig); print("F5 done")


# ------------------------------------------------------------------ F6 / F7 / F8: copies and crops of existing images
def fig_f6_f7_f8():
    shutil.copy(REPO / "data/amsterdam/de_pijp/crops/v4/c4/c4_gallery_test_50_fixed_builder.png", OUT / "F6_c4_targets_gallery.png")
    a = Image.open(REPO / "runs/diagnostics/c4_segment_L2_4k_b128_disc3_relV2_cosine4k_seed4/pred_step003999.png").convert("RGB")
    b = Image.open(REPO / "runs/diagnostics/c4n_bld_det_L4_16k_store100k/pred_step015999.png").convert("RGB")
    def rows(im, which, header=48):
        h = (im.size[1] - header) // 8
        out = Image.new("RGB", (im.size[0], h * len(which)), "white")
        for k, r in enumerate(which):
            out.paste(im.crop((0, header + r * h, im.size[0], header + (r + 1) * h)), (0, k * h))
        return out
    ra, rb = rows(a, [0, 2, 3, 5]), rows(b, [0, 4, 5, 7]); H = max(ra.size[1], rb.size[1])
    canvas = Image.new("RGB", (ra.size[0] + rb.size[0] + 30, H + 40), "white")
    canvas.paste(ra, (0, 40)); canvas.paste(rb, (ra.size[0] + 30, 40))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas); d.text((10, 12), "c4a segment model @ 4k steps  (crop | target | P(fg))", fill="black"); d.text((ra.size[0] + 40, 12), "c4b detour model, 4 blocks, 100k store @ 16k steps", fill="black")
    canvas.save(OUT / "F7_c4_predictions.png")
    shutil.copy(REPO / "runs/diagnostics/c4n_segdet_L4_16k_seed4_store100k/compare_c4_multi.png", OUT / "F8a_c4_levers_curves.png")
    runs = [("½ segment + ½ detour", "c4n_segdet_L4_16k_seed4_store100k"), ("detour only", "c4n_det_only_L4_16k_store100k"),
            ("½ building + ½ detour", "c4n_bld_det_L4_16k_store100k"), ("⅓ building + ⅓ segment + ⅓ detour", "c4n_bld_seg_det_L4_16k_store100k")]
    labels = ["straight (≤1.05)", "one bend (1.05–1.3)", "several corners (>1.3)"]; keys = ["iou_dilated_ratio_1.00_1.05", "iou_dilated_ratio_1.05_1.30", "iou_dilated_ratio_gt_1.30"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(3); w = 0.2
    for i, (name, run) in enumerate(runs):
        last = [json.loads(l) for l in open(REPO / f"runs/diagnostics/{run}/eval.jsonl")][-1]
        ax.bar(x + (i - 1.5) * w, [last[k] for k in keys], w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.5, 1.0); ax.set_ylabel("IoU after 1-cell dilation (val, detour)")
    ax.set_title("Shortest path around buildings, by how much the route bends (4 blocks, 100k store, 16k steps)"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "F8b_c4_strata.png"); plt.close(fig); print("F6/F7/F8 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["f1", "f2", "f3", "f4", "f5", "f678"]
    for w in which:
        {"f1": fig_f1, "f2": fig_f2, "f3": fig_f3, "f4": fig_f4, "f5": fig_f5, "f678": fig_f6_f7_f8}[w]()
