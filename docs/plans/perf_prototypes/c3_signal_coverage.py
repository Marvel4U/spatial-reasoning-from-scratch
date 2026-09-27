"""Positive-label coverage until the transition, per task/run.
coverage(cell) = expected number of times a given output cell carried a positive label
               = fg_share x batch x steps_to_transition.
For m1 also count the cells that carry the *distance-discriminative* signal: food&drink cells
inside the disc (positives) and food&drink cells outside it (negatives that only the marker explains)."""
import sys, os, json, importlib
from pathlib import Path
REPO = Path.home() / "Github/spatial_reasoning_LLM_artifact"
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, config
led = {r["id"]: r for r in json.load(open(REPO / "runs/experiments.json"))["runs"]}
# run id, transition step (first clear move of val IoU / val loss), batch
RUNS = [
    ("c3_within_20m_8k_rand4", 1000), ("within_50m_6k_rand2", 4500), ("within_75m_rand1", 6500), ("within_100m_rand1", 6500),
    ("within_50m_rand2_disc3_relV2", 1500), ("within_75m_rand1_disc3_relV2", 1500), ("within_100m_rand1_disc3_relV2", 1500),
    ("food_drink_within_100m_rand1_disc3_relV2", None), ("food_drink_within_100m_L4_8k_b32_disc3_relV2", 6500),
]
from spatial_data.c3_targets import _within_radius_pix, _cells_from_pix
print(f"{'run':46s} {'B':>3} {'K':>2} {'fg cells/sample':>15} {'fg share':>9} {'estab-out cells':>15} {'T':>5} {'coverage':>9} {'disc-signal cov':>15}")
for run, T in RUNS:
    importlib.reload(config)
    for k, v in led[run]["overrides"].items():
        if hasattr(config, k): setattr(config, k, v)
    config.verbose = False
    sys.modules.pop("data", None)
    import data
    data.setup_device(); data.load_data()
    tl = data.train_loader
    fg = 0.0; est_out = 0.0; n = 0
    for _ in range(40):
        img, tgt, *_ = tl.next_batch()
        b = tgt.shape[0]; fg += tgt.float().flatten(1).sum(1).sum().item(); n += b
        if "food_drink" in run:
            # rebuild disc & estab at cell level from the loader's last markers/crops
            mc, mr = tl._last_marker_col, tl._last_marker_row if hasattr(tl, "_last_marker_col") else (None, None)
    fg_per = fg / n
    B = config.batch_size; K = led[run]["overrides"].get("c3_n_markers", "?")
    cov = fg_per / 4096 * B * T if T else float("nan")
    est_out_per = ""
    if "food_drink" in run:
        # estab==3 cells outside the disc: sample directly from the crops with fresh markers
        from spatial_data.dataset_c3_gpu import sample_markers
        crops = tl.crops; N = crops.shape[0]; g = torch.Generator(device=crops.device); g.manual_seed(0)
        ix = torch.randint(0, N, (512,), device=crops.device, generator=g)
        mc, mr = sample_markers(512, 1, 0.0, 256, g, crops.device)
        estab = crops[ix, 2] == 3; disc = _within_radius_pix(mc, mr, 100.0, h=256, w=256)
        est_c = _cells_from_pix(estab).bool(); disc_c = _cells_from_pix(disc).bool()
        est_in = (est_c & disc_c).float().flatten(1).sum(1).mean().item(); est_o = (est_c & ~disc_c).float().flatten(1).sum(1).mean().item()
        est_out_per = f"{est_o:15.1f}"
        sig = (est_in + est_o) / 4096 * B * T if T else float("nan")
        print(f"{run:46s} {B:>3} {str(K):>2} {fg_per:15.1f} {fg_per/4096:9.4f} {est_out_per} {str(T):>5} {cov:9.1f} {sig:15.1f}")
    else:
        print(f"{run:46s} {B:>3} {str(K):>2} {fg_per:15.1f} {fg_per/4096:9.4f} {'':15s} {str(T):>5} {cov:9.1f}")
