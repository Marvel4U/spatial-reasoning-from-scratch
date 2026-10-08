#!/usr/bin/env python3
"""Wall time of the wave search on the network (2 Oct 2026): per pass at batch 1 and batched, and per whole
free-running route including the loop around it. Usage (from harness/): PYTHONPATH=. ../.venv/bin/python diagnostics/wave_rollout_timing.py RUN_ID"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
import config, data, grid_vit_adapter
from checkpoints import checkpoint_model_config
from diagnostics import wave_search_rollout as ws
from diagnostics.wave_rollout_eval import STATE_KEY, step
from plain_gpt_module.checkpoint import read_checkpoint
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from spatial_data.c4_tasks import TOK_DONE, TOK_SEARCH, WAVE_HORIZON_M
from spatial_data.dataset_c4_gpu import _wave_start_disc
from spatial_data.wave_search_schedule import n_goal_from_m

run = sys.argv[1]
meta = json.loads((Path(config.diagnostics_dir) / run / "meta.json").read_text()); ws._apply_meta_config(meta)
for k, v in ws._overrides_for_run(run).items():
    if hasattr(config, k): setattr(config, k, v)
config.sync_in_chans_from_encoding(); data.setup_device()
import spatial_data.channels as _ch
_ch.MARKER_RADIUS_PX = int(config.marker_radius_px)
loader, vr = ws.build_wave_rollout_loader()
ck = ws._resolve_checkpoint(run, None); doc = read_checkpoint(ck, map_location="cpu")
model = LocalGridViT(LocalGridViTConfig(**checkpoint_model_config(doc))).to(data.device); grid_vit_adapter.load_checkpoint(ck, model=model); model.train(False)
g = config.grid_out_size; crop = config.img_size; fg = model.config.foreground_class
n_par = sum(p.numel() for p in model.parameters())
goal = np.load(Path(config.c4_routes_dir()) / "c5_wave_val.npz")["goal_m"]

def sync(): torch.cuda.synchronize()
img_t, _, tid, cond, _ = loader.fetch_wave_pass(0, "wave_first")
out = {"run": run, "params_M": round(n_par / 1e6, 2), "gpu": torch.cuda.get_device_name(0), "dtype": str(data.dtype)}
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
    for B in (1, 16, 128):
        x = img_t[None].repeat(B, 1, 1, 1); t = tid[None].repeat(B); c = cond[None].repeat(B, 1)
        for _ in range(5): ws.model_forward(model, x, None, t, c)
        sync(); t0 = time.perf_counter(); n = 30
        for _ in range(n): ws.model_forward(model, x, None, t, c)
        sync(); dt = (time.perf_counter() - t0) / n
        out[f"forward_ms_batch{B}"] = round(1000 * dt, 2); out[f"passes_per_s_batch{B}"] = round(B / dt, 1)
# whole free-running routes, one at a time, as the rollout eval does (grid forward + token forward per pass)
times, passes, lens = [], [], []
for rix in range(60):
    if not np.isfinite(goal[rix]): continue
    n_goal = n_goal_from_m(float(goal[rix]), WAVE_HORIZON_M)
    base = {tok: loader.fetch_wave_pass(rix, key)[:4] for tok, key in STATE_KEY.items()}
    pad = _wave_start_disc(loader.route["markers"][rix:rix + 1], g, crop)[0]; state = TOK_SEARCH
    sync(); t0 = time.perf_counter(); k = 0
    for _ in range(2 * n_goal + 4):
        im, _, ti, co = base[state]
        pad, tok = step(model, im, pad, ti, co, crop, g, fg, "cuda"); k += 1
        if tok == TOK_DONE or tok not in STATE_KEY: break
        state = tok
    sync(); times.append(time.perf_counter() - t0); passes.append(k); lens.append(float(goal[rix]))
times, passes = np.array(times[5:]), np.array(passes[5:])           # the first routes warm up
out.update(routes=len(times), route_ms_mean=round(1000 * times.mean(), 1), route_ms_median=round(1000 * float(np.median(times)), 1),
           route_ms_max=round(1000 * times.max(), 1), passes_mean=round(float(passes.mean()), 2), ms_per_pass_in_loop=round(1000 * times.sum() / passes.sum(), 2),
           walking_m_mean=round(float(np.mean(lens[5:])), 1))
print(json.dumps(out, indent=1))
(Path(config.diagnostics_dir) / run / "wave_rollout_timing.json").write_text(json.dumps(out, indent=1))
