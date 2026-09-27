"""c4a segment train step: sweep batch_size × L2/L4 on CUDA (read-only).

Usage (repo venv, from repo root):
  python docs/plans/perf_prototypes/profile_c4_batch_size.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import torch
import config
import data
import grid_vit_adapter as A
from spatial_batch import forward as model_forward

config.task_rung = "c4"
config.data_source = "worldsnap"
config.cropset = "v4"
config.encoding_mode = "onehot_v4"
config.c4_task_mode = "single"
config.c4_task_key = "segment"
config.c4_eval_task_key = "segment"
config.c4_mix_c3_task_keys = None
config.marker_radius_px = 3
config.rel_pos_bias = "all"
config.c4_grid_loss = "ce_dice"
config.worldsnap_max_train_items = 200
config.worldsnap_max_val_items = 50
config.use_compile = False
config.verbose = False
config.use_gpu = True

BATCH_SIZES = (8, 16, 24, 32, 48, 64, 96, 128)
N_WARMUP = 3
N_TIMED = 40


def try_config(*, enc_n_layer: int, batch_size: int) -> dict | None:
    config.batch_size = batch_size
    config.enc_n_layer = enc_n_layer
    data.train_loader = data.val_loader = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = data.device
    try:
        data.setup_device()
        data.load_data()
        model = A.build_model().to(dev)
        opt = A.configure_optimizer(model)
        loader = data.train_loader

        def step():
            img, tgt, task_id, cond_ids = loader.next_batch()
            img, tgt = img.to(dev), tgt.to(dev)
            if task_id is not None:
                task_id = task_id.to(dev)
            if cond_ids is not None:
                cond_ids = cond_ids.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=data.dtype):
                _, loss = model_forward(model, img, tgt, task_id, cond_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(N_WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N_TIMED
        peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
        params = sum(p.numel() for p in model.parameters())
        return {
            "enc_n_layer": enc_n_layer,
            "batch_size": batch_size,
            "ms_per_step": dt * 1e3,
            "samples_s": batch_size / dt,
            "peak_vram_mb": peak_mb,
            "params_m": params / 1e6,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise
    finally:
        data.train_loader = data.val_loader = None
        if getattr(data, "m", None) is not None:
            data.m = None
        if getattr(data, "optimizer", None) is not None:
            data.optimizer = None


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(f"device={torch.cuda.get_device_name()} c4a segment in_chans=16 P=16\n")
    rows = []
    for layers in (2, 4):
        print(f"--- enc_n_layer={layers} ---")
        for bs in BATCH_SIZES:
            r = try_config(enc_n_layer=layers, batch_size=bs)
            if r is None:
                print(f"  batch={bs:3d}  OOM")
            else:
                rows.append(r)
                print(
                    f"  batch={bs:3d}  {r['ms_per_step']:6.1f} ms/step  "
                    f"{r['samples_s']:6.0f} samp/s  VRAM {r['peak_vram_mb']:6.0f} MB  "
                    f"{r['params_m']:.2f}M params"
                )
        print()
    if rows:
        best_l2 = max((r for r in rows if r["enc_n_layer"] == 2), key=lambda x: x["samples_s"])
        best_l4 = max((r for r in rows if r["enc_n_layer"] == 4), key=lambda x: x["samples_s"])
        print(
            f"best L2: batch={best_l2['batch_size']} → {best_l2['samples_s']:.0f} samp/s "
            f"({best_l2['peak_vram_mb']:.0f} MB peak)"
        )
        print(
            f"best L4: batch={best_l4['batch_size']} → {best_l4['samples_s']:.0f} samp/s "
            f"({best_l4['peak_vram_mb']:.0f} MB peak)"
        )


if __name__ == "__main__":
    main()
