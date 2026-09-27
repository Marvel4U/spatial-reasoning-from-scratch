"""Sweep batch_size on CUDA (Step 6); read-only, no repo writes."""
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

config.data_source = "worldsnap"
config.worldsnap_gpu_resident = True
config.verbose = False
config.use_compile = False
data.setup_device()
dev = data.device


def try_batch(bs: int) -> dict | None:
    config.batch_size = bs
    torch.cuda.empty_cache()
    try:
        data.load_data()
        model = A.build_model().to(dev)
        opt = A.configure_optimizer(model)
        loader = data.train_loader
        img, tgt = loader.next_batch()
        img, tgt = img.to(dev), tgt.to(dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=data.dtype):
                _, loss = model(img, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 30
        peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
        return {
            "batch_size": bs,
            "ms_per_step": dt * 1e3,
            "samples_s": bs / dt,
            "peak_vram_mb": peak_mb,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise
    finally:
        data.train_loader = data.val_loader = None
        data.m = data.optimizer = None


print(f"device={dev} P={config.patch_size} compile={config.use_compile}")
for bs in (16, 32, 64, 128, 256):
    r = try_batch(bs)
    if r is None:
        print(f"batch_size={bs:3d}  OOM")
    else:
        print(
            f"batch_size={r['batch_size']:3d}  {r['ms_per_step']:.1f} ms/step  "
            f"{r['samples_s']:.0f} samples/s  peak VRAM {r['peak_vram_mb']:.0f} MB"
        )
