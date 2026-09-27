"""Read-only profiling of the local harness: where does wall time go? Writes nothing into the repo."""
import sys, time, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch
import config
config.data_source = "worldsnap"; config.cropset = "v2"; config.taskset = "t0_point_v0"
config.val_same_as_train = False; config.verbose = False
config.worldsnap_gpu_resident = True
for k, v in dict(worldsnap_max_train_items=None, worldsnap_max_val_items=None).items():
    if hasattr(config, k): setattr(config, k, v)
import data, grid_vit_adapter as A, eval_spatial
data.setup_device() if hasattr(data, "setup_device") else None
t0 = time.perf_counter(); data.load_data(); print(f"load_data: {time.perf_counter()-t0:.1f}s  (batch={config.batch_size}, P={config.patch_size}, enc={config.encoding_mode}, dtype={data.dtype})")
dev = data.device; sync = torch.cuda.synchronize

def timeit(fn, n):
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - t) / n

def run(use_compile, label):
    config.use_compile = use_compile
    model = A.build_model(); opt = A.configure_optimizer(model)
    loader = data.train_loader; B = config.batch_size
    img, tgt = loader.next_batch(); img, tgt = img.to(dev), tgt.to(dev)
    def gpu_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=data.dtype):
            _, loss = model(img, tgt)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    t = time.perf_counter(); [gpu_step() for _ in range(3)]; sync(); warm = time.perf_counter() - t
    t_gpu = timeit(gpu_step, 30)
    def fwd_only():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype): model(img, tgt)
    t_fwd = timeit(fwd_only, 30)
    print(f"[{label}] warmup(3 steps) {warm:.1f}s | GPU train step {t_gpu*1e3:.1f} ms = {B/t_gpu:.0f} samples/s if data were free | fwd-only {t_fwd*1e3:.1f} ms")
    return model, t_gpu

model, t_gpu = run(False, "eager")
loader = data.train_loader; B = config.batch_size
t_batch = timeit(lambda: loader.next_batch(), 20)
b = loader.next_batch()
t_h2d = timeit(lambda: (b[0].to(dev), b[1].to(dev)), 20)
print(f"next_batch (CPU build) {t_batch*1e3:.1f} ms | host->device copy {t_h2d*1e3:.1f} ms | GPU step {t_gpu*1e3:.1f} ms")
tot = t_batch + t_h2d + t_gpu
print(f"=> one training step ~{tot*1e3:.0f} ms = {B/tot:.0f} samples/s; data share {100*(t_batch+t_h2d)/tot:.0f}%")

# per-sample cost breakdown of batch building
import random
from spatial_data import dataset_c0 as D, channels as C
idx = loader.index; rng = random.Random(0)
t_sample = timeit(lambda: idx.sample(rng), 200)
print(f"one sample build: {t_sample*1e3:.2f} ms  (x{B} = {t_sample*B*1e3:.1f} ms; torch.stack rest)")

# eval costs
t = time.perf_counter(); A.estimate_loss(model, eval_iters=config.eval_iters); sync(); t_est = time.perf_counter() - t
t = time.perf_counter(); eval_spatial.eval_loader(model, data.val_loader, max_batches=20); sync(); t_sp20 = time.perf_counter() - t
t = time.perf_counter(); eval_spatial.eval_loader(model, data.val_loader, max_batches=10); sync(); t_sp10 = time.perf_counter() - t
print(f"estimate_loss({config.eval_iters} iters x 2 splits) {t_est:.2f}s | spatial eval 20 batches {t_sp20:.2f}s | 10 batches {t_sp10:.2f}s")
per_interval_train = config.eval_interval * tot
print(f"per eval interval ({config.eval_interval} steps): train {per_interval_train:.1f}s vs eval {t_est:.1f}s + spatial(10) {t_sp10:.1f}s => eval share {100*(t_est+t_sp10)/(per_interval_train+t_est+t_sp10):.0f}%")

# how much of spatial eval is the python per-sample loop?
imgs = [loader.next_batch() for _ in range(3)]
def fwd_batches():
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=data.dtype):
        for i, tg in imgs: model(i.to(dev), tg.to(dev))
print(f"spatial eval per batch {t_sp20/20*1e3:.0f} ms vs data+forward per batch {(t_batch+timeit(fwd_batches,3)/3)*1e3:.0f} ms  -> remainder = python metric loop with .item() syncs")

try:
    run(True, "compiled")
except Exception as e:
    print("compile failed:", type(e).__name__, str(e)[:300])
