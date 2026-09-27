import sys, time, os, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO / "harness"); sys.path.insert(0, str(REPO / "harness")); sys.path.insert(0, str(REPO))
import torch, numpy as np
import config
config.data_source = "worldsnap"; config.val_same_as_train = False; config.verbose = False
import data
data.setup_device()
data.load_data()
L = data.train_loader; idx = L.index; rng = random.Random(1)
print("materialized:", idx._materialized is not None, "| torch threads:", torch.get_num_threads(), "| cpu count:", os.cpu_count())
def T(fn, n=20):
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1e3
samples = [idx.sample(rng) for _ in range(32)]
print("sample dtype/shape:", samples[0][0].dtype, tuple(samples[0][0].shape), samples[0][1].dtype, tuple(samples[0][1].shape))
print(f"32x sample(): {T(lambda: [idx.sample(rng) for _ in range(32)]):.1f} ms")
print(f"stack imgs:   {T(lambda: torch.stack([s[0] for s in samples])):.1f} ms")
print(f"stack tgts:   {T(lambda: torch.stack([s[1] for s in samples])):.1f} ms")
print(f"next_batch:   {T(L.next_batch):.1f} ms")
# pieces of one sample
item = idx.items[0]
from spatial_data import channels as C
planes = idx.layer_cache.get(item.labels_path) if idx.layer_cache is not None else None
print("cached plane dtype/shape:", planes[0].dtype, planes[0].shape)
print(f"  scalar_layers_from_arrays: {T(lambda: C.scalar_layers_from_arrays(planes), 200):.3f} ms")
print(f"  marker_plane:              {T(lambda: C.marker_plane(256, 256, 10, 10), 200):.3f} ms")
print(f"  build_input_from_layers:   {T(lambda: C.build_input_from_layers(planes, encoding=idx.encoding, marker_row=10, marker_col=10), 200):.3f} ms")
for nt in (1, 4):
    torch.set_num_threads(nt)
    print(f"threads={nt}: next_batch {T(L.next_batch):.1f} ms | 32x sample {T(lambda: [idx.sample(rng) for _ in range(32)]):.1f} ms")
# the fast alternative: whole split on GPU as uint8, batch built by indexing on the GPU
paths = sorted({it.labels_path for it in idx.items}); pid = {p: i for i, p in enumerate(paths)}
stack = np.stack([np.stack(idx.layer_cache.get(p), 0) for p in paths])          # (n_crops, 3, 256, 256) uint8
print("stacked crops:", stack.shape, stack.dtype, f"{stack.nbytes/1e6:.0f} MB")
dev = "cuda"; G = torch.from_numpy(stack).to(dev)
crop_ix = torch.tensor([pid[it.labels_path] for it in idx.items], device=dev)
rows = torch.tensor([it.marker_row for it in idx.items], device=dev); cols = torch.tensor([it.marker_col for it in idx.items], device=dev)
def gpu_batch(B=32):
    j = torch.randint(0, len(crop_ix), (B,), device=dev)
    img = torch.zeros(B, 4, 256, 256, device=dev)
    img[:, :3] = G[crop_ix[j]].float() / 3.0
    ar = torch.arange(B, device=dev)
    img[ar, 3, rows[j], cols[j]] = 1.0
    tgt = torch.zeros(B, 64, 64, dtype=torch.long, device=dev); tgt[ar, rows[j] // 4, cols[j] // 4] = 1
    return img, tgt
gpu_batch(); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(100): gpu_batch()
torch.cuda.synchronize(); print(f"GPU-side batch build: {(time.perf_counter()-t)/100*1e3:.2f} ms per batch of 32")
a = L.next_batch()[0]; print("value check vs harness scaling: harness max", float(a[:, :3].max()), "| gpu", float(gpu_batch()[0][:, :3].max()))
