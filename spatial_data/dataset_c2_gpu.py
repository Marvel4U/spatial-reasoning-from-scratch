"""GPU-resident c2 batches: pixel-AND targets, fixed crops or on-the-fly district windows.

Two train sources. "fixed_crops" is what c1 does (the 2,000 saved train crops) and keeps
c1/c2 comparable. "district_random" cuts fresh 256 m windows out of the whole district at
1 m/px, rejecting every window that touches a val/test block — the same rule crops.py used,
so the held-out ground stays held out while the number of distinct train crops is unbounded.
Validation is always the deterministic full grid (every val crop x every active task).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .c2_tasks import C2TaskSpec
from .channels import Encoding
from .dataset_c1_gpu import build_image_from_crops

LAYER_KEYS = ("noise", "surface", "estab")
CROP_PX = 256


def load_crop_planes(crops_dir: Path, split: str, device: torch.device,
                     *, max_items: int | None = None) -> torch.Tensor:
    """Saved crops -> (N, 3, 256, 256) uint8 on `device` (merged noise/surface/estab only).

    Cropset v4 npz files also store noise7 and sun; c1/c2 regression reads only these three.
    """
    files = sorted(Path(crops_dir, split).glob("*_labels.npz"))[:max_items]
    if not files:
        raise FileNotFoundError(f"no {split} crops under {crops_dir}")
    stack = np.stack([np.stack([np.load(f)[f"{k}_256"] for k in LAYER_KEYS]) for f in files])
    return torch.from_numpy(stack.astype(np.uint8)).to(device)


def load_district_planes(crops_dir: Path, device: torch.device) -> torch.Tensor:
    """The whole district at 1 m/px -> (3, H, W) uint8 on `device`."""
    npz = np.load(Path(crops_dir) / "district_labels_1m.npz")
    return torch.from_numpy(np.stack([npz[k] for k in LAYER_KEYS]).astype(np.uint8)).to(device)


def held_out_rects(crops_dir: Path) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) in metres from the district NW corner for every val/test block."""
    split = json.loads(Path(crops_dir, "split.json").read_text())
    return [tuple(b["offset_m_from_district_nw"]) for b in split["blocks"] if b["split"] != "train"]


def c2_targets_from_planes(layers_u8: torch.Tensor, cond_ids: torch.Tensor,
                           *, grid_size: int = 64, foreground_class: int = 1) -> torch.Tensor:
    """(B,3,H,W) uint8 + (B,2) atom ids (-1 = always true) -> (B,G,G) long targets.

    Vectorised over the batch: gather each sample's two class planes, compare, AND at
    pixel level, then 4x4 majority. No Python loop, so a batch costs one pass.
    """
    b, _, h, w = layers_u8.shape
    g = grid_size
    s = h // g
    lay = torch.where(cond_ids >= 0, cond_ids.clamp_min(0) // 4, torch.zeros_like(cond_ids))
    cls = torch.where(cond_ids >= 0, cond_ids.clamp_min(0) % 4, torch.full_like(cond_ids, -1))
    sel = layers_u8.gather(1, lay.view(b, 2, 1, 1).expand(b, 2, h, w).long())
    hold = (sel.long() == cls.view(b, 2, 1, 1)) | (cond_ids < 0).view(b, 2, 1, 1)
    pix = hold[:, 0] & hold[:, 1]
    blk = pix.reshape(b, g, s, g, s).permute(0, 1, 3, 2, 4).reshape(b, g, g, s * s)
    tgt = torch.zeros(b, g, g, dtype=torch.long, device=layers_u8.device)
    tgt[blk.sum(-1) >= (s * s + 1) // 2] = foreground_class
    return tgt


def cond_id_table(task_specs: list[C2TaskSpec], device: torch.device) -> torch.Tensor:
    """(n_tasks, 2) atom ids with -1 padding for single-atom tasks."""
    return torch.tensor([t.padded_cond_ids() for t in task_specs], dtype=torch.long, device=device)


class C2DistrictSampler:
    """Uniform 256x256 windows on the 1 m grid that touch no held-out block."""

    def __init__(self, district: torch.Tensor, rects: list[tuple[float, float, float, float]],
                 seed: int):
        self.district = district
        self.device = district.device
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._rects = torch.tensor(rects or [[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32,
                                   device=self.device)
        self._has_rects = bool(rects)
        _, h, w = district.shape
        self.max_dy = h - CROP_PX
        self.max_dx = w - CROP_PX
        self._offs = torch.arange(CROP_PX, device=self.device)
        gy, gx = torch.meshgrid(torch.arange(self.max_dy + 1, device=self.device),
                                torch.arange(self.max_dx + 1, device=self.device), indexing="ij")
        # Exact size of the sampling pool — the honest "how many train crops exist" number.
        self.n_origins = int(self.accepts(gx.reshape(-1), gy.reshape(-1)).sum().item())

    def sample_origins(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rejection sampling, vectorised: draw a surplus, keep the accepted, top up."""
        dxs: list[torch.Tensor] = []
        dys: list[torch.Tensor] = []
        got = 0
        while got < n:
            k = 2 * (n - got) + 32
            dx = torch.randint(0, self.max_dx + 1, (k,), device=self.device, generator=self._gen)
            dy = torch.randint(0, self.max_dy + 1, (k,), device=self.device, generator=self._gen)
            ok = self.accepts(dx, dy)
            dxs.append(dx[ok])
            dys.append(dy[ok])
            got += int(ok.sum().item())
        return torch.cat(dxs)[:n], torch.cat(dys)[:n]

    def accepts(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """True where the window [dx, dx+256) x [dy, dy+256) misses every held-out block."""
        if not self._has_rects:
            return torch.ones_like(dx, dtype=torch.bool)
        x0, y0, x1, y1 = (self._rects[:, i].view(1, -1) for i in range(4))
        fx, fy = dx.float().view(-1, 1), dy.float().view(-1, 1)
        hit = (fx < x1) & (fx + CROP_PX > x0) & (fy < y1) & (fy + CROP_PX > y0)
        return ~hit.any(dim=1)

    def windows(self, n: int) -> torch.Tensor:
        dx, dy = self.sample_origins(n)
        rows = (dy.view(-1, 1) + self._offs).long()
        cols = (dx.view(-1, 1) + self._offs).long()
        return self.district[:, rows[:, :, None], cols[:, None, :]].permute(1, 0, 2, 3).contiguous()


class C2GpuTrainLoader:
    """Random (crop, task) batches. Returns (img, target, task_index, cond_ids)."""

    def __init__(self, task_specs: list[C2TaskSpec], encoding: Encoding, grid_size: int,
                 img_size: int, batch_size: int, seed: int, *, crops: torch.Tensor | None = None,
                 sampler: C2DistrictSampler | None = None,
                 train_task_indices: list[int] | None = None):
        if (crops is None) == (sampler is None):
            raise ValueError("give exactly one of crops= (fixed_crops) or sampler= (district_random)")
        self.crops = crops
        self.sampler = sampler
        self.device = crops.device if crops is not None else sampler.device
        self.task_specs = task_specs
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._cond = cond_id_table(task_specs, self.device)
        self._pick = torch.tensor(
            train_task_indices if train_task_indices is not None else list(range(len(task_specs))),
            dtype=torch.long, device=self.device,
        )

    def __len__(self) -> int:
        """Distinct source crops: the fixed set, or every admissible district window origin."""
        return int(self.crops.shape[0]) if self.crops is not None else self.sampler.n_origins

    def next_batch(self):
        if self.crops is not None:
            ix = torch.randint(0, self.crops.shape[0], (self.B,), device=self.device, generator=self._gen)
            layers = self.crops[ix]
        else:
            layers = self.sampler.windows(self.B)
        j = torch.randint(0, self._pick.shape[0], (self.B,), device=self.device, generator=self._gen)
        task_index = self._pick[j]
        return self._build(layers, task_index)

    def _build(self, layers: torch.Tensor, task_index: torch.Tensor):
        cond_ids = self._cond[task_index]
        img = build_image_from_crops(layers, self.encoding)
        tgt = c2_targets_from_planes(layers, cond_ids, grid_size=self.grid_size)
        return img, tgt, task_index, cond_ids


class C2GpuValLoader:
    """Deterministic full grid: every val crop x every active task, task-major, fixed order."""

    def __init__(self, crops: torch.Tensor, task_specs: list[C2TaskSpec], encoding: Encoding,
                 grid_size: int, img_size: int, batch_size: int):
        self.crops = crops
        self.device = crops.device
        self.task_specs = task_specs
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self.n_crops = int(crops.shape[0])
        self.n_tasks = len(task_specs)
        self._cond = cond_id_table(task_specs, self.device)
        self._pos = 0
        idx = torch.arange(self.n_crops * self.n_tasks, device=self.device)
        self._task_of = idx // self.n_crops   # task-major: task 0 over all crops, then task 1
        self._crop_of = idx % self.n_crops

    def __len__(self) -> int:
        return self.n_crops * self.n_tasks

    def reset(self) -> None:
        self._pos = 0

    def next_batch(self):
        n = len(self)
        lo = self._pos
        hi = min(lo + self.B, n)
        self._pos = hi % n
        sl = slice(lo, hi)
        layers = self.crops[self._crop_of[sl]]
        task_index = self._task_of[sl]
        cond_ids = self._cond[task_index]
        img = build_image_from_crops(layers, self.encoding)
        tgt = c2_targets_from_planes(layers, cond_ids, grid_size=self.grid_size)
        return img, tgt, task_index, cond_ids

    def iter_batches(self):
        """One full deterministic pass, from the start."""
        self.reset()
        for _ in range((len(self) + self.B - 1) // self.B):
            yield self.next_batch()
        self.reset()
