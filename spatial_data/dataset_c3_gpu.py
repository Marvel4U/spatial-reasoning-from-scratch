"""GPU-resident c3 batches: v4 noise7 planes + relative / marker-relation targets."""
from __future__ import annotations

import torch

from .c3_targets import c3_targets_for_spec
from .c3_tasks import C3TaskSpec, c3_task_specs_for_keys
from .channels import Encoding
from .dataset_c1_gpu import build_image_from_crops

C3_LAYER_NPZ = ("noise7_256", "surface_256", "estab_256")


def load_crop_planes_v4(crops_dir, split: str, device: torch.device,
                        *, max_items: int | None = None) -> torch.Tensor:
    from pathlib import Path
    import numpy as np
    files = sorted(Path(crops_dir, split).glob("*_labels.npz"))[:max_items]
    if not files:
        raise FileNotFoundError(f"no {split} crops under {crops_dir}")
    stack = np.stack([np.stack([np.load(f)[k] for k in C3_LAYER_NPZ]) for f in files])
    return torch.from_numpy(stack.astype(np.uint8)).to(device)


def sample_markers(b: int, k: int, min_dist: float, size: int, gen: torch.Generator,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,K) marker cols/rows, uniform in [0, size), every pair >= min_dist apart.

    Rejection per sample: draw K, redraw the sample while any pair is too close. With
    K = 4 and min_dist = 40 px in a 256 px crop the acceptance rate is high, so the
    loop ends after a few rounds; a cap avoids an infinite loop for impossible settings.
    """
    col = torch.randint(0, size, (b, k), device=device, generator=gen)
    row = torch.randint(0, size, (b, k), device=device, generator=gen)
    for _ in range(200):
        d2 = (col[:, :, None] - col[:, None, :]) ** 2 + (row[:, :, None] - row[:, None, :]) ** 2
        d2 = d2.float() + torch.eye(k, device=device) * 1e9
        bad = (d2.min(dim=2).values.min(dim=1).values < min_dist * min_dist)
        if not bad.any():
            return col, row
        nb = int(bad.sum().item())
        col[bad] = torch.randint(0, size, (nb, k), device=device, generator=gen)
        row[bad] = torch.randint(0, size, (nb, k), device=device, generator=gen)
    raise RuntimeError(f"could not place {k} markers >= {min_dist} px apart in {size} px")


def sample_markers_random_k_per_sample(
    b: int,
    k_min: int,
    k_max: int,
    min_dist: float,
    size: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, k_max) marker cols/rows; unused slots -1. Each sample gets K_i ~ Uniform[k_min, k_max]."""
    if k_min < 1 or k_max < k_min:
        raise ValueError(f"need 1 <= k_min <= k_max, got {k_min}, {k_max}")
    col = torch.full((b, k_max), -1, device=device, dtype=torch.long)
    row = torch.full((b, k_max), -1, device=device, dtype=torch.long)
    for bi in range(b):
        ki = int(torch.randint(k_min, k_max + 1, (1,), device=device, generator=gen).item())
        c, r = sample_markers(1, ki, min_dist, size, gen, device)
        col[bi, :ki] = c[0, :ki]
        row[bi, :ki] = r[0, :ki]
    return col, row


class C3GpuTrainLoader:
    def __init__(
        self,
        spec: C3TaskSpec,
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        seed: int,
        *,
        crops: torch.Tensor,
        marker_col: torch.Tensor | None = None,
        marker_row: torch.Tensor | None = None,
        n_markers: int = 0,
        marker_min_dist_px: float = 0.0,
        n_markers_random: tuple[int, int] | None = None,
    ):
        self.crops = crops
        self.device = crops.device
        self.spec = spec
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        # n_markers > 0: ignore the fixed per-crop marker and draw K fresh markers per batch
        self.n_markers = int(n_markers)
        self.marker_min_dist_px = float(marker_min_dist_px)
        self.n_markers_random = n_markers_random
        if n_markers_random is not None:
            lo, hi = n_markers_random
            if lo < 1 or hi < lo:
                raise ValueError(f"bad n_markers_random {n_markers_random}")
        if spec.show_marker_plane and (self.n_markers > 0 or n_markers_random is not None):
            self._marker_col = self._marker_row = None
        elif spec.show_marker_plane:
            if marker_col is None or marker_row is None:
                raise ValueError("marker plane task requires marker_col and marker_row")
            self._marker_col = marker_col
            self._marker_row = marker_row
        else:
            self._marker_col = None
            self._marker_row = None
        a, r = spec.padded_cond_ids()
        self._cond = torch.tensor([[a, r]], dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return int(self.crops.shape[0])

    def _build(self, layers: torch.Tensor, ix: torch.Tensor):
        b = layers.shape[0]
        task_index = torch.zeros(b, dtype=torch.long, device=self.device)
        cond_ids = self._cond.expand(b, -1)
        mc = mr = None
        if self.spec.show_marker_plane and self.n_markers_random is not None:
            lo, hi = self.n_markers_random
            mc, mr = sample_markers_random_k_per_sample(
                b, lo, hi, self.marker_min_dist_px, self.img_size, self._gen, self.device,
            )
        elif self.spec.show_marker_plane and self.n_markers > 0:
            mc, mr = sample_markers(b, self.n_markers, self.marker_min_dist_px, self.img_size,
                                    self._gen, self.device)
        elif self._marker_col is not None:
            mc = self._marker_col[ix]
            mr = self._marker_row[ix]
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c3_targets_for_spec(
            layers, self.spec, grid_size=self.grid_size, marker_col=mc, marker_row=mr,
        )
        return img, tgt, task_index, cond_ids

    def next_batch(self):
        ix = torch.randint(0, self.crops.shape[0], (self.B,), device=self.device, generator=self._gen)
        return self._build(self.crops[ix], ix)


class C3GpuValLoader:
    def __init__(
        self,
        crops: torch.Tensor,
        spec: C3TaskSpec,
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        *,
        marker_col: torch.Tensor | None = None,
        marker_row: torch.Tensor | None = None,
    ):
        self.crops = crops
        self.device = crops.device
        self.spec = spec
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self.n_crops = int(crops.shape[0])
        if spec.show_marker_plane:
            if marker_col is None or marker_row is None:
                raise ValueError("marker plane task requires marker_col and marker_row")
            self._marker_col = marker_col
            self._marker_row = marker_row
        else:
            self._marker_col = None
            self._marker_row = None
        a, r = spec.padded_cond_ids()
        self._cond = torch.tensor([[a, r]], dtype=torch.long, device=self.device)
        self._pos = 0

    def __len__(self) -> int:
        return self.n_crops

    def reset(self) -> None:
        self._pos = 0

    def _build(self, ix: torch.Tensor):
        layers = self.crops[ix]
        b = layers.shape[0]
        task_index = torch.zeros(b, dtype=torch.long, device=self.device)
        cond_ids = self._cond.expand(b, -1)
        mc = mr = None
        if self._marker_col is not None:
            mc = self._marker_col[ix]
            mr = self._marker_row[ix]
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c3_targets_for_spec(
            layers, self.spec, grid_size=self.grid_size, marker_col=mc, marker_row=mr,
        )
        return img, tgt, task_index, cond_ids

    def next_batch(self):
        n = len(self)
        lo = self._pos
        hi = min(lo + self.B, n)
        self._pos = hi % n
        ix = torch.arange(lo, hi, device=self.device)
        return self._build(ix)

    def iter_batches(self):
        self.reset()
        for _ in range((len(self) + self.B - 1) // self.B):
            yield self.next_batch()
        self.reset()


def c3_cond_id_table(task_specs: list[C3TaskSpec], device: torch.device) -> torch.Tensor:
    return torch.tensor([s.padded_cond_ids() for s in task_specs], dtype=torch.long, device=device)


def c3_targets_mixed_batch(
    layers_u8: torch.Tensor,
    task_index: torch.Tensor,
    task_specs: list[C3TaskSpec],
    *,
    grid_size: int,
    marker_col: torch.Tensor | None,
    marker_row: torch.Tensor | None,
) -> torch.Tensor:
    b = layers_u8.shape[0]
    parts = []
    for i in range(b):
        ti = int(task_index[i].item())
        mc = marker_col[i : i + 1] if marker_col is not None else None
        mr = marker_row[i : i + 1] if marker_row is not None else None
        parts.append(
            c3_targets_for_spec(
                layers_u8[i : i + 1],
                task_specs[ti],
                grid_size=grid_size,
                marker_col=mc,
                marker_row=mr,
            )
        )
    return torch.cat(parts, dim=0)


class C3MultiTaskGpuTrainLoader:
    """Random (crop, c3 task) batches; shared marker plane when specs use it."""

    def __init__(
        self,
        task_keys: list[str],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        seed: int,
        *,
        crops: torch.Tensor,
        marker_col: torch.Tensor,
        marker_row: torch.Tensor,
        n_markers: int = 0,
        marker_min_dist_px: float = 0.0,
        n_markers_random: tuple[int, int] | None = None,
        train_task_weights: list[float] | None = None,
    ):
        self.task_specs = c3_task_specs_for_keys(task_keys)
        self.crops = crops
        self.device = crops.device
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self.n_markers = int(n_markers)
        self.marker_min_dist_px = float(marker_min_dist_px)
        self.n_markers_random = n_markers_random
        self._marker_col = marker_col
        self._marker_row = marker_row
        self._cond = c3_cond_id_table(self.task_specs, self.device)
        nt = len(self.task_specs)
        if train_task_weights is None:
            self._task_weights = None
        else:
            if len(train_task_weights) != nt:
                raise ValueError(f"train_task_weights len {len(train_task_weights)} != n_tasks {nt}")
            w = torch.tensor(train_task_weights, dtype=torch.float, device=self.device)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"train_task_weights must be non-negative with positive sum, got {train_task_weights}")
            self._task_weights = w / w.sum()

    def __len__(self) -> int:
        return int(self.crops.shape[0])

    def _markers_for_batch(self, b: int, ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_markers_random is not None:
            lo, hi = self.n_markers_random
            return sample_markers_random_k_per_sample(
                b, lo, hi, self.marker_min_dist_px, self.img_size, self._gen, self.device,
            )
        if self.n_markers > 0:
            return sample_markers(
                b, self.n_markers, self.marker_min_dist_px, self.img_size, self._gen, self.device,
            )
        return self._marker_col[ix], self._marker_row[ix]

    def _sample_task_index(self, batch_size: int) -> torch.Tensor:
        nt = len(self.task_specs)
        if self._task_weights is None:
            return torch.randint(0, nt, (batch_size,), device=self.device, generator=self._gen)
        return torch.multinomial(self._task_weights, batch_size, replacement=True, generator=self._gen)

    def _build(self, layers: torch.Tensor, ix: torch.Tensor, task_index: torch.Tensor):
        b = layers.shape[0]
        mc, mr = self._markers_for_batch(b, ix)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c3_targets_mixed_batch(
            layers, task_index, self.task_specs,
            grid_size=self.grid_size, marker_col=mc, marker_row=mr,
        )
        cond_ids = self._cond[task_index]
        return img, tgt, task_index, cond_ids

    def next_batch(self):
        n = self.crops.shape[0]
        ix = torch.randint(0, n, (self.B,), device=self.device, generator=self._gen)
        task_index = self._sample_task_index(self.B)
        return self._build(self.crops[ix], ix, task_index)


class C3MultiTaskGpuValLoader:
    """Task-major val: all crops for task 0, then task 1, …"""

    def __init__(
        self,
        crops: torch.Tensor,
        task_keys: list[str],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        *,
        marker_col: torch.Tensor,
        marker_row: torch.Tensor,
    ):
        self.crops = crops
        self.device = crops.device
        self.task_specs = c3_task_specs_for_keys(task_keys)
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self.n_crops = int(crops.shape[0])
        self.n_tasks = len(self.task_specs)
        self._marker_col = marker_col
        self._marker_row = marker_row
        self._cond = c3_cond_id_table(self.task_specs, self.device)
        self._pos = 0
        idx = torch.arange(self.n_crops * self.n_tasks, device=self.device)
        self._task_of = idx // self.n_crops
        self._crop_of = idx % self.n_crops

    def __len__(self) -> int:
        return self.n_crops * self.n_tasks

    def reset(self) -> None:
        self._pos = 0

    def _build(self, sl: slice):
        layers = self.crops[self._crop_of[sl]]
        task_index = self._task_of[sl]
        ix = self._crop_of[sl]
        mc = self._marker_col[ix]
        mr = self._marker_row[ix]
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c3_targets_mixed_batch(
            layers, task_index, self.task_specs,
            grid_size=self.grid_size, marker_col=mc, marker_row=mr,
        )
        cond_ids = self._cond[task_index]
        return img, tgt, task_index, cond_ids

    def next_batch(self):
        n = len(self)
        lo = self._pos
        hi = min(lo + self.B, n)
        self._pos = hi % n
        return self._build(slice(lo, hi))

    def iter_batches(self):
        self.reset()
        for _ in range((len(self) + self.B - 1) // self.B):
            yield self.next_batch()
        self.reset()

    def iter_batches_for_task(self, task_index: int, max_batches: int | None = None):
        """One task's crops only — for CE snap on task-major multi val (m1 not in first batches)."""
        if task_index < 0 or task_index >= self.n_tasks:
            raise ValueError(f"task_index {task_index} out of range [0, {self.n_tasks})")
        lo_g = task_index * self.n_crops
        hi_g = lo_g + self.n_crops
        pos = 0
        n_batches = 0
        while lo_g + pos < hi_g:
            if max_batches is not None and n_batches >= max_batches:
                break
            sl_lo = lo_g + pos
            sl_hi = min(sl_lo + self.B, hi_g)
            pos += sl_hi - sl_lo
            yield self._build(slice(sl_lo, sl_hi))
            n_batches += 1
