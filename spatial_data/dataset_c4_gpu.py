"""GPU-resident c4 batches: v4 district windows + fixed route samples from c4 npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .c4_tasks import (
    C4TargetKind,
    C4TaskSpec,
    FOREGROUND,
    c4_task_specs_for_keys,
    path_pix_to_grid_numpy,
    rasterise_segment_numpy,
)
from .channels import Encoding
from .dataset_c1_gpu import build_image_from_crops

C4_LAYER_KEYS = ("noise7", "surface", "estab")
C4_RGB_LAYER_KEYS = ("noise", "surface", "estab")  # 4-level planes drawn into the crop PNG
CROP_PX = 256


def load_district_planes_v4(crops_dir: Path, device: torch.device) -> torch.Tensor:
    npz = np.load(Path(crops_dir) / "district_labels_1m.npz")
    stack = np.stack([npz[k] for k in C4_LAYER_KEYS]).astype(np.uint8)
    return torch.from_numpy(stack).to(device)


def load_district_rgb_planes_v4(crops_dir: Path, device: torch.device) -> torch.Tensor:
    """Whole district RGB stack (3, H, W) uint8 — same keys as crop PNG / c3 pred strips."""
    npz = np.load(Path(crops_dir) / "district_labels_1m.npz")
    stack = np.stack([npz[k] for k in C4_RGB_LAYER_KEYS]).astype(np.uint8)
    return torch.from_numpy(stack).to(device)


def load_c4_route_index(routes_dir: Path, split: str, device: torch.device,
                        *, max_items: int | None = None) -> dict[str, torch.Tensor]:
    path = Path(routes_dir) / f"c4_routes_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    n = z["origin"].shape[0]
    if max_items is not None:
        n = min(n, max_items)
    origin = torch.from_numpy(z["origin"][:n].astype(np.int64)).to(device)
    markers = torch.from_numpy(z["markers"][:n].astype(np.float32)).to(device)
    path_pix = torch.from_numpy(z["path_pix"][:n].astype(np.int64)).to(device)
    n_pix = torch.from_numpy(z["n_pix"][:n].astype(np.int64)).to(device)
    detour_ratio = torch.from_numpy(z["detour_ratio"][:n].astype(np.float32)).to(device)
    straight_px = torch.from_numpy(z["straight_px"][:n].astype(np.float32)).to(device)
    path_px = torch.from_numpy(z["path_px"][:n].astype(np.float32)).to(device)
    stratum = torch.from_numpy(z["stratum"][:n].astype(np.int64)).to(device)
    return dict(
        origin=origin, markers=markers, path_pix=path_pix, n_pix=n_pix,
        detour_ratio=detour_ratio, straight_px=straight_px, path_px=path_px, stratum=stratum,
    )


def _cut_windows(district: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """district (3,H,W), origin (B,2) dx,dy -> (B,3,CROP,CROP). One gather, no per-sample sync."""
    ar = torch.arange(CROP_PX, device=district.device)
    ys = origin[:, 1].view(-1, 1) + ar.view(1, -1)          # (B, CROP)
    xs = origin[:, 0].view(-1, 1) + ar.view(1, -1)
    out = district[:, ys[:, :, None], xs[:, None, :]]        # (3, B, CROP, CROP)
    return out.permute(1, 0, 2, 3).contiguous()


def _marker_cols_rows(markers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,2,2) float xy -> (B,2) long col/row."""
    col = markers[..., 0].floor().clamp(0, CROP_PX - 1).long()
    row = markers[..., 1].floor().clamp(0, CROP_PX - 1).long()
    return col, row


def _scatter_cells(tgt: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, valid: torch.Tensor) -> None:
    """Mark the 4 m cell of every valid (x, y) pixel in tgt (B, g, g) in place."""
    b, g, _ = tgt.shape
    s = CROP_PX // g
    gc = (xs // s).clamp(0, g - 1); gr = (ys // s).clamp(0, g - 1)
    bi = torch.arange(b, device=tgt.device).view(b, 1).expand_as(gc)
    flat = tgt.view(b, g * g)
    flat[bi[valid], (gr * g + gc)[valid]] = FOREGROUND


def _segment_targets(markers: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Same sampling as rasterise_segment_numpy (L = ceil(2 |b-a|) steps, t = i / L), vectorised."""
    b = markers.shape[0]; dev = markers.device
    a = markers[:, 0]; q = markers[:, 1]
    L = (q - a).norm(dim=1).mul(2).ceil().clamp(min=1)                       # (B,)
    Lmax = int(L.max().item()) if b else 1                                     # one sync per batch
    i = torch.arange(Lmax + 1, device=dev, dtype=torch.float32).view(1, -1)   # (1, Lmax+1)
    t = (i / L.view(-1, 1)).clamp(max=1.0)                                     # (B, Lmax+1)
    valid = i <= L.view(-1, 1)
    pts = a[:, None, :] + t[..., None] * (q - a)[:, None, :]
    xs = pts[..., 0].floor().long().clamp(0, CROP_PX - 1)
    ys = pts[..., 1].floor().long().clamp(0, CROP_PX - 1)
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=dev)
    _scatter_cells(tgt, xs, ys, valid)
    return tgt


def _path_targets(path_pix: torch.Tensor, n_pix: torch.Tensor, grid_size: int) -> torch.Tensor:
    """(B, P, 2) padded int pixels (-1 = unused) -> (B, g, g), any pixel per cell."""
    b, P, _ = path_pix.shape; dev = path_pix.device
    idx = torch.arange(P, device=dev).view(1, -1)
    valid = (idx < n_pix.view(-1, 1)) & (path_pix[..., 0] >= 0) & (path_pix[..., 1] >= 0)
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=dev)
    _scatter_cells(tgt, path_pix[..., 0].clamp(min=0), path_pix[..., 1].clamp(min=0), valid)
    return tgt


def _building_targets(layers: torch.Tensor, grid_size: int) -> torch.Tensor:
    """(B,3,CROP,CROP) uint8 planes -> (B,g,g): cell is building if >= half its pixels are class 3."""
    b = layers.shape[0]; g = grid_size; s = CROP_PX // g
    bld = (layers[:, 1] == 3).view(b, g, s, g, s).sum(dim=(2, 4))
    return (bld >= (s * s) // 2).long() * FOREGROUND


def _targets_for_batch(
    spec: C4TaskSpec,
    *,
    markers: torch.Tensor,
    path_pix: torch.Tensor,
    n_pix: torch.Tensor,
    grid_size: int,
    device: torch.device,
    layers: torch.Tensor | None = None,
) -> torch.Tensor:
    if spec.target_kind == C4TargetKind.SEGMENT:
        return _segment_targets(markers, grid_size)
    if spec.target_kind == C4TargetKind.BUILDING:
        if layers is None:
            raise ValueError("building targets need the window layers")
        return _building_targets(layers, grid_size)
    return _path_targets(path_pix, n_pix, grid_size)


def c4_cond_id_table(task_specs: list[C4TaskSpec], device: torch.device) -> torch.Tensor:
    return torch.tensor([s.padded_cond_ids() for s in task_specs], dtype=torch.long, device=device)


def c4_targets_mixed_batch(
    task_index: torch.Tensor,
    task_specs: list[C4TaskSpec],
    *,
    markers: torch.Tensor,
    path_pix: torch.Tensor,
    n_pix: torch.Tensor,
    grid_size: int,
    device: torch.device,
    layers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Every target kind for the whole batch, then a per-sample select: no Python loop."""
    kinds = [sp.target_kind for sp in task_specs]
    kind_ix = torch.tensor([{C4TargetKind.SEGMENT: 0, C4TargetKind.DETOUR_PATH: 1, C4TargetKind.BUILDING: 2}[k] for k in kinds],
                           dtype=torch.long, device=device)[task_index]
    out = _path_targets(path_pix, n_pix, grid_size)
    if C4TargetKind.SEGMENT in kinds:
        out = torch.where((kind_ix == 0).view(-1, 1, 1), _segment_targets(markers, grid_size), out)
    if C4TargetKind.BUILDING in kinds:
        if layers is None:
            raise ValueError("building targets need the window layers")
        out = torch.where((kind_ix == 2).view(-1, 1, 1), _building_targets(layers, grid_size), out)
    return out


class C4GpuTrainLoader:
    def __init__(
        self,
        spec: C4TaskSpec,
        encoding: Encoding,
        grid_size: int,
        batch_size: int,
        seed: int,
        *,
        district: torch.Tensor,
        route_index: dict[str, torch.Tensor],
    ):
        self.district = district
        self.route = route_index
        self.device = district.device
        self.spec = spec
        self.encoding = encoding
        self.grid_size = grid_size
        self.B = batch_size
        self.n_routes = int(route_index["origin"].shape[0])
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        a, r = spec.padded_cond_ids()
        self._cond = torch.tensor([[a, r]], dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.n_routes

    def _gather(self, ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = self.route["origin"][ix]
        markers = self.route["markers"][ix]
        path_pix = self.route["path_pix"][ix]
        n_pix = self.route["n_pix"][ix]
        layers = _cut_windows(self.district, origin)
        return layers, markers, path_pix, n_pix, origin

    def _build(self, ix: torch.Tensor):
        b = ix.shape[0]
        layers, markers, path_pix, n_pix, _ = self._gather(ix)
        mc, mr = _marker_cols_rows(markers)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = _targets_for_batch(
            self.spec, markers=markers, path_pix=path_pix, n_pix=n_pix,
            grid_size=self.grid_size, device=self.device, layers=layers,
        )
        task_index = torch.zeros(b, dtype=torch.long, device=self.device)
        cond_ids = self._cond.expand(b, -1)
        return img, tgt, task_index, cond_ids

    def next_batch(self):
        ix = torch.randint(0, self.n_routes, (self.B,), device=self.device, generator=self._gen)
        return self._build(ix)


class C4GpuValLoader:
    def __init__(
        self,
        spec: C4TaskSpec,
        encoding: Encoding,
        grid_size: int,
        batch_size: int,
        *,
        district: torch.Tensor,
        route_index: dict[str, torch.Tensor],
    ):
        self.district = district
        self.route = route_index
        self.device = district.device
        self.spec = spec
        self.encoding = encoding
        self.grid_size = grid_size
        self.B = batch_size
        self.n_routes = int(route_index["origin"].shape[0])
        a, r = spec.padded_cond_ids()
        self._cond = torch.tensor([[a, r]], dtype=torch.long, device=self.device)
        self._pos = 0

    def __len__(self) -> int:
        return self.n_routes

    def reset(self) -> None:
        self._pos = 0

    def _build(self, ix: torch.Tensor):
        b = ix.shape[0]
        origin = self.route["origin"][ix]
        markers = self.route["markers"][ix]
        path_pix = self.route["path_pix"][ix]
        n_pix = self.route["n_pix"][ix]
        layers = _cut_windows(self.district, origin)
        mc, mr = _marker_cols_rows(markers)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = _targets_for_batch(
            self.spec, markers=markers, path_pix=path_pix, n_pix=n_pix,
            grid_size=self.grid_size, device=self.device, layers=layers,
        )
        task_index = torch.zeros(b, dtype=torch.long, device=self.device)
        cond_ids = self._cond.expand(b, -1)
        return img, tgt, task_index, cond_ids

    def _route_task_from_flat(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return flat, torch.zeros_like(flat)

    def _meta_for_routes(self, route_ix: torch.Tensor) -> dict[str, torch.Tensor]:
        r = self.route
        return {
            "route_ix": route_ix,
            "detour_ratio": r["detour_ratio"][route_ix],
            "straight_px": r["straight_px"][route_ix],
            "path_px": r["path_px"][route_ix],
            "stratum": r["stratum"][route_ix],
            "markers": r["markers"][route_ix],
        }

    def next_batch(self):
        n = len(self)
        lo, hi = self._pos, min(self._pos + self.B, n)
        self._pos = hi % n if n else 0
        ix = torch.arange(lo, hi, device=self.device)
        return self._build(ix)

    def next_batch_with_meta(self):
        n = len(self)
        lo, hi = self._pos, min(self._pos + self.B, n)
        self._pos = hi % n if n else 0
        flat = torch.arange(lo, hi, device=self.device)
        route_ix, task_ix = self._route_task_from_flat(flat)
        img, tgt, task_index, cond_ids = self._build(route_ix)
        return img, tgt, task_index, cond_ids, {**self._meta_for_routes(route_ix), "task_ix": task_ix}

    def fetch_at(self, flat: int):
        flat_t = torch.tensor([flat], device=self.device, dtype=torch.long)
        route_ix, task_ix = self._route_task_from_flat(flat_t)
        img, tgt, task_index, cond_ids = self._build(route_ix)
        return img[0], tgt[0], task_index[0], cond_ids[0], {**self._meta_for_routes(route_ix), "task_ix": task_ix[0]}

    def iter_batches(self):
        self.reset()
        for _ in range((len(self) + self.B - 1) // self.B):
            yield self.next_batch()
        self.reset()


class C4MultiTaskGpuTrainLoader:
    def __init__(
        self,
        task_keys: list[str],
        encoding: Encoding,
        grid_size: int,
        batch_size: int,
        seed: int,
        *,
        district: torch.Tensor,
        route_index: dict[str, torch.Tensor],
        train_task_weights: list[float] | None = None,
    ):
        self.task_specs = c4_task_specs_for_keys(task_keys)
        self.district = district
        self.route = route_index
        self.device = district.device
        self.encoding = encoding
        self.grid_size = grid_size
        self.B = batch_size
        self.n_routes = int(route_index["origin"].shape[0])
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._cond = c4_cond_id_table(self.task_specs, self.device)
        nt = len(self.task_specs)
        if train_task_weights is None:
            self._task_weights = None
        else:
            if len(train_task_weights) != nt:
                raise ValueError(f"train_task_weights len {len(train_task_weights)} != n_tasks {nt}")
            w = torch.tensor(train_task_weights, dtype=torch.float, device=self.device)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"bad train_task_weights {train_task_weights}")
            self._task_weights = w / w.sum()

    def __len__(self) -> int:
        return self.n_routes

    def _sample_task_index(self, batch_size: int) -> torch.Tensor:
        nt = len(self.task_specs)
        if self._task_weights is None:
            return torch.randint(0, nt, (batch_size,), device=self.device, generator=self._gen)
        return torch.multinomial(self._task_weights, batch_size, replacement=True, generator=self._gen)

    def next_batch(self):
        ix = torch.randint(0, self.n_routes, (self.B,), device=self.device, generator=self._gen)
        b = ix.shape[0]
        origin = self.route["origin"][ix]
        markers = self.route["markers"][ix]
        path_pix = self.route["path_pix"][ix]
        n_pix = self.route["n_pix"][ix]
        layers = _cut_windows(self.district, origin)
        mc, mr = _marker_cols_rows(markers)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        task_index = self._sample_task_index(b)
        tgt = c4_targets_mixed_batch(
            task_index, self.task_specs, markers=markers, path_pix=path_pix, n_pix=n_pix,
            grid_size=self.grid_size, device=self.device, layers=layers,
        )
        cond_ids = self._cond[task_index]
        return img, tgt, task_index, cond_ids


class C4MultiTaskGpuValLoader:
    """Full val grid: every route × every active c4 task."""

    def __init__(
        self,
        task_keys: list[str],
        encoding: Encoding,
        grid_size: int,
        batch_size: int,
        *,
        district: torch.Tensor,
        route_index: dict[str, torch.Tensor],
    ):
        self.task_specs = c4_task_specs_for_keys(task_keys)
        self.district = district
        self.route = route_index
        self.device = district.device
        self.encoding = encoding
        self.grid_size = grid_size
        self.B = batch_size
        self.n_routes = int(route_index["origin"].shape[0])
        self.n_tasks = len(self.task_specs)
        self._cond = c4_cond_id_table(self.task_specs, self.device)
        self._pos = 0

    def __len__(self) -> int:
        return self.n_routes * self.n_tasks

    def reset(self) -> None:
        self._pos = 0

    def _route_task_from_flat(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return flat // self.n_tasks, flat % self.n_tasks

    def _meta_for_routes(self, route_ix: torch.Tensor) -> dict[str, torch.Tensor]:
        r = self.route
        return {
            "route_ix": route_ix,
            "detour_ratio": r["detour_ratio"][route_ix],
            "straight_px": r["straight_px"][route_ix],
            "path_px": r["path_px"][route_ix],
            "stratum": r["stratum"][route_ix],
            "markers": r["markers"][route_ix],
        }

    def _build_from_flat(self, flat: torch.Tensor):
        route_ix, task_index = self._route_task_from_flat(flat)
        origin = self.route["origin"][route_ix]
        markers = self.route["markers"][route_ix]
        path_pix = self.route["path_pix"][route_ix]
        n_pix = self.route["n_pix"][route_ix]
        layers = _cut_windows(self.district, origin)
        mc, mr = _marker_cols_rows(markers)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c4_targets_mixed_batch(
            task_index, self.task_specs, markers=markers, path_pix=path_pix, n_pix=n_pix,
            grid_size=self.grid_size, device=self.device, layers=layers,
        )
        cond_ids = self._cond[task_index]
        return img, tgt, task_index, cond_ids, route_ix

    def next_batch(self):
        total = len(self)
        lo, hi = self._pos, min(self._pos + self.B, total)
        self._pos = hi % total if total else 0
        flat = torch.arange(lo, hi, device=self.device)
        img, tgt, task_index, cond_ids, route_ix = self._build_from_flat(flat)
        return img, tgt, task_index, cond_ids

    def next_batch_with_meta(self):
        total = len(self)
        lo, hi = self._pos, min(self._pos + self.B, total)
        self._pos = hi % total if total else 0
        flat = torch.arange(lo, hi, device=self.device)
        img, tgt, task_index, cond_ids, route_ix = self._build_from_flat(flat)
        _, task_ix = self._route_task_from_flat(flat)
        return img, tgt, task_index, cond_ids, {**self._meta_for_routes(route_ix), "task_ix": task_ix}

    def fetch_at(self, flat: int):
        flat_t = torch.tensor([flat], device=self.device, dtype=torch.long)
        img, tgt, task_index, cond_ids, route_ix = self._build_from_flat(flat_t)
        _, task_ix = self._route_task_from_flat(flat_t)
        return img[0], tgt[0], task_index[0], cond_ids[0], {**self._meta_for_routes(route_ix), "task_ix": task_ix[0]}
