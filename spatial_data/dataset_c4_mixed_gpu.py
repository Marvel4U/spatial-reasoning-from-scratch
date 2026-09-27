"""Train batches mixing c3 marker tasks (v4 crops) with c4 route tasks (district windows)."""
from __future__ import annotations

import torch

from .c3_targets import c3_targets_for_spec
from .c3_tasks import C3TaskSpec, c3_task_specs_for_keys
from .c4_tasks import C4TaskSpec, c4_task_specs_for_keys
from .channels import Encoding
from .dataset_c1_gpu import build_image_from_crops
from .dataset_c3_gpu import c3_cond_id_table, sample_markers
from .dataset_c4_gpu import (
    C4GpuValLoader,
    _cut_windows,
    _marker_cols_rows,
    _targets_for_batch,
    c4_cond_id_table,
)


class C4C3MixGpuTrainLoader:
    """Random (c3 crop task | c4 route task) per sample; shared v4 encoding + marker plane."""

    def __init__(
        self,
        c3_task_keys: list[str],
        c4_task_keys: list[str],
        encoding: Encoding,
        grid_size: int,
        img_size: int,
        batch_size: int,
        seed: int,
        *,
        c3_crops: torch.Tensor,
        c4_district: torch.Tensor,
        c4_route_index: dict[str, torch.Tensor],
        c3_n_markers: int = 2,
        c3_marker_min_dist_px: float = 40.0,
        train_task_weights: list[float] | None = None,
    ):
        self.c3_specs: list[C3TaskSpec] = c3_task_specs_for_keys(c3_task_keys)
        self.c4_specs: list[C4TaskSpec] = c4_task_specs_for_keys(c4_task_keys)
        self.n_c3 = len(self.c3_specs)
        self.n_c4 = len(self.c4_specs)
        self.n_tasks = self.n_c3 + self.n_c4
        self.c3_crops = c3_crops
        self.c4_district = c4_district
        self.c4_route = c4_route_index
        self.device = c3_crops.device
        self.encoding = encoding
        self.grid_size = grid_size
        self.img_size = img_size
        self.B = batch_size
        self.c3_n_markers = int(c3_n_markers)
        self.c3_marker_min_dist_px = float(c3_marker_min_dist_px)
        self.n_c3_crops = int(c3_crops.shape[0])
        self.n_c4_routes = int(c4_route_index["origin"].shape[0])
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        c3_cond = c3_cond_id_table(self.c3_specs, self.device)
        c4_cond = c4_cond_id_table(self.c4_specs, self.device)
        self._cond = torch.cat([c3_cond, c4_cond], dim=0)
        if train_task_weights is None:
            self._task_weights = None
        else:
            if len(train_task_weights) != self.n_tasks:
                raise ValueError(f"train_task_weights len {len(train_task_weights)} != {self.n_tasks}")
            w = torch.tensor(train_task_weights, dtype=torch.float, device=self.device)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"bad train_task_weights {train_task_weights}")
            self._task_weights = w / w.sum()

    def __len__(self) -> int:
        return max(self.n_c3_crops, self.n_c4_routes)

    def _sample_task_index(self, batch_size: int) -> torch.Tensor:
        if self._task_weights is None:
            return torch.randint(0, self.n_tasks, (batch_size,), device=self.device, generator=self._gen)
        return torch.multinomial(self._task_weights, batch_size, replacement=True, generator=self._gen)

    def _build_c3(self, spec: C3TaskSpec, crop_ix: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layers = self.c3_crops[crop_ix : crop_ix + 1]
        mc, mr = sample_markers(
            1, self.c3_n_markers, self.c3_marker_min_dist_px, self.img_size, self._gen, self.device,
        )
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = c3_targets_for_spec(
            layers, spec, grid_size=self.grid_size, marker_col=mc, marker_row=mr,
        )
        return img[0], tgt[0], mc[0], mr[0]

    def _build_c4(self, spec: C4TaskSpec, route_ix: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ix = torch.tensor([route_ix], device=self.device, dtype=torch.long)
        origin = self.c4_route["origin"][ix]
        markers = self.c4_route["markers"][ix]
        path_pix = self.c4_route["path_pix"][ix]
        n_pix = self.c4_route["n_pix"][ix]
        layers = _cut_windows(self.c4_district, origin)
        mc, mr = _marker_cols_rows(markers)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        tgt = _targets_for_batch(
            spec, markers=markers, path_pix=path_pix, n_pix=n_pix,
            grid_size=self.grid_size, device=self.device, layers=layers,
        )
        return img[0], tgt[0], mc[0], mr[0]

    def next_batch(self):
        task_index = self._sample_task_index(self.B)
        imgs, tgts, tids, cids = [], [], [], []
        for i in range(self.B):
            ti = int(task_index[i].item())
            if ti < self.n_c3:
                crop_ix = int(torch.randint(0, self.n_c3_crops, (1,), device=self.device, generator=self._gen).item())
                img, tgt, _, _ = self._build_c3(self.c3_specs[ti], crop_ix)
            else:
                ci = ti - self.n_c3
                route_ix = int(torch.randint(0, self.n_c4_routes, (1,), device=self.device, generator=self._gen).item())
                img, tgt, _, _ = self._build_c4(self.c4_specs[ci], route_ix)
            imgs.append(img)
            tgts.append(tgt)
            tids.append(ti)
            cids.append(self._cond[ti])
        return (
            torch.stack(imgs),
            torch.stack(tgts),
            torch.tensor(tids, device=self.device, dtype=torch.long),
            torch.stack(cids),
        )


def c4_segment_val_loader(
    encoding: Encoding,
    grid_size: int,
    batch_size: int,
    *,
    district: torch.Tensor,
    route_index: dict[str, torch.Tensor],
):
    from .c4_tasks import get_c4_task
    return C4GpuValLoader(
        get_c4_task("segment"), encoding, grid_size, batch_size,
        district=district, route_index=route_index,
    )
