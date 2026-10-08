"""GPU-resident c4 batches: v4 district windows + fixed route samples from c4 npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .c4_tasks import (
    C4TargetKind,
    C4TaskSpec,
    FOREGROUND,
    IGNORE_TARGET,
    TRAJ_KIND_INFO,
    WAVE_KINDS,
    WAVE_HORIZON_M,
    TOK_SEARCH,
    TOK_BACK,
    TOK_DRAW,
    TOK_DONE,
    PATH_WIDTH_CELLS,
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


def store_crop_px(routes_dir: Path) -> int:
    """Window size of a route store in px (1 px = 1 m): store.json, written by c4_routes since 29 Sep 2026;
    older stores have none and are 256."""
    import json
    p = Path(routes_dir) / "store.json"
    return int(json.loads(p.read_text())["crop_px"]) if p.is_file() else CROP_PX


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
        poly=torch.from_numpy(z["poly"][:n].astype(np.float32)).to(device),
        n_verts=torch.from_numpy(z["n_verts"][:n].astype(np.int64)).to(device),
        **_load_fans(Path(routes_dir), split, n, device),
        **_load_loose(Path(routes_dir), split, n, device),
        **_load_traj(Path(routes_dir), split, n, device),
        **_load_wave(Path(routes_dir), split, n, device),
        detour_ratio=detour_ratio, straight_px=straight_px, path_px=path_px, stratum=stratum,
        crop_px=store_crop_px(Path(routes_dir)),
    )


def _load_fans(routes_dir: Path, split: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """C5 rung 1: visible-corner fans per marker, if `c5_fans_{split}.npz` exists next to the store."""
    p = routes_dir / f"c5_fans_{split}.npz"
    if not p.is_file():
        return {}
    z = np.load(p)
    return dict(fan_xy=torch.from_numpy(z["fan_xy"][:n]).to(device), fan_n=torch.from_numpy(z["fan_n"][:n].astype(np.int64)).to(device))


def _load_loose(routes_dir: Path, split: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """C5 rung 4: loose paths next to the store (`spatial_data/c5_loose.py`), if built."""
    p = routes_dir / f"c5_loose_{split}.npz"
    if not p.is_file():
        return {}
    z = np.load(p)
    return dict(loose_poly=torch.from_numpy(z["loose_poly"][:n]).to(device), loose_n=torch.from_numpy(z["loose_n"][:n].astype(np.int64)).to(device))


def _load_traj(routes_dir: Path, split: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """C5 key trajectories (`spatial_data/c5_trajectories.py`), if built: ragged pixel lists per sample."""
    out = {}
    for sfx in ("", "_r40"):
        p = routes_dir / f"c5_traj{sfx}_{split}.npz"
        if not p.is_file():
            continue
        z = np.load(p)
        if len(z["offset"]) - 1 < n:      # built for the first routes only: usable by runs that load at most that many
            print(f"  note: {p.name} covers the first {len(z['offset']) - 1} routes, the run loads {n}: trajectory tasks{sfx} unavailable")
            continue
        off = z["offset"][: n + 1].astype(np.int64)
        out[f"traj_pix{sfx}"] = torch.from_numpy(z["pix"][: off[-1]].astype(np.int64)).to(device)
        out[f"traj_tag{sfx}"] = torch.from_numpy(z["pix_tag"][: off[-1]].astype(np.int64)).to(device)
        out[f"traj_off{sfx}"] = torch.from_numpy(off).to(device)
        for r in (2, 3, 4):                   # later rounds (marker 0 only), stored apart by the builder
            if f"pix{r}" in z.files:
                offr = z[f"offset{r}"][: n + 1].astype(np.int64)
                out[f"traj_pix{sfx}_r{r}"] = torch.from_numpy(z[f"pix{r}"][: offr[-1]].astype(np.int64)).to(device)
                out[f"traj_off{sfx}_r{r}"] = torch.from_numpy(offr).to(device)
    return out


def _traj_gather(route: dict, ix: torch.Tensor, sfx: str, rnd: int, marker: int | None):
    """Pixels (P,2) and their sample index (P,) of round `rnd` for samples ix (marker filter for round 1)."""
    key = f"traj_pix{sfx}" if rnd == 1 else f"traj_pix{sfx}_r{rnd}"
    if key not in route:
        raise ValueError(f"trajectory targets need c5_traj{sfx}_{{split}}.npz (round {rnd}) next to the route store "
                         "(spatial_data/c5_trajectories.py)")
    b = ix.shape[0]; dev = ix.device
    off = route[f"traj_off{sfx}" if rnd == 1 else f"traj_off{sfx}_r{rnd}"]
    lo = off[ix]; cnt = off[ix + 1] - lo
    total = int(cnt.sum().item())
    if total == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    sid = torch.repeat_interleave(torch.arange(b, device=dev), cnt)
    first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    pos = torch.arange(total, device=dev) - first + torch.repeat_interleave(lo, cnt)
    px = route[key][pos]
    if rnd == 1 and marker is not None:      # one marker only: its pixels carry tag marker * 32 + trajectory
        keep = (route[f"traj_tag{sfx}"][pos] // 32) == marker
        px, sid = px[keep], sid[keep]
    return px, sid


def _traj_scratch(route: dict, ix: torch.Tensor, crop: int, sfx: str, upto: int, marker: int | None) -> torch.Tensor:
    """(B,crop,crop) float: the pixels of rounds 1..upto drawn 1:1 (the scratchpad input of step upto+1)."""
    b = ix.shape[0]; out = torch.zeros(b, crop, crop, device=ix.device)
    for r in range(1, upto + 1):
        px, sid = _traj_gather(route, ix, sfx, r, marker)
        if px.shape[0]:
            out.view(b, -1)[sid, px[:, 1].clamp(0, crop - 1) * crop + px[:, 0].clamp(0, crop - 1)] = 1.0
    return out


def _traj_targets(route: dict, ix: torch.Tensor, grid_size: int, crop: int = CROP_PX, marker: int | None = None,
                  sfx: str = "", rnd: int = 1) -> torch.Tensor:
    """(B,g,g): cells any key-trajectory pixel of round `rnd` of sample ix[b] falls in. One sync per round."""
    b = ix.shape[0]
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=ix.device)
    px, sid = _traj_gather(route, ix, sfx, rnd, marker)
    if px.shape[0] == 0:
        return tgt
    s = crop // grid_size
    gc = (px[:, 0] // s).clamp(0, grid_size - 1); gr = (px[:, 1] // s).clamp(0, grid_size - 1)
    tgt.view(b, -1)[sid, gr * grid_size + gc] = FOREGROUND
    return tgt


_ONE_MARKER_KINDS = tuple(k for k, (_, m, _r) in TRAJ_KIND_INFO.items() if m is not None)
_STEP_KINDS = tuple(k for k, (_, _m, r) in TRAJ_KIND_INFO.items() if r >= 2)


def _one_marker(img: torch.Tensor, layers: torch.Tensor, encoding, mc: torch.Tensor, mr: torch.Tensor,
                sel: torch.Tensor) -> torch.Tensor:
    """For samples sel (B,) bool: marker plane with marker 0 only (slot 1 = -1 = unused). Repainted after the image
    is built, so the loaders' random streams stay as they were."""
    if not bool(sel.any()):
        return img
    mc1, mr1 = mc.clone(), mr.clone(); mc1[:, 1] = -1; mr1[:, 1] = -1
    one = build_image_from_crops(layers, encoding, marker_col=mc1, marker_row=mr1)
    img = img.clone(); img[sel, 15] = one[sel, 15]
    return img


def _load_wave(routes_dir: Path, split: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """C5 R5 (`spatial_data/c5_wave.py`), if built: the 2 m walking-distance field from the start, the goal's
    distance and the way back (ragged pixels)."""
    p = routes_dir / f"c5_wave_{split}.npz"
    if not p.is_file():
        return {}
    z = np.load(p)
    if len(z["goal_m"]) < n:            # built for the first routes only: usable by runs that load at most that many
        print(f"  note: {p.name} covers the first {len(z['goal_m'])} routes, the run loads {n}: wave tasks unavailable")
        return {}
    off = z["offset"][: n + 1].astype(np.int64)
    dist = z["dist"][:n]
    dist_dev = device
    if dist.dtype == np.uint16:         # stores of windows above 256 px (c5_wave.dist_dtype): torch cannot index
        dist = dist.view(np.int16)      # uint16 on the GPU; viewed as int16 the 'not reached' 65535 reads -1
        # 128 KB per route (2.6 GB for 20k, 7.9 GB for 60k): too much next to a 1024-token model on 12 GB, so the
        # field stays in pinned CPU memory and a batch fetches its rows (_wave_dist_rows; 16 MB per 128 samples)
        dist_dev = torch.device("cpu")
    wd = torch.from_numpy(np.ascontiguousarray(dist))
    wd = wd.pin_memory() if (dist_dev.type == "cpu" and torch.device(device).type == "cuda") else wd.to(dist_dev)
    return dict(wave_dist=wd,
                wave_goal_m=torch.from_numpy(z["goal_m"][:n].astype(np.float32)).to(device),
                wave_path=torch.from_numpy(z["path"][: off[-1]].astype(np.int64)).to(device),
                wave_off=torch.from_numpy(off).to(device))


def _wave_dist_rows(route: dict, ix: torch.Tensor) -> torch.Tensor:
    """route["wave_dist"][ix] on ix's device, wherever the field lives (GPU for 256 m stores, pinned CPU for larger)."""
    d = route["wave_dist"]
    if d.device == ix.device:
        return d[ix]
    return d[ix.cpu()].to(ix.device, non_blocking=True)


def _wave_cell_minmax(dist2: torch.Tensor, grid: int):
    """(B,S,S) uint8 (2 m units, 255 unreached) or int16 (larger windows, -1 unreached) -> dmin, dmax (B,g,g) metres
    per output cell (inf unreached)."""
    d = dist2.float() * 2.0
    unreached = (dist2 == 255) if dist2.dtype == torch.uint8 else (dist2 < 0)
    d = torch.where(unreached, torch.full_like(d, float("inf")), d)
    b, S, _ = d.shape; s = S // grid
    blk = d.view(b, grid, s, grid, s)
    return blk.amin(dim=(2, 4)), blk.amax(dim=(2, 4)) + 2.9


def _wave_fronts(dmin: torch.Tensor, dmax: torch.Tensor, k: torch.Tensor, horizon: float) -> torch.Tensor:
    """(B,g,g) bool: rings j*horizon for j = 1..k[b] (crossing rule)."""
    out = torch.zeros_like(dmin, dtype=torch.bool)
    kmax = int(k.max().item()) if k.numel() else 0
    for j in range(1, kmax + 1):
        out |= (dmin < j * horizon) & (dmax >= j * horizon) & (k >= j).view(-1, 1, 1)
    return out


def _wave_start_disc(markers: torch.Tensor, grid: int, crop: int) -> torch.Tensor:
    """(B,g,g) bool: the 3x3 cells around the start's cell."""
    b = markers.shape[0]; dev = markers.device; s = crop // grid
    cx = (markers[:, 0, 0] // s).long().clamp(0, grid - 1); cy = (markers[:, 0, 1] // s).long().clamp(0, grid - 1)
    gx = torch.arange(grid, device=dev).view(1, 1, -1); gy = torch.arange(grid, device=dev).view(1, -1, 1)
    return ((gx - cx.view(-1, 1, 1)).abs() <= 1) & ((gy - cy.view(-1, 1, 1)).abs() <= 1)


def _wave_path_cells(route: dict, ix: torch.Tensor, grid: int, crop: int, lo_m: torch.Tensor | None = None,
                     hi_m: torch.Tensor | None = None, thick: bool = True) -> torch.Tensor:
    """(B,g,g) bool: cells of the way back goal -> start; only pixels whose walking distance from the start lies in
    [lo_m[b], hi_m[b]] when given; widened to PATH_WIDTH_CELLS when thick."""
    b = ix.shape[0]; dev = ix.device
    off = route["wave_off"]; lo = off[ix]; cnt = off[ix + 1] - lo
    out = torch.zeros(b, grid, grid, dtype=torch.bool, device=dev)
    total = int(cnt.sum().item())
    if total == 0:
        return out
    sid = torch.repeat_interleave(torch.arange(b, device=dev), cnt)
    first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    pos = torch.arange(total, device=dev) - first + torch.repeat_interleave(lo, cnt)
    px = route["wave_path"][pos]
    if lo_m is not None:
        d2 = _wave_dist_rows(route, ix)                                    # (B,S,S) uint8 / int16, 2 m units
        S = d2.shape[1]; sub_ = crop // S
        dpx = d2[sid, (px[:, 1] // sub_).clamp(0, S - 1), (px[:, 0] // sub_).clamp(0, S - 1)].float() * 2.0
        keep = (dpx >= lo_m[sid]) & (dpx <= hi_m[sid])
        px, sid = px[keep], sid[keep]
        if px.shape[0] == 0:
            return out
    s = crop // grid
    out.view(b, -1)[sid, (px[:, 1] // s).clamp(0, grid - 1) * grid + (px[:, 0] // s).clamp(0, grid - 1)] = True
    if thick and PATH_WIDTH_CELLS > 1:
        k = PATH_WIDTH_CELLS
        out = F_max_pool(out.float().unsqueeze(1), k, 1, k // 2).squeeze(1) > 0
    return out


def F_max_pool(x, k, stride, pad):
    import torch.nn.functional as F
    return F.max_pool2d(x, k, stride=stride, padding=pad)


def _wave_post(img, tgt, kinds, route, route_ix, grid_size, crop, gen, *, force_k_in: int | None = None, force_j: int | None = None,
               pad_noise: dict | None = None, own_pads: dict | None = None, cleanup: dict | None = None):
    """The wave step tasks. Forward (first/step/reach): pad in = start disc + fronts after k passes, target = pad after
    k+1 passes, token SEARCH (BACK when the new front reaches the goal). Backward (back/origin): pad in = disc +
    fronts 1..j + thick path from the goal down to front j, target = disc + fronts 1..j-1 + path down to front j-1,
    token BACK (DRAW when j = 1, the start is reached). Final (draw): pad in = disc + whole thick path, target = the
    thick path, token DONE. Returns img, tgt, tok, pad_in_cells (the teacher's clean pad in).
    Training only (gen given), in this order: (1) own_pads (spatial_data/c5_own_pads.py): with probability p the
    clean pad in is replaced by the network's own output of the pass that produces this input; (2) pad_noise on
    whatever pad is now in (teacher's or own). The target is always the teacher's clean next pad.
    Cleanup pass (spatial_data/c5_cleanup.py, config c5_wave_cleanup), training only: when given, the draw samples
    are governed by it instead of own_pads: with probability cleanup['p'] the pad in is the network's own drawing
    entering the draw pass and the target is the shortest chain through it (the teacher's path where that drawing
    fails the success rule). The other passes keep own_pads. The pad noise then hits the cleanup input like any
    other (the rollout's draw input is as noisy as any other pad in)."""
    if "wave_dist" not in route:
        raise ValueError("wave tasks need c5_wave_{split}.npz next to the route store (spatial_data/c5_wave.py)")
    dev = img.device; b = img.shape[0]
    kind_t = torch.tensor([WAVE_KINDS.index(k) if k in WAVE_KINDS else -1 for k in kinds], device=dev)   # 0 first 1 step 2 reach 3 back 4 origin 5 draw 6 fwd 7 bwd
    sel = kind_t >= 0
    dmin, dmax = _wave_cell_minmax(_wave_dist_rows(route, route_ix), grid_size)
    gm = route["wave_goal_m"][route_ix]
    n_goal = torch.where(torch.isfinite(gm), (gm / WAVE_HORIZON_M).floor().long() + 1, torch.ones_like(gm, dtype=torch.long))
    u = _uniform(route_ix, 6, gen)
    span = (n_goal - 2).clamp(min=0)                                                # wave_step: k in 1..n_goal-2
    k_step = torch.where(span > 0, 1 + (u * span.float()).floor().long().clamp(max=(span - 1).clamp(min=0)), torch.zeros_like(span))
    fwd = (kind_t <= 2) | (kind_t == 6)
    k_any = (u * n_goal.float()).floor().long().clamp(max=(n_goal - 1).clamp(min=0))   # wave_fwd: k in 0..n_goal-1
    k_in = torch.where(kind_t == 0, torch.zeros_like(n_goal), torch.where(kind_t == 1, k_step,
           torch.where(kind_t == 6, k_any, n_goal - 1)))
    if force_k_in is not None:
        k_in = torch.where(kind_t == 1, torch.full_like(n_goal, int(force_k_in)), k_in)
    markers = route["markers"][route_ix]
    disc = _wave_start_disc(markers, grid_size, crop)
    # forward passes
    pad_in = disc | _wave_fronts(dmin, dmax, k_in, WAVE_HORIZON_M)
    pad_out = disc | _wave_fronts(dmin, dmax, k_in + 1, WAVE_HORIZON_M)
    # backward passes: j = the outermost front still on the pad (n_goal .. 1); wave_back draws j at random in 2..n_goal
    u2 = _uniform(route_ix, 7, gen)
    j_back = 2 + (u2 * (n_goal - 1).clamp(min=1).float()).floor().long().clamp(max=(n_goal - 2).clamp(min=0))
    j_back = torch.minimum(j_back, n_goal)                                             # n_goal == 1 -> j = 1
    j_any = 1 + (u2 * n_goal.float()).floor().long().clamp(max=(n_goal - 1).clamp(min=0))   # wave_bwd: j in 1..n_goal
    j = torch.where(kind_t == 3, j_back, torch.where(kind_t == 7, j_any, torch.ones_like(n_goal)))
    if force_j is not None:
        j = torch.where(kind_t == 3, torch.full_like(n_goal, int(force_j)), j)
    bwd = (kind_t == 3) | (kind_t == 4) | (kind_t == 7)
    if bool(bwd.any()):
        R = WAVE_HORIZON_M; inf = torch.full_like(gm, float("inf"))
        path_in = _wave_path_cells(route, route_ix, grid_size, crop, lo_m=j.float() * R, hi_m=inf)
        path_out = _wave_path_cells(route, route_ix, grid_size, crop, lo_m=(j - 1).float() * R, hi_m=inf)
        b_in = disc | _wave_fronts(dmin, dmax, j, R) | path_in
        b_out = disc | _wave_fronts(dmin, dmax, j - 1, R) | path_out
        pad_in = torch.where(bwd.view(-1, 1, 1), b_in, pad_in); pad_out = torch.where(bwd.view(-1, 1, 1), b_out, pad_out)
    is_draw = kind_t == 5
    if bool(is_draw.any()):
        whole = _wave_path_cells(route, route_ix, grid_size, crop)
        pad_in = torch.where(is_draw.view(-1, 1, 1), disc | whole, pad_in); pad_out = torch.where(is_draw.view(-1, 1, 1), whole, pad_out)
    pad_clean = pad_in
    if gen is not None and own_pads is not None:                                     # training only, before the noise
        from .c5_own_pads import gather_own_pads, own_pad_source_slot
        slot = own_pad_source_slot(kind_t, k_in, j, n_goal)
        if cleanup is not None:                                                      # draw samples belong to the cleanup option
            slot = torch.where(is_draw, torch.full_like(slot, -1), slot)
        use, own = gather_own_pads(own_pads, route_ix, slot, gen)
        pad_in = torch.where(use.view(-1, 1, 1), own, pad_in)
    if gen is not None and cleanup is not None:                                      # training only, before the noise
        from .c5_cleanup import gather_cleanup
        use_c, c_in, c_out = gather_cleanup(cleanup, route_ix, is_draw, gen)
        pad_in = torch.where(use_c.view(-1, 1, 1), c_in, pad_in)
        pad_out = torch.where(use_c.view(-1, 1, 1), c_out, pad_out)
    if gen is not None and pad_noise is not None:                                    # training only, see _wave_pad_noise
        pad_in = torch.where(sel.view(-1, 1, 1), _wave_pad_noise(pad_in, gen, **pad_noise), pad_in)
    s = crop // grid_size
    scratch = pad_in.float().repeat_interleave(s, 1).repeat_interleave(s, 2)
    img = img.clone(); img[sel, 16] = scratch[sel]
    tgt = torch.where(sel.view(-1, 1, 1), pad_out.long() * FOREGROUND, tgt)
    tok = torch.full((b,), -1, dtype=torch.long, device=dev)
    tok = torch.where(fwd & (kind_t >= 0), torch.full_like(tok, TOK_SEARCH), tok)
    tok = torch.where(kind_t == 2, torch.full_like(tok, TOK_BACK), tok)            # the new front reaches the goal
    tok = torch.where((kind_t == 6) & (k_in + 1 >= n_goal), torch.full_like(tok, TOK_BACK), tok)
    tok = torch.where(kind_t == 7, torch.where(j > 1, torch.full_like(tok, TOK_BACK), torch.full_like(tok, TOK_DRAW)), tok)
    tok = torch.where(kind_t == 3, torch.where(j > 1, torch.full_like(tok, TOK_BACK), torch.full_like(tok, TOK_DRAW)), tok)
    tok = torch.where(kind_t == 4, torch.full_like(tok, TOK_DRAW), tok)
    tok = torch.where(kind_t == 5, torch.full_like(tok, TOK_DONE), tok)
    return img, tgt, tok, torch.where(sel.view(-1, 1, 1), pad_clean, torch.zeros_like(pad_clean))


_WAVE_PAD_SHIFT_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift_wave_pad_cells(pad: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """One-cell translation with zero fill (no wrap). pad: (g,g) bool."""
    g = pad.shape[-1]
    out = torch.zeros_like(pad)
    y0, y1 = max(0, dy), g + min(0, dy)
    sy0, sy1 = max(0, -dy), g - max(0, dy)
    x0, x1 = max(0, dx), g + min(0, dx)
    sx0, sx1 = max(0, -dx), g - max(0, dx)
    out[y0:y1, x0:x1] = pad[sy0:sy1, sx0:sx1]
    return out


def _wave_pad_noise(pad: torch.Tensor, gen: torch.Generator, p: float = 0.5, thicken: float = 0.5, drop: float = 0.05,
                    add: float = 0.002, shift: float = 0.0) -> torch.Tensor:
    """Perturb a share p of the pads (B,g,g) bool the way the network's own output tends to differ from the teacher's:
    with probability `thicken` the drawing is widened by one cell (the halo); with `shift` a random cardinal 1-cell
    translation (misplaced front); a share `drop` of lit cells is removed; a share `add` of all cells is speckle-lit."""
    b = pad.shape[0]; dev = pad.device
    pick = torch.rand(b, device=dev, generator=gen) < p
    if not bool(pick.any()):
        return pad
    out = pad.clone()
    thick = F_max_pool(pad.float().unsqueeze(1), 3, 1, 1).squeeze(1) > 0
    do_thick = (torch.rand(b, device=dev, generator=gen) < thicken) & pick
    out = torch.where(do_thick.view(-1, 1, 1), thick, out)
    if shift > 0:
        do_shift = (torch.rand(b, device=dev, generator=gen) < shift) & pick
        for i in torch.nonzero(do_shift, as_tuple=False).view(-1).tolist():
            k = int(torch.randint(0, len(_WAVE_PAD_SHIFT_DIRS), (1,), generator=gen, device=dev).item())
            dy, dx = _WAVE_PAD_SHIFT_DIRS[k]
            out[i] = _shift_wave_pad_cells(out[i], dy, dx)
    r = torch.rand(pad.shape, device=dev, generator=gen)
    out = out & ~((r < drop) & pick.view(-1, 1, 1))
    out = out | ((r > 1.0 - add) & pick.view(-1, 1, 1))
    return out


def _fans(route: dict, ix: torch.Tensor):
    return (route["fan_xy"][ix], route["fan_n"][ix]) if "fan_xy" in route else (None, None)


def _raster_segments_cells(a: torch.Tensor, q: torch.Tensor, valid: torch.Tensor, grid_size: int, crop: int = CROP_PX) -> torch.Tensor:
    """Segments a->q, (B,S,2) each with a (B,S) validity mask -> (B,g,g) cell mask, same sampling rule as
    rasterise_segment_numpy (L = ceil(2 |q-a|) steps, t = i / L), fully batched."""
    b, S, _ = a.shape; dev = a.device
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=dev)
    if S == 0 or not bool(valid.any()):
        return tgt
    L = (q - a).norm(dim=-1).mul(2).ceil().clamp(min=1)                      # (B,S)
    Lmax = int(L[valid].max().item())                                          # one sync per batch
    i = torch.arange(Lmax + 1, device=dev, dtype=torch.float32).view(1, 1, -1)
    t = (i / L.unsqueeze(-1)).clamp(max=1.0)
    ok = (i <= L.unsqueeze(-1)) & valid.unsqueeze(-1)                          # (B,S,Lmax+1)
    pts = a.unsqueeze(2) + t.unsqueeze(-1) * (q - a).unsqueeze(2)              # (B,S,Lmax+1,2)
    xs = pts[..., 0].floor().long().clamp(0, crop - 1).view(b, -1)
    ys = pts[..., 1].floor().long().clamp(0, crop - 1).view(b, -1)
    _scatter_cells(tgt, xs, ys, ok.view(b, -1), crop)
    return tgt


def _rays_targets(markers: torch.Tensor, fan_xy: torch.Tensor, fan_n: torch.Tensor, grid_size: int, crop: int = CROP_PX) -> torch.Tensor:
    """markers (B,2,2), fan_xy (B,2,K,2), fan_n (B,2) -> (B,g,g): rays from each marker to its visible corners."""
    b, _, K, _ = fan_xy.shape
    a = markers.unsqueeze(2).expand(b, 2, K, 2).reshape(b, 2 * K, 2)
    q = fan_xy.reshape(b, 2 * K, 2)
    valid = (torch.arange(K, device=fan_n.device).view(1, 1, K) < fan_n.unsqueeze(-1)).reshape(b, 2 * K)
    return _raster_segments_cells(a, q, valid, grid_size, crop)


def polyline_scratch(poly: torch.Tensor, n_verts: torch.Tensor, *, size: int = CROP_PX) -> torch.Tensor:
    """(B,V,2) polylines (x, y in window px, -1 padded) with n_verts (B,) -> (B,size,size) float scratchpad,
    1 px lines, 0.5 px sampling. C5 rungs 2-4 render their given paths with this."""
    b, V, _ = poly.shape; dev = poly.device
    out = torch.zeros(b, size, size, device=dev)
    if V < 2:
        return out
    a, q = poly[:, :-1], poly[:, 1:]
    valid = torch.arange(V - 1, device=dev).view(1, -1) < (n_verts.view(-1, 1) - 1)
    if not bool(valid.any()):
        return out
    L = (q - a).norm(dim=-1).mul(2).ceil().clamp(min=1)
    Lmax = int(L[valid].max().item())
    i = torch.arange(Lmax + 1, device=dev, dtype=torch.float32).view(1, 1, -1)
    t = (i / L.unsqueeze(-1)).clamp(max=1.0); ok = (i <= L.unsqueeze(-1)) & valid.unsqueeze(-1)
    pts = a.unsqueeze(2) + t.unsqueeze(-1) * (q - a).unsqueeze(2)
    xs = pts[..., 0].floor().long().clamp(0, size - 1).view(b, -1); ys = pts[..., 1].floor().long().clamp(0, size - 1).view(b, -1)
    bi = torch.arange(b, device=dev).view(b, 1).expand_as(xs); okf = ok.view(b, -1)
    out.view(b, -1)[bi[okf], (ys * size + xs)[okf]] = 1.0
    return out


_LENGTH_KINDS = (C4TargetKind.LENGTH_LINE, C4TargetKind.LENGTH_PATH)
_SCRATCH_KINDS = _LENGTH_KINDS + (C4TargetKind.TIGHTEN,)


def _uniform(route_ix: torch.Tensor, salt: int, gen: torch.Generator | None) -> torch.Tensor:
    """Per-sample U(0,1): from the generator in training, a fixed hash of the route index in validation."""
    if gen is not None:
        return torch.rand(route_ix.shape[0], device=route_ix.device, generator=gen)
    # 32-bit integer mix (the "lowbias32" hash), so neighbouring route indices get unrelated values
    M = 0xFFFFFFFF
    x = (route_ix.long() * 1000003 + salt * 7919 + 12345) & M
    x = x ^ (x >> 16); x = (x * 0x7FEB352D) & M
    x = x ^ (x >> 15); x = (x * 0x846CA68B) & M
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF).double().div(float(1 << 24)).float()


def _c5_post(img: torch.Tensor, tgt: torch.Tensor, kinds: list, route: dict, route_ix: torch.Tensor,
             grid_size: int, gen: torch.Generator | None, *, wave_force_k_in: int | None = None,
             wave_force_j: int | None = None, pad_noise: dict | None = None, own_pads: dict | None = None,
             cleanup: dict | None = None):
    crop = int(route.get("crop_px", CROP_PX))
    """C5 rungs 2-4. Tighten: the loose path on the scratchpad (channel 16), markers kept.
    Length: the line / path on the scratchpad, markers hidden, no mask target (every cell IGNORE_TARGET,
    so only the token is trained) and the length-bin token target; -1 elsewhere.
    kinds: the C4TargetKind of every sample (list of length B)."""
    from .c4_tasks import length_bin
    b = img.shape[0]; dev = img.device
    tok = torch.full((b,), -1, dtype=torch.long, device=dev)
    pad_cells = None
    if any(k in WAVE_KINDS for k in kinds):
        if img.shape[1] < 17:
            raise ValueError("wave tasks need the 17-channel encoding onehot_v4_sp (scratchpad at channel 16)")
        img, tgt, tok, pad_cells = _wave_post(
            img, tgt, kinds, route, route_ix, grid_size, crop, gen,
            force_k_in=wave_force_k_in, force_j=wave_force_j, pad_noise=pad_noise, own_pads=own_pads, cleanup=cleanup,
        )
        if all(k in WAVE_KINDS for k in kinds):
            return img, tgt, tok, pad_cells
    is_step = torch.tensor([k in _STEP_KINDS for k in kinds], device=dev)
    if bool(is_step.any()):
        if img.shape[1] < 17:
            raise ValueError("trajectory step tasks need the 17-channel encoding onehot_v4_sp (scratchpad at channel 16)")
        img = img.clone()
        for kind in set(k for k in kinds if k in _STEP_KINDS):
            sfx, marker, rnd = TRAJ_KIND_INFO[kind]
            sel = torch.tensor([k == kind for k in kinds], device=dev)
            sc = _traj_scratch(route, route_ix, crop, sfx, rnd - 1, marker)
            img[sel, 16] = sc[sel]
    is_tight = torch.tensor([k == C4TargetKind.TIGHTEN for k in kinds], device=dev)
    if bool(is_tight.any()):
        if img.shape[1] < 17:
            raise ValueError("tighten needs the 17-channel encoding onehot_v4_sp (scratchpad at channel 16)")
        if "loose_poly" not in route:
            raise ValueError("tighten needs c5_loose_{split}.npz next to the route store (spatial_data/c5_loose.py)")
        sc = polyline_scratch(route["loose_poly"][route_ix], route["loose_n"][route_ix], size=crop)
        img = img.clone(); img[is_tight, 16] = sc[is_tight]     # markers stay: they are the fixed endpoints
    is_line = torch.tensor([k == C4TargetKind.LENGTH_LINE for k in kinds], device=dev)
    is_path = torch.tensor([k == C4TargetKind.LENGTH_PATH for k in kinds], device=dev)
    sel = is_line | is_path
    if not bool(sel.any()):
        return img, tgt, tok, pad_cells
    if img.shape[1] < 17:
        raise ValueError("length tasks need the 17-channel encoding onehot_v4_sp (scratchpad at channel 16)")
    V = route["poly"].shape[1]
    poly = torch.full((b, V, 2), -1.0, device=dev); nv = torch.zeros(b, dtype=torch.long, device=dev)
    length = torch.zeros(b, device=dev)
    markers = route["markers"][route_ix]
    if bool(is_line.any()):
        # a random sub-segment of the marker-to-marker line: lengths spread from ~4 m to the full distance
        u0, u1 = _uniform(route_ix, 1, gen), _uniform(route_ix, 2, gen)
        # the full marker-to-marker line in half the cases, a random sub-segment otherwise
        full = _uniform(route_ix, 5, gen) < 0.5
        t0 = torch.where(full, torch.zeros_like(u0), 0.7 * u0)
        t1 = torch.where(full, torch.ones_like(u1), t0 + 0.1 + (1.0 - t0 - 0.1) * u1)
        a, q = markers[:, 0], markers[:, 1]
        p0 = a + t0.unsqueeze(1) * (q - a); p1 = a + t1.unsqueeze(1) * (q - a)
        li = is_line
        poly[li, 0] = p0[li]; poly[li, 1] = p1[li]; nv[li] = 2
        length[li] = (p1 - p0).norm(dim=1)[li]
    if bool(is_path.any()):
        # the stored shortest path, in half the cases cut to a random prefix of its vertices
        pp = route["poly"][route_ix]; nn = route["n_verts"][route_ix]
        u, w = _uniform(route_ix, 3, gen), _uniform(route_ix, 4, gen)
        # full path in half the cases, otherwise a prefix that keeps at least one corner when the route has one
        kmin = torch.minimum(torch.full_like(nn, 3), nn)
        span = (nn - kmin + 1).clamp(min=1)
        k = torch.where(w < 0.5, nn, kmin + (u * span.float()).floor().long().clamp(max=(span - 1)))
        k = torch.minimum(k.clamp(min=2), nn)
        idx = torch.arange(V, device=dev).view(1, -1)
        keep = idx < k.view(-1, 1)
        pth = torch.where(keep.unsqueeze(-1), pp, torch.full_like(pp, -1.0))
        seg = (pth[:, 1:] - pth[:, :-1]).norm(dim=-1) * (keep[:, 1:]).float()
        pi = is_path
        poly[pi] = pth[pi]; nv[pi] = k[pi]; length[pi] = seg.sum(dim=1)[pi]
    scratch = polyline_scratch(poly, nv, size=crop)
    img = img.clone()
    img[sel, 16] = scratch[sel]
    img[sel, 15] = 0.0                                          # markers hidden: the length must be read off the scratchpad
    tgt = torch.where(sel.view(-1, 1, 1), torch.full_like(tgt, IGNORE_TARGET), tgt)
    tok = torch.where(sel, length_bin(length), tok)
    return img, tgt, tok, pad_cells


def _cut_windows(district: torch.Tensor, origin: torch.Tensor, crop: int = CROP_PX) -> torch.Tensor:
    """district (3,H,W), origin (B,2) dx,dy -> (B,3,crop,crop). One gather, no per-sample sync."""
    ar = torch.arange(crop, device=district.device)
    ys = origin[:, 1].view(-1, 1) + ar.view(1, -1)          # (B, CROP)
    xs = origin[:, 0].view(-1, 1) + ar.view(1, -1)
    out = district[:, ys[:, :, None], xs[:, None, :]]        # (3, B, CROP, CROP)
    return out.permute(1, 0, 2, 3).contiguous()


def _marker_cols_rows(markers: torch.Tensor, crop: int = CROP_PX) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,2,2) float xy -> (B,2) long col/row."""
    col = markers[..., 0].floor().clamp(0, crop - 1).long()
    row = markers[..., 1].floor().clamp(0, crop - 1).long()
    return col, row


def _scatter_cells(tgt: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, valid: torch.Tensor, crop: int = CROP_PX) -> None:
    """Mark the cell of every valid (x, y) pixel in tgt (B, g, g) in place (cell = crop / g px)."""
    b, g, _ = tgt.shape
    s = crop // g
    gc = (xs // s).clamp(0, g - 1); gr = (ys // s).clamp(0, g - 1)
    bi = torch.arange(b, device=tgt.device).view(b, 1).expand_as(gc)
    flat = tgt.view(b, g * g)
    flat[bi[valid], (gr * g + gc)[valid]] = FOREGROUND


def _segment_targets(markers: torch.Tensor, grid_size: int, crop: int = CROP_PX) -> torch.Tensor:
    """Same sampling as rasterise_segment_numpy (L = ceil(2 |b-a|) steps, t = i / L), vectorised."""
    b = markers.shape[0]; dev = markers.device
    a = markers[:, 0]; q = markers[:, 1]
    L = (q - a).norm(dim=1).mul(2).ceil().clamp(min=1)                       # (B,)
    Lmax = int(L.max().item()) if b else 1                                     # one sync per batch
    i = torch.arange(Lmax + 1, device=dev, dtype=torch.float32).view(1, -1)   # (1, Lmax+1)
    t = (i / L.view(-1, 1)).clamp(max=1.0)                                     # (B, Lmax+1)
    valid = i <= L.view(-1, 1)
    pts = a[:, None, :] + t[..., None] * (q - a)[:, None, :]
    xs = pts[..., 0].floor().long().clamp(0, crop - 1)
    ys = pts[..., 1].floor().long().clamp(0, crop - 1)
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=dev)
    _scatter_cells(tgt, xs, ys, valid, crop)
    return tgt


def _path_targets(path_pix: torch.Tensor, n_pix: torch.Tensor, grid_size: int, crop: int = CROP_PX) -> torch.Tensor:
    """(B, P, 2) padded int pixels (-1 = unused) -> (B, g, g), any pixel per cell."""
    b, P, _ = path_pix.shape; dev = path_pix.device
    idx = torch.arange(P, device=dev).view(1, -1)
    valid = (idx < n_pix.view(-1, 1)) & (path_pix[..., 0] >= 0) & (path_pix[..., 1] >= 0)
    tgt = torch.zeros(b, grid_size, grid_size, dtype=torch.long, device=dev)
    _scatter_cells(tgt, path_pix[..., 0].clamp(min=0), path_pix[..., 1].clamp(min=0), valid, crop)
    return tgt


def _building_targets(layers: torch.Tensor, grid_size: int) -> torch.Tensor:
    """(B,3,CROP,CROP) uint8 planes -> (B,g,g): cell is building if >= half its pixels are class 3."""
    b = layers.shape[0]; g = grid_size; s = layers.shape[-1] // g
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
    fan_xy: torch.Tensor | None = None,
    fan_n: torch.Tensor | None = None,
    crop: int = CROP_PX,
    traj: tuple | None = None,
) -> torch.Tensor:
    if spec.target_kind in TRAJ_KIND_INFO:
        sfx, marker, rnd = TRAJ_KIND_INFO[spec.target_kind]
        return _traj_targets(*traj, grid_size, crop, marker=marker, sfx=sfx, rnd=rnd)
    if spec.target_kind in WAVE_KINDS:
        return torch.zeros(markers.shape[0], grid_size, grid_size, dtype=torch.long, device=markers.device)
    if spec.target_kind == C4TargetKind.TIGHTEN:
        return _path_targets(path_pix, n_pix, grid_size, crop)
    if spec.target_kind in (C4TargetKind.LENGTH_LINE, C4TargetKind.LENGTH_PATH):
        return torch.full((markers.shape[0], grid_size, grid_size), IGNORE_TARGET, dtype=torch.long, device=markers.device)
    if spec.target_kind == C4TargetKind.RAYS:
        if fan_xy is None:
            raise ValueError("rays targets need c5_fans_{split}.npz next to the route store (spatial_data/c5_fans.py)")
        return _rays_targets(markers, fan_xy, fan_n, grid_size, crop)
    if spec.target_kind == C4TargetKind.SEGMENT:
        return _segment_targets(markers, grid_size, crop)
    if spec.target_kind == C4TargetKind.BUILDING:
        if layers is None:
            raise ValueError("building targets need the window layers")
        return _building_targets(layers, grid_size)
    return _path_targets(path_pix, n_pix, grid_size, crop)


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
    fan_xy: torch.Tensor | None = None,
    fan_n: torch.Tensor | None = None,
    crop: int = CROP_PX,
    traj: tuple | None = None,
) -> torch.Tensor:
    """Every target kind for the whole batch, then a per-sample select: no Python loop."""
    kinds = [sp.target_kind for sp in task_specs]
    kind_ix = torch.tensor([{C4TargetKind.SEGMENT: 0, C4TargetKind.DETOUR_PATH: 1, C4TargetKind.BUILDING: 2, C4TargetKind.RAYS: 3,
                             C4TargetKind.LENGTH_LINE: 4, C4TargetKind.LENGTH_PATH: 5, C4TargetKind.TIGHTEN: 1,
                             C4TargetKind.TRAJ: 6, C4TargetKind.TRAJ_ONE: 7, C4TargetKind.TRAJ_ONE_R40: 8,
                             C4TargetKind.TRAJ_STEP2_R40: 9, C4TargetKind.TRAJ_STEP3_R40: 10, C4TargetKind.TRAJ_STEP4_R40: 11,
                             C4TargetKind.WAVE_FIRST: 12, C4TargetKind.WAVE_STEP: 13, C4TargetKind.WAVE_REACH: 14,
                             C4TargetKind.WAVE_BACK: 15, C4TargetKind.WAVE_ORIGIN: 16, C4TargetKind.WAVE_DRAW: 17,
                             C4TargetKind.WAVE_FWD: 18, C4TargetKind.WAVE_BWD: 19}[k] for k in kinds],
                           dtype=torch.long, device=device)[task_index]
    out = _path_targets(path_pix, n_pix, grid_size, crop)
    if any(k in WAVE_KINDS for k in kinds):                       # filled by the post step
        out = torch.where((kind_ix >= 12).view(-1, 1, 1), torch.zeros_like(out), out)
    if C4TargetKind.SEGMENT in kinds:
        out = torch.where((kind_ix == 0).view(-1, 1, 1), _segment_targets(markers, grid_size, crop), out)
    if C4TargetKind.BUILDING in kinds:
        if layers is None:
            raise ValueError("building targets need the window layers")
        out = torch.where((kind_ix == 2).view(-1, 1, 1), _building_targets(layers, grid_size), out)
    if C4TargetKind.RAYS in kinds:
        if fan_xy is None:
            raise ValueError("rays targets need c5_fans_{split}.npz next to the route store")
        out = torch.where((kind_ix == 3).view(-1, 1, 1), _rays_targets(markers, fan_xy, fan_n, grid_size, crop), out)
    for kind, code in ((C4TargetKind.TRAJ, 6), (C4TargetKind.TRAJ_ONE, 7), (C4TargetKind.TRAJ_ONE_R40, 8),
                       (C4TargetKind.TRAJ_STEP2_R40, 9), (C4TargetKind.TRAJ_STEP3_R40, 10), (C4TargetKind.TRAJ_STEP4_R40, 11)):
        if kind in kinds:
            sfx, marker, rnd = TRAJ_KIND_INFO[kind]
            out = torch.where((kind_ix == code).view(-1, 1, 1),
                              _traj_targets(*traj, grid_size, crop, marker=marker, sfx=sfx, rnd=rnd), out)
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
        self.crop = int(route_index.get("crop_px", CROP_PX))
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
        fx, fn = _fans(self.route, ix)
        layers = _cut_windows(self.district, origin, self.crop)
        return layers, markers, path_pix, n_pix, origin

    def _build(self, ix: torch.Tensor):
        b = ix.shape[0]
        layers, markers, path_pix, n_pix, _ = self._gather(ix)
        fx, fn = _fans(self.route, ix)
        mc, mr = _marker_cols_rows(markers, self.crop)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        img = _one_marker(img, layers, self.encoding, mc, mr,
                          torch.full((b,), self.spec.target_kind in _ONE_MARKER_KINDS, device=self.device))
        tgt = _targets_for_batch(
            self.spec, markers=markers, path_pix=path_pix, n_pix=n_pix, fan_xy=fx, fan_n=fn, traj=(self.route, ix),
            grid_size=self.grid_size, device=self.device, layers=layers, crop=self.crop,
        )
        img, tgt, tok, pad_cells = _c5_post(img, tgt, [self.spec.target_kind] * b, self.route, ix, self.grid_size, self._gen, pad_noise=getattr(self, 'pad_noise', None),
                                             own_pads=getattr(self, 'own_pads', None), cleanup=getattr(self, 'cleanup', None))
        task_index = torch.zeros(b, dtype=torch.long, device=self.device)
        cond_ids = self._cond.expand(b, -1)
        if getattr(self, "return_tokens", False):
            return img, tgt, task_index, cond_ids, tok
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
        self.crop = int(route_index.get("crop_px", CROP_PX))
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
        fx, fn = _fans(self.route, ix)
        layers = _cut_windows(self.district, origin, self.crop)
        mc, mr = _marker_cols_rows(markers, self.crop)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        img = _one_marker(img, layers, self.encoding, mc, mr,
                          torch.full((b,), self.spec.target_kind in _ONE_MARKER_KINDS, device=self.device))
        tgt = _targets_for_batch(
            self.spec, markers=markers, path_pix=path_pix, n_pix=n_pix, fan_xy=fx, fan_n=fn, traj=(self.route, ix),
            grid_size=self.grid_size, device=self.device, layers=layers, crop=self.crop,
        )
        img, tgt, tok, pad_cells = _c5_post(img, tgt, [self.spec.target_kind] * b, self.route, ix, self.grid_size, None)
        self._last_tok = tok; self._last_pad = pad_cells
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
        return img, tgt, task_index, cond_ids, {**self._meta_for_routes(route_ix), "task_ix": task_ix, "tok_tgt": getattr(self, "_last_tok", None), "pad_cells": getattr(self, "_last_pad", None)}

    def fetch_at(self, flat: int):
        flat_t = torch.tensor([flat], device=self.device, dtype=torch.long)
        route_ix, task_ix = self._route_task_from_flat(flat_t)
        img, tgt, task_index, cond_ids = self._build(route_ix)
        return img[0], tgt[0], task_index[0], cond_ids[0], {**self._meta_for_routes(route_ix), "task_ix": task_ix[0],
                                                            "tok_tgt": self._last_tok}

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
        self.crop = int(route_index.get("crop_px", CROP_PX))
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._cond = c4_cond_id_table(self.task_specs, self.device)
        self._kinds = [sp.target_kind for sp in self.task_specs]
        self._any_length = any(k in _SCRATCH_KINDS or k in _STEP_KINDS or k in WAVE_KINDS for k in self._kinds)
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
        fx, fn = _fans(self.route, ix)
        layers = _cut_windows(self.district, origin, self.crop)
        mc, mr = _marker_cols_rows(markers, self.crop)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        task_index = self._sample_task_index(b)
        is_one = torch.tensor([k in _ONE_MARKER_KINDS for k in self._kinds], device=self.device)[task_index]
        img = _one_marker(img, layers, self.encoding, mc, mr, is_one)
        tgt = c4_targets_mixed_batch(
            task_index, self.task_specs, markers=markers, path_pix=path_pix, n_pix=n_pix, fan_xy=fx, fan_n=fn,
            grid_size=self.grid_size, device=self.device, layers=layers, crop=self.crop, traj=(self.route, ix),
        )
        kinds = [self._kinds[int(t)] for t in task_index.tolist()] if self._any_length else None
        tok = torch.full((b,), -1, dtype=torch.long, device=self.device)
        if kinds is not None:
            img, tgt, tok, pad_cells = _c5_post(img, tgt, kinds, self.route, ix, self.grid_size, self._gen, pad_noise=getattr(self, 'pad_noise', None),
                                             own_pads=getattr(self, 'own_pads', None), cleanup=getattr(self, 'cleanup', None))
        cond_ids = self._cond[task_index]
        if getattr(self, "return_tokens", False):
            return img, tgt, task_index, cond_ids, tok
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
        self.crop = int(route_index.get("crop_px", CROP_PX))
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
        fx, fn = _fans(self.route, route_ix)
        layers = _cut_windows(self.district, origin, self.crop)
        mc, mr = _marker_cols_rows(markers, self.crop)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        is_one = torch.tensor([sp.target_kind in _ONE_MARKER_KINDS for sp in self.task_specs], device=self.device)[task_index]
        img = _one_marker(img, layers, self.encoding, mc, mr, is_one)
        tgt = c4_targets_mixed_batch(
            task_index, self.task_specs, markers=markers, path_pix=path_pix, n_pix=n_pix, fan_xy=fx, fan_n=fn,
            grid_size=self.grid_size, device=self.device, layers=layers, crop=self.crop, traj=(self.route, route_ix),
        )
        kinds = [self.task_specs[int(t)].target_kind for t in task_index.tolist()]
        img, tgt, tok, pad_cells = _c5_post(img, tgt, kinds, self.route, route_ix, self.grid_size, None)
        self._last_tok = tok; self._last_pad = pad_cells
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
        return img, tgt, task_index, cond_ids, {**self._meta_for_routes(route_ix), "task_ix": task_ix, "tok_tgt": getattr(self, "_last_tok", None), "pad_cells": getattr(self, "_last_pad", None)}

    def fetch_at(self, flat: int):
        flat_t = torch.tensor([flat], device=self.device, dtype=torch.long)
        img, tgt, task_index, cond_ids, route_ix = self._build_from_flat(flat_t)
        _, task_ix = self._route_task_from_flat(flat_t)
        return img[0], tgt[0], task_index[0], cond_ids[0], {**self._meta_for_routes(route_ix), "task_ix": task_ix[0],
                                                            "tok_tgt": self._last_tok}

    def fetch_wave_pass(
        self,
        route_ix: int,
        task_key: str,
        *,
        force_k_in: int | None = None,
        force_j: int | None = None,
    ):
        """One route, one wave pass (deterministic k/j for rollout)."""
        ti = next(i for i, sp in enumerate(self.task_specs) if sp.key == task_key)
        flat_t = torch.tensor([route_ix * self.n_tasks + ti], device=self.device, dtype=torch.long)
        route_ix_t, task_index = self._route_task_from_flat(flat_t)
        origin = self.route["origin"][route_ix_t]
        markers = self.route["markers"][route_ix_t]
        path_pix = self.route["path_pix"][route_ix_t]
        n_pix = self.route["n_pix"][route_ix_t]
        fx, fn = _fans(self.route, route_ix_t)
        layers = _cut_windows(self.district, origin, self.crop)
        mc, mr = _marker_cols_rows(markers, self.crop)
        img = build_image_from_crops(layers, self.encoding, marker_col=mc, marker_row=mr)
        is_one = torch.tensor([self.task_specs[int(t)].target_kind in _ONE_MARKER_KINDS for t in task_index], device=self.device)
        img = _one_marker(img, layers, self.encoding, mc, mr, is_one)
        tgt = c4_targets_mixed_batch(
            task_index, self.task_specs, markers=markers, path_pix=path_pix, n_pix=n_pix, fan_xy=fx, fan_n=fn,
            grid_size=self.grid_size, device=self.device, layers=layers, crop=self.crop, traj=(self.route, route_ix_t),
        )
        kinds = [self.task_specs[int(t)].target_kind for t in task_index.tolist()]
        img, tgt, tok, pad_cells = _c5_post(
            img, tgt, kinds, self.route, route_ix_t, self.grid_size, None,
            wave_force_k_in=force_k_in, wave_force_j=force_j,
        )
        cond_ids = self._cond[task_index]
        return img[0], tgt[0], task_index[0], cond_ids[0], int(tok.view(-1)[0].item())
