"""Spatial grid eval: loss, IoU, exact-cell, and c0 decomposition metrics."""
from __future__ import annotations

import torch

import config
import data
from spatial_batch import forward as model_forward, next_batch as loader_next_batch


def _subcells() -> int:
    patch_grid = config.img_size // config.patch_size
    return config.grid_out_size // patch_grid


def _layer_plane_from_img(img: torch.Tensor, layer_index: int) -> torch.Tensor:
    """(B, C, H, W) → (B, H, W) int class indices."""
    if config.encoding_mode == "scalar":
        return (img[:, layer_index] * 3.0).round().long().clamp(0, 3)
    if config.encoding_mode == "onehot_v4":
        bases = (0, 7, 11)
        widths = (7, 4, 4)
        base, w = bases[layer_index], widths[layer_index]
        return img[:, base : base + w].argmax(dim=1)
    base = layer_index * 4
    return img[:, base : base + 4].argmax(dim=1)


def _pure_cell_mask_batched(layer_plane: torch.Tensor) -> torch.Tensor:
    """(B, H, W) → (B, G, G) bool: uniform class within each output cell."""
    g = config.grid_out_size
    h = config.img_size
    s = h // g
    b = layer_plane.shape[0]
    blk = layer_plane.reshape(b, g, s, g, s).permute(0, 1, 3, 2, 4).reshape(b, g, g, s * s)
    return blk.max(dim=-1).values == blk.min(dim=-1).values


def _c1_metrics_batch(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pure: torch.Tensor,
    *,
    fg_class: int,
) -> dict[str, torch.Tensor]:
    pred_fg = pred == fg_class
    tgt_fg = tgt == fg_class
    inter = (pred_fg & tgt_fg).sum(dim=(1, 2)).float()
    union = (pred_fg | tgt_fg).sum(dim=(1, 2)).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
    tgt_n = tgt_fg.sum(dim=(1, 2))
    pred_n = pred_fg.sum(dim=(1, 2))
    nonempty = tgt_n > 0
    empty = ~nonempty
    empty_fp = empty & (pred_n > 0)
    tp = inter
    fp = (pred_fg & ~tgt_fg).sum(dim=(1, 2)).float()
    fn = (~pred_fg & tgt_fg).sum(dim=(1, 2)).float()
    prec = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    rec = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
    pure_flat = pure.reshape(pure.shape[0], -1)
    tgt_flat = tgt_fg.reshape(tgt_fg.shape[0], -1)
    pred_flat = pred_fg.reshape(pred_fg.shape[0], -1)
    def _masked_iou(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = mask & (tgt_flat | pred_flat)
        has = m.any(dim=1)
        inter_m = (pred_flat & tgt_flat & m).sum(dim=1).float()
        union_m = ((pred_flat | tgt_flat) & m).sum(dim=1).float()
        iou_m = torch.where(union_m > 0, inter_m / union_m, torch.ones_like(inter_m))
        return iou_m, has
    pure_mask = pure_flat
    mixed_mask = ~pure_flat
    iou_pure, has_pure = _masked_iou(pure_mask)
    iou_mixed, has_mixed = _masked_iou(mixed_mask)
    return {
        "iou_sum": iou[nonempty].sum() if nonempty.any() else torch.zeros((), device=iou.device),
        "iou_n": nonempty.sum(),
        "empty_n": empty.sum(),
        "empty_fp": empty_fp.sum(),
        "empty_pred_cells_sum": pred_n[empty].float().sum() if empty.any() else torch.zeros((), device=pred.device),
        "prec_sum": prec[nonempty].sum() if nonempty.any() else torch.zeros((), device=prec.device),
        "rec_sum": rec[nonempty].sum() if nonempty.any() else torch.zeros((), device=rec.device),
        "iou_pure_sum": iou_pure[has_pure].sum() if has_pure.any() else torch.zeros((), device=iou.device),
        "iou_pure_n": has_pure.sum(),
        "iou_mixed_sum": iou_mixed[has_mixed].sum() if has_mixed.any() else torch.zeros((), device=iou.device),
        "iou_mixed_n": has_mixed.sum(),
        "n_grids": torch.tensor(pred.shape[0], device=pred.device),
    }


def _pure_for_c1_batch(img: torch.Tensor, task_ids: torch.Tensor | None, layer_index: int) -> torch.Tensor:
    if task_ids is None:
        return _pure_cell_mask_batched(_layer_plane_from_img(img, layer_index))
    keys = config.c1_active_task_keys or config.c1_task_keys_active()
    from spatial_data.c1_tasks import get_c1_task
    b = img.shape[0]
    pure = torch.zeros(b, config.grid_out_size, config.grid_out_size, dtype=torch.bool, device=img.device)
    for i in range(b):
        ti = int(task_ids[i].item())
        li = get_c1_task(keys[ti], config.cropset).layer_index
        pure[i] = _pure_cell_mask_batched(_layer_plane_from_img(img[i : i + 1], li))[0]
    return pure


@torch.no_grad()
def eval_loader_c1(model, loader, *, max_batches: int | None = None) -> dict:
    from spatial_data.c1_tasks import get_c1_task

    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    keys = config.c1_active_task_keys or config.c1_task_keys_active()
    single_task = keys[0] if len(keys) == 1 else None
    model.eval()
    fg_class = 1 if config.num_grid_classes > 1 else 0
    n_tasks = len(keys)
    acc_dev = device if str(device).startswith("cuda") else torch.device("cpu")
    acc = {
        k: torch.zeros((), device=acc_dev)
        for k in ("iou_sum", "prec_sum", "rec_sum", "iou_pure_sum", "iou_mixed_sum", "empty_pred_cells_sum")
    }
    acc.update({k: torch.zeros((), device=acc_dev, dtype=torch.long) for k in (
        "iou_n", "empty_n", "empty_fp", "iou_pure_n", "iou_mixed_n", "n_grids",
    )})
    per_task: dict[int, dict] = {
        ti: {"iou_sum": 0.0, "iou_n": 0, "n": 0} for ti in range(n_tasks)
    }
    held_out_ix: set[int] = set()
    ho_correct_sum = 0.0
    ho_correct_n = 0
    ho_wrong_sum = 0.0
    ho_wrong_n = 0
    if config.c1_task_mode == "factorised_twelve":
        from spatial_data.c1_tasks import compositional_wrong_task_index, held_out_task_indices
        held_out_ix = set(held_out_task_indices(keys, config.c1_held_out_task_keys))
    n_batches = 0
    loss_sum = 0.0
    n_iter = max_batches if max_batches is not None else 1
    for _ in range(n_iter):
        img, tgt, task_id, cond_ids = loader_next_batch(loader)
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, loss = model_forward(model, img, tgt, task_id, cond_ids)
        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        li = get_c1_task(keys[0], config.cropset).layer_index if single_task else 1
        pure = _pure_for_c1_batch(img, task_id, li)
        m = _c1_metrics_batch(pred, tgt, pure, fg_class=fg_class)
        for k in ("iou_sum", "prec_sum", "rec_sum", "iou_pure_sum", "iou_mixed_sum", "empty_pred_cells_sum"):
            acc[k] = acc[k] + m[k]
        for k in ("iou_n", "empty_n", "empty_fp", "iou_pure_n", "iou_mixed_n"):
            acc[k] = acc[k] + m[k]
        acc["n_grids"] = acc["n_grids"] + m["n_grids"]
        if task_id is not None:
            pred_fg = pred == fg_class
            tgt_fg = tgt == fg_class
            inter = (pred_fg & tgt_fg).sum(dim=(1, 2)).float()
            union = (pred_fg | tgt_fg).sum(dim=(1, 2)).float()
            iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
            nonempty = tgt_fg.sum(dim=(1, 2)) > 0
            for bi in range(img.shape[0]):
                ti = int(task_id[bi].item())
                per_task[ti]["n"] += 1
                if nonempty[bi]:
                    per_task[ti]["iou_sum"] += iou[bi].item()
                    per_task[ti]["iou_n"] += 1
        if held_out_ix and task_id is not None:
            from spatial_data.c1_tasks import compositional_wrong_task_index
            for bi in range(img.shape[0]):
                ti = int(task_id[bi].item())
                if ti not in held_out_ix:
                    continue
                if not nonempty[bi]:
                    continue
                ho_correct_sum += iou[bi].item()
                ho_correct_n += 1
                wrong_tid = compositional_wrong_task_index(ti, keys, cropset=config.cropset)
                wrong_id = torch.tensor([wrong_tid], device=device, dtype=torch.long)
                with torch.autocast(device_type=device_type, dtype=data.dtype):
                    logits_w, _ = model_forward(model, img[bi : bi + 1], tgt[bi : bi + 1], wrong_id)
                pred_w = logits_w.argmax(dim=-1)
                m_w = _c1_metrics_batch(pred_w, tgt[bi : bi + 1], pure[bi : bi + 1], fg_class=fg_class)
                if int(m_w["iou_n"].item()) > 0:
                    ho_wrong_sum += (m_w["iou_sum"] / m_w["iou_n"]).item()
                    ho_wrong_n += 1
        n_batches += 1
    model.train()
    if n_batches == 0:
        raise RuntimeError("eval_loader_c1: no batches")
    iou_n_i = int(acc["iou_n"].item())
    pure_n_i = int(acc["iou_pure_n"].item())
    mixed_n_i = int(acc["iou_mixed_n"].item())
    empty_n_i = int(acc["empty_n"].item())
    n_grids = int(acc["n_grids"].item())
    out = {
        "batches": n_batches,
        "mean_loss": loss_sum / n_batches,
        "mean_iou_fg": (acc["iou_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "iou_nonempty_n": iou_n_i,
        "precision_fg": (acc["prec_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "recall_fg": (acc["rec_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "mean_iou_pure_cells": (acc["iou_pure_sum"] / acc["iou_pure_n"]).item() if pure_n_i else 0.0,
        "mean_iou_mixed_cells": (acc["iou_mixed_sum"] / acc["iou_mixed_n"]).item() if mixed_n_i else 0.0,
        "empty_target_frac": empty_n_i / n_grids if n_grids else 0.0,
        "empty_target_fp_rate": (acc["empty_fp"].float() / acc["empty_n"]).item() if empty_n_i else 0.0,
        "empty_target_pred_cells_mean": (
            (acc["empty_pred_cells_sum"] / acc["empty_n"]).item() if empty_n_i else 0.0
        ),
        "n_grids": n_grids,
        "c1_task": single_task or config.c1_task_mode,
    }
    for ti, key in enumerate(keys):
        pt = per_task[ti]
        out[f"iou_{key}"] = (pt["iou_sum"] / pt["iou_n"]) if pt["iou_n"] else 0.0
    if held_out_ix:
        out["iou_held_out_correct"] = ho_correct_sum / ho_correct_n if ho_correct_n else 0.0
        out["iou_held_out_wrong_task"] = ho_wrong_sum / ho_wrong_n if ho_wrong_n else 0.0
        out["held_out_composition_gap"] = out["iou_held_out_correct"] - out["iou_held_out_wrong_task"]
        out["held_out_eval_n"] = ho_correct_n
    return out


def _c3_loader_batches(loader, max_batches: int | None, task_index_filter: int | None = None):
    if task_index_filter is not None and hasattr(loader, "iter_batches_for_task"):
        yield from loader.iter_batches_for_task(task_index_filter, max_batches)
        return
    if max_batches is None and hasattr(loader, "iter_batches"):
        yield from loader.iter_batches()
    else:
        yield from _iter_batches(loader, max_batches)


@torch.no_grad()
def _c3_iou_accum(
    model,
    loader,
    *,
    max_batches: int | None,
    encoder_self_attn_only: bool = False,
    encoder_patch_local_only: bool = False,
    task_index_filter: int | None = None,
) -> dict:
    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    fg_class = 1 if config.num_grid_classes > 1 else 0
    acc_dev = device if str(device).startswith("cuda") else torch.device("cpu")
    acc = {
        k: torch.zeros((), device=acc_dev)
        for k in ("iou_sum", "prec_sum", "rec_sum", "iou_pure_sum", "iou_mixed_sum", "empty_pred_cells_sum")
    }
    acc.update({k: torch.zeros((), device=acc_dev, dtype=torch.long) for k in (
        "iou_n", "empty_n", "empty_fp", "iou_pure_n", "iou_mixed_n", "n_grids",
    )})
    n_batches = 0
    loss_sum = 0.0
    for img, tgt, task_id, cond_ids in _c3_loader_batches(
        loader, max_batches, task_index_filter=task_index_filter,
    ):
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, loss = model_forward(
                model, img, tgt, task_id, cond_ids,
                encoder_self_attn_only=encoder_self_attn_only,
                encoder_patch_local_only=encoder_patch_local_only,
            )
        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        # c3 task_id is not c1 task index; pure/mixed diagnostics use estab plane on v4.
        pure = _pure_for_c1_batch(img, None, 2)
        m = _c1_metrics_batch(pred, tgt, pure, fg_class=fg_class)
        for k in ("iou_sum", "prec_sum", "rec_sum", "iou_pure_sum", "iou_mixed_sum", "empty_pred_cells_sum"):
            acc[k] = acc[k] + m[k]
        for k in ("iou_n", "empty_n", "empty_fp", "iou_pure_n", "iou_mixed_n"):
            acc[k] = acc[k] + m[k]
        acc["n_grids"] = acc["n_grids"] + m["n_grids"]
        n_batches += 1
    if n_batches == 0:
        raise RuntimeError("eval_loader_c3: no batches")
    iou_n_i = int(acc["iou_n"].item())
    empty_n_i = int(acc["empty_n"].item())
    n_grids = int(acc["n_grids"].item())
    pure_n_i = int(acc["iou_pure_n"].item())
    mixed_n_i = int(acc["iou_mixed_n"].item())
    return {
        "batches": n_batches,
        "mean_loss": loss_sum / n_batches,
        "mean_iou_fg": (acc["iou_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "iou_nonempty_n": iou_n_i,
        "precision_fg": (acc["prec_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "recall_fg": (acc["rec_sum"] / acc["iou_n"]).item() if iou_n_i else 0.0,
        "mean_iou_pure_cells": (acc["iou_pure_sum"] / acc["iou_pure_n"]).item() if pure_n_i else 0.0,
        "mean_iou_mixed_cells": (acc["iou_mixed_sum"] / acc["iou_mixed_n"]).item() if mixed_n_i else 0.0,
        "empty_target_frac": empty_n_i / n_grids if n_grids else 0.0,
        "empty_target_fp_rate": (acc["empty_fp"].float() / acc["empty_n"]).item() if empty_n_i else 0.0,
        "empty_target_pred_cells_mean": (
            (acc["empty_pred_cells_sum"] / acc["empty_n"]).item() if empty_n_i else 0.0
        ),
        "n_grids": n_grids,
    }


_C3_GEOMETRY_ORACLE_CACHE: dict | None = None


def reset_c3_geometry_oracle_cache() -> None:
    global _C3_GEOMETRY_ORACLE_CACHE
    _C3_GEOMETRY_ORACLE_CACHE = None


@torch.no_grad()
def _c3_geometry_oracle_iou(loader, *, max_batches: int | None = None) -> dict | None:
    """Marker tasks: paint disc from stored marker col/row; score vs loader targets (C3_PLAN N1)."""
    from spatial_data.c3_tasks import get_c3_task
    from spatial_data.c3_targets import c3_targets_for_spec

    if config.c3_is_multi_task():
        return None
    spec = get_c3_task(config.c3_eval_key())
    if not spec.show_marker_plane or not hasattr(loader, "_marker_col"):
        return None
    acc_dev = loader.device
    label_match = torch.zeros((), device=acc_dev, dtype=torch.long)
    label_total = torch.zeros((), device=acc_dev, dtype=torch.long)
    n_batches = 0
    n_crops = len(loader)
    bsz = loader.B
    pos = 0
    while pos < n_crops:
        if max_batches is not None and n_batches >= max_batches:
            break
        hi = min(pos + bsz, n_crops)
        ix = torch.arange(pos, hi, device=loader.device)
        pos = hi
        layers = loader.crops[ix]
        mc = loader._marker_col[ix]
        mr = loader._marker_row[ix]
        oracle = c3_targets_for_spec(
            layers, spec, grid_size=loader.grid_size, marker_col=mc, marker_row=mr,
        )
        _, tgt, _, _ = loader._build(ix)
        label_match = label_match + (tgt == oracle).sum(dtype=torch.long)
        label_total = label_total + tgt.numel()
        n_batches += 1
    if n_batches == 0:
        return None
    agree = (label_match.float() / label_total.clamp(min=1)).item()
    return {
        "c3_geometry_oracle_iou": agree,
        "c3_geometry_oracle_n": int(label_total.item()),
        "c3_loader_label_agreement": agree,
        "c3_geometry_oracle_batches": n_batches,
    }


@torch.no_grad()
def eval_loader_c3(model, loader, *, max_batches: int | None = None, snap: bool = False) -> dict:
    """c3 step 4: full val + ablations; ``snap=True`` = one forward pass (train CE snap only)."""
    from spatial_data.c3_tasks import LOCAL_CEILING

    model.eval()
    key = config.c3_eval_key()
    task_ix = None
    if config.c3_is_multi_task():
        task_ix = config.c3_keys_active().index(key)
    if snap:
        full = _c3_iou_accum(
            model, loader, max_batches=max_batches, task_index_filter=task_ix,
        )
        model.train()
        return {**full, "c3_task": key}
    global _C3_GEOMETRY_ORACLE_CACHE
    if max_batches is None:
        geo = _c3_geometry_oracle_iou(loader, max_batches=None)
        _C3_GEOMETRY_ORACLE_CACHE = geo
    elif _C3_GEOMETRY_ORACLE_CACHE is not None:
        geo = _C3_GEOMETRY_ORACLE_CACHE
    else:
        geo = _c3_geometry_oracle_iou(loader, max_batches=1)
        _C3_GEOMETRY_ORACLE_CACHE = geo
    full = _c3_iou_accum(
        model, loader, max_batches=max_batches, encoder_self_attn_only=False, task_index_filter=task_ix,
    )
    patch_local = _c3_iou_accum(
        model, loader, max_batches=max_batches, encoder_patch_local_only=True, task_index_filter=task_ix,
    )
    diagonal = _c3_iou_accum(
        model, loader, max_batches=max_batches, encoder_self_attn_only=True, task_index_filter=task_ix,
    )
    model.train()
    ceiling = LOCAL_CEILING.get(key)
    ablated_iou = patch_local["mean_iou_fg"]
    out = {
        **full,
        "c3_task": key,
        "c3_local_ceiling": ceiling,
        "mean_iou_fg_patch_local": ablated_iou,
        "mean_iou_fg_self_attn_only": diagonal["mean_iou_fg"],
        "c3_ablation_drop": full["mean_iou_fg"] - ablated_iou,
        "c3_ablation_gap_vs_ceiling": ablated_iou - ceiling if ceiling is not None else None,
    }
    if config.c3_is_multi_task():
        for i, tk in enumerate(config.c3_keys_active()):
            out[f"iou_{tk}"] = _c3_iou_accum(
                model, loader, max_batches=max_batches, task_index_filter=i,
            )["mean_iou_fg"]
    if geo:
        out.update(geo)
    return out


def _grid_metrics_batch(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    am: torch.Tensor,
    tidx: torch.Tensor,
    *,
    fg_class: int,
    s: int,
    G: int,
) -> dict[str, torch.Tensor]:
    """Batched c0 grid metrics; tensors stay on device until the eval ends."""
    b = pred.shape[0]
    device = pred.device
    pred_fg = pred == fg_class
    tgt_fg = tgt == fg_class
    inter = (pred_fg & tgt_fg).sum(dim=(1, 2)).float()
    union = (pred_fg | tgt_fg).sum(dim=(1, 2)).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
    pred_n_fg = pred_fg.sum(dim=(1, 2))
    tgt_n_fg = tgt_fg.sum(dim=(1, 2))
    pred_flat_idx = pred_fg.flatten(1).float().argmax(dim=1)
    exact = (tgt_n_fg == 1) & (pred_n_fg == 1) & (pred_flat_idx == tidx)
    tr = tidx // G
    tc = tidx % G
    ar = am // G
    ac = am % G
    batch_ix = torch.arange(b, device=device)
    true_cell_recall = pred[batch_ix, tr, tc] == fg_class
    global_argmax_exact = am == tidx
    patch_hit = (ar // s == tr // s) & (ac // s == tc // s)
    subcell_given_patch = patch_hit & (am == tidx)
    rows = torch.arange(G, device=device).view(1, G, 1)
    cols = torch.arange(G, device=device).view(1, 1, G)
    pr = tr // s
    pc = tc // s
    in_true_patch = (rows // s == pr.view(b, 1, 1)) & (cols // s == pc.view(b, 1, 1))
    in_patch_sum = (pred_fg & in_true_patch).sum(dim=(1, 2)).float()
    fg_count = pred_n_fg.float()
    share = torch.where(fg_count > 0, in_patch_sum / fg_count, torch.nan)
    has_fg = fg_count > 0
    dist = torch.sqrt((ar - tr).float().square() + (ac - tc).float().square())
    return {
        "iou_sum": iou.sum(),
        "exact_cell": exact.sum(),
        "pred_fg_counts": pred_n_fg,
        "true_cell_recall": true_cell_recall.sum(),
        "global_argmax_exact": global_argmax_exact.sum(),
        "patch_hit": patch_hit.sum(),
        "subcell_hit_given_patch": subcell_given_patch.sum(),
        "fg_in_true_patch_sum": share[has_fg].nansum(),
        "fg_in_true_patch_n": has_fg.sum(),
        "argmax_dist_sum": dist.sum(),
        "argmax_dists": dist,
    }


@torch.no_grad()
def eval_loader_c2_grid(model, loader, *, max_batches: int | None = None) -> dict:
    """c2: val crop × task grid (full pass if max_batches is None); see harness/eval_c2.py."""
    from eval_c2 import evaluate_c2
    keys = config.c2_active_task_keys or config.c2_task_keys_active()
    res = evaluate_c2(
        model, loader,
        task_keys=keys,
        weights=config.c2_fg_weights_active(),
        encoding=config.encoding_mode,
        held_out=config.c2_held_out_keys_active(),
        autocast_dtype=data.dtype,
        max_batches=max_batches,
    )
    total_batches = (len(loader) + loader.B - 1) // loader.B
    used_batches = total_batches if max_batches is None else min(max_batches, total_batches)
    return {
        "batches": used_batches,
        "n_grids": used_batches * loader.B,
        "mean_iou_fg": res["mean_trained"].get("iou_rule", 0.0),
        "mean_loss": float("nan"),
        "c2_task_set": config.c2_task_set,
        "c2": res,
    }


def diagnostics_val_max_batches(ce_eval_iters: int) -> int | None:
    """Batch cap for tier-2 `{run_id}.eval.jsonl` rows tied to CE eval iters.

    c2 validation is task-major (all crops for task 0, then task 1, …). A partial pass
    only scores the first one–two tasks; `evaluate_c2` leaves the rest at zero IoU.
    """
    if config.task_rung in ("c2", "c4"):
        return None
    return ce_eval_iters


@torch.no_grad()
def eval_loader(model, loader, *, max_batches: int | None = None, snap: bool = False) -> dict:
    if config.task_rung == "c2":
        return eval_loader_c2_grid(model, loader, max_batches=max_batches)
    if config.task_rung == "c3":
        return eval_loader_c3(model, loader, max_batches=max_batches, snap=snap)
    if config.task_rung == "c4":
        from eval_c4 import eval_loader_c4
        return eval_loader_c4(model, loader, max_batches=max_batches, snap=snap)
    if config.task_rung == "c1":
        return eval_loader_c1(model, loader, max_batches=max_batches)
    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    model.eval()
    n_batches = 0
    loss_sum = 0.0
    fg_class = 1 if config.num_grid_classes > 1 else 0
    s = _subcells()
    block_cells = s * s
    acc_device = device if str(device).startswith("cuda") else torch.device("cpu")
    iou_sum = torch.zeros((), device=acc_device)
    exact_cell = torch.zeros((), device=acc_device, dtype=torch.long)
    true_cell_recall = torch.zeros((), device=acc_device, dtype=torch.long)
    global_argmax_exact = torch.zeros((), device=acc_device, dtype=torch.long)
    patch_hit = torch.zeros((), device=acc_device, dtype=torch.long)
    subcell_hit_given_patch = torch.zeros((), device=acc_device, dtype=torch.long)
    fg_in_true_patch_sum = torch.zeros((), device=acc_device)
    fg_in_true_patch_n = torch.zeros((), device=acc_device, dtype=torch.long)
    argmax_dist_sum = torch.zeros((), device=acc_device)
    pred_fg_counts_parts: list[torch.Tensor] = []
    argmax_dist_parts: list[torch.Tensor] = []
    n_grids = 0
    for img, tgt, task_id, cond_ids in _iter_batches(loader, max_batches):
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, loss = model_forward(model, img, tgt, task_id, cond_ids)
        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        b, g, _ = pred.shape
        if config.num_grid_classes > 1:
            score = (logits[..., fg_class] - logits[..., 0]).reshape(b, -1)
        else:
            score = logits.reshape(b, -1)
        am = score.argmax(dim=1)
        tidx = tgt.reshape(b, -1).argmax(dim=1)
        m = _grid_metrics_batch(pred, tgt, am, tidx, fg_class=fg_class, s=s, G=g)
        iou_sum = iou_sum + m["iou_sum"]
        exact_cell = exact_cell + m["exact_cell"]
        true_cell_recall = true_cell_recall + m["true_cell_recall"]
        global_argmax_exact = global_argmax_exact + m["global_argmax_exact"]
        patch_hit = patch_hit + m["patch_hit"]
        subcell_hit_given_patch = subcell_hit_given_patch + m["subcell_hit_given_patch"]
        fg_in_true_patch_sum = fg_in_true_patch_sum + m["fg_in_true_patch_sum"]
        fg_in_true_patch_n = fg_in_true_patch_n + m["fg_in_true_patch_n"]
        argmax_dist_sum = argmax_dist_sum + m["argmax_dist_sum"]
        pred_fg_counts_parts.append(m["pred_fg_counts"])
        argmax_dist_parts.append(m["argmax_dists"])
        n_grids += b
        n_batches += 1
    model.train()
    if n_batches == 0:
        raise RuntimeError("eval_loader: no batches")
    counts = torch.cat(pred_fg_counts_parts).float().cpu()
    dists = torch.cat(argmax_dist_parts).float().cpu()
    patch_hit_i = int(patch_hit.item())
    fg_n_i = int(fg_in_true_patch_n.item())
    return {
        "batches": n_batches,
        "mean_loss": loss_sum / n_batches,
        "mean_iou_fg": (iou_sum / n_grids).item(),
        "exact_cell_accuracy": (exact_cell.float() / n_grids).item(),
        "n_grids": n_grids,
        "subcells_per_patch": s,
        "pred_fg_cells_mean": counts.mean().item() if n_grids else 0.0,
        "pred_fg_cells_median": counts.median().item() if n_grids else 0.0,
        "pred_fg_full_patch_frac": (counts == block_cells).float().mean().item() if n_grids else 0.0,
        "pred_fg_zero_frac": (counts == 0).float().mean().item() if n_grids else 0.0,
        "true_cell_recall": (true_cell_recall.float() / n_grids).item(),
        "pred_fg_in_true_patch_mean": (
            (fg_in_true_patch_sum / fg_in_true_patch_n).item() if fg_n_i else 0.0
        ),
        "global_argmax_exact": (global_argmax_exact.float() / n_grids).item(),
        "global_argmax_patch_hit": (patch_hit.float() / n_grids).item(),
        "subcell_hit_given_patch": (
            (subcell_hit_given_patch.float() / patch_hit).item() if patch_hit_i else 0.0
        ),
        "global_argmax_dist_cells_mean": (argmax_dist_sum / n_grids).item(),
        "global_argmax_dist_cells_median": dists.median().item() if n_grids else 0.0,
    }


def _iter_batches(loader, max_batches: int | None):
    if max_batches is None:
        while True:
            yield loader_next_batch(loader)
    else:
        for _ in range(max_batches):
            yield loader_next_batch(loader)


def _single_fg_index(fg_mask: torch.Tensor) -> tuple[int, int] | None:
    nz = fg_mask.nonzero(as_tuple=False)
    if nz.size(0) != 1:
        return None
    return int(nz[0, 0].item()), int(nz[0, 1].item())


def _grid_metrics_batch_reference(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    am: torch.Tensor,
    tidx: torch.Tensor,
    *,
    fg_class: int,
    s: int,
    G: int,
) -> dict:
    """Per-sample loop (parity reference for tests)."""
    b = pred.shape[0]
    iou_sum = 0.0
    exact_cell = 0
    pred_fg_counts: list[int] = []
    true_cell_recall = 0
    global_argmax_exact = 0
    patch_hit = 0
    subcell_hit_given_patch = 0
    fg_in_true_patch_sum = 0.0
    fg_in_true_patch_n = 0
    argmax_dist_sum = 0.0
    argmax_dists: list[float] = []
    for bi in range(b):
        p, t = pred[bi], tgt[bi]
        pred_fg = p == fg_class
        tgt_fg = t == fg_class
        inter = (pred_fg & tgt_fg).sum().item()
        union = (pred_fg | tgt_fg).sum().item()
        iou_sum += inter / union if union > 0 else 1.0
        pi = _single_fg_index(pred_fg)
        ti = _single_fg_index(tgt_fg)
        exact_cell += int(ti is not None and pi == ti)
        n_fg = int(pred_fg.sum().item())
        pred_fg_counts.append(n_fg)
        tr, tc = (tidx[bi] // G).item(), (tidx[bi] % G).item()
        ar, ac = (am[bi] // G).item(), (am[bi] % G).item()
        true_cell_recall += int(p[tr, tc].item() == fg_class)
        global_argmax_exact += int(am[bi].item() == tidx[bi].item())
        ph = (ar // s == tr // s) and (ac // s == tc // s)
        patch_hit += int(ph)
        if ph:
            subcell_hit_given_patch += int(am[bi].item() == tidx[bi].item())
        fg_idx = pred_fg.nonzero(as_tuple=False)
        if fg_idx.size(0) > 0:
            in_patch = (
                (fg_idx[:, 0] // s == tr // s) & (fg_idx[:, 1] // s == tc // s)
            ).float().mean().item()
            fg_in_true_patch_sum += in_patch
            fg_in_true_patch_n += 1
        d = float(((ar - tr) ** 2 + (ac - tc) ** 2) ** 0.5)
        argmax_dist_sum += d
        argmax_dists.append(d)
    counts = torch.tensor(pred_fg_counts, dtype=torch.float32)
    patch_hit_i = patch_hit
    return {
        "mean_iou_fg": iou_sum / b,
        "exact_cell_accuracy": exact_cell / b,
        "pred_fg_cells_mean": counts.mean().item(),
        "pred_fg_cells_median": counts.median().item(),
        "pred_fg_full_patch_frac": (counts == s * s).float().mean().item(),
        "pred_fg_zero_frac": (counts == 0).float().mean().item(),
        "true_cell_recall": true_cell_recall / b,
        "pred_fg_in_true_patch_mean": (
            fg_in_true_patch_sum / fg_in_true_patch_n if fg_in_true_patch_n else 0.0
        ),
        "global_argmax_exact": global_argmax_exact / b,
        "global_argmax_patch_hit": patch_hit / b,
        "subcell_hit_given_patch": (
            subcell_hit_given_patch / patch_hit_i if patch_hit_i else 0.0
        ),
        "global_argmax_dist_cells_mean": argmax_dist_sum / b,
        "global_argmax_dist_cells_median": (
            torch.tensor(argmax_dists).median().item() if argmax_dists else 0.0
        ),
    }


def eval_model(model, *, max_batches: int | None = 20) -> dict:
    return {
        "meta": {
            "task_rung": config.task_rung,
            "code_version": config.code_version,
            "num_classes": config.num_grid_classes,
        },
        "val": eval_loader(model, data.val_loader, max_batches=max_batches),
    }


def print_c0_detail(val: dict, *, prefix: str = "    c0 detail:") -> None:
    s = int(val.get("subcells_per_patch", _subcells()))
    chance = 1.0 / (s * s)
    print(
        f"{prefix} pred_fg mean/median={val['pred_fg_cells_mean']:.1f}/"
        f"{val['pred_fg_cells_median']:.0f} full_patch({s}×{s})="
        f"{val['pred_fg_full_patch_frac']:.2f} recall={val['true_cell_recall']:.3f} "
        f"argmax_patch={val['global_argmax_patch_hit']:.3f} "
        f"subcell|patch={val['subcell_hit_given_patch']:.3f} (chance={chance:.3f}) "
        f"global_argmax_exact={val['global_argmax_exact']:.3f}"
    )


def print_c2_detail(val: dict, *, prefix: str = "    c2 detail:") -> None:
    print(f"{prefix} mean rule IoU (trained tasks)={val['mean_iou_fg']:.4f} "
          f"({val['batches']} batches, ~{val['n_grids']} pairs)")


def print_c1_detail(val: dict, *, prefix: str = "    c1 detail:") -> None:
    print(
        f"{prefix} task={val.get('c1_task', '?')} "
        f"IoU={val.get('mean_iou_fg', 0):.3f} "
        f"IoU_pure={val.get('mean_iou_pure_cells', 0):.3f} "
        f"IoU_mixed={val.get('mean_iou_mixed_cells', 0):.3f} "
        f"P/R={val.get('precision_fg', 0):.3f}/{val.get('recall_fg', 0):.3f} "
        f"empty_fp={val.get('empty_target_fp_rate', 0):.3f}"
    )
    keys = config.c1_active_task_keys or []
    if len(keys) > 1:
        parts = [f"{k}={val.get(f'iou_{k}', 0):.3f}" for k in keys if f"iou_{k}" in val]
        if parts:
            print(f"{prefix} per_task: " + " ".join(parts))
    if "iou_held_out_correct" in val:
        print(
            f"{prefix} held_out: correct={val['iou_held_out_correct']:.3f} "
            f"wrong_task={val['iou_held_out_wrong_task']:.3f} "
            f"gap={val['held_out_composition_gap']:.3f} (n={val.get('held_out_eval_n', 0)})"
        )


def print_report(result: dict):
    v = result["val"]
    if config.task_rung == "c2":
        from eval_c2 import print_table
        print(f"  spatial eval (c2): mean rule IoU over trained tasks={v['mean_iou_fg']:.4f} "
              f"({v['n_grids']} crop x task pairs)")
        print_table(v["c2"])
        return
    if config.task_rung == "c3":
        ceil = v.get("c3_local_ceiling")
        gap = (v["mean_iou_fg"] - ceil) if ceil is not None else None
        print(
            f"  spatial eval (c3): task={v.get('c3_task')} loss={v['mean_loss']:.4f} "
            f"IoU_fg={v['mean_iou_fg']:.4f} ({v['n_grids']} crops, nonempty n={v['iou_nonempty_n']})"
        )
        if ceil is not None:
            print(
                f"    vs local-only ceiling {ceil:.3f} → gap {gap:+.3f}; "
                f"empty_frac={v['empty_target_frac']:.3f} empty_fp={v['empty_target_fp_rate']:.3f}"
            )
        else:
            print(
                f"    (no local-only ceiling for this task); "
                f"empty_frac={v['empty_target_frac']:.3f} empty_fp={v['empty_target_fp_rate']:.3f}"
            )
        iou_pl = v.get("mean_iou_fg_patch_local")
        if iou_pl is not None:
            drop = v.get("c3_ablation_drop", 0.0)
            vs_ceil = v.get("c3_ablation_gap_vs_ceiling")
            line = f"    patch-local IoU={iou_pl:.4f} (drop {drop:+.3f} vs full)"
            if vs_ceil is not None:
                line += f"; patch-local vs ceiling {vs_ceil:+.3f}"
            print(line)
        iou_diag = v.get("mean_iou_fg_self_attn_only")
        if iou_diag is not None:
            print(f"    strict diagonal IoU={iou_diag:.4f} (cond tokens isolated; diagnostic only)")
        if v.get("c3_geometry_oracle_iou") is not None:
            print(
                f"    geometry oracle IoU={v['c3_geometry_oracle_iou']:.4f} "
                f"(loader label agreement {v.get('c3_loader_label_agreement', 0):.6f})"
            )
        if config.c3_is_multi_task():
            parts = [
                f"{tk}={v.get(f'iou_{tk}', 0):.3f}"
                for tk in config.c3_keys_active()
                if f"iou_{tk}" in v
            ]
            if parts:
                print(f"    per_task IoU: " + " ".join(parts))
        return
    if config.task_rung == "c4":
        from eval_c4 import print_c4_detail
        print(
            f"  spatial eval (c4): loss={v['mean_loss']:.4f} "
            f"IoU_fg={v['mean_iou_fg']:.4f} conn={v.get('connectivity', 0):.4f} "
            f"({v['n_grids']} val items, {v['batches']} batches)"
        )
        print_c4_detail(v)
        return
    if config.task_rung == "c1":
        print(
            f"  spatial eval: loss={v['mean_loss']:.4f} "
            f"IoU_fg={v['mean_iou_fg']:.4f} "
            f"({v['n_grids']} grids, {v['batches']} batches)"
        )
        print_c1_detail(v)
        return
    print(
        f"  spatial eval: loss={v['mean_loss']:.4f} "
        f"IoU_fg={v['mean_iou_fg']:.4f} exact_cell={v['exact_cell_accuracy']:.4f} "
        f"({v['n_grids']} grids, {v['batches']} batches)"
    )
    print_c0_detail(v)


def flatten_for_eval_jsonl(val: dict) -> dict:
    """Spec-style flat keys for runs/diagnostics/{run_id}/eval.jsonl (plot_run §9 #4)."""
    if config.task_rung == "c2":
        from eval_c2 import flatten_for_eval_jsonl as flatten_c2
        return {"IoU_fg": round(val["mean_iou_fg"], 6), **flatten_c2(val["c2"])}
    if config.task_rung == "c4":
        from eval_c4 import flatten_for_eval_jsonl as flatten_c4
        return flatten_c4(val)
    base = {
        "mean_loss": round(val["mean_loss"], 6),
        "IoU_fg": round(val["mean_iou_fg"], 6),
    }
    if config.task_rung == "c3":
        out = {
            "mean_loss": round(val["mean_loss"], 6),
            "IoU_fg": round(val["mean_iou_fg"], 6),
            "c3_task": val.get("c3_task", config.c3_task_key),
            "c3_local_ceiling": val.get("c3_local_ceiling"),
            "empty_target_frac": round(val.get("empty_target_frac", 0.0), 6),
            "empty_target_fp_rate": round(val.get("empty_target_fp_rate", 0.0), 6),
        }
        if "mean_iou_fg_patch_local" in val:
            out["IoU_fg_patch_local"] = round(val["mean_iou_fg_patch_local"], 6)
            out["c3_ablation_drop"] = round(val["c3_ablation_drop"], 6)
            g = val.get("c3_ablation_gap_vs_ceiling")
            if g is not None:
                out["c3_ablation_gap_vs_ceiling"] = round(g, 6)
        if "mean_iou_fg_self_attn_only" in val:
            out["IoU_fg_strict_diagonal"] = round(val["mean_iou_fg_self_attn_only"], 6)
        if val.get("c3_geometry_oracle_iou") is not None:
            out["c3_geometry_oracle_iou"] = round(val["c3_geometry_oracle_iou"], 6)
            out["c3_loader_label_agreement"] = round(val.get("c3_loader_label_agreement", 0.0), 6)
        for layer in (0, 1):
            for k in (f"attn_sw_mass/{layer}", f"attn_sw_ratio/{layer}"):
                if k in val:
                    out[k] = round(float(val[k]), 6)
        return out
    if config.task_rung == "c1":
        base.update({
            "c1_task": val.get("c1_task", config.c1_task_key),
            "IoU_pure_cells": round(val.get("mean_iou_pure_cells", 0.0), 6),
            "IoU_mixed_cells": round(val.get("mean_iou_mixed_cells", 0.0), 6),
            "precision_fg": round(val.get("precision_fg", 0.0), 6),
            "recall_fg": round(val.get("recall_fg", 0.0), 6),
            "empty_target_fp_rate": round(val.get("empty_target_fp_rate", 0.0), 6),
            "empty_target_frac": round(val.get("empty_target_frac", 0.0), 6),
        })
        keys = config.c1_active_task_keys or config.c1_task_keys_active()
        for k in keys:
            fk = f"iou_{k}"
            if fk in val:
                base[fk] = round(float(val[fk]), 6)
        for hk in ("iou_held_out_correct", "iou_held_out_wrong_task", "held_out_composition_gap"):
            if hk in val:
                base[hk] = round(float(val[hk]), 6)
        if "held_out_eval_n" in val:
            base["held_out_eval_n"] = int(val["held_out_eval_n"])
        return base
    s = int(val.get("subcells_per_patch", _subcells()))
    block = s * s
    base.update({
        "exact_cell": round(val["exact_cell_accuracy"], 6),
        "subcells_per_patch": s,
        "n_fg_mean": round(val["pred_fg_cells_mean"], 4),
        "n_fg_median": round(val["pred_fg_cells_median"], 4),
        f"frac_n_fg_eq_{block}": round(val["pred_fg_full_patch_frac"], 4),
        "recall": round(val["true_cell_recall"], 6),
        "argmax_hit": round(val["global_argmax_exact"], 6),
        "patch_hit": round(val["global_argmax_patch_hit"], 6),
        "subcell_hit_given_patch": round(val["subcell_hit_given_patch"], 6),
        "argmax_dist_mean": round(val["global_argmax_dist_cells_mean"], 4),
        "argmax_dist_median": round(val["global_argmax_dist_cells_median"], 4),
        "pred_fg_in_true_patch_mean": round(val["pred_fg_in_true_patch_mean"], 4),
        "pred_fg_zero_frac": round(val["pred_fg_zero_frac"], 4),
    })
    return base


def spatial_summary_for_run_log(val: dict) -> dict:
    """Flat keys for runs/experiments.json summary."""
    if config.task_rung == "c2":
        out = {"val_iou_fg": val.get("mean_iou_fg")}
        for group in ("mean_trained", "mean_held_out"):
            for name, v in val["c2"].get(group, {}).items():
                out[f"val_c2_{group}_{name}"] = v
        c2 = val.get("c2") or {}
        for hk in ("iou_held_out_correct", "iou_held_out_wrong_description", "held_out_description_gap"):
            if hk in c2:
                out[f"val_c2_{hk}"] = c2[hk]
        return out
    if config.task_rung == "c3":
        summary = {
            "val_iou_fg": val.get("mean_iou_fg"),
            "val_c3_task": val.get("c3_task"),
            "val_c3_local_ceiling": val.get("c3_local_ceiling"),
            "val_c3_gap_vs_local_ceiling": (
                val.get("mean_iou_fg", 0) - val["c3_local_ceiling"]
                if val.get("c3_local_ceiling") is not None
                else None
            ),
            "val_empty_target_fp_rate": val.get("empty_target_fp_rate"),
        }
        if "mean_iou_fg_patch_local" in val:
            summary["val_iou_fg_patch_local"] = val.get("mean_iou_fg_patch_local")
            summary["val_c3_ablation_drop"] = val.get("c3_ablation_drop")
            summary["val_c3_ablation_gap_vs_ceiling"] = val.get("c3_ablation_gap_vs_ceiling")
        if "mean_iou_fg_self_attn_only" in val:
            summary["val_iou_fg_strict_diagonal"] = val.get("mean_iou_fg_self_attn_only")
        return summary
    if config.task_rung == "c4":
        return {
            "val_iou_fg": val.get("mean_iou_fg"),
            "val_c4_connectivity": val.get("connectivity"),
            "val_c4_length_ratio": val.get("length_ratio"),
            "val_c4_iou_pred_vs_segment": val.get("iou_pred_vs_segment"),
        }
    if config.task_rung == "c1":
        summary = {
            "val_iou_fg": val.get("mean_iou_fg"),
            "val_iou_pure_cells": val.get("mean_iou_pure_cells"),
            "val_iou_mixed_cells": val.get("mean_iou_mixed_cells"),
            "val_precision_fg": val.get("precision_fg"),
            "val_recall_fg": val.get("recall_fg"),
            "val_empty_target_fp_rate": val.get("empty_target_fp_rate"),
        }
        if "iou_held_out_correct" in val:
            summary["val_iou_held_out_correct"] = val.get("iou_held_out_correct")
            summary["val_iou_held_out_wrong_task"] = val.get("iou_held_out_wrong_task")
            summary["val_held_out_composition_gap"] = val.get("held_out_composition_gap")
        return summary
    mapping = {
        "mean_iou_fg": "val_iou_fg",
        "exact_cell_accuracy": "val_exact_cell",
        "pred_fg_cells_mean": "val_pred_fg_cells_mean",
        "pred_fg_full_patch_frac": "val_pred_fg_full_patch_frac",
        "true_cell_recall": "val_true_cell_recall",
        "global_argmax_patch_hit": "val_global_argmax_patch_hit",
        "subcell_hit_given_patch": "val_subcell_hit_given_patch",
        "global_argmax_exact": "val_global_argmax_exact",
    }
    return {dst: val[src] for src, dst in mapping.items() if src in val}
