"""c4 route eval: IoU, connectivity, length ratio, detour strata, segment shortcut."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage

import config
import data
from spatial_batch import forward as model_forward
from spatial_data.c4_tasks import (
    IGNORE_TARGET,
    LENGTH_BIN_M,
    TOL_M,
    STRATUM_LABELS,
    TASK_KEYS,
    path_pix_to_grid_numpy,
    m_per_cell,
    rasterise_segment_numpy,
)


def _grid_size() -> int:
    return config.grid_out_size


#: straight marker distance bins in metres (eval only): the 256 m stores hold 40-250 m, the 512 m store 40-500 m
LENGTH_BINS_M = ((40.0, 120.0), (120.0, 250.0), (250.0, 500.0))


def _len_label(straight_m: float) -> str | None:
    for lo, hi in LENGTH_BINS_M:
        if lo <= straight_m < hi:
            return f"len_{lo:.0f}_{hi:.0f}"
    return None


def _cell_m(grid: int | None = None) -> float:
    """Metres per output cell: window size (config.img_size px at 1 m/px) over the grid."""
    return config.img_size / (grid or _grid_size())


def _tol_cells() -> int:
    """TOL_M in cells: 1 at 4 m cells (256 px / 64, 512 px / 128), 2 at 2 m cells (256 px / 128)."""
    return max(1, int(round(TOL_M / _cell_m())))


def token_logits(model, img, task_id, cond_ids):
    """(B, n_token_classes) from the readout slot, or None when the model has no token head."""
    core = getattr(model, "_orig_mod", model)
    if getattr(core, "token_head", None) is None:
        return None
    full = core._encode_full(img, task_id, cond_ids=cond_ids)
    return core.token_head(full[:, core.config.readout_token_index, :])


def _marker_grid_cells(markers_xy: np.ndarray, *, grid: int | None = None, img: int | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    grid = grid or _grid_size(); img = img or config.img_size
    s = img // grid
    out = []
    for k in range(2):
        gc = int(np.floor(markers_xy[k, 0] / s))
        gr = int(np.floor(markers_xy[k, 1] / s))
        out.append((int(np.clip(gr, 0, grid - 1)), int(np.clip(gc, 0, grid - 1))))
    return out[0], out[1]


def _component_ids_touching_marker(pred: np.ndarray, gr: int, gc: int) -> set[int]:
    g = pred.shape[0]
    lab, _ = ndimage.label(pred, structure=np.ones((3, 3), dtype=int))  # 8-connected: diagonal lines count
    ids: set[int] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = gr + dr, gc + dc
            if 0 <= r < g and 0 <= c < g and pred[r, c]:
                ids.add(int(lab[r, c]))
    ids.discard(0)
    return ids


def pred_connects_markers(pred_fg: np.ndarray, markers_xy: np.ndarray) -> bool:
    m0, m1 = _marker_grid_cells(markers_xy)
    s0 = _component_ids_touching_marker(pred_fg, m0[0], m0[1])
    s1 = _component_ids_touching_marker(pred_fg, m1[0], m1[1])
    return bool(s0 & s1)


def pred_path_length_m(pred_fg: np.ndarray, markers_xy: np.ndarray) -> float:
    m0, m1 = _marker_grid_cells(markers_xy)
    s0 = _component_ids_touching_marker(pred_fg, m0[0], m0[1])
    s1 = _component_ids_touching_marker(pred_fg, m1[0], m1[1])
    shared = s0 & s1
    if not shared:
        return 0.0
    lab, _ = ndimage.label(pred_fg, structure=np.ones((3, 3), dtype=int))
    cid = next(iter(shared))
    return float((lab == cid).sum()) * _cell_m(pred_fg.shape[0])


def _dilate(a: np.ndarray) -> np.ndarray:
    k = 2 * _tol_cells() + 1
    return ndimage.binary_dilation(a, structure=np.ones((k, k), dtype=bool))


def tolerant_line_metrics(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float, float]:
    """For thin line targets: IoU after dilating both by TOL_M (4 m); precision and recall within 4 m.
    At grid 64 this is one cell, as before; the *_1cell metric names are kept for continuity."""
    pd, td = _dilate(pred), _dilate(tgt)
    iou_d = _iou_np(pd, td)
    prec = float(np.logical_and(pred, td).sum() / pred.sum()) if pred.sum() else 0.0
    rec = float(np.logical_and(tgt, pd).sum() / tgt.sum()) if tgt.sum() else 1.0
    return iou_d, prec, rec


def _iou_np(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def segment_grid_from_markers(markers_xy: np.ndarray) -> np.ndarray:
    pix = rasterise_segment_numpy(markers_xy[0], markers_xy[1], crop_px=config.img_size)
    return path_pix_to_grid_numpy(pix, grid=_grid_size(), img=config.img_size)


@dataclass
class _Bucket:
    iou_sum: float = 0.0
    iou_n: int = 0
    conn_sum: float = 0.0
    len_ratio_sum: float = 0.0
    len_ratio_n: int = 0
    shortcut_iou_sum: float = 0.0
    shortcut_n: int = 0
    iou_dil_sum: float = 0.0
    prec1_sum: float = 0.0
    rec1_sum: float = 0.0

    def add(
        self,
        *,
        iou: float,
        connected: bool,
        len_ratio: float | None,
        shortcut_iou: float | None,
        iou_dil: float = 0.0,
        prec1: float = 0.0,
        rec1: float = 0.0,
    ) -> None:
        self.iou_sum += iou
        self.iou_n += 1
        self.iou_dil_sum += iou_dil
        self.prec1_sum += prec1
        self.rec1_sum += rec1
        self.conn_sum += float(connected)
        if connected and len_ratio is not None:
            self.len_ratio_sum += len_ratio
            self.len_ratio_n += 1
        if shortcut_iou is not None:
            self.shortcut_iou_sum += shortcut_iou
            self.shortcut_n += 1

    def mean_iou(self) -> float:
        return self.iou_sum / self.iou_n if self.iou_n else 0.0

    def mean_conn(self) -> float:
        return self.conn_sum / self.iou_n if self.iou_n else 0.0

    def mean_iou_dil(self) -> float:
        return self.iou_dil_sum / self.iou_n if self.iou_n else 0.0

    def mean_prec1(self) -> float:
        return self.prec1_sum / self.iou_n if self.iou_n else 0.0

    def mean_rec1(self) -> float:
        return self.rec1_sum / self.iou_n if self.iou_n else 0.0

    def mean_len_ratio(self) -> float | None:
        return self.len_ratio_sum / self.len_ratio_n if self.len_ratio_n else None

    def mean_shortcut_iou(self) -> float | None:
        return self.shortcut_iou_sum / self.shortcut_n if self.shortcut_n else None


@dataclass
class _C4Accum:
    overall: _Bucket = field(default_factory=_Bucket)
    by_task: dict[str, _Bucket] = field(default_factory=dict)
    by_stratum: dict[int, _Bucket] = field(default_factory=dict)
    by_len: dict[str, _Bucket] = field(default_factory=dict)
    loss_sum: float = 0.0
    n_batches: int = 0
    n_grids: int = 0
    rays_cells: int = 0
    rays_on_bld: int = 0
    tok: dict[str, list[int]] = field(default_factory=dict)   # task -> [n, exact, within one bin, sum abs error in bins]
    wave: dict[str, list[float]] = field(default_factory=dict)  # task -> [n, recall on copied cells, recall on new cells, false fg on empty]
    wave_rule: dict[str, list[float]] = field(default_factory=dict)

    def bucket_task(self, key: str) -> _Bucket:
        if key not in self.by_task:
            self.by_task[key] = _Bucket()
        return self.by_task[key]

    def bucket_stratum(self, si: int) -> _Bucket:
        if si not in self.by_stratum:
            self.by_stratum[si] = _Bucket()
        return self.by_stratum[si]


def _gt_length_m(task_key: str, straight_px: float, path_px: float) -> float:
    return float(straight_px if task_key == "segment" else path_px)


def wave_decision_tau(task_key: str) -> float:
    """P(fg) threshold undoing dampened fg CE weight (balancing memo §3)."""
    w = config.c4_fg_weight(task_key)          # the loss's weight, c5_wave_fg_weight_override included
    return w / (1.0 + w)


def _wave_rule_tau_by_key(keys: list[str]) -> dict[str, float]:
    return {k: wave_decision_tau(k) for k in keys if k.startswith("wave_")}


def _iter_c4_val(loader, max_batches: int | None):
    if not hasattr(loader, "next_batch_with_meta"):
        raise TypeError("c4 eval requires a C4GpuValLoader or C4MultiTaskGpuValLoader")
    loader.reset()
    n_batches = 0
    n_expected = (len(loader) + loader.B - 1) // loader.B
    while n_batches < n_expected:
        if max_batches is not None and n_batches >= max_batches:
            break
        yield loader.next_batch_with_meta()
        n_batches += 1
    loader.reset()


@torch.no_grad()
def eval_loader_c4(model, loader, *, max_batches: int | None = None, snap: bool = False) -> dict:
    """Full c4 val metrics; ``snap=True`` skips strata/shortcut (train CE snap)."""
    model.eval()
    device = data.device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    fg_class = 1 if config.num_grid_classes > 1 else 0
    keys = config.c4_keys_active()
    if hasattr(loader, "spec") and not hasattr(loader, "task_specs"):
        keys = [loader.spec.key]      # single-task val loader (eval on one task of a mixed run): task_ix is always 0
    key_by_ix = {i: k for i, k in enumerate(keys)}
    detour_ix = keys.index("detour") if "detour" in keys else None
    acc = _C4Accum()
    wave_rule_eval = bool(getattr(config, "c5_wave_eval_decision_rule", False))
    wave_tau = _wave_rule_tau_by_key(keys) if wave_rule_eval else {}

    for img, tgt, task_id, cond_ids, meta in _iter_c4_val(loader, max_batches):
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        tok_tgt = meta.get("tok_tgt")
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, loss = model_forward(model, img, tgt, task_id, cond_ids)
            tl = token_logits(model, img, task_id, cond_ids) if tok_tgt is not None and bool((tok_tgt >= 0).any()) else None
        tok_pred = tl.argmax(dim=-1).cpu().numpy() if tl is not None else None
        tok_np = tok_tgt.cpu().numpy() if tok_tgt is not None else None
        pad_cells = meta.get("pad_cells"); pad_np = pad_cells.cpu().numpy() if pad_cells is not None else None
        acc.loss_sum += loss.item()
        acc.n_batches += 1
        pred = logits.argmax(dim=-1)
        prob_fg_b = (
            torch.softmax(logits.float(), dim=-1)[:, :, :, fg_class]
            if wave_rule_eval
            else None
        )
        b = pred.shape[0]
        acc.n_grids += b
        markers_b = meta["markers"].detach().cpu().numpy()
        strata = meta["stratum"].detach().cpu().numpy()
        straight = meta["straight_px"].detach().cpu().numpy()
        path_len = meta["path_px"].detach().cpu().numpy()
        task_ix = meta["task_ix"].detach().cpu().numpy()

        g = pred.shape[1]; s = config.img_size // g
        for i in range(b):
            tk = key_by_ix[int(task_ix[i])]
            if tok_np is not None and tok_np[i] >= 0 and tok_pred is not None:
                t = acc.tok.setdefault(tk, [0, 0, 0, 0]); err = abs(int(tok_pred[i]) - int(tok_np[i]))
                t[0] += 1; t[1] += int(err == 0); t[2] += int(err <= 1); t[3] += err
            tgt_np = tgt[i].cpu().numpy()
            if (tgt_np == IGNORE_TARGET).all():
                continue                                      # token-only sample: no mask metrics
            pred_fg = (pred[i].cpu().numpy() == fg_class)
            tgt_fg = (tgt_np == fg_class)
            if tk.startswith("wave_") and pad_np is not None:
                copied = pad_np[i] & tgt_fg; new = tgt_fg & ~pad_np[i]; empty = ~tgt_fg
                w = acc.wave.setdefault(tk, [0, 0.0, 0.0, 0.0]); w[0] += 1
                w[1] += float(pred_fg[copied].mean()) if copied.any() else 1.0
                w[2] += float(pred_fg[new].mean()) if new.any() else 1.0
                w[3] += float(pred_fg[empty].mean()) if empty.any() else 0.0
                if wave_rule_eval and tk in wave_tau:
                    tau = wave_tau[tk]
                    pred_r = (prob_fg_b[i].cpu().numpy() > tau)
                    wr = acc.wave_rule.setdefault(tk, [0, 0.0, 0.0, 0.0]); wr[0] += 1
                    wr[1] += float(pred_r[copied].mean()) if copied.any() else 1.0
                    wr[2] += float(pred_r[new].mean()) if new.any() else 1.0
                    wr[3] += float(pred_r[empty].mean()) if empty.any() else 0.0
            mk = markers_b[i]
            iou = _iou_np(pred_fg, tgt_fg)
            iou_d, prec1, rec1 = tolerant_line_metrics(pred_fg, tgt_fg)
            if tk in ("rays", "tighten") or tk.startswith("trajectories"):
                bld = (img[i, 10].float().view(g, s, g, s).sum(dim=(1, 3)) >= (s * s) // 2).cpu().numpy()   # surface class 3, majority per cell
                acc.rays_cells += int(pred_fg.sum()); acc.rays_on_bld += int((pred_fg & bld).sum())
            connected = pred_connects_markers(pred_fg, mk)
            gt_m = _gt_length_m(tk, straight[i], path_len[i])
            pred_m = pred_path_length_m(pred_fg, mk) if connected else 0.0
            len_ratio = (pred_m / gt_m) if connected and gt_m > 0 else None
            seg_iou = None
            if detour_ix is not None and int(task_ix[i]) == detour_ix:
                seg = segment_grid_from_markers(mk)
                seg_iou = _iou_np(pred_fg, seg)
            si = int(strata[i])
            buckets = [acc.overall, acc.bucket_task(tk), acc.bucket_stratum(si)]
            ll = _len_label(float(straight[i]))
            if ll is not None:
                buckets.append(acc.by_len.setdefault(ll, _Bucket()))
            for bucket in buckets:
                bucket.add(iou=iou, connected=connected, len_ratio=len_ratio, shortcut_iou=seg_iou,
                           iou_dil=iou_d, prec1=prec1, rec1=rec1)

    if acc.n_batches == 0:
        raise RuntimeError("eval_loader_c4: no batches")
    model.train()
    out = {
        "batches": acc.n_batches,
        "n_grids": acc.n_grids,
        "mean_loss": acc.loss_sum / acc.n_batches,
        "mean_iou_fg": acc.overall.mean_iou(),
        "connectivity": acc.overall.mean_conn(),
        "iou_dilated": acc.overall.mean_iou_dil(),
        "rays_on_building_share": (acc.rays_on_bld / acc.rays_cells) if acc.rays_cells else None,
        "precision_1cell": acc.overall.mean_prec1(),
        "recall_1cell": acc.overall.mean_rec1(),
        "length_ratio": acc.overall.mean_len_ratio(),
        "iou_pred_vs_segment": acc.overall.mean_shortcut_iou(),
        "c4_tasks": keys,
        "grid": _grid_size(),
    }
    for tk, (n, rc, rn, fe) in acc.wave.items():
        if n:
            out[f"wave_copied_recall_{tk}"] = rc / n
            out[f"wave_new_recall_{tk}"] = rn / n
            out[f"wave_false_fg_{tk}"] = fe / n
    if wave_rule_eval:
        for tk, tau in wave_tau.items():
            out[f"wave_decision_tau_{tk}"] = tau
        for tk, (n, rc, rn, fe) in acc.wave_rule.items():
            if n:
                out[f"wave_copied_recall_rule_{tk}"] = rc / n
                out[f"wave_new_recall_rule_{tk}"] = rn / n
                out[f"wave_false_fg_rule_{tk}"] = fe / n
    for tk, (n, ex, w1, err) in acc.tok.items():
        if n:
            out[f"tok_acc_{tk}"] = ex / n
            out[f"tok_within1_{tk}"] = w1 / n
            out[f"tok_mae_m_{tk}"] = LENGTH_BIN_M * err / n
    for tk, bucket in acc.by_task.items():
        out[f"iou_{tk}"] = bucket.mean_iou()
        out[f"iou_dilated_{tk}"] = bucket.mean_iou_dil()
        out[f"precision_1cell_{tk}"] = bucket.mean_prec1()
        out[f"recall_1cell_{tk}"] = bucket.mean_rec1()
        out[f"connectivity_{tk}"] = bucket.mean_conn()
        lr = bucket.mean_len_ratio()
        if lr is not None:
            out[f"length_ratio_{tk}"] = lr
        if tk == "detour":
            sc = bucket.mean_shortcut_iou()
            if sc is not None:
                out["iou_pred_vs_segment_detour"] = sc
    for ll, bucket in acc.by_len.items():
        out[f"iou_{ll}"] = bucket.mean_iou()
        out[f"iou_dilated_{ll}"] = bucket.mean_iou_dil()
        out[f"connectivity_{ll}"] = bucket.mean_conn()
        out[f"n_{ll}"] = bucket.iou_n
    for si, bucket in acc.by_stratum.items():
        label = STRATUM_LABELS[si] if si < len(STRATUM_LABELS) else f"stratum_{si}"
        out[f"iou_{label}"] = bucket.mean_iou()
        out[f"iou_dilated_{label}"] = bucket.mean_iou_dil()
        out[f"connectivity_{label}"] = bucket.mean_conn()
        lr = bucket.mean_len_ratio()
        if lr is not None:
            out[f"length_ratio_{label}"] = lr
        sc = bucket.mean_shortcut_iou()
        if sc is not None:
            out[f"iou_pred_vs_segment_{label}"] = sc
    if snap:
        return {k: out[k] for k in ("batches", "n_grids", "mean_loss", "mean_iou_fg") if k in out}
    return out


def flatten_for_eval_jsonl(val: dict) -> dict:
    row = {
        "mean_loss": round(val["mean_loss"], 6),
        "IoU_fg": round(val["mean_iou_fg"], 6),
        "c4_connectivity": round(val.get("connectivity", 0.0), 6),
        "c4_iou_dilated": round(val.get("iou_dilated", 0.0), 6),
        **({"c5_rays_on_building": round(val["rays_on_building_share"], 6)} if val.get("rays_on_building_share") is not None else {}),
        "c4_precision_1cell": round(val.get("precision_1cell", 0.0), 6),
        "c4_recall_1cell": round(val.get("recall_1cell", 0.0), 6),
    }
    if val.get("length_ratio") is not None:
        row["c4_length_ratio"] = round(val["length_ratio"], 6)
    if val.get("iou_pred_vs_segment") is not None:
        row["c4_iou_pred_vs_segment"] = round(val["iou_pred_vs_segment"], 6)
    for tk in val.get("c4_tasks", TASK_KEYS):
        for fk in (f"iou_{tk}", f"iou_dilated_{tk}"):
            if fk in val:
                row[fk] = round(float(val[fk]), 6)
    for label in STRATUM_LABELS:
        for fk in (f"iou_{label}", f"iou_dilated_{label}"):
            if fk in val:
                row[fk] = round(float(val[fk]), 6)
    for fk, v in val.items():
        if fk.startswith(("iou_len_", "iou_dilated_len_", "connectivity_len_", "n_len_")):
            row[fk] = round(float(v), 6)
        if fk.startswith(
            (
                "wave_copied_recall_",
                "wave_new_recall_",
                "wave_false_fg_",
                "wave_decision_tau_",
            )
        ):
            row[f"c5_{fk}"] = round(float(v), 6)
        if fk.startswith(
            ("wave_copied_recall_rule_", "wave_new_recall_rule_", "wave_false_fg_rule_")
        ):
            row[f"c5_{fk}"] = round(float(v), 6)
        if fk.startswith(("tok_acc_", "tok_within1_", "tok_mae_m_")):
            row[f"c5_{fk}"] = round(float(v), 6)
    return row


def print_c4_detail(val: dict, *, prefix: str = "    c4 detail:") -> None:
    parts = [f"conn={val.get('connectivity', 0):.3f}"]
    if val.get("length_ratio") is not None:
        parts.append(f"len_ratio={val['length_ratio']:.3f}")
    if val.get("iou_pred_vs_segment") is not None:
        parts.append(f"IoU(pred,segment)={val['iou_pred_vs_segment']:.3f}")
    print(prefix + " " + " ".join(parts))
    per = [f"{tk}={val.get(f'iou_{tk}', 0):.3f}" for tk in val.get("c4_tasks", []) if f"iou_{tk}" in val]
    if per:
        print(prefix + " per_task IoU: " + " ".join(per))
    tok = [f"{tk}: acc={val[f'tok_acc_{tk}']:.3f} within1={val[f'tok_within1_{tk}']:.3f} mae={val[f'tok_mae_m_{tk}']:.1f}m"
           for tk in val.get("c4_tasks", []) if f"tok_acc_{tk}" in val]
    if tok:
        print(prefix + " length token: " + " | ".join(tok))
    for label in STRATUM_LABELS:
        fk = f"iou_{label}"
        if fk in val:
            line = f"{label}: IoU={val[fk]:.3f} conn={val.get(f'connectivity_{label}', 0):.3f}"
            if f"length_ratio_{label}" in val:
                line += f" len={val[f'length_ratio_{label}']:.3f}"
            if f"iou_pred_vs_segment_{label}" in val:
                line += f" segIoU={val[f'iou_pred_vs_segment_{label}']:.3f}"
            print(prefix + " " + line)
