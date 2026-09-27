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
    M_PER_CELL,
    STRATUM_LABELS,
    TASK_KEYS,
    path_pix_to_grid_numpy,
    rasterise_segment_numpy,
)


def _grid_size() -> int:
    return config.grid_out_size


def _marker_grid_cells(markers_xy: np.ndarray, *, grid: int = 64, img: int = 256) -> tuple[tuple[int, int], tuple[int, int]]:
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
    return float((lab == cid).sum()) * M_PER_CELL


def _dilate(a: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(a, structure=np.ones((3, 3), dtype=bool))


def tolerant_line_metrics(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, float, float]:
    """For one-cell-wide targets: IoU after dilating both by one cell; precision and recall within one cell."""
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
    pix = rasterise_segment_numpy(markers_xy[0], markers_xy[1])
    return path_pix_to_grid_numpy(pix)


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
    loss_sum: float = 0.0
    n_batches: int = 0
    n_grids: int = 0

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
    key_by_ix = {i: k for i, k in enumerate(keys)}
    detour_ix = keys.index("detour") if "detour" in keys else None
    acc = _C4Accum()

    for img, tgt, task_id, cond_ids, meta in _iter_c4_val(loader, max_batches):
        img, tgt = img.to(device), tgt.to(device)
        if task_id is not None:
            task_id = task_id.to(device)
        if cond_ids is not None:
            cond_ids = cond_ids.to(device)
        with torch.autocast(device_type=device_type, dtype=data.dtype):
            logits, loss = model_forward(model, img, tgt, task_id, cond_ids)
        acc.loss_sum += loss.item()
        acc.n_batches += 1
        pred = logits.argmax(dim=-1)
        b = pred.shape[0]
        acc.n_grids += b
        markers_b = meta["markers"].detach().cpu().numpy()
        strata = meta["stratum"].detach().cpu().numpy()
        straight = meta["straight_px"].detach().cpu().numpy()
        path_len = meta["path_px"].detach().cpu().numpy()
        task_ix = meta["task_ix"].detach().cpu().numpy()

        for i in range(b):
            tk = key_by_ix[int(task_ix[i])]
            pred_fg = (pred[i].cpu().numpy() == fg_class)
            tgt_fg = (tgt[i].cpu().numpy() == fg_class)
            mk = markers_b[i]
            iou = _iou_np(pred_fg, tgt_fg)
            iou_d, prec1, rec1 = tolerant_line_metrics(pred_fg, tgt_fg)
            connected = pred_connects_markers(pred_fg, mk)
            gt_m = _gt_length_m(tk, straight[i], path_len[i])
            pred_m = pred_path_length_m(pred_fg, mk) if connected else 0.0
            len_ratio = (pred_m / gt_m) if connected and gt_m > 0 else None
            seg_iou = None
            if detour_ix is not None and int(task_ix[i]) == detour_ix:
                seg = segment_grid_from_markers(mk)
                seg_iou = _iou_np(pred_fg, seg)
            si = int(strata[i])
            for bucket in (acc.overall, acc.bucket_task(tk), acc.bucket_stratum(si)):
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
        "precision_1cell": acc.overall.mean_prec1(),
        "recall_1cell": acc.overall.mean_rec1(),
        "length_ratio": acc.overall.mean_len_ratio(),
        "iou_pred_vs_segment": acc.overall.mean_shortcut_iou(),
        "c4_tasks": keys,
    }
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
    for label in STRATUM_LABELS:
        fk = f"iou_{label}"
        if fk in val:
            line = f"{label}: IoU={val[fk]:.3f} conn={val.get(f'connectivity_{label}', 0):.3f}"
            if f"length_ratio_{label}" in val:
                line += f" len={val[f'length_ratio_{label}']:.3f}"
            if f"iou_pred_vs_segment_{label}" in val:
                line += f" segIoU={val[f'iou_pred_vs_segment_{label}']:.3f}"
            print(prefix + " " + line)
