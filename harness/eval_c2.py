"""Per-task c2 evaluation (C2_PLAN §2 + §6 B1; balancing memo §6). Kept out of eval_spatial.py for length.

Never pools tasks: a conjunction and its parts have very different positive shares, so a
pooled mean hides exactly what c2 is about. Each task is scored at the plain 0.5 cut-off,
at the decision rule P(fg) > w/(1+w) that undoes the training weight, and at the best
cut-off of a sweep (oracle, diagnostic only). For two-atom tasks the wrong cells are
attributed to the atom whose own majority mask failed, which says whether errors come
from the weaker part or from the combination itself.
"""
from __future__ import annotations

import torch

SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)
_RULE_METRICS = ("iou_rule", "precision", "recall", "iou_pure_cells", "iou_mixed_cells")


def _planes_from_img(img: torch.Tensor, encoding: str) -> torch.Tensor:
    """(B, C, H, W) model input -> (B, 3, H, W) class indices 0..3."""
    if encoding == "scalar":
        return (img[:, :3] * 3.0).round().long().clamp(0, 3)
    return torch.stack([img[:, 4 * li: 4 * li + 4].argmax(dim=1) for li in range(3)], dim=1)


def _cell_counts(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, H, W) bool -> (B, G, G) count of true pixels per cell."""
    b, h, _ = mask.shape
    s = h // grid
    blk = mask.reshape(b, grid, s, grid, s).permute(0, 1, 3, 2, 4).reshape(b, grid, grid, s * s)
    return blk.sum(-1)


def atom_masks(planes: torch.Tensor, cond_ids: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (majority mask per atom (B,2,G,G) bool, pixel-AND mask (B,H,W) bool).

    A padding atom (-1) is "always true", so single-atom tasks flow through unchanged.
    """
    b, _, h, w = planes.shape
    s = h // grid
    lay = cond_ids.clamp_min(0) // 4
    cls = torch.where(cond_ids >= 0, cond_ids.clamp_min(0) % 4, torch.full_like(cond_ids, -1))
    sel = planes.gather(1, lay.view(b, 2, 1, 1).expand(b, 2, h, w))
    hold = (sel == cls.view(b, 2, 1, 1)) | (cond_ids < 0).view(b, 2, 1, 1)
    maj = torch.stack([_cell_counts(hold[:, i], grid) >= (s * s + 1) // 2 for i in range(2)], dim=1)
    return maj, hold[:, 0] & hold[:, 1]


def _masked_iou(pred_f: torch.Tensor, tgt_f: torch.Tensor, mask_f: torch.Tensor):
    m = mask_f & (tgt_f | pred_f)
    inter = (pred_f & tgt_f & m).sum(1).float()
    union = ((pred_f | tgt_f) & m).sum(1).float()
    return torch.where(union > 0, inter / union, torch.ones_like(inter)), m.any(1)


def new_accumulator() -> dict:
    a = {f"sweep_{t}": 0.0 for t in SWEEP}
    a.update({k: 0.0 for k in _RULE_METRICS})
    a.update({"n_nonempty": 0, "n_pure": 0, "n_mixed": 0, "n_empty": 0, "empty_fp": 0,
              "n_samples": 0, "wrong": 0, "only_a": 0, "only_b": 0, "both": 0, "neither": 0})
    return a


@torch.no_grad()
def accumulate_batch(acc: dict, prob_fg: torch.Tensor, tgt: torch.Tensor,
                     maj: torch.Tensor, pix_and: torch.Tensor, rule_thr: float) -> None:
    """Fold one single-task slice of a batch into that task's accumulator."""
    g = tgt.shape[-1]
    tgt_fg = tgt == 1
    nonempty = tgt_fg.flatten(1).any(1)
    acc["n_samples"] += int(tgt.shape[0])
    acc["n_empty"] += int((~nonempty).sum().item())
    for t in SWEEP:
        pr = prob_fg > t
        inter = (pr & tgt_fg).flatten(1).sum(1).float()
        union = (pr | tgt_fg).flatten(1).sum(1).float()
        iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
        acc[f"sweep_{t}"] += float(iou[nonempty].sum().item())
    pr = prob_fg > rule_thr
    if (~nonempty).any():
        acc["empty_fp"] += int(pr[~nonempty].flatten(1).any(1).sum().item())
    inter = (pr & tgt_fg).flatten(1).sum(1).float()
    union = (pr | tgt_fg).flatten(1).sum(1).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
    tp = inter
    fp = (pr & ~tgt_fg).flatten(1).sum(1).float()
    fn = (~pr & tgt_fg).flatten(1).sum(1).float()
    prec = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    rec = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
    acc["iou_rule"] += float(iou[nonempty].sum().item())
    acc["precision"] += float(prec[nonempty].sum().item())
    acc["recall"] += float(rec[nonempty].sum().item())
    acc["n_nonempty"] += int(nonempty.sum().item())
    s = pix_and.shape[-1] // g
    cnt = _cell_counts(pix_and, g)
    pure = (cnt == 0) | (cnt == s * s)
    pf, tf, mf = pr.flatten(1), tgt_fg.flatten(1), pure.flatten(1)
    iou_p, has_p = _masked_iou(pf, tf, mf)
    iou_m, has_m = _masked_iou(pf, tf, ~mf)
    acc["iou_pure_cells"] += float(iou_p[has_p].sum().item())
    acc["n_pure"] += int(has_p.sum().item())
    acc["iou_mixed_cells"] += float(iou_m[has_m].sum().item())
    acc["n_mixed"] += int(has_m.sum().item())
    wrong = pr != tgt_fg
    a_false, b_false = ~maj[:, 0], ~maj[:, 1]
    acc["wrong"] += int(wrong.sum().item())
    acc["only_a"] += int((wrong & a_false & ~b_false).sum().item())
    acc["only_b"] += int((wrong & b_false & ~a_false).sum().item())
    acc["both"] += int((wrong & a_false & b_false).sum().item())
    acc["neither"] += int((wrong & ~a_false & ~b_false).sum().item())


def _rule_iou_batch(prob_fg: torch.Tensor, tgt: torch.Tensor, rule_thr: float) -> torch.Tensor:
    """Per-sample rule IoU on non-empty targets; empty samples get NaN."""
    tgt_fg = tgt == 1
    nonempty = tgt_fg.flatten(1).any(1)
    pr = prob_fg > rule_thr
    inter = (pr & tgt_fg).flatten(1).sum(1).float()
    union = (pr | tgt_fg).flatten(1).sum(1).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
    return torch.where(nonempty, iou, torch.full_like(iou, float("nan")))


def finish(acc: dict, rule_thr: float) -> dict:
    n = max(acc["n_nonempty"], 1)
    sweep = {t: acc[f"sweep_{t}"] / n for t in SWEEP}
    best_thr = max(sweep, key=lambda t: sweep[t])
    wrong = max(acc["wrong"], 1)
    return {
        "iou_plain": sweep[0.5],
        "iou_rule": acc["iou_rule"] / n,
        "iou_oracle": sweep[best_thr],
        "oracle_cutoff": best_thr,
        "rule_cutoff": rule_thr,
        "precision": acc["precision"] / n,
        "recall": acc["recall"] / n,
        "empty_target_fp_rate": acc["empty_fp"] / acc["n_empty"] if acc["n_empty"] else 0.0,
        "empty_target_frac": acc["n_empty"] / max(acc["n_samples"], 1),
        "iou_pure_cells": acc["iou_pure_cells"] / max(acc["n_pure"], 1),
        "iou_mixed_cells": acc["iou_mixed_cells"] / max(acc["n_mixed"], 1),
        "err_only_a": acc["only_a"] / wrong,
        "err_only_b": acc["only_b"] / wrong,
        "err_both": acc["both"] / wrong,
        "err_neither": acc["neither"] / wrong,
        "n_nonempty": acc["n_nonempty"],
        "n_samples": acc["n_samples"],
    }


@torch.no_grad()
def evaluate_c2(model, loader, *, task_keys: list[str], weights: tuple[float, ...],
                encoding: str, held_out: tuple[str, ...] = (), autocast_dtype=None,
                max_batches: int | None = None) -> dict:
    """Full deterministic pass over `loader` (val crops x tasks); per-task metrics."""
    rule = [w / (1.0 + w) for w in weights]
    accs = {ti: new_accumulator() for ti in range(len(task_keys))}
    hold = set(held_out) & set(task_keys)
    held_out_ix = {i for i, k in enumerate(task_keys) if k in hold}
    wrong_acc = {k: {"correct": 0.0, "wrong": 0.0, "n": 0} for k in hold}
    ho_correct_sum = ho_wrong_sum = 0.0
    ho_n = 0
    was_training = model.training
    model.train(False)
    from spatial_batch import forward as model_forward
    from spatial_data.c2_tasks import get_c2_task, wrong_cond_ids_padded

    n_batches = 0
    for img, tgt, task_index, cond_ids in loader.iter_batches():
        dev_type = "cuda" if str(img.device).startswith("cuda") else "cpu"
        if autocast_dtype is not None:
            with torch.autocast(device_type=dev_type, dtype=autocast_dtype):
                out = model_forward(model, img, tgt, task_index, cond_ids)
        else:
            out = model_forward(model, img, tgt, task_index, cond_ids)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.softmax(logits.float(), dim=-1)[..., 1]
        planes = _planes_from_img(img, encoding)
        maj, pix_and = atom_masks(planes, cond_ids, tgt.shape[-1])
        for ti in torch.unique(task_index).tolist():
            m = task_index == ti
            accumulate_batch(accs[ti], prob[m], tgt[m], maj[m], pix_and[m], rule[ti])
        if held_out_ix:
            for bi in range(img.shape[0]):
                ti = int(task_index[bi].item())
                key = task_keys[ti]
                if ti not in held_out_ix:
                    continue
                sl = slice(bi, bi + 1)
                iou_c = _rule_iou_batch(prob[sl], tgt[sl], rule[ti])
                if torch.isnan(iou_c).all():
                    continue
                w_pair = wrong_cond_ids_padded(get_c2_task(key), slot=0)
                wrong_c = cond_ids[sl].clone()
                wrong_c[0, 0], wrong_c[0, 1] = w_pair[0], w_pair[1]
                if autocast_dtype is not None:
                    with torch.autocast(device_type=dev_type, dtype=autocast_dtype):
                        out_w = model_forward(model, img[sl], tgt[sl], task_index[sl], wrong_c)
                else:
                    out_w = model_forward(model, img[sl], tgt[sl], task_index[sl], wrong_c)
                logits_w = out_w[0] if isinstance(out_w, tuple) else out_w
                prob_w = torch.softmax(logits_w.float(), dim=-1)[..., 1]
                iou_w = _rule_iou_batch(prob_w, tgt[sl], rule[ti]).item()
                iou_c_v = iou_c.item()
                wrong_acc[key]["correct"] += iou_c_v
                wrong_acc[key]["wrong"] += iou_w
                wrong_acc[key]["n"] += 1
                ho_correct_sum += iou_c_v
                ho_wrong_sum += iou_w
                ho_n += 1
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train(was_training)
    per_task = {task_keys[ti]: finish(accs[ti], rule[ti]) for ti in accs}
    out = {"tasks": per_task,
           "mean_trained": mean_over(per_task, [k for k in task_keys if k not in hold])}
    if hold:
        out["mean_held_out"] = mean_over(per_task, [k for k in task_keys if k in hold])
        out["held_out_keys"] = sorted(hold)
        if ho_n:
            out["held_out_wrong_description"] = {
                k: {
                    "iou_rule_correct": v["correct"] / v["n"],
                    "iou_rule_wrong": v["wrong"] / v["n"],
                    "gap": (v["correct"] - v["wrong"]) / v["n"],
                    "n_nonempty": v["n"],
                    "wrong_cond_slot": 0,
                }
                for k, v in wrong_acc.items() if v["n"]
            }
            out["iou_held_out_correct"] = ho_correct_sum / ho_n
            out["iou_held_out_wrong_description"] = ho_wrong_sum / ho_n
            out["held_out_description_gap"] = out["iou_held_out_correct"] - out["iou_held_out_wrong_description"]
            out["held_out_wrong_description_n"] = ho_n
    return out


def mean_over(per_task: dict, keys: list[str]) -> dict:
    if not keys:
        return {}
    fields = ("iou_plain", "iou_rule", "iou_oracle", "precision", "recall",
              "empty_target_fp_rate", "iou_pure_cells", "iou_mixed_cells")
    return {f: sum(per_task[k][f] for k in keys) / len(keys) for f in fields}


def flatten_for_eval_jsonl(result: dict) -> dict:
    """`c2/<task_key>/<metric>` keys for runs/diagnostics/*.eval.jsonl."""
    flat: dict[str, float] = {}
    for key, m in result["tasks"].items():
        for name, v in m.items():
            flat[f"c2/{key}/{name}"] = round(float(v), 6) if isinstance(v, float) else v
    for group in ("mean_trained", "mean_held_out"):
        for name, v in result.get(group, {}).items():
            flat[f"c2/{group}/{name}"] = round(float(v), 6)
    for hk in ("iou_held_out_correct", "iou_held_out_wrong_description", "held_out_description_gap"):
        if hk in result:
            flat[f"c2/{hk}"] = round(float(result[hk]), 6)
    if "held_out_wrong_description_n" in result:
        flat["c2/held_out_wrong_description_n"] = int(result["held_out_wrong_description_n"])
    return flat


def print_table(result: dict, *, prefix: str = "  c2") -> None:
    head = (f"{'task':22s} {'thr':>5s} {'plain':>6s} {'rule':>6s} {'oracle':>6s} {'@':>5s} "
            f"{'P':>5s} {'R':>5s} {'emptFP':>6s} {'pure':>5s} {'mixed':>5s}  errA/B/both/none")
    print(f"{prefix}: per-task (IoU on non-empty targets)")
    print(f"{prefix}  {head}")
    for key, m in result["tasks"].items():
        print(f"{prefix}  {key:22s} {m['rule_cutoff']:5.3f} {m['iou_plain']:6.3f} "
              f"{m['iou_rule']:6.3f} {m['iou_oracle']:6.3f} {m['oracle_cutoff']:5.2f} "
              f"{m['precision']:5.2f} {m['recall']:5.2f} {m['empty_target_fp_rate']:6.2f} "
              f"{m['iou_pure_cells']:5.2f} {m['iou_mixed_cells']:5.2f}  "
              f"{m['err_only_a']:.2f}/{m['err_only_b']:.2f}/{m['err_both']:.2f}/{m['err_neither']:.2f}")
    for group in ("mean_trained", "mean_held_out"):
        g = result.get(group)
        if g:
            print(f"{prefix}  {group}: plain {g['iou_plain']:.3f} | rule {g['iou_rule']:.3f} "
                  f"| oracle {g['iou_oracle']:.3f}")
    wd = result.get("held_out_wrong_description")
    if wd:
        print(f"{prefix}: held-out wrong-description (score vs **correct** target, corrupt atom slot 0)")
        for key, m in wd.items():
            print(f"{prefix}  {key:22s} correct={m['iou_rule_correct']:.3f} "
                  f"wrong={m['iou_rule_wrong']:.3f} gap={m['gap']:+.3f} (n={m['n_nonempty']})")
        print(f"{prefix}  pooled: correct={result['iou_held_out_correct']:.3f} "
              f"wrong={result['iou_held_out_wrong_description']:.3f} "
              f"gap={result['held_out_description_gap']:+.3f} (n={result['held_out_wrong_description_n']})")
