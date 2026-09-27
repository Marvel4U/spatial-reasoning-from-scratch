"""Unpack harness batch loaders (c0: 2-tuple, c1: 3-tuple, c2: 4-tuple)."""
from __future__ import annotations

import torch


def next_batch(
    loader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch = loader.next_batch()
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    img, tgt = batch
    return img, tgt, None, None


def forward(
    model,
    img,
    tgt,
    task_id: torch.Tensor | None = None,
    cond_ids: torch.Tensor | None = None,
    *,
    encoder_self_attn_only: bool = False,
    encoder_patch_local_only: bool = False,
):
    kw: dict = {
        "encoder_self_attn_only": encoder_self_attn_only,
        "encoder_patch_local_only": encoder_patch_local_only,
    }
    if task_id is not None:
        kw["task_id"] = task_id
    if cond_ids is not None:
        kw["cond_ids"] = cond_ids
    if tgt is None and task_id is None and cond_ids is None and not encoder_self_attn_only:
        return model(img, None)
    if tgt is None:
        return model(img, None, **kw)
    if task_id is None and cond_ids is None and not encoder_self_attn_only:
        return model(img, tgt)
    return model(img, tgt, **kw)
