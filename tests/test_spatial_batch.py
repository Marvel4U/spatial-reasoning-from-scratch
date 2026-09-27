"""spatial_batch tuple unpacking."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from spatial_batch import forward, next_batch  # noqa: E402


class _L:
    def __init__(self, batch):
        self._batch = batch

    def next_batch(self):
        return self._batch


def test_next_batch_lengths():
    img = torch.zeros(2, 4, 8, 8)
    tgt = torch.zeros(2, 64, 64, dtype=torch.long)
    tid = torch.zeros(2, dtype=torch.long)
    cid = torch.zeros(2, 2, dtype=torch.long)
    a, b, c, d = next_batch(_L((img, tgt)))
    assert c is None and d is None
    a, b, c, d = next_batch(_L((img, tgt, tid)))
    assert d is None
    a, b, c, d = next_batch(_L((img, tgt, tid, cid)))
    assert d is not None
