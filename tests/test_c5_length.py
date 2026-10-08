"""C5 rungs 2-3: length tasks through the real loaders (skips without the 100k store)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuda required", allow_module_level=True)

from spatial_data.c4_tasks import get_c4_task, length_bin, N_LENGTH_BINS
from spatial_data.dataset_c4_gpu import (load_district_planes_v4, load_c4_route_index, C4GpuValLoader,
                                         C4MultiTaskGpuTrainLoader)

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data/amsterdam/de_pijp/crops/v4"
STORE = CROPS / "c4_100k"
pytestmark = pytest.mark.skipif(not (STORE / "c4_routes_val.npz").is_file(), reason="c4_100k store missing")


def _setup():
    dev = torch.device("cuda")
    return load_district_planes_v4(CROPS, dev), load_c4_route_index(STORE, "val", dev)


def test_length_bin():
    assert length_bin(0.0) == 0 and length_bin(9.99) == 0 and length_bin(10.0) == 1 and length_bin(1e4) == N_LENGTH_BINS - 1


def test_length_path_full_matches_store():
    district, val = _setup()
    vl = C4GpuValLoader(get_c4_task("length_path"), "onehot_v4_sp", 64, 32, district=district, route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    tok = meta["tok_tgt"]
    assert tok is not None and (tok >= 0).all() and (tok < N_LENGTH_BINS).all()
    assert img[:, 15].sum() == 0, "markers hidden for length tasks"
    assert (img[:, 16].flatten(1).sum(1) > 0).all(), "scratchpad drawn"
    assert (tgt == -100).all(), "no mask target: every cell ignored"
    # samples whose drawn path is the full stored path must carry the stored length's bin
    full = [i for i in range(img.shape[0]) if int(img[i, 16].sum()) >= int(0.9 * float(meta["path_px"][i]))]
    assert full, "some full paths expected in a batch of 32"
    for i in full:
        assert abs(int(tok[i]) - int(length_bin(meta["path_px"][i]))) <= 1


def test_length_line_token_matches_drawn_segment():
    district, val = _setup()
    vl = C4GpuValLoader(get_c4_task("length_line"), "onehot_v4_sp", 64, 32, district=district, route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    tok = meta["tok_tgt"].cpu().numpy()
    for i in range(img.shape[0]):
        ys, xs = np.nonzero(img[i, 16].cpu().numpy())
        extent = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())       # drawn pixels span the segment
        assert abs(tok[i] - int(length_bin(extent))) <= 1, (i, tok[i], extent)


def test_mixed_train_loader_returns_tokens():
    district, val = _setup()
    tl = C4MultiTaskGpuTrainLoader(["segment", "length_line", "length_path"], "onehot_v4_sp", 64, 64, 3,
                                   district=district, route_index=val)
    tl.return_tokens = True
    img, tgt, tix, cond, tok = tl.next_batch()
    seg = tix == 0
    assert (tok[seg] == -1).all() and (tok[~seg] >= 0).all()
    assert img[seg, 15].flatten(1).sum(1).gt(0).all(), "segment samples keep their markers"
    assert img[seg, 16].sum() == 0, "segment samples have an empty scratchpad"
