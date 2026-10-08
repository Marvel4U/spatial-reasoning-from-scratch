"""C5 rung 4: tighten task through the real loaders (skips without the store or the loose paths)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuda required", allow_module_level=True)

from spatial_data.c4_tasks import get_c4_task, C4TargetKind
from spatial_data.dataset_c4_gpu import (load_district_planes_v4, load_c4_route_index, C4GpuValLoader,
                                         C4MultiTaskGpuTrainLoader, polyline_scratch, _path_targets)

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data/amsterdam/de_pijp/crops/v4"
STORE = CROPS / "c4_100k"
pytestmark = pytest.mark.skipif(not (STORE / "c5_loose_val.npz").is_file(), reason="loose paths not built")


def _setup():
    dev = torch.device("cuda")
    return load_district_planes_v4(CROPS, dev), load_c4_route_index(STORE, "val", dev)


def test_tighten_spec():
    assert get_c4_task("tighten").target_kind == C4TargetKind.TIGHTEN


def test_tighten_batch_contents():
    district, val = _setup()
    vl = C4GpuValLoader(get_c4_task("tighten"), "onehot_v4_sp", 64, 32, district=district, route_index=val)
    img, tgt, tix, cond, meta = vl.next_batch_with_meta()
    ix = meta["route_ix"]
    assert (img[:, 15].flatten(1).sum(1) > 0).all(), "markers shown for tighten"
    ref = polyline_scratch(val["loose_poly"][ix], val["loose_n"][ix])
    assert torch.equal(img[:, 16], ref), "scratchpad = loose path"
    assert torch.equal(tgt, _path_targets(val["path_pix"][ix], val["n_pix"][ix], 64)), "target = stored tight path"
    assert (meta["tok_tgt"] == -1).all(), "no token target for tighten"


def test_loose_is_never_shorter_than_tight():
    z = np.load(STORE / "c5_loose_val.npz"); r = np.load(STORE / "c4_routes_val.npz")
    assert (z["loose_px"] >= r["path_px"] - 1e-3).all()


def test_mixed_loader_with_tighten():
    district, val = _setup()
    tl = C4MultiTaskGpuTrainLoader(["segment", "tighten"], "onehot_v4_sp", 64, 64, 5, district=district, route_index=val)
    img, tgt, tix, cond = tl.next_batch()
    assert img[tix == 0, 16].sum() == 0 and (img[tix == 1, 16].flatten(1).sum(1) > 0).all()
