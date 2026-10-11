"""C3_PLAN N1: geometry oracle on val loader (marker tasks)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"
DIST = ROOT / "data" / "amsterdam" / "de_pijp"


def _has_crops(split: str) -> bool:
    """True if the v4 cropset has *_labels.npz crop files for `split` (what load_crop_planes_v4 reads)."""
    return any((CROPS / split).glob("*_labels.npz"))


def _has_markers(split: str) -> bool:
    """Crop files plus the t0 marker index that marker_table_for_split reads."""
    return _has_crops(split) and (DIST / "tasks" / "t0_point_v0" / f"{split}.jsonl").is_file()


def test_geometry_oracle_within_20m_val():
    if not (_has_markers("val") and _has_markers("train")):
        pytest.skip("v4 train/val crops or t0_point_v0 marker index missing")
    if not __import__("torch").cuda.is_available():
        pytest.skip("cuda required")
    import config
    import data
    import eval_spatial

    data.setup_device()
    config.task_rung = "c3"
    config.data_source = "worldsnap"
    config.cropset = "v4"
    config.c3_task_key = "within_20m"
    config.encoding_mode = "onehot_v4"
    config.sync_in_chans_from_encoding()
    config.batch_size = 16
    data.load_data()
    geo = eval_spatial._c3_geometry_oracle_iou(data.val_loader, max_batches=None)
    assert geo is not None
    assert geo["c3_loader_label_agreement"] == 1.0
    assert geo["c3_geometry_oracle_iou"] >= 0.999
