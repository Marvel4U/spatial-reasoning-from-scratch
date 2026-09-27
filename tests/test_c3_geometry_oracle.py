"""C3_PLAN N1: geometry oracle on val loader (marker tasks)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

CROPS = ROOT / "data" / "amsterdam" / "de_pijp" / "crops" / "v4"


def test_geometry_oracle_within_20m_val():
    if not CROPS.is_dir():
        pytest.skip("v4 missing")
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
