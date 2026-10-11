"""End-to-end harness path for each C5 experiment spec: data → forward → diagnostics hooks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("cuda required", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness"
for p in (str(ROOT), str(HARNESS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
import data  # noqa: E402
import experiments  # noqa: E402
import grid_vit_adapter  # noqa: E402
from spatial_batch import forward as model_forward, next_batch as loader_next_batch  # noqa: E402
from diagnostics.probe_batch import ProbeBatch  # noqa: E402
from diagnostics.pred_strip import save_pred_strip  # noqa: E402
from diagnostics.tier2_eval import Tier2EvalDiagnostics  # noqa: E402
from spatial_data.c4_sidecars import c4_train_val_sidecars_ok  # noqa: E402

STORE = ROOT / "data/amsterdam/de_pijp/crops/v4/c4_100k"
pytestmark = pytest.mark.skipif(
    not (STORE / "c4_routes_val.npz").is_file(),
    reason="c4_100k route store missing",
)


def _apply_overrides(overrides: dict) -> dict:
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(config, k)
        setattr(config, k, v)
    return saved


def _restore_overrides(saved: dict) -> None:
    for k, v in saved.items():
        setattr(config, k, v)


def _rerun_smoke_cases():
    return [(s["id"], s) for s in experiments.c5_rerun_queue()]


@pytest.mark.parametrize("run_id,spec", _rerun_smoke_cases(), ids=[x[0] for x in _rerun_smoke_cases()])
def test_c5_harness_pipeline_smoke(run_id: str, spec: dict, tmp_path: Path):
    crops = ROOT / "data/amsterdam/de_pijp/crops/v4"
    routes_dir = crops / spec["overrides"]["c4_routes_subdir"]
    need = [crops / "district_labels_1m.npz", routes_dir / "c4_routes_train.npz", routes_dir / "c4_routes_val.npz"]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.is_file()]
    if spec["overrides"].get("c4_mix_c3_task_keys") and not any((crops / "train").glob("*_labels.npz")):
        missing.append("v4 train crops (*_labels.npz)")
    if missing:
        pytest.skip(f"{run_id}: needs {missing}")
    keys = spec["overrides"]["c4_task_keys"]
    assert c4_train_val_sidecars_ok(routes_dir, keys), f"{run_id}: missing train/val sidecars under {routes_dir} for {keys}"
    g = spec["overrides"].get("grid_out_size", config.grid_out_size)

    saved = _apply_overrides(spec["overrides"])
    config.diagnostics_pred_enabled = True
    try:
        config.sync_in_chans_from_encoding()
        data.setup_device()
        data.load_data()
        train_b = loader_next_batch(data.train_loader)
        val_b = loader_next_batch(data.val_loader)
        assert len(train_b) in (4, 5), f"{run_id}: train batch tuple len {len(train_b)}"
        assert len(val_b) in (4, 5), f"{run_id}: val batch tuple len {len(val_b)}"
        assert config.grid_out_size == g, run_id

        model = grid_vit_adapter.build_model(device=data.device).to(data.device)
        img, tgt, tid, cid, *rest = train_b
        tok_tgt = rest[0] if rest else None
        dev = data.device
        img, tgt = img.to(dev), tgt.to(dev)
        tid = tid.to(dev) if tid is not None else None
        cid = cid.to(dev) if cid is not None else None
        if tok_tgt is not None:
            tok_tgt = tok_tgt.to(dev)
        logits, loss = model_forward(model, img, tgt, tid, cid, tok_tgt=tok_tgt)
        assert logits.shape[:3] == tgt.shape, run_id
        assert loss is not None and loss.ndim == 0, run_id
        if "tokonly" in run_id:
            from spatial_data.c4_tasks import IGNORE_TARGET
            assert (tgt == IGNORE_TARGET).any(), f"{run_id}: expect some IGNORE_TARGET cells (token-only length)"

        probe = ProbeBatch.build_c4_from_val_loader(data.val_loader)
        assert all(ix < data.val_loader.n_routes for ix in probe.item_indices), probe.item_indices
        if "tokonly" in run_id:
            pred_len = sum(1 for i in probe.pred_indices if int(probe.tok_tgt[i]) >= 0)
            pred_mask = len(probe.pred_indices) - pred_len
            assert pred_len >= 1 and pred_mask >= 1, (pred_len, pred_mask, probe.pred_indices)
        save_pred_strip(model, probe, step=0, out_dir=tmp_path, run_id=spec["id"])
        assert list(tmp_path.glob("pred_step*.png")), run_id

        t2 = Tier2EvalDiagnostics(spec["id"], out_dir=tmp_path)
        ref_png = t2.maybe_save_c4_reference_pred_strip(model, 0)
        ref_dir = crops / config.diagnostics_pred_c4_reference_subdir
        from spatial_data.c4_sidecars import c4_val_sidecars_ok

        if not c4_val_sidecars_ok(ref_dir, keys):
            assert ref_png is None, f"{run_id}: reference strip should skip when {ref_dir} lacks sidecars"
    finally:
        _restore_overrides(saved)
        data.m = None
        data.optimizer = None
        data.train_loader = None
        data.val_loader = None
