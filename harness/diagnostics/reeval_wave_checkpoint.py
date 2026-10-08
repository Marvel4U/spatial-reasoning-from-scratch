#!/usr/bin/env python3
"""Full val CE eval + tier-2 row (incl. decision rule) for an existing checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config
import data
import eval_spatial
import grid_vit_adapter
from diagnostics.paths import run_dir
from diagnostics.tier2_eval import Tier2EvalDiagnostics


def _overrides_for_run(run_id: str) -> dict:
    path = Path(config.RESULTS_PATH)
    if not path.is_file():
        return {}
    for r in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
        if r.get("id") == run_id:
            return dict(r.get("overrides") or {})
    return {}


def _replace_eval_row(eval_path: Path, row: dict) -> None:
    rows: list[dict] = []
    if eval_path.is_file():
        for line in eval_path.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    step = row["step"]
    rows = [r for r in rows if r.get("step") != step]
    rows.append(row)
    rows.sort(key=lambda r: int(r["step"]))
    with eval_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-run val eval on a harness checkpoint (current eval code)")
    p.add_argument("run_id")
    p.add_argument("--step", type=int, default=11999)
    p.add_argument(
        "--no-decision-rule",
        action="store_true",
        help="leave c5_wave_eval_decision_rule as in saved run overrides",
    )
    args = p.parse_args(argv)
    ckpt = Path(config.harness_checkpoint_dir) / f"{args.run_id}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    saved: dict = {}
    for k, v in _overrides_for_run(args.run_id).items():
        if hasattr(config, k):
            saved[k] = getattr(config, k)
            setattr(config, k, v)
    if not args.no_decision_rule:
        saved["c5_wave_eval_decision_rule"] = getattr(config, "c5_wave_eval_decision_rule", False)
        config.c5_wave_eval_decision_rule = True
    try:
        config.sync_in_chans_from_encoding()
        data.setup_device()
        data.load_data()
        model = grid_vit_adapter.build_model()
        grid_vit_adapter.load_checkpoint(ckpt, model=model)
        model.eval()
        iters = config.eval_iters_final
        losses = grid_vit_adapter.estimate_loss(model, eval_iters=iters)
        t2 = Tier2EvalDiagnostics(args.run_id)
        t2.record_ce_eval(
            args.step,
            model,
            eval_iters=iters,
            train_loss=losses["train"],
            val_loss=losses["val"],
        )
        eval_path = run_dir(args.run_id) / "eval.jsonl"
        last = json.loads(eval_path.read_text(encoding="utf-8").strip().split("\n")[-1])
        last["reeval_at"] = datetime.now(timezone.utc).isoformat()
        _replace_eval_row(eval_path, last)
        print(f"updated {eval_path} @ step {args.step}")
        print(
            f"  val={losses['val']:.4f}  rule new step="
            f"{last.get('c5_wave_new_recall_rule_wave_step')}  dil step={last.get('iou_dilated_wave_step')}"
        )
        result = eval_spatial.eval_model(model, max_batches=None)
        eval_spatial.print_report(result)
        return 0
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


if __name__ == "__main__":
    raise SystemExit(main())
