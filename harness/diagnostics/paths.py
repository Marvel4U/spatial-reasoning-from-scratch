"""Layout: runs/diagnostics/{run_id}/tier1.jsonl, eval.jsonl, meta.json, dashboard.png, pred_step*.png."""
from __future__ import annotations

from pathlib import Path

import config

SKIP_TOP_LEVEL = frozenset({"archive", "l6c_task_gallery", "_adhoc"})


def diagnostics_root() -> Path:
    return Path(config.diagnostics_dir)


def run_dir(run_id: str, *, create: bool = False) -> Path:
    p = diagnostics_root() / run_id
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def tier1_path(run_id: str) -> Path:
    return run_dir(run_id) / "tier1.jsonl"


def eval_path(run_id: str) -> Path:
    return run_dir(run_id) / "eval.jsonl"


def meta_path(run_id: str) -> Path:
    return run_dir(run_id) / "meta.json"


def dashboard_path(run_id: str) -> Path:
    return run_dir(run_id) / "dashboard.png"


def summary_path(run_id: str) -> Path:
    return run_dir(run_id) / "summary.png"


def harness_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "harness.log"


def compare_path(primary_run_id: str, other_run_id: str) -> Path:
    return run_dir(primary_run_id) / f"compare_{other_run_id}.png"


def _legacy_tier1(run_id: str) -> Path:
    return diagnostics_root() / f"{run_id}.jsonl"


def _legacy_eval(run_id: str) -> Path:
    return diagnostics_root() / f"{run_id}.eval.jsonl"


def _legacy_meta(run_id: str) -> Path:
    return diagnostics_root() / f"{run_id}.meta.json"


def _legacy_dashboard(run_id: str) -> Path:
    return diagnostics_root() / f"{run_id}_plots.png"


def _legacy_summary(run_id: str) -> Path:
    return run_dir(run_id) / "diagnostics_summary.png"


def resolve_tier1(run_id: str) -> Path | None:
    for p in (tier1_path(run_id), _legacy_tier1(run_id)):
        if p.is_file():
            return p
    return None


def resolve_eval(run_id: str) -> Path | None:
    for p in (eval_path(run_id), _legacy_eval(run_id)):
        if p.is_file():
            return p
    return None


def resolve_meta(run_id: str) -> Path | None:
    for p in (meta_path(run_id), _legacy_meta(run_id)):
        if p.is_file():
            return p
    return None
