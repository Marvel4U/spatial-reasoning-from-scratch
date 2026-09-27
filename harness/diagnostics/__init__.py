"""Training diagnostics (ANALYSIS_SUITE_SPEC tier 1+)."""
from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class RunDiagnostics:
    tier1: "Tier1Diagnostics"
    tier2: "Tier2EvalDiagnostics"
    ckpt: "CheckpointSeries"


def maybe_run_diagnostics(run_id: str) -> RunDiagnostics | None:
    if not config.diagnostics_enabled:
        return None
    from .checkpoint_series import CheckpointSeries
    from .tier1 import Tier1Diagnostics
    from .tier2_eval import Tier2EvalDiagnostics
    t1 = Tier1Diagnostics(run_id)
    t2 = Tier2EvalDiagnostics(run_id, meta_path=t1.meta_path)
    return RunDiagnostics(t1, t2, CheckpointSeries(run_id))


def maybe_tier1(run_id: str):
    d = maybe_run_diagnostics(run_id)
    return d.tier1 if d else None
