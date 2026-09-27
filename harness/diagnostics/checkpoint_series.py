"""Tier 2(d): log-spaced weights-only checkpoints."""
from __future__ import annotations

from pathlib import Path

import config
from plain_gpt_module.checkpoint import save_checkpoint


class CheckpointSeries:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = Path(config.diagnostics_checkpoint_dir) / run_id
        self.steps = set(int(s) for s in config.diagnostics_checkpoint_steps)

    def should_save(self, step: int) -> bool:
        return step in self.steps

    def save(self, step: int, model) -> Path | None:
        if not self.should_save(step):
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"step{step}.pt"
        save_checkpoint(
            path, model, optimizer=None, training_stats=None,
            model_config=None, extra={"run_id": self.run_id, "step": step},
            verbose=False,
        )
        config.log(f"  diagnostics: checkpoint series → {path}", force=True)
        return path
