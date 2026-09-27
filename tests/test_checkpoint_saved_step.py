import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from grid_vit_adapter import best_checkpoint_path, checkpoint_saved_step, max_checkpoint_steps
from plain_gpt_module.checkpoint import save_checkpoint


def test_series_step_from_filename_and_extra(tmp_path):
    series = tmp_path / "run_a"
    series.mkdir()
    p = series / "step500.pt"
    save_checkpoint(
        p,
        torch.nn.Linear(2, 2),
        training_stats=None,
        extra={"step": 500},
        verbose=False,
    )
    main = tmp_path / "run_a.pt"
    assert checkpoint_saved_step(p) == 500
    assert max_checkpoint_steps(main) == 500
    assert best_checkpoint_path(main) == p
