import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

from eval_c4 import wave_decision_tau


def test_wave_decision_tau_from_stats(monkeypatch):
    import config

    monkeypatch.setattr(config, "c4_weight_alpha", 0.5)
    monkeypatch.setattr(config, "c4_task_stats", lambda: {"tasks": {"wave_step": {"f": 0.019}}})
    t = wave_decision_tau("wave_step")
    assert 0.5 < t < 1.0
    assert abs(t - 7.0 / 8.0) < 0.05
