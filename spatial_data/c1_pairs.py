"""Fixed (crop, task) val pairs for reproducible c1 eval (C1_PLAN §8)."""
from __future__ import annotations

import random


def fixed_val_pairs(
    n_crops: int,
    n_tasks: int,
    n_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    if n_crops < 1 or n_tasks < 1:
        raise ValueError("n_crops and n_tasks must be >= 1")
    rng = random.Random(seed)
    return [(rng.randrange(n_crops), rng.randrange(n_tasks)) for _ in range(n_pairs)]
