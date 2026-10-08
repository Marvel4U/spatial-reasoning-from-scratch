"""C5 npz sidecars next to a c4 route store (fans for rays, loose paths for tighten)."""
from __future__ import annotations

from pathlib import Path


def _missing_sidecars_for_split(routes_dir: Path, keys: set[str], split: str) -> list[str]:
    missing: list[str] = []
    if "rays" in keys and not (routes_dir / f"c5_fans_{split}.npz").is_file():
        missing.append(f"c5_fans_{split}.npz")
    if "tighten" in keys and not (routes_dir / f"c5_loose_{split}.npz").is_file():
        missing.append(f"c5_loose_{split}.npz")
    if any(k.startswith("wave_") for k in keys) and not (routes_dir / f"c5_wave_{split}.npz").is_file():
        missing.append(f"c5_wave_{split}.npz")
    for k in keys:
        if k.startswith("trajectories"):
            sfx = "_r40" if k.endswith("_r40") else ""
            if not (routes_dir / f"c5_traj{sfx}_{split}.npz").is_file():
                missing.append(f"c5_traj{sfx}_{split}.npz")
    return missing


def c4_val_sidecars_ok(routes_dir: Path | str, task_keys: list[str]) -> bool:
    """True if every active task can build val targets from files under ``routes_dir``."""
    return not _missing_sidecars_for_split(Path(routes_dir), set(task_keys), "val")


def c4_train_val_sidecars_ok(routes_dir: Path | str, task_keys: list[str]) -> bool:
    """True if train and val splits have sidecars for all active C5 tasks (harness CE eval uses both)."""
    p = Path(routes_dir)
    keys = set(task_keys)
    for split in ("train", "val"):
        if _missing_sidecars_for_split(p, keys, split):
            return False
    return True


def assert_c4_train_val_sidecars(routes_dir: Path | str, task_keys: list[str]) -> None:
    """Fail fast before step 0 if sidecars are missing (common when a long train build is still running)."""
    p = Path(routes_dir)
    keys = set(task_keys)
    missing: list[str] = []
    for split in ("train", "val"):
        missing.extend(_missing_sidecars_for_split(p, keys, split))
    if missing:
        raise SystemExit(
            f"C5 sidecars missing under {p}:\n  "
            + "\n  ".join(sorted(set(missing)))
            + "\nBuild them before training (see spatial_data/c5_trajectories.py, c5_fans, c5_loose)."
        )
