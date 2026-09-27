"""Remove on-disk artifacts for one harness run_id (diagnostics + checkpoints)."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import config
from diagnostics.paths import diagnostics_root, run_dir

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_run_id(run_id: str) -> None:
    if not run_id or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id {run_id!r} (use letters, digits, _, -)")


def paths_for_run(
    run_id: str,
    *,
    diag_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    series_dir: Path | None = None,
    include_main_checkpoint: bool = True,
) -> list[Path]:
    """All file/dir paths this helper may delete for run_id (existing or not)."""
    _validate_run_id(run_id)
    root = Path(diag_dir or config.diagnostics_dir)
    checkpoint_dir = Path(checkpoint_dir or config.harness_checkpoint_dir)
    series_dir = Path(series_dir or config.diagnostics_checkpoint_dir)
    out: list[Path] = [
        run_dir(run_id),
        root / f"{run_id}.jsonl",
        root / f"{run_id}.eval.jsonl",
        root / f"{run_id}.meta.json",
        root / f"{run_id}_plots.png",
        root / f"{run_id}_harness.log",
        series_dir / run_id,
    ]
    if include_main_checkpoint:
        out.append(checkpoint_dir / f"{run_id}.pt")
    for p in sorted(root.glob(f"{run_id}_vs_*_compare.png")):
        out.append(p)
    for p in sorted(root.glob(f"*_vs_{run_id}_compare.png")):
        out.append(p)
    rd = run_dir(run_id)
    if rd.is_dir():
        for p in sorted(rd.glob("compare_*.png")):
            out.append(p)
    return out


def delete_run_artifacts(
    run_id: str,
    *,
    dry_run: bool = False,
    include_main_checkpoint: bool = True,
    diag_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    series_dir: Path | None = None,
) -> list[Path]:
    """
    Delete diagnostics under runs/diagnostics/{run_id}/, legacy flat files, checkpoint series,
    and optionally runs/checkpoints/{run_id}.pt.

    Does not edit runs/experiments.json.
    """
    removed: list[Path] = []
    rd = run_dir(run_id)
    if rd.is_dir():
        removed.append(rd)
        if not dry_run:
            shutil.rmtree(rd)
    for path in paths_for_run(
        run_id,
        diag_dir=diag_dir,
        checkpoint_dir=checkpoint_dir,
        series_dir=series_dir,
        include_main_checkpoint=include_main_checkpoint,
    ):
        if path == rd or not path.exists():
            continue
        removed.append(path)
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Delete diagnostics and checkpoint files for one harness run_id",
    )
    p.add_argument("run_id", help="experiment id (e.g. c0_worldsnap_P16_2k)")
    p.add_argument("-n", "--dry-run", action="store_true", help="list paths only")
    p.add_argument(
        "--keep-main-checkpoint",
        action="store_true",
        help="keep runs/checkpoints/{run_id}.pt (still removes step*.pt series)",
    )
    p.add_argument("--diag-dir", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        paths = delete_run_artifacts(
            args.run_id,
            dry_run=args.dry_run,
            include_main_checkpoint=not args.keep_main_checkpoint,
            diag_dir=args.diag_dir,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    label = "would remove" if args.dry_run else "removed"
    if not paths:
        print(f"no files found for run_id={args.run_id!r}")
        return 0
    for path in paths:
        print(f"{label}: {path}")
    print(f"{label} {len(paths)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
