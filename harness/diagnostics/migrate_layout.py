"""One-time (or repeat-safe) move flat diagnostics files into runs/diagnostics/{run_id}/."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parents[1]
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import config
from diagnostics.paths import (
    SKIP_TOP_LEVEL,
    compare_path,
    dashboard_path,
    diagnostics_root,
    eval_path,
    harness_log_path,
    meta_path,
    run_dir,
    summary_path,
    tier1_path,
)

_COMPARE_RE = re.compile(r"^(.+)_vs_(.+)_compare\.png$")


def _discover_run_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for f in root.glob("*.jsonl"):
        name = f.name
        if name.endswith(".eval.jsonl"):
            ids.add(name[: -len(".eval.jsonl")])
        elif name.endswith(".jsonl"):
            ids.add(name[: -len(".jsonl")])
    for f in root.glob("*.meta.json"):
        ids.add(f.name[: -len(".meta.json")])
    for d in root.iterdir():
        if not d.is_dir() or d.name in SKIP_TOP_LEVEL:
            continue
        if any(d.glob("pred_step*.png")) or any(
            (d / n).is_file() for n in ("tier1.jsonl", "eval.jsonl", "meta.json")
        ):
            ids.add(d.name)
    return ids


def _move(src: Path, dst: Path, *, dry_run: bool) -> bool:
    if not src.is_file() or src == dst:
        return False
    if dst.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  would move {src.name} → {dst.relative_to(diagnostics_root())}")
    else:
        shutil.move(str(src), str(dst))
        print(f"  moved {src.name} → {dst.relative_to(diagnostics_root())}")
    return True


def migrate_run(run_id: str, *, dry_run: bool) -> int:
    root = diagnostics_root()
    n = 0
    rd = run_dir(run_id, create=not dry_run)
    pairs = [
        (root / f"{run_id}.jsonl", tier1_path(run_id)),
        (root / f"{run_id}.eval.jsonl", eval_path(run_id)),
        (root / f"{run_id}.meta.json", meta_path(run_id)),
        (root / f"{run_id}_plots.png", dashboard_path(run_id)),
        (root / f"{run_id}_harness.log", harness_log_path(run_id)),
        (rd / "diagnostics_summary.png", summary_path(run_id)),
    ]
    for src, dst in pairs:
        if _move(src, dst, dry_run=dry_run):
            n += 1
    return n


def migrate_compare_pngs(*, dry_run: bool) -> int:
    root = diagnostics_root()
    n = 0
    for f in sorted(root.glob("*_vs_*_compare.png")):
        m = _COMPARE_RE.match(f.name)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        dst = compare_path(a, b)
        if _move(f, dst, dry_run=dry_run):
            n += 1
    return n


def migrate_adhoc_pngs(*, dry_run: bool) -> int:
    """Root-level view/gallery PNGs → _adhoc/."""
    root = diagnostics_root()
    adhoc = root / "_adhoc"
    n = 0
    for pat in ("view_*.png", "compare_*.png", "gallery_*.png", "c1_*.png", "series_*.png"):
        for f in root.glob(pat):
            if f.parent != root:
                continue
            dst = adhoc / f.name
            if _move(f, dst, dry_run=dry_run):
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Move diagnostics artifacts into per-run folders")
    p.add_argument("-n", "--dry-run", action="store_true")
    p.add_argument("--run-id", default=None, help="single run (default: all discovered)")
    args = p.parse_args(argv)
    root = diagnostics_root()
    ids = {args.run_id} if args.run_id else _discover_run_ids(root)
    total = 0
    for rid in sorted(ids):
        if rid in SKIP_TOP_LEVEL:
            continue
        print(f"{rid}:")
        total += migrate_run(rid, dry_run=args.dry_run)
    total += migrate_compare_pngs(dry_run=args.dry_run)
    total += migrate_adhoc_pngs(dry_run=args.dry_run)
    print(f"{'would move' if args.dry_run else 'moved'} {total} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
