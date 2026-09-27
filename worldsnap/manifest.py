"""Snapshot manifest: provenance for every stored layer.

WHY
---
The artifact's credibility rests on verifiable answers, and a verifiable answer
is only as good as the traceability of the data it was computed from. Open geo
datasets are *mutable*: 3DBAG republishes, PDOK re-surveys, city portals
explicitly do not preserve history. Six months from now "the building heights in
De Pijp" is meaningless unless we recorded which release we downloaded and can
prove the bytes did not change. Hence one JSON per district, one entry per
layer, with source URL, licence, download time, provider-stated vintage, CRS,
bbox, feature count and a sha256 of the file on disk.

The file is written sorted and with fixed indentation so that a re-run producing
identical data produces an identical diff (apart from the timestamp), which
makes accidental data drift visible in review.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file, as lowercase hex."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string with 'Z'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def layer_entry(
    *,
    layer: str,
    source_name: str,
    source_url: str,
    licence: str,
    vintage: str,
    crs: str,
    bbox: tuple[float, float, float, float],
    bbox_crs: str,
    feature_count: int,
    file_path: Path,
    district_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one manifest entry. ``vintage`` is whatever the *provider* states.

    ``file_path`` is recorded relative to ``district_dir`` so the manifest stays
    valid if the snapshot tree is moved or copied to another machine.
    """
    return {
        "layer": layer,
        "source_name": source_name,
        "source_url": source_url,
        "licence": licence,
        "downloaded_utc": utc_now_iso(),
        "vintage": vintage,
        "crs": crs,
        "bbox": [round(v, 3) for v in bbox],
        "bbox_crs": bbox_crs,
        "feature_count": feature_count,
        "file": file_path.relative_to(district_dir).as_posix(),
        "file_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
        **(extra or {}),
    }


def district_dir(data_root: Path, city: str, district: str) -> Path:
    """``data/<city>/<district>/``, created if missing."""
    d = data_root / city / district
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_manifest(district_dir_path: Path, entry: dict[str, Any]) -> Path:
    """Insert or replace the entry for ``entry['layer']`` and rewrite the file."""
    path = district_dir_path / MANIFEST_NAME
    doc: dict[str, Any] = {"layers": {}}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("layers", {})
    doc["layers"][entry["layer"]] = entry
    doc["layers"] = dict(sorted(doc["layers"].items()))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
