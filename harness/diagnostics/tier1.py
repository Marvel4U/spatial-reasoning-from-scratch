"""Tier 1 JSONL: timing, semantic groups, update ratios (ANALYSIS_SUITE_SPEC §5)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

import config
from plain_gpt_module import unwrap_compiled
from plain_gpt_module.local_config import LocalGridViTConfig
from run_log import config_snapshot

from .groups import GroupSpec, build_semantic_groups, patch_embed_decomposition, view_group_tensor
from .paths import meta_path, run_dir, tier1_path


def input_channel_names() -> list[str]:
    if config.encoding_mode == "scalar":
        return ["noise", "surface", "estab", "marker"]
    from spatial_data.channels import NUM_CLASSES_PER_LAYER, NUM_NOISE7_CLASSES
    names: list[str] = []
    if config.encoding_mode == "onehot_v4":
        for c in range(NUM_NOISE7_CLASSES):
            names.append(f"noise7_c{c}")
        for layer in ("surface", "estab"):
            for c in range(NUM_CLASSES_PER_LAYER):
                names.append(f"{layer}_c{c}")
        names.append("marker")
        return names
    for layer in ("noise", "surface", "estab"):
        for c in range(NUM_CLASSES_PER_LAYER):
            names.append(f"{layer}_c{c}")
    names.append("marker")
    return names


def _json_float(v: float, digits: int) -> float | None:
    if not isinstance(v, (int, float)):
        return v
    v = float(v)
    if not math.isfinite(v):
        return None
    return round(v, digits)


class Tier1Diagnostics:
    """Append to runs/diagnostics/{run_id}/tier1.jsonl on recorded steps."""

    def __init__(
        self,
        run_id: str,
        *,
        interval: int | None = None,
        record_step0: bool | None = None,
        out_dir: Path | None = None,
    ):
        self.run_id = run_id
        self.interval = max(1, interval if interval is not None else config.diagnostics_interval)
        self.record_step0 = (
            config.diagnostics_record_step0 if record_step0 is None else record_step0
        )
        _ = out_dir  # canonical layout ignores custom out_dir
        run_dir(run_id, create=True)
        self.jsonl_path = tier1_path(run_id)
        self.meta_path = meta_path(run_id)
        self._groups: list[GroupSpec] | None = None
        self._cfg: LocalGridViTConfig | None = None
        self._channels: list[str] = input_channel_names()
        self._write_meta([])

    def prepare(self, model) -> None:
        core = unwrap_compiled(model)
        if not hasattr(core, "config"):
            raise TypeError("model must expose LocalGridViT.config")
        self._cfg = core.config
        self._channels = input_channel_names()
        self._groups = build_semantic_groups(core, self._cfg, self._channels)
        self._write_meta([g.key for g in self._groups])
        config.log(f"  diagnostics: tier1 {len(self._groups)} param groups", force=True)

    def _write_meta(self, parameter_groups: list[str]) -> None:
        doc = {
            "run_id": self.run_id,
            "interval": self.interval,
            "record_step0": self.record_step0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot(),
            "parameter_groups": parameter_groups,
            "input_channels": self._channels,
            "probe_sample_ids": None,
            "diagnostics_eval_interval": config.diagnostics_eval_interval,
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")

    def should_record(self, step: int) -> bool:
        if step == 0 and self.record_step0:
            return True
        return step > 0 and step % self.interval == 0

    def _require_groups(self) -> tuple[list[GroupSpec], LocalGridViTConfig]:
        if self._groups is None or self._cfg is None:
            raise RuntimeError("tier1.prepare(model) must run before recorded steps")
        return self._groups, self._cfg

    def collect_pre_clip(self, model) -> dict[str, float]:
        groups, cfg = self._require_groups()
        core = unwrap_compiled(model)
        out = patch_embed_decomposition(cfg, core.patch_embed.proj.weight, self._channels)
        for g in groups:
            w = view_group_tensor(g.param.data, g.channel, cfg)
            out[f"param_rms/{g.key}"] = w.pow(2).mean().sqrt().item()
            if g.param.grad is not None:
                gv = view_group_tensor(g.param.grad, g.channel, cfg)
                out[f"grad_norm/{g.key}"] = gv.norm().item()
        return out

    def snapshot_weights(self, model) -> dict[str, torch.Tensor]:
        groups, cfg = self._require_groups()
        return {
            g.key: view_group_tensor(g.param.data, g.channel, cfg).clone()
            for g in groups
        }

    def update_ratios(self, model, before: dict[str, torch.Tensor]) -> dict[str, float]:
        groups, cfg = self._require_groups()
        out: dict[str, float] = {}
        for g in groups:
            now = view_group_tensor(g.param.data, g.channel, cfg)
            b = before[g.key]
            bs = b.std(unbiased=False).item()
            a = (now - b).std(unbiased=False).item()
            if bs > 1e-12:
                out[f"update_ratio/{g.key}"] = math.log10(max(a, 1e-12) / bs)
            else:
                out[f"update_ratio/{g.key}"] = float("-inf") if a > 0 else float("nan")
        return out

    def record_step(
        self,
        *,
        step: int,
        loss: float,
        lr: float,
        t_data: float,
        t_step: float,
        batch_size: int,
        grad_norm_total: float,
        pre_clip: dict[str, float],
        update_ratios: dict[str, float],
    ) -> None:
        total = t_data + t_step
        row = {
            "step": step,
            "loss": round(float(loss), 6),
            "lr": float(lr),
            "grad_norm_total": round(float(grad_norm_total), 6),
            "t_data": round(t_data, 6),
            "t_step": round(t_step, 6),
            "gpu_starvation": round(t_data / total, 4) if total > 0 else 0.0,
            "samples_per_s": round(batch_size / total, 2) if total > 0 else 0.0,
            **{k: _json_float(v, 6) for k, v in pre_clip.items()},
            **{k: _json_float(v, 4) for k, v in update_ratios.items()},
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
