"""Tier 2: c0 eval JSONL, probe metrics (b), pred PNGs (c)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config
import data
import eval_spatial

from .paths import eval_path, run_dir
from .pred_strip import save_pred_strip
from .probe_batch import ProbeBatch
from .probe_metrics import probe_forward_metrics


class Tier2EvalDiagnostics:
    def __init__(self, run_id: str, *, out_dir: Path | None = None, meta_path: Path | None = None):
        self.run_id = run_id
        if out_dir is not None:
            self.run_dir = Path(out_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = self.run_dir / "eval.jsonl"
        else:
            self.run_dir = run_dir(run_id, create=True)
            self.jsonl_path = eval_path(run_id)
        self.meta_path = meta_path
        self._interval = config.diagnostics_eval_interval
        self._probe: ProbeBatch | None = None
        config.log(
            f"  diagnostics: tier2 eval → {self.jsonl_path} "
            f"(every CE eval" + ("" if self._interval is None else f", step≡0 mod {self._interval}") + ")",
            force=True,
        )

    def should_record(self, step: int) -> bool:
        if self._interval is None:
            return True
        return step % self._interval == 0

    def _ensure_probe(self) -> ProbeBatch:
        if self._probe is None:
            self._probe = ProbeBatch.build()
            if self.meta_path is not None:
                self._probe.update_meta(self.meta_path)
        return self._probe

    def record_ce_eval(
        self,
        step: int,
        model,
        *,
        eval_iters: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        val_batches = eval_spatial.diagnostics_val_max_batches(eval_iters)
        val = eval_spatial.eval_loader(model, data.val_loader, max_batches=val_batches)
        row = {
            "step": step,
            "at": datetime.now(timezone.utc).isoformat(),
            "train_loss_eval": round(train_loss, 6),
            "val_loss_eval": round(val_loss, 6),
            "eval_iters": eval_iters,
            "val_eval_batches": val["batches"],
            "val_eval_full_grid": val_batches is None,
            "batches": val["batches"],
            "n_grids": val["n_grids"],
            **eval_spatial.flatten_for_eval_jsonl(val),
        }
        if config.diagnostics_probe_enabled:
            probe = self._ensure_probe()
            row.update(probe_forward_metrics(model, probe))
        if config.diagnostics_pred_enabled and self.should_record(step):
            probe = self._ensure_probe()
            png = save_pred_strip(model, probe, step=step, out_dir=self.run_dir, run_id=self.run_id)
            config.log(f"  diagnostics: pred strip → {png}", force=True)
        rollout_row = self.maybe_wave_rollout(model, step)
        if rollout_row:
            row.update(rollout_row)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def maybe_wave_rollout(self, model, step: int) -> dict | None:
        if not config.diagnostics_wave_rollout_enabled or config.task_rung != "c4":
            return None
        keys = list(config.c4_task_keys) if config.c4_is_multi_task() else [config.c4_task_key]
        if not any(k.startswith("wave_") for k in keys):
            return None
        if not self.should_record(step):
            return None
        from spatial_data.c4_sidecars import c4_val_sidecars_ok

        routes_dir = config.c4_routes_dir()
        if not c4_val_sidecars_ok(routes_dir, list(keys)):
            return None
        from .wave_search_rollout import (
            build_wave_rollout_loader,
            run_rollout,
            save_rollout_figure,
            wave_rollout_route_indices,
            wave_rollout_summary,
        )
        loader, val_routes = build_wave_rollout_loader()
        g = config.grid_out_size
        crop = int(val_routes.get("crop_px", config.img_size))
        from .wave_search_rollout import wave_rollout_start_pass_for_route

        start_cfg = config.diagnostics_wave_rollout_start_pass
        merged: dict = {}
        primary = True
        for route_ix in wave_rollout_route_indices():
            if route_ix < 0 or route_ix >= loader.n_routes:
                config.log(f"  diagnostics: wave rollout route {route_ix} out of range 0..{loader.n_routes - 1}", force=True)
                continue
            start_pass = wave_rollout_start_pass_for_route(route_ix, start_cfg)
            rows = run_rollout(
                model, loader, route_ix=route_ix, grid=g, crop=crop, start_pass=start_pass,
            )
            out = self.run_dir / f"wave_rollout_route{route_ix:03d}_step{step:06d}.png"
            save_rollout_figure(
                rows, run_id=self.run_id, route_ix=route_ix, out=out, val_loader=loader,
                start_pass=start_pass,
            )
            summary = wave_rollout_summary(rows)
            suffix = "" if primary else f"_r{route_ix}"
            for k, v in summary.items():
                merged[f"{k}{suffix}"] = round(v, 4) if isinstance(v, float) else v
            merged[f"wave_rollout_route{suffix}"] = route_ix
            config.log(
                f"  diagnostics: wave rollout route {route_ix} → {out.name} "
                f"(IoU {summary['wave_rollout_mask_iou_mean']:.2f}, tok {summary['wave_rollout_tok_acc']:.2f})",
                force=True,
            )
            primary = False
        return merged or None

    def maybe_save_c4_reference_pred_strip(self, model, step: int):
        """Same probe slot indices on the default (20k) val store — for alt-route-store training runs."""
        ref = config.diagnostics_pred_c4_reference_subdir
        if (
            config.task_rung != "c4"
            or not config.diagnostics_pred_enabled
            or not ref
            or config.c4_routes_subdir == ref
        ):
            return None
        import data
        from spatial_data.c4_sidecars import c4_val_sidecars_ok

        from .probe_batch import ProbeBatch

        routes_dir = config.c3_crops_dir() / ref
        keys = list(config.c4_task_keys) if config.c4_is_multi_task() else [config.c4_task_key]
        if not c4_val_sidecars_ok(routes_dir, keys):
            config.log(
                f"  diagnostics: skip reference pred strip (store {ref!r} lacks C5 sidecars for {keys})",
                force=True,
            )
            return None
        ref_loader = data.build_c4_val_loader(ref)
        probe = ProbeBatch.build_c4_from_val_loader(ref_loader)
        suffix = f"_routes_{ref}"
        png = save_pred_strip(
            model,
            probe,
            step=step,
            out_dir=self.run_dir,
            run_id=self.run_id,
            c4_val_loader=ref_loader,
            filename_suffix=suffix,
            title_suffix=f" (val routes {ref!r})",
        )
        config.log(f"  diagnostics: pred strip (reference val store) → {png}", force=True)
        return png
