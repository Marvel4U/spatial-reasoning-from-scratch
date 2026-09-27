"""Run spatial experiments; append results to runs/experiments.json."""
import time
from pathlib import Path

import torch

import config
import data
import eval_spatial
import experiments
import grid_vit_adapter
import qual
from diagnostics import maybe_run_diagnostics
from config_report import print_config_report
from experiment_skip import SkipListener
from run_log import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_ARCH_VERSION,
    RunRecorder,
    find_run_by_id,
    initial_record,
    now_iso,
    record_for_resume,
    should_skip_finished_run,
    training_goal_met,
    training_setup_fingerprint,
    training_setup_matches,
)


def _apply_overrides(overrides: dict) -> dict:
    saved = {}
    for k, v in overrides.items():
        if k == "code_version":
            raise KeyError("code_version is set in config.py only")
        if not hasattr(config, k):
            raise KeyError(f"unknown config key in overrides: {k}")
        saved[k] = getattr(config, k)
        setattr(config, k, v)
    return saved


def _restore_overrides(saved: dict):
    for k, v in saved.items():
        setattr(config, k, v)


def _normalize_spatial_eval(cfg) -> dict | None:
    if not cfg:
        return None
    if cfg is True:
        cfg = {}
    mb = cfg.get("max_batches", 20)
    out = {"max_batches": None if mb is None else int(mb)}
    snap = cfg.get("train_snap_batches")
    if snap is None:
        snap = 3
    out["train_snap_batches"] = int(snap)
    return out


def _resolve_spec(spec: dict) -> dict:
    return {
        "id": spec["id"],
        "architecture": spec.get("architecture", DEFAULT_ARCHITECTURE),
        "arch_version": spec.get("arch_version", DEFAULT_ARCH_VERSION),
        "train_steps": spec.get("train_steps", config.t_train),
        "overrides": dict(spec.get("overrides") or {}),
        "notes": spec.get("notes", ""),
        "init_checkpoint": spec.get("init_checkpoint"),
        "load_optimizer": spec.get("load_optimizer", False),
        "reset_step_counter": spec.get("reset_step_counter", False),
        "force_restart": spec.get("force_restart", False),
        "spatial_eval": spec.get("spatial_eval"),
        "lr_horizon_steps": spec.get("lr_horizon_steps"),
    }


def run_experiment(spec: dict, results_path: Path | None = None) -> dict:
    results_path = results_path or Path(experiments.RESULTS_PATH)
    spec = _resolve_spec(spec)
    run_id = spec["id"]
    arch = spec["architecture"]
    if arch != grid_vit_adapter.ARCHITECTURE:
        raise ValueError(f"unsupported architecture: {arch}")
    arch_version = spec["arch_version"]
    train_steps = spec["train_steps"]
    overrides = spec["overrides"]
    init_ckpt = spec.get("init_checkpoint")
    load_optimizer = spec.get("load_optimizer", False)
    reset_step_counter = spec.get("reset_step_counter", False)
    force_restart = spec.get("force_restart", False)
    lr_horizon_steps = spec.get("lr_horizon_steps") or train_steps
    spatial_eval_cfg = _normalize_spatial_eval(spec.get("spatial_eval"))

    print(f"\n{'=' * 60}\nexperiment: {run_id}  ({arch}@v{arch_version})\n{'=' * 60}")

    saved = _apply_overrides(overrides)
    data.load_data()
    train_n, val_n = data.train_sample_count(), data.val_sample_count()
    started = now_iso()
    recorder = None
    skip_listener = None
    run_t0 = time.perf_counter()
    ckpt_path = Path(config.harness_checkpoint_dir) / f"{run_id}.pt"
    save_ckpt = config.harness_save_checkpoints
    resume_ckpt = (
        grid_vit_adapter.best_checkpoint_path(ckpt_path) if save_ckpt else None
    )
    ckpt_steps = (
        grid_vit_adapter.max_checkpoint_steps(ckpt_path) if save_ckpt else None
    )
    prior = find_run_by_id(results_path, run_id)
    fingerprint = training_setup_fingerprint(
        architecture=arch,
        arch_version=arch_version,
        overrides=overrides,
        train_samples=train_n,
        val_samples=val_n,
        init_checkpoint=init_ckpt,
    )
    auto_resume = (
        config.harness_auto_resume
        and not force_restart
        and not init_ckpt
        and save_ckpt
        and resume_ckpt is not None
        and ckpt_steps is not None
        and not training_goal_met(ckpt_steps, train_steps)
        and prior is not None
        and training_setup_matches(prior, fingerprint)
    )
    skip_finished = should_skip_finished_run(
        force_restart=force_restart,
        train_steps=train_steps,
        overrides=overrides,
        prior=prior,
        ckpt_steps=ckpt_steps,
        ckpt_path=ckpt_path if save_ckpt else None,
    )

    saved_skip_val = config.eval_skip_val_in_ce_eval
    try:
        if skip_finished:
            print(
                f"  complete:  checkpoint @ step {ckpt_steps} "
                f"(target {train_steps}) — skipping experiment"
            )
            return prior if prior is not None else {"id": run_id, "skipped": True}
        if auto_resume:
            record = record_for_resume(prior, train_steps=train_steps, started=started)
            if save_ckpt:
                record["checkpoint"] = str(ckpt_path)
            recorder = RunRecorder(results_path, record)
            recorder.flush("resumed")
        else:
            if force_restart:
                print("  start:     fresh (force_restart=True)")
            elif resume_ckpt is not None and prior is not None and not training_setup_matches(prior, fingerprint):
                print("  start:     fresh (training setup differs from prior run — check overrides)")
            data.reset_training_stats()
            recorder = RunRecorder(results_path, initial_record(
                run_id,
                source="harness",
                architecture=arch,
                arch_version=arch_version,
                train_steps=train_steps,
                overrides=overrides,
                notes=spec["notes"],
                started=started,
                train_samples=train_n,
                val_samples=val_n,
                init_checkpoint=init_ckpt,
            ))
            if save_ckpt:
                recorder.record["checkpoint"] = str(ckpt_path)
            recorder.flush("started")

        torch.manual_seed(config.seed)
        model = grid_vit_adapter.build_model(device=data.device)
        model = model.to(data.device)
        opt = grid_vit_adapter.configure_optimizer(model)
        if auto_resume:
            grid_vit_adapter.load_checkpoint(resume_ckpt, model=model, optimizer=opt)
            print(f"  resume:    {resume_ckpt} @ step {ckpt_steps}")
        elif init_ckpt:
            init_path = Path(init_ckpt)
            grid_vit_adapter.load_checkpoint(init_path, model=model, optimizer=opt if load_optimizer else None)
            if not load_optimizer:
                opt = grid_vit_adapter.configure_optimizer(model)
            if reset_step_counter:
                data.reset_training_stats()
        data.m = model
        data.optimizer = opt
        n_params = sum(p.numel() for p in model.parameters())
        recorder.record["summary"]["n_params"] = n_params
        recorder.flush("model ready")

        print_config_report(model, train_n, val_n, train_steps=train_steps)
        if save_ckpt:
            print(f"  checkpoint: {ckpt_path}")
        if config.experiment_skip_enabled:
            print("  skip:      press 'c' to skip this run")
            skip_listener = SkipListener()
            skip_listener.start()

        snap_batches = (
            spatial_eval_cfg.get("train_snap_batches", 0) if spatial_eval_cfg else 0
        )
        config.eval_skip_val_in_ce_eval = snap_batches > 0

        diag = maybe_run_diagnostics(run_id)

        def on_eval(step, losses, secs_since_last_eval, eval_iters=None):
            losses = dict(losses)
            timing = {"snap_s": 0.0, "log_s": 0.0}
            if snap_batches > 0:
                t_snap = time.perf_counter()
                snap = eval_spatial.eval_loader(
                    model, data.val_loader, max_batches=snap_batches, snap=True,
                )
                losses["val"] = snap["mean_loss"]
                timing["snap_s"] = time.perf_counter() - t_snap
            t_log = time.perf_counter()
            training_time_s = time.perf_counter() - run_t0
            recorder.on_eval(step, losses, secs_since_last_eval, training_time_s)
            if diag is not None and diag.tier2.should_record(step):
                diag.tier2.record_ce_eval(
                    step, model,
                    eval_iters=eval_iters if eval_iters is not None else config.eval_iters,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                )
            timing["log_s"] = time.perf_counter() - t_log
            return losses, timing

        should_stop = skip_listener.should_stop if skip_listener else None
        start_step = data.training_stats["steps"] + 1 if data.training_stats else 0
        remaining = train_steps - start_step
        if remaining <= 0:
            print(f"  complete:  already at target step — skipping training")
            stats = dict(data.training_stats)
            stats["skipped"] = False
        else:
            stats = grid_vit_adapter.train(
                remaining, model=model, opt=opt, save=save_ckpt, on_eval=on_eval,
                should_stop=should_stop, checkpoint_path=str(ckpt_path),
                lr_horizon_steps=lr_horizon_steps,
                tier1=diag.tier1 if diag else None,
                ckpt_series=diag.ckpt if diag else None,
            )
        if diag is not None and not stats.get("skipped"):
            diag.tier2.maybe_save_c4_reference_pred_strip(model, stats["steps"])
        if spatial_eval_cfg:
            print(f"\n--- spatial eval @ step {stats['steps']} ---")
            result = eval_spatial.eval_model(model, max_batches=spatial_eval_cfg["max_batches"])
            eval_spatial.print_report(result)
            recorder.on_spatial_eval(stats["steps"], result)
        sample = qual.run_qual_sample(model=model)
        recorder.set_sample(sample)
        if stats.get("skipped"):
            recorder.skip(stats)
        else:
            recorder.complete(stats)
        return recorder.record
    except KeyboardInterrupt:
        if recorder is not None:
            recorder.abort("keyboard interrupt")
        raise
    except Exception as e:
        if recorder is not None:
            recorder.abort(str(e))
        raise
    finally:
        if skip_listener is not None:
            skip_listener.stop()
        config.eval_skip_val_in_ce_eval = saved_skip_val
        _restore_overrides(saved)


def main():
    data.setup_device()
    specs = getattr(experiments, "RUN", None) or experiments.TO_RUN
    if not specs:
        print("RUN is empty — edit experiments.py")
        return
    for spec in specs:
        run_experiment(spec)


if __name__ == "__main__":
    main()
