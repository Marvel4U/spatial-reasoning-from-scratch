"""Manual single run (set config.training = True to train for config.t_train steps)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

print("train.py: loading torch …", flush=True)
import torch

import config
import data
import experiments
import grid_vit_adapter
import qual
from spatial_batch import forward as model_forward, next_batch as loader_next_batch
from diagnostics import maybe_run_diagnostics
from config_report import print_config_report
from run_log import RunRecorder, initial_record, manual_run_id, now_iso


def _parse_args():
    p = argparse.ArgumentParser(description="Spatial harness manual train / smoke")
    p.add_argument("--quiet", action="store_true", help="disable stage logs (config.verbose=False)")
    p.add_argument("-v", "--verbose", action="store_true", help="force verbose (default on)")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.quiet:
        config.verbose = False
    elif args.verbose:
        config.verbose = True

    config.log("=== train.py ===", force=True)
    config.log(f"  repo_root={config.REPO_ROOT}")
    config.log(f"  cwd={Path.cwd()}")
    config.log(f"  torch={torch.__version__} verbose={config.verbose} training={config.training}")

    t0 = time.perf_counter()
    data.setup_device()
    config.log(f"  setup_device done (+{time.perf_counter() - t0:.2f}s)")

    t1 = time.perf_counter()
    data.load_data()
    config.log(f"  load_data done (+{time.perf_counter() - t1:.2f}s)")

    torch.manual_seed(config.seed)
    config.log("  building model …")
    t2 = time.perf_counter()
    model = grid_vit_adapter.build_model(device=data.device)
    config.log(f"  build_model done (+{time.perf_counter() - t2:.2f}s)")

    config.log("  optimizer …")
    optimizer = grid_vit_adapter.configure_optimizer(model)
    model = model.to(data.device)
    data.m = model
    data.optimizer = optimizer

    config.log(f"  forward smoke batch_size={config.batch_size} …")
    t3 = time.perf_counter()
    img, tgt, task_id, cond_ids = loader_next_batch(data.train_loader)
    config.log(f"  batch shapes img={tuple(img.shape)} tgt={tuple(tgt.shape)}")
    img, tgt = img.to(data.device), tgt.to(data.device)
    if task_id is not None:
        task_id = task_id.to(data.device)
    if cond_ids is not None:
        cond_ids = cond_ids.to(data.device)
    with data.ctx:
        _, loss = model_forward(model, img, tgt, task_id, cond_ids)
    if device_sync := str(data.device).startswith("cuda"):
        torch.cuda.synchronize()
    config.log(f"  smoke forward loss={loss.item():.4f} (+{time.perf_counter() - t3:.2f}s)")

    print_config_report(model, data.train_sample_count(), data.val_sample_count())

    recorder = None
    run_t0 = time.perf_counter()
    if config.training:
        config.log(f"  training for {config.t_train} steps …", force=True)
        results_path = Path(experiments.RESULTS_PATH)
        run_id = manual_run_id()
        recorder = RunRecorder(results_path, initial_record(
            run_id,
            source="manual",
            train_steps=config.t_train,
            notes="train.py",
            started=now_iso(),
            n_params=sum(p.numel() for p in model.parameters()),
            train_samples=data.train_sample_count(),
            val_samples=data.val_sample_count(),
        ))
        recorder.flush("started")

        diag = maybe_run_diagnostics(run_id)

        def on_eval(step, losses, secs_since_last_eval, eval_iters=None):
            if recorder is not None:
                recorder.on_eval(step, losses, secs_since_last_eval, time.perf_counter() - run_t0)
            if diag is not None and diag.tier2.should_record(step):
                diag.tier2.record_ce_eval(
                    step, model,
                    eval_iters=eval_iters if eval_iters is not None else config.eval_iters,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                )

        try:
            stats = grid_vit_adapter.train(
                config.t_train, model=model, opt=optimizer,
                on_eval=on_eval if (recorder or diag) else None,
                tier1=diag.tier1 if diag else None,
                ckpt_series=diag.ckpt if diag else None,
            )
            sample = qual.run_qual_sample(model=model)
            if recorder is not None:
                recorder.set_sample(sample)
                recorder.complete(stats)
        except KeyboardInterrupt:
            if recorder is not None:
                recorder.abort("keyboard interrupt")
            raise
    else:
        config.log("  config.training=False — skipping train loop; running qual …", force=True)
        qual.run_qual_sample(model=model)

    config.log(f"=== train.py done (total {time.perf_counter() - t0:.2f}s) ===", force=True)


if __name__ == "__main__":
    main()
