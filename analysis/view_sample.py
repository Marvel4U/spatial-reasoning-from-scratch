"""CLI for §13 static c0 task viewer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from plain_gpt_module.local_config import LocalGridViTConfig

from .c0_adapter import predict_sample
from .c1_adapter import predict_c1_sample
from .geometry import C0Geometry
from .load_model import load_vit_checkpoint
from .load_sample import Split, load_synthetic_sample, load_worldsnap_sample, repo_root
from .load_c1_sample import C1LoadedSample, load_c1_worldsnap_sample
from .render import render_compare, render_sample
from .render_c1 import render_c1_compare, render_c1_sample


def _resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_forward(device: torch.device, fn):
    _device_sync(device)
    t0 = time.perf_counter()
    out = fn()
    _device_sync(device)
    return out, time.perf_counter() - t0


def _time_plot(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _print_timing(
    *,
    device: torch.device | None,
    forward_s: float | None,
    plot_s: float,
    n_forward: int = 1,
) -> None:
    parts: list[str] = []
    if forward_s is not None and device is not None:
        label = device.type
        ms = forward_s * 1000
        if n_forward > 1:
            parts.append(f"forward ({label}) {ms:.1f} ms total ({n_forward}×, {ms / n_forward:.1f} ms/item)")
        else:
            parts.append(f"forward ({label}) {ms:.1f} ms")
    parts.append(f"plot+save {plot_s * 1000:.1f} ms")
    print("timing: " + ", ".join(parts))


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _encoding_from_in_chans(in_chans: int) -> str:
    if in_chans == 4:
        return "scalar"
    if in_chans == 13:
        return "onehot"
    raise ValueError(f"unsupported in_chans={in_chans} for viewer")


def _load_sample(args, *, cfg: LocalGridViTConfig | None = None) -> "LoadedSample":
    img_size = cfg.img_size if cfg else args.img_size
    grid_size = cfg.grid_out_size if cfg else args.grid_size
    encoding = _encoding_from_in_chans(cfg.in_chans) if cfg else args.encoding
    if args.synthetic:
        return load_synthetic_sample(
            split=args.split,
            item_index=args.item,
            seed=args.seed,
            img_size=img_size,
            grid_size=grid_size,
        )
    return load_worldsnap_sample(
        split=args.split,
        item_index=args.item,
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
        district_dir=args.district_dir,
        cropset=args.cropset,
        taskset=args.taskset,
    )


def _load_c1_sample(args, *, cfg: LocalGridViTConfig | None = None) -> C1LoadedSample:
    img_size = cfg.img_size if cfg else args.img_size
    grid_size = cfg.grid_out_size if cfg else args.grid_size
    encoding = _encoding_from_in_chans(cfg.in_chans) if cfg else args.encoding
    return load_c1_worldsnap_sample(
        split=args.split,
        crop_index=args.item,
        task_key=args.c1_task,
        encoding=encoding,
        img_size=img_size,
        grid_size=grid_size,
        district_dir=args.district_dir,
        cropset=args.cropset,
    )


def _geom_for_sample(sample, patch_size: int | None, cfg: LocalGridViTConfig | None) -> C0Geometry:
    if cfg is not None:
        return C0Geometry.from_config(cfg)
    p = patch_size or 16
    return C0Geometry(
        patch_size=p,
        img_size=sample.meta.img_size,
        grid_size=sample.meta.grid_size,
        n_patch=sample.meta.img_size // p,
        subcells=sample.meta.grid_size // (sample.meta.img_size // p),
    )


def _run_single(args) -> Path:
    out = args.out or Path("runs/diagnostics") / f"view_{args.split}{args.item:04d}.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.no_model:
        if args.task_rung == "c1":
            sample_c1 = _load_c1_sample(args)
            geom = _geom_for_sample(sample_c1, args.patch_size, None)

            def _plot_c1():
                _save_figure(
                    render_c1_sample(sample_c1, geom, cfg=None, prob=None, err_rgb=None),
                    out,
                )

            plot_s = _time_plot(_plot_c1)
            _print_timing(device=None, forward_s=None, plot_s=plot_s)
            print(f"wrote {out} (--no-model c1 {args.c1_task})")
            return out
        sample = _load_sample(args)
        geom = _geom_for_sample(sample, args.patch_size, None)

        def _plot():
            _save_figure(
                render_sample(sample, geom, cfg=None, prob=None, metrics=None, ckpt_label=""),
                out,
            )

        plot_s = _time_plot(_plot)
        _print_timing(device=None, forward_s=None, plot_s=plot_s)
        print(f"wrote {out} (--no-model)")
        return out

    if not args.ckpt:
        raise SystemExit("provide --ckpt or use --no-model")
    device = _resolve_device(args.use_gpu)
    model, cfg, _ = load_vit_checkpoint(args.ckpt, device=device)
    ckpt_stem = Path(args.ckpt).stem

    if args.task_rung == "c1":
        sample_c1 = _load_c1_sample(args, cfg=cfg)
        geom = C0Geometry.from_config(cfg)

        (_, prob, err, metrics), fwd_s = _time_forward(
            device,
            lambda: predict_c1_sample(model, sample_c1, cfg, device=device),
        )

        def _plot_c1():
            _save_figure(
                render_c1_sample(
                    sample_c1, geom, cfg=cfg, prob=prob, err_rgb=err,
                    metrics=metrics, ckpt_label=ckpt_stem,
                ),
                out,
            )

        plot_s = _time_plot(_plot_c1)
        _print_timing(device=device, forward_s=fwd_s, plot_s=plot_s)
        print(f"wrote {out} (c1 {args.c1_task} IoU={metrics['iou']:.4f})")
        return out

    sample = _load_sample(args, cfg=cfg)
    geom = _geom_for_sample(sample, args.patch_size, cfg)

    (_, prob, metrics), fwd_s = _time_forward(
        device,
        lambda: predict_sample(model, sample, cfg, device=device),
    )

    def _plot():
        _save_figure(
            render_sample(
                sample, geom, cfg=cfg, prob=prob, metrics=metrics, ckpt_label=ckpt_stem,
            ),
            out,
        )

    plot_s = _time_plot(_plot)
    _print_timing(device=device, forward_s=fwd_s, plot_s=plot_s)
    print(f"wrote {out}")
    return out


def _run_compare(args) -> Path:
    device = _resolve_device(args.use_gpu)
    out = args.out or Path("runs/diagnostics") / f"compare_{args.split}{args.item:04d}.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.task_rung == "c1":
        sample_c1 = None
        geom = None
        rows_c1: list = []
        fwd_total = 0.0
        for ckpt in args.ckpt:
            model, cfg, _ = load_vit_checkpoint(ckpt, device=device)
            sample_c1 = _load_c1_sample(args, cfg=cfg)
            geom = C0Geometry.from_config(cfg)
            (_, prob, err, metrics), fwd_s = _time_forward(
                device,
                lambda m=model, s=sample_c1, c=cfg: predict_c1_sample(m, s, c, device=device),
            )
            fwd_total += fwd_s
            rows_c1.append((prob, err, metrics, Path(ckpt).stem))
        assert sample_c1 is not None and geom is not None

        def _plot_c1():
            _save_figure(render_c1_compare(sample_c1, geom, rows_c1), out)

        plot_s = _time_plot(_plot_c1)
        _print_timing(device=device, forward_s=fwd_total, plot_s=plot_s, n_forward=len(rows_c1))
        print(f"wrote {out} ({len(rows_c1)} checkpoints, c1)")
        return out

    rows = []
    sample = None
    fwd_total = 0.0
    for ckpt in args.ckpt:
        model, cfg, _ = load_vit_checkpoint(ckpt, device=device)
        sample = _load_sample(args, cfg=cfg)
        (_, prob, metrics), fwd_s = _time_forward(
            device,
            lambda m=model, s=sample, c=cfg: predict_sample(m, s, c, device=device),
        )
        fwd_total += fwd_s
        rows.append((C0Geometry.from_config(cfg), cfg, prob, metrics, Path(ckpt).stem))
    assert sample is not None

    def _plot():
        _save_figure(render_compare(sample, rows), out)

    plot_s = _time_plot(_plot)
    _print_timing(device=device, forward_s=fwd_total, plot_s=plot_s, n_forward=len(rows))
    print(f"wrote {out} ({len(rows)} checkpoints)")
    return out


def _run_gallery(args) -> Path:
    device = _resolve_device(args.use_gpu)
    if not args.ckpt:
        raise SystemExit("gallery requires --ckpt")
    model, cfg, _ = load_vit_checkpoint(args.ckpt, device=device)
    want = args.gallery
    max_scan = max(want * 8, 128)
    scores: list[tuple[float, int, torch.Tensor]] = []
    fwd_total = 0.0
    n_fwd = 0
    for idx in range(max_scan):
        try:
            if args.task_rung == "c1":
                if args.synthetic:
                    raise IndexError
                sample_c1 = load_c1_worldsnap_sample(
                    split=args.split, crop_index=idx, task_key=args.c1_task,
                    encoding=_encoding_from_in_chans(cfg.in_chans),
                    img_size=cfg.img_size, grid_size=cfg.grid_out_size,
                    district_dir=args.district_dir, cropset=args.cropset,
                )
            elif args.synthetic:
                sample = load_synthetic_sample(
                    split=args.split, item_index=idx, seed=args.seed,
                    img_size=cfg.img_size, grid_size=cfg.grid_out_size,
                )
            else:
                sample = load_worldsnap_sample(
                    split=args.split, item_index=idx, encoding=args.encoding,
                    img_size=cfg.img_size, grid_size=cfg.grid_out_size,
                    district_dir=args.district_dir, cropset=args.cropset, taskset=args.taskset,
                )
        except IndexError:
            break
        if args.task_rung == "c1":
            (_, prob, _, metrics), fwd_s = _time_forward(
                device,
                lambda s=sample_c1: predict_c1_sample(model, s, cfg, device=device),
            )
            if args.sort == "iou":
                key = (1.0 - metrics["iou"]) if args.worst else -metrics["iou"]
            elif args.sort == "patch_hit":
                key = -metrics["iou"]
            else:
                key = 1.0 - metrics["iou"]
        else:
            (_, prob, metrics), fwd_s = _time_forward(
                device,
                lambda s=sample: predict_sample(model, s, cfg, device=device),
            )
            key = metrics["argmax_dist_cells"]
            if args.sort == "patch_hit":
                key = -float(metrics["patch_hit"])
        fwd_total += fwd_s
        n_fwd += 1
        scores.append((key, idx, prob))
    scores = scores[:max_scan]
    reverse = args.worst
    scores.sort(key=lambda t: t[0], reverse=reverse)
    scores = scores[:want]
    if not scores:
        raise SystemExit("no gallery items loaded")
    cols = min(6, len(scores))
    rows_n = (len(scores) + cols - 1) // cols
    out = args.out or Path("runs/diagnostics") / f"gallery_{Path(args.ckpt).stem}.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _plot():
        fig, axes = plt.subplots(rows_n, cols, figsize=(2.2 * cols, 2.2 * rows_n), dpi=100)
        axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        for ax, (key, idx, prob) in zip(axes_flat, scores):
            ax.imshow(prob.detach().cpu().numpy(), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"i={idx} d={key:.1f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes_flat[len(scores):]:
            ax.axis("off")
        fig.suptitle(f"gallery {args.sort} {'worst' if args.worst else 'best'}", fontsize=10)
        fig.tight_layout()
        _save_figure(fig, out)

    plot_s = _time_plot(_plot)
    _print_timing(device=device, forward_s=fwd_total, plot_s=plot_s, n_forward=n_fwd)
    print(f"wrote {out} ({len(scores)} panels)")
    return out


def _run_series(args) -> Path:
    series_dir = Path(args.ckpt_series)
    ckpts = sorted(series_dir.glob("step*.pt"), key=lambda p: int(p.stem.replace("step", "")))
    if not ckpts:
        raise SystemExit(f"no step*.pt in {series_dir}")
    args.ckpt = [str(p) for p in ckpts]
    return _run_compare(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="c0/c1 static task viewer (ANALYSIS_SUITE_SPEC §13)")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--item", type=int, default=0, help="c0: jsonl index; c1: crop index in split folder")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-model", action="store_true", help="panels A–D only")
    p.add_argument("--ckpt", type=Path, nargs="*", help="checkpoint(s); multiple → compare mode")
    p.add_argument("--ckpt-series", type=Path, help="directory of step*.pt (over-time strip)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--encoding", choices=("scalar", "onehot"), default="scalar")
    p.add_argument("--patch-size", type=int, default=None, help="when --no-model (else from ckpt)")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--district-dir", type=Path, default=None)
    p.add_argument("--cropset", default="v2")
    p.add_argument("--taskset", default="t0_point_v0")
    p.add_argument("--task-rung", choices=("c0", "c1"), default="c0")
    p.add_argument("--c1-task", default="building", help="c1 task key (spatial_data/c1_tasks.py)")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--gallery", type=int, default=0, metavar="N")
    p.add_argument("--sort", choices=("argmax_dist", "patch_hit", "iou"), default="argmax_dist")
    p.add_argument("--worst", action="store_true", help="gallery: highest distance first")
    return p


def main(argv: list[str] | None = None) -> int:
    if str(repo_root()) not in sys.path:
        sys.path.insert(0, str(repo_root()))
    args = build_parser().parse_args(argv)
    if args.district_dir is None:
        args.district_dir = repo_root() / "data" / "amsterdam" / "de_pijp"
    if args.gallery:
        _run_gallery(args)
    elif args.ckpt_series:
        _run_series(args)
    elif args.ckpt and len(args.ckpt) > 1:
        _run_compare(args)
    else:
        if args.ckpt and len(args.ckpt) == 1:
            args.ckpt = args.ckpt[0]
        _run_single(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
