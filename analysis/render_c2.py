"""c2 task viewer — same A–G layout as c1 (render_c1)."""
from __future__ import annotations

from matplotlib.figure import Figure

from plain_gpt_module.local_config import LocalGridViTConfig

from .c1_adapter import focal_grid_cell
from .geometry import C0Geometry
from .load_c2_sample import C2LoadedSample
from .load_c1_sample import C1LoadedSample
from .render_c1 import render_c1_sample


def _as_c1(sample: C2LoadedSample) -> C1LoadedSample:
    from .load_c1_sample import C1SampleMeta

    m = sample.meta
    meta = C1SampleMeta(
        split=m.split,
        crop_index=m.crop_index,
        crop_id=m.crop_id,
        task_key=m.task_key,
        encoding=m.encoding,
        img_size=m.img_size,
        grid_size=m.grid_size,
    )
    return C1LoadedSample(
        img=sample.img, target=sample.target, meta=meta, label_planes=sample.label_planes,
    )


def render_c2_sample(
    sample: C2LoadedSample,
    geom: C0Geometry,
    *,
    cfg: LocalGridViTConfig | None = None,
    prob=None,
    err_rgb=None,
    metrics=None,
    ckpt_label: str = "",
    title_suffix: str = "",
) -> Figure:
    fig = render_c1_sample(
        _as_c1(sample), geom, cfg=cfg, prob=prob, err_rgb=err_rgb,
        metrics=metrics, ckpt_label=ckpt_label,
    )
    if title_suffix:
        fig.suptitle(f"{sample.meta.task_key}  {title_suffix}", fontsize=11, y=0.98)
    return fig
