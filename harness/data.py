"""Device setup and c0 batch loaders (synthetic or worldsnap via spatial_data)."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

import config

if str(config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT))

device: torch.device | None = None
dtype = torch.bfloat16
ctx = None
train_loader = None
val_loader = None
_n_val_crops: int | None = None  # worldsnap crop count (c1 val pairs can exceed this)
training_stats: dict | None = None
m = None
optimizer = None


def setup_device():
    global device, ctx, dtype
    if config.cpu_num_threads is not None:
        torch.set_num_threads(config.cpu_num_threads)
    config.log(
        f"data.setup_device: use_gpu={config.use_gpu} cuda_available={torch.cuda.is_available()} "
        f"cpu_threads={torch.get_num_threads()}"
    )
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        config.log(f"  device=cuda ({torch.cuda.get_device_name(0)}) dtype=bfloat16")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        ctx = torch.autocast(device_type="cpu", enabled=False)
        if config.use_gpu and not torch.cuda.is_available():
            config.log("  warning: use_gpu=True but CUDA unavailable — using cpu", force=True)
        config.log("  device=cpu dtype=float32")
    return device


def reset_training_stats():
    global training_stats
    training_stats = None


def _make_c0_sample(rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    h = w = config.img_size
    g = config.grid_out_size
    layers = torch.tensor(rng.choices([0.0, 1 / 3, 2 / 3, 1.0], k=3 * h * w))
    layers = layers.view(3, h, w)
    mr, mc = rng.randrange(h), rng.randrange(w)
    marker = torch.zeros(1, h, w)
    marker[0, mr, mc] = 1.0
    img = torch.cat([layers, marker], dim=0)
    tgt = torch.zeros(g, g, dtype=torch.long)
    gr, gc = mr * g // h, mc * g // w
    tgt[gr, gc] = 1
    return img, tgt


class GridBatchLoader:
    """Fixed pool of (image, target_grid) samples; random batch with replacement."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int):
        if not samples:
            raise ValueError("samples must be non-empty")
        self.samples = samples
        self.B = batch_size

    def __len__(self) -> int:
        return len(self.samples)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = [random.randrange(len(self.samples)) for _ in range(self.B)]
        imgs = torch.stack([self.samples[i][0] for i in idx])
        tgts = torch.stack([self.samples[i][1] for i in idx])
        return imgs, tgts


def _build_synthetic_pool(n: int, seed: int, *, label: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    config.log(f"  data: building {label} pool n={n} (seed={seed}) …")
    t0 = time.perf_counter()
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    step = max(1, n // 10)
    for i in range(n):
        out.append(_make_c0_sample(rng))
        if config.verbose and (i + 1) % step == 0:
            config.log(f"  data:   {label} {i + 1}/{n}")
    config.log(f"  data: {label} pool done in {time.perf_counter() - t0:.1f}s")
    return out


def _load_synthetic():
    global train_loader, val_loader
    train_pool = _build_synthetic_pool(
        config.synthetic_train_samples, config.seed, label="train",
    )
    if config.val_same_as_train:
        config.log("  data: val_same_as_train — val pool shares train samples")
        val_pool = train_pool
    else:
        val_pool = _build_synthetic_pool(
            config.synthetic_val_samples, config.seed + 1, label="val",
        )
    train_loader = GridBatchLoader(train_pool, config.batch_size)
    val_loader = GridBatchLoader(val_pool, config.batch_size)


def _load_worldsnap_c1():
    global train_loader, val_loader, _n_val_crops
    from spatial_data.c1_pairs import fixed_val_pairs
    from spatial_data.c1_tasks import get_c1_task, task_keys_for_mode, task_specs_for_keys
    from spatial_data.dataset_c1 import (
        C1BatchLoader,
        C1CropIndex,
        C1MultiTaskBatchLoader,
        list_crop_items,
        resolve_c1_task,
    )
    from spatial_data.dataset_c1_gpu import (
        C1GpuBatchLoader,
        C1GpuResidentStore,
        C1MultiTaskGpuBatchLoader,
    )
    from spatial_data.layer_cache import LayerCache

    config.c1_active_task_keys = config.c1_task_keys_active()
    multi = config.c1_is_multi_task()
    task_keys = config.c1_active_task_keys
    task_specs = task_specs_for_keys(task_keys, cropset=config.cropset)
    use_gpu_loader = (
        config.worldsnap_gpu_resident
        and device is not None
        and device.type == "cuda"
    )
    district = config.district_data_dir()
    crops_dir = district / "crops" / config.cropset
    mode_note = f"mode={config.c1_task_mode} tasks={task_keys}" if multi else f"task={task_keys[0]!r}"
    config.log(f"  worldsnap c1: {mode_note} encoding={config.encoding_mode}")
    t0 = time.perf_counter()
    train_items = list_crop_items(crops_dir, "train")
    val_items = train_items if config.val_same_as_train else list_crop_items(crops_dir, "val")
    if config.worldsnap_max_train_items is not None:
        train_items = train_items[: config.worldsnap_max_train_items]
    if config.worldsnap_max_val_items is not None:
        val_items = val_items[: config.worldsnap_max_val_items]
    _n_val_crops = len(val_items)
    enc = config.encoding_mode
    g, h = config.grid_out_size, config.img_size
    layer_cache = None
    if config.worldsnap_cache_layers:
        layer_cache = LayerCache()
        paths = {it.labels_path for it in train_items}
        paths.update(it.labels_path for it in val_items)
        layer_cache.warm(paths, log=config.log)
    val_pairs = None
    train_task_ix = None
    if multi:
        val_pairs = fixed_val_pairs(
            len(val_items), len(task_specs), config.c1_val_pairs, config.seed + 99,
        )
        train_task_ix = config.c1_train_task_indices_active()
    gpu_note = ", gpu_resident_batches" if use_gpu_loader else ""
    config.log(
        f"  worldsnap c1: {len(train_items)} train + {len(val_items)} val crops "
        f"({time.perf_counter() - t0:.1f}s{gpu_note}"
        + (f", val_pairs={len(val_pairs)}" if val_pairs else "")
        + (f", train_tasks={len(train_task_ix)}" if train_task_ix is not None else "")
        + ")"
    )

    def _crop_ix_tensor(items: list) -> torch.Tensor:
        path_id = {p: i for i, p in enumerate(sorted({it.labels_path.resolve() for it in items}))}
        return torch.tensor(
            [path_id[it.labels_path.resolve()] for it in items],
            dtype=torch.long,
            device=device if use_gpu_loader else "cpu",
        )

    if multi:
        if use_gpu_loader:
            lc = layer_cache or LayerCache()
            if layer_cache is None:
                lc.warm({it.labels_path for it in train_items + val_items}, log=config.log)
            train_store = C1GpuResidentStore.from_items(train_items, lc, device, log=config.log)
            val_store = train_store if config.val_same_as_train else C1GpuResidentStore.from_items(
                val_items, lc, device, log=config.log,
            )
            train_loader = C1MultiTaskGpuBatchLoader(
                train_store, _crop_ix_tensor(train_items), task_specs, enc, g, h,
                config.batch_size, config.seed, train_task_indices=train_task_ix,
            )
            val_loader = C1MultiTaskGpuBatchLoader(
                val_store, _crop_ix_tensor(val_items), task_specs, enc, g, h,
                config.batch_size, config.seed + 1, fixed_pairs=val_pairs,
            )
        else:
            train_loader = C1MultiTaskBatchLoader(
                train_items, task_specs, enc, g, h, config.batch_size, config.seed,
                layer_cache=layer_cache, train_task_indices=train_task_ix,
            )
            val_loader = C1MultiTaskBatchLoader(
                val_items, task_specs, enc, g, h, config.batch_size, config.seed + 1,
                layer_cache=layer_cache, fixed_pairs=val_pairs,
            )
        return

    task = resolve_c1_task(config.c1_task_key, cropset=config.cropset)

    def _make_index(items) -> C1CropIndex:
        return C1CropIndex(items, task, enc, g, h, layer_cache=layer_cache)

    train_index = _make_index(train_items)
    val_index = train_index if config.val_same_as_train else _make_index(val_items)

    def _make_loader(index: C1CropIndex, seed: int):
        if not use_gpu_loader:
            return C1BatchLoader(index, config.batch_size, seed)
        lc = layer_cache
        if lc is None:
            lc = LayerCache()
            lc.warm({it.labels_path for it in index.items}, log=config.log)
        store = C1GpuResidentStore.from_items(index.items, lc, device, log=config.log)
        crop_ix = _crop_ix_tensor(index.items).to(device)
        return C1GpuBatchLoader(store, crop_ix, index, config.batch_size, seed)

    train_loader = _make_loader(train_index, config.seed)
    val_loader = _make_loader(val_index, config.seed + 1)


def _load_worldsnap_c2():
    """c2: GPU-resident batches (fixed crops or on-the-fly district windows) + full val grid."""
    global train_loader, val_loader, _n_val_crops
    from spatial_data.c2_tasks import get_c2_task
    from spatial_data.dataset_c2_gpu import (
        C2DistrictSampler,
        C2GpuTrainLoader,
        C2GpuValLoader,
        held_out_rects,
        load_crop_planes,
        load_district_planes,
    )

    if device is None or device.type != "cuda":
        raise RuntimeError("c2 loaders are GPU-resident; set config.use_gpu and run on cuda")
    config.c2_active_task_keys = config.c2_task_keys_active()
    keys = config.c2_active_task_keys
    specs = [get_c2_task(k) for k in keys]
    crops_dir = config.c2_crops_dir()
    enc, g, h = config.encoding_mode, config.grid_out_size, config.img_size
    t0 = time.perf_counter()
    val_planes = load_crop_planes(
        crops_dir, "val", device, max_items=config.worldsnap_max_val_items,
    )
    _n_val_crops = int(val_planes.shape[0])
    train_task_ix = config.c2_train_task_indices_active()
    if config.c2_train_source == "district_random":
        sampler = C2DistrictSampler(
            load_district_planes(crops_dir, device), held_out_rects(crops_dir), config.seed,
        )
        train_loader = C2GpuTrainLoader(
            specs, enc, g, h, config.batch_size, config.seed,
            sampler=sampler, train_task_indices=train_task_ix,
        )
    elif config.c2_train_source == "fixed_crops":
        train_planes = load_crop_planes(
            crops_dir, "train", device, max_items=config.worldsnap_max_train_items,
        )
        train_loader = C2GpuTrainLoader(
            specs, enc, g, h, config.batch_size, config.seed,
            crops=train_planes, train_task_indices=train_task_ix,
        )
    else:
        raise ValueError(f"unknown c2_train_source={config.c2_train_source!r}")
    val_loader = C2GpuValLoader(val_planes, specs, enc, g, h, config.batch_size)
    config.log(
        f"  worldsnap c2: set={config.c2_task_set} tasks={len(keys)} "
        f"train_source={config.c2_train_source} encoding={enc} "
        f"val_grid={_n_val_crops}x{len(keys)}={len(val_loader)} "
        f"({time.perf_counter() - t0:.1f}s)"
        + (f", train_tasks={len(train_task_ix)}" if train_task_ix is not None else "")
    )


def _load_worldsnap_c3():
    import spatial_data.channels as _ch
    from eval_spatial import reset_c3_geometry_oracle_cache

    reset_c3_geometry_oracle_cache()
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)  # experiment override -> plane function
    global train_loader, val_loader, _n_val_crops
    from spatial_data.c3_tasks import get_c3_task
    from spatial_data.c3_markers import marker_tensors
    from spatial_data.dataset_c3_gpu import (
        C3GpuTrainLoader,
        C3GpuValLoader,
        C3MultiTaskGpuTrainLoader,
        C3MultiTaskGpuValLoader,
        load_crop_planes_v4,
    )

    if device is None or device.type != "cuda":
        raise RuntimeError("c3 loaders are GPU-resident; set config.use_gpu and run on cuda")
    multi = config.c3_is_multi_task()
    keys = list(config.c3_task_keys) if multi else [config.c3_task_key]
    config.c3_active_task_keys = keys
    spec = get_c3_task(keys[0])
    crops_dir = config.c3_crops_dir()
    district_dir = config.district_data_dir()
    enc, g, h = config.encoding_mode, config.grid_out_size, config.img_size
    t0 = time.perf_counter()
    val_planes = load_crop_planes_v4(
        crops_dir, "val", device, max_items=config.worldsnap_max_val_items,
    )
    _n_val_crops = int(val_planes.shape[0])
    train_planes = load_crop_planes_v4(
        crops_dir, "train", device, max_items=config.worldsnap_max_train_items,
    )
    need_marker = multi or spec.show_marker_plane
    mcol = mrow = val_mcol = val_mrow = None
    if need_marker:
        mcol, mrow = marker_tensors(crops_dir, district_dir, "train", device)
        val_mcol, val_mrow = marker_tensors(crops_dir, district_dir, "val", device)
    rnd = None
    rmax = int(getattr(config, "c3_n_markers_random_max", 0) or 0)
    rmin = int(getattr(config, "c3_n_markers_random_min", 0) or 0)
    if rmax >= rmin and rmax >= 1 and rmin >= 1:
        rnd = (rmin, rmax)
    if multi:
        assert mcol is not None and mrow is not None and val_mcol is not None and val_mrow is not None
        tw = config.c3_train_task_weights
        train_loader = C3MultiTaskGpuTrainLoader(
            keys, enc, g, h, config.batch_size, config.seed,
            crops=train_planes, marker_col=mcol, marker_row=mrow,
            n_markers=config.c3_n_markers, marker_min_dist_px=config.c3_marker_min_dist_m,
            n_markers_random=rnd, train_task_weights=tw,
        )
        val_loader = C3MultiTaskGpuValLoader(
            val_planes, keys, enc, g, h, config.batch_size,
            marker_col=val_mcol, marker_row=val_mrow,
        )
        wnote = f" weights={tw}" if tw else ""
        task_note = f"multi tasks={keys}{wnote}"
    else:
        train_loader = C3GpuTrainLoader(
            spec, enc, g, h, config.batch_size, config.seed,
            crops=train_planes, marker_col=mcol, marker_row=mrow,
            n_markers=config.c3_n_markers, marker_min_dist_px=config.c3_marker_min_dist_m,
            n_markers_random=rnd,
        )
        val_loader = C3GpuValLoader(
            val_planes, spec, enc, g, h, config.batch_size,
            marker_col=val_mcol, marker_row=val_mrow,
        )
        task_note = f"task={config.c3_task_key}"
    config.log(
        f"  worldsnap c3: {task_note} encoding={enc} "
        f"train={len(train_loader)} val={len(val_loader)} ({time.perf_counter() - t0:.1f}s)"
    )


def _load_worldsnap_c4():
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)
    global train_loader, val_loader, _n_val_crops
    from spatial_data.c4_tasks import get_c4_task
    from spatial_data.dataset_c4_gpu import (
        C4GpuTrainLoader,
        C4GpuValLoader,
        C4MultiTaskGpuTrainLoader,
        C4MultiTaskGpuValLoader,
        load_c4_route_index,
        load_district_planes_v4,
    )

    if device is None or device.type != "cuda":
        raise RuntimeError("c4 loaders are GPU-resident; set config.use_gpu and run on cuda")
    multi = config.c4_is_multi_task()
    c4_keys = list(config.c4_task_keys) if multi else [config.c4_task_key]
    mix_c3 = list(config.c4_mix_c3_task_keys or [])
    spec = get_c4_task(c4_keys[0])
    crops_dir = config.c3_crops_dir()
    routes_dir = config.c4_routes_dir()
    enc, g = config.encoding_mode, config.grid_out_size
    t0 = time.perf_counter()
    district = load_district_planes_v4(crops_dir, device)
    val_routes = load_c4_route_index(
        routes_dir, "val", device, max_items=config.worldsnap_max_val_items,
    )
    _n_val_crops = None  # val = route×task pairs (len(val_loader)), not route count alone
    train_routes = load_c4_route_index(
        routes_dir, "train", device, max_items=config.worldsnap_max_train_items,
    )
    if config.c4_max_straight_m is not None:
        def _filt(ri):
            keep = ri["straight_px"] <= float(config.c4_max_straight_m)
            return {k: v[keep] for k, v in ri.items()}
        n_tr, n_va = train_routes["origin"].shape[0], val_routes["origin"].shape[0]
        train_routes, val_routes = _filt(train_routes), _filt(val_routes)
        config.log(f"  c4 max_straight_m={config.c4_max_straight_m}: train {n_tr}->{train_routes['origin'].shape[0]}, "
                   f"val {n_va}->{val_routes['origin'].shape[0]}")
    eval_key = config.c4_eval_task_key

    def _val_loader():
        if eval_key == "segment":
            from spatial_data.dataset_c4_mixed_gpu import c4_segment_val_loader
            return c4_segment_val_loader(
                enc, g, config.batch_size, district=district, route_index=val_routes,
            )
        if eval_key == "detour":
            return C4GpuValLoader(
                get_c4_task("detour"), enc, g, config.batch_size,
                district=district, route_index=val_routes,
            )
        if eval_key != "segment" and multi:
            return C4MultiTaskGpuValLoader(
                c4_keys, enc, g, config.batch_size,
                district=district, route_index=val_routes,
            )
        return C4GpuValLoader(
            get_c4_task(eval_key), enc, g, config.batch_size,
            district=district, route_index=val_routes,
        )

    if mix_c3:
        from spatial_data.dataset_c3_gpu import load_crop_planes_v4
        from spatial_data.dataset_c4_mixed_gpu import C4C3MixGpuTrainLoader
        train_crops = load_crop_planes_v4(
            crops_dir, "train", device, max_items=config.worldsnap_max_train_items,
        )
        k3 = int(config.c3_n_markers) if config.c3_n_markers > 0 else 2
        train_loader = C4C3MixGpuTrainLoader(
            mix_c3, c4_keys, enc, g, config.img_size, config.batch_size, config.seed,
            c3_crops=train_crops, c4_district=district, c4_route_index=train_routes,
            c3_n_markers=k3, c3_marker_min_dist_px=config.c3_marker_min_dist_m,
            train_task_weights=config.c4_train_task_weights,
        )
        val_loader = _val_loader()
        config.c4_active_task_keys = mix_c3 + c4_keys
        tw = config.c4_train_task_weights
        wnote = f" weights={tw}" if tw else ""
        task_note = f"mix c3={mix_c3} c4={c4_keys}{wnote} val={eval_key}"
    elif multi:
        tw = config.c4_train_task_weights
        train_loader = C4MultiTaskGpuTrainLoader(
            c4_keys, enc, g, config.batch_size, config.seed,
            district=district, route_index=train_routes, train_task_weights=tw,
        )
        val_loader = _val_loader()
        config.c4_active_task_keys = c4_keys
        wnote = f" weights={tw}" if tw else ""
        task_note = f"multi tasks={c4_keys}{wnote} val={eval_key}"
    else:
        train_loader = C4GpuTrainLoader(
            spec, enc, g, config.batch_size, config.seed,
            district=district, route_index=train_routes,
        )
        val_loader = _val_loader()
        config.c4_active_task_keys = c4_keys
        task_note = f"task={config.c4_task_key} val={eval_key}"
    val_n = len(val_loader)
    train_n = len(train_loader)
    config.log(
        f"  worldsnap c4: {task_note} encoding={enc} "
        f"train={train_n} val={val_n} ({time.perf_counter() - t0:.1f}s)"
    )


def build_c4_val_loader(routes_subdir: str):
    """C4 val loader for an alternate route store (diagnostics only; does not replace val_loader)."""
    import spatial_data.channels as _ch
    _ch.MARKER_RADIUS_PX = int(config.marker_radius_px)
    from spatial_data.c4_tasks import get_c4_task
    from spatial_data.dataset_c4_gpu import (
        C4GpuValLoader,
        C4MultiTaskGpuValLoader,
        load_c4_route_index,
        load_district_planes_v4,
    )
    if device is None or device.type != "cuda":
        raise RuntimeError("c4 loaders are GPU-resident; set config.use_gpu and run on cuda")
    multi = config.c4_is_multi_task()
    c4_keys = list(config.c4_task_keys) if multi else [config.c4_task_key]
    crops_dir = config.c3_crops_dir()
    routes_dir = crops_dir / routes_subdir
    enc, g = config.encoding_mode, config.grid_out_size
    district = load_district_planes_v4(crops_dir, device)
    val_routes = load_c4_route_index(
        routes_dir, "val", device, max_items=config.worldsnap_max_val_items,
    )
    if config.c4_max_straight_m is not None:
        keep = val_routes["straight_px"] <= float(config.c4_max_straight_m)
        val_routes = {k: v[keep] for k, v in val_routes.items()}
    eval_key = config.c4_eval_task_key
    if eval_key == "segment":
        from spatial_data.dataset_c4_mixed_gpu import c4_segment_val_loader
        return c4_segment_val_loader(
            enc, g, config.batch_size, district=district, route_index=val_routes,
        )
    if eval_key == "detour":
        return C4GpuValLoader(
            get_c4_task("detour"), enc, g, config.batch_size,
            district=district, route_index=val_routes,
        )
    if eval_key != "segment" and multi:
        return C4MultiTaskGpuValLoader(
            c4_keys, enc, g, config.batch_size,
            district=district, route_index=val_routes,
        )
    return C4GpuValLoader(
        get_c4_task(eval_key), enc, g, config.batch_size,
        district=district, route_index=val_routes,
    )


def _load_worldsnap():
    global train_loader, val_loader
    if config.task_rung == "c4":
        _load_worldsnap_c4()
        return
    if config.task_rung == "c3":
        _load_worldsnap_c3()
        return
    if config.task_rung == "c2":
        _load_worldsnap_c2()
        return
    if config.task_rung == "c1":
        _load_worldsnap_c1()
        return
    from spatial_data.dataset_c0 import (
        C0BatchLoader,
        C0JsonlIndex,
        default_task_paths,
        load_c0_items,
        materialize_c0_tensors,
    )
    from spatial_data.layer_cache import LayerCache

    use_gpu_loader = (
        config.worldsnap_gpu_resident
        and device is not None
        and device.type == "cuda"
    )
    if config.worldsnap_gpu_resident and not use_gpu_loader:
        config.log(
            "  worldsnap: worldsnap_gpu_resident=True but no CUDA — using CPU C0BatchLoader",
            force=True,
        )
    district = config.district_data_dir()
    crops_dir = district / "crops" / config.cropset
    task_paths = default_task_paths(district, config.taskset)
    config.log(
        f"  worldsnap: district={district.name} cropset={config.cropset} "
        f"taskset={config.taskset} encoding={config.encoding_mode}"
    )
    t0 = time.perf_counter()
    train_items = load_c0_items(
        task_paths["train"], crops_dir, max_items=config.worldsnap_max_train_items,
    )
    enc = config.encoding_mode
    g, h = config.grid_out_size, config.img_size
    val_items: list | None = None
    if not config.val_same_as_train:
        val_items = load_c0_items(
            task_paths["val"], crops_dir, max_items=config.worldsnap_max_val_items,
        )
    layer_cache = None
    if config.worldsnap_cache_layers:
        layer_cache = LayerCache()
        all_paths = {it.labels_path for it in train_items}
        if val_items is not None:
            all_paths.update(it.labels_path for it in val_items)
        layer_cache.warm(all_paths, log=config.log)
    mat_limit = config.worldsnap_materialize_items

    def _make_index(items: list, label: str) -> C0JsonlIndex:
        mat = None
        if mat_limit is not None and len(items) <= mat_limit:
            if layer_cache is None:
                layer_cache_local = LayerCache()
                layer_cache_local.warm({it.labels_path for it in items}, log=config.log)
            else:
                layer_cache_local = layer_cache
            config.log(f"  worldsnap: materializing {len(items)} {label} tensors (limit={mat_limit})")
            mat = materialize_c0_tensors(
                items, enc, g, h, layer_cache_local, log=config.log,
            )
        return C0JsonlIndex(
            items, enc, g, h, layer_cache=layer_cache, materialized=mat,
        )

    train_index = _make_index(train_items, "train")
    if config.val_same_as_train:
        config.log("  data: val_same_as_train — val uses train item index")
        val_index = train_index
    else:
        val_index = _make_index(val_items, "val")
    n_val = len(val_index)
    cap_note = ""
    if config.worldsnap_max_train_items is not None:
        cap_note = f" (train cap={config.worldsnap_max_train_items})"
    cache_note = ""
    if layer_cache is not None:
        cache_note = f", layer_cache={len(layer_cache)} npz"
    if train_index._materialized is not None:
        cache_note += ", train materialized"
    if val_index is not train_index and val_index._materialized is not None:
        cache_note += ", val materialized"
    gpu_note = ", gpu_resident_batches" if use_gpu_loader else ""
    config.log(
        f"  worldsnap: indexed {len(train_items)} train + {n_val} val items{cap_note} "
        f"({time.perf_counter() - t0:.1f}s{cache_note}{gpu_note})"
    )

    def _make_loader(index: C0JsonlIndex, seed: int, label: str):
        if not use_gpu_loader:
            return C0BatchLoader(index, config.batch_size, seed)
        from spatial_data.dataset_c0_gpu import C0GpuBatchLoader, C0GpuResidentStore
        lc = layer_cache
        if lc is None:
            lc = LayerCache()
            lc.warm({it.labels_path for it in index.items}, log=config.log)
        store = C0GpuResidentStore.from_items(index.items, lc, device, log=config.log)
        if str(device).startswith("cuda"):
            alloc_mb = torch.cuda.memory_allocated(device) / 1e6
            config.log(f"  worldsnap: {label} gpu_store cuda allocated ~{alloc_mb:.0f} MB")
        return C0GpuBatchLoader(store, index, config.batch_size, seed)

    train_loader = _make_loader(train_index, config.seed, "train")
    val_loader = _make_loader(val_index, config.seed + 1, "val")


def load_data():
    global train_loader, val_loader, _n_val_crops
    _n_val_crops = None
    if config.task_rung == "c2":
        config.ensure_c2_encoding()
    elif config.task_rung == "c3":
        config.ensure_c3_encoding()
    elif config.task_rung == "c4":
        config.ensure_c4_encoding()
    else:
        config.sync_in_chans_from_encoding()
    config.log(
        f"data.load_data: source={config.data_source!r} batch={config.batch_size} "
        f"in_chans={config.in_chans}"
    )
    if config.data_source == "synthetic":
        _load_synthetic()
    elif config.data_source == "worldsnap":
        _load_worldsnap()
    else:
        raise ValueError(f"unknown data_source={config.data_source!r}")
    config.log(
        f"data.load_data: ready train={len(train_loader)} val={len(val_loader)} "
        f"img=({config.in_chans},{config.img_size},{config.img_size}) "
        f"grid={config.grid_out_size}×{config.grid_out_size}"
    )


def train_sample_count() -> int:
    if train_loader is None:
        raise RuntimeError("call load_data() first")
    return len(train_loader)


def val_sample_count() -> int:
    if val_loader is None:
        raise RuntimeError("call load_data() first")
    return len(val_loader)


def val_crop_count() -> int:
    """Number of val crops (jsonl rows or npz files), not c1 (crop, task) pairs."""
    if val_loader is None:
        raise RuntimeError("call load_data() first")
    if _n_val_crops is not None:
        return _n_val_crops
    return len(val_loader)
