"""Print config / scale sanity check at train startup."""
import config


def print_config_report(model, train_samples: int, val_samples: int, train_steps: int | None = None):
    n_params = sum(p.numel() for p in model.parameters())
    steps = train_steps if train_steps is not None else config.t_train
    samples_per_step = config.batch_size
    samples_seen = steps * samples_per_step
    g = config.img_size // config.patch_size
    print("=== config report (local grid ViT) ===")
    print(f"  code:      v{config.code_version}")
    print(f"  model:     enc_n_layer={config.enc_n_layer}, n_embd={config.n_embd}, h_heads={config.h_heads}")
    print(f"  grid:      in_chans={config.in_chans}, img={config.img_size}, patch={config.patch_size}, "
          f"out={config.grid_out_size}×{config.grid_out_size}, K={config.num_grid_classes}")
    print(f"  patch grid g={g}, subcells={config.grid_out_size // g}")
    print(f"  params:    {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  data:      {config.data_source}, rung={config.task_rung}, encoding={config.encoding_mode}")
    print(f"  train:     batch={config.batch_size}, lr={config.learning_rate:g}, steps={steps}")
    print(f"  compile:   {'torch.compile' if config.use_compile else 'off'}")
    print(f"  pool:      train {train_samples:,} samples, val {val_samples:,} samples")
    print(f"  this run:  ~{samples_seen:,} sample presentations")
    print("=====================")
