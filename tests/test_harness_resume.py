from run_log import (
    _resume_config_equal,
    should_skip_finished_run,
    training_setup_matches,
)


def test_config_list_tuple_roundtrip():
    prior_cfg = {"batch_size": 32, "diagnostics_checkpoint_steps": [0, 100, 1000]}
    fp_cfg = {"batch_size": 32, "diagnostics_checkpoint_steps": (0, 100, 1000)}
    assert _resume_config_equal(prior_cfg, fp_cfg)


def test_diagnostics_steps_ignored_for_resume_match():
    prior = {
        "architecture": "local_grid_vit",
        "arch_version": 1,
        "overrides": {"batch_size": 32},
        "init_checkpoint": None,
        "data": {"train_samples": 10, "val_samples": 5},
        "config": {
            "batch_size": 32,
            "diagnostics_checkpoint_steps": [0, 100],
            "diagnostics_enabled": True,
        },
    }
    fp = {
        "architecture": "local_grid_vit",
        "arch_version": 1,
        "overrides": {"batch_size": 32},
        "init_checkpoint": None,
        "data": {"train_samples": 10, "val_samples": 5},
        "config": {
            "batch_size": 32,
            "diagnostics_checkpoint_steps": (0, 100, 500),
            "diagnostics_enabled": False,
        },
    }
    assert training_setup_matches(prior, fp)


def test_skip_finished_when_ckpt_meets_target():
    assert should_skip_finished_run(
        force_restart=False,
        train_steps=8000,
        overrides={"a": 1},
        prior={"overrides": {"a": 1}, "aborted": True},
        ckpt_steps=7999,
    )
    assert not should_skip_finished_run(
        force_restart=False,
        train_steps=8000,
        overrides={"a": 1},
        prior={"overrides": {"a": 2}},
        ckpt_steps=7999,
    )


def test_skip_finished_prior_missing_new_override_keys():
    assert should_skip_finished_run(
        force_restart=False,
        train_steps=4000,
        overrides={"batch_size": 128, "seed": 4, "c4_task_key": "segment"},
        prior={"overrides": {"batch_size": 128, "c4_task_key": "segment"}},
        ckpt_steps=3999,
    )


def test_skip_finished_ignores_diagnostics_override_toggle():
    assert should_skip_finished_run(
        force_restart=False,
        train_steps=4000,
        overrides={"batch_size": 128, "diagnostics_enabled": True},
        prior={"overrides": {"batch_size": 128, "diagnostics_enabled": False}, "aborted": True},
        ckpt_steps=3999,
    )
    assert training_setup_matches(
        {
            "architecture": "local_grid_vit",
            "arch_version": 1,
            "overrides": {"batch_size": 128, "diagnostics_enabled": False},
            "init_checkpoint": None,
            "data": {"train_samples": 20000, "val_samples": 500},
            "config": {"batch_size": 128},
        },
        {
            "architecture": "local_grid_vit",
            "arch_version": 1,
            "overrides": {"batch_size": 128, "diagnostics_enabled": True},
            "init_checkpoint": None,
            "data": {"train_samples": 20000, "val_samples": 500},
            "config": {"batch_size": 128},
        },
    )
