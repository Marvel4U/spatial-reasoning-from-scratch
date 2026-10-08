from plain_gpt_module.wave_beta_schedule import beta_thirds_decay


def test_beta_thirds_decay_12k():
    t = 12000
    assert beta_thirds_decay(0, t, beta_high=5.0) == 5.0
    assert beta_thirds_decay(3999, t, beta_high=5.0) == 5.0
    assert beta_thirds_decay(4000, t, beta_high=5.0) == 5.0
    mid = beta_thirds_decay(6000, t, beta_high=5.0)
    assert 0.0 < mid < 5.0
    assert beta_thirds_decay(8000, t, beta_high=5.0) == 0.0
    assert beta_thirds_decay(11999, t, beta_high=5.0) == 0.0
