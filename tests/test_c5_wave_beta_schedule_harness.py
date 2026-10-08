"""Beta schedule must use experiment train goal, not default config.t_train."""
from plain_gpt_module.wave_beta_schedule import beta_thirds_decay


def test_step2000_of_16k_is_beta_high():
    assert beta_thirds_decay(2000, 16000, beta_high=5.0) == 5.0


def test_step2000_of_default_500_is_zero():
    assert beta_thirds_decay(2000, 500, beta_high=5.0) == 0.0
