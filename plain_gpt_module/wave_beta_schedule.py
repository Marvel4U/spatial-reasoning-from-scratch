"""Schedule for ``wave_change_ce_beta`` during long C5 wave runs."""


def beta_thirds_decay(step: int, total_steps: int, *, beta_high: float) -> float:
    """First third at ``beta_high``, middle third linear → 0, last third at 0."""
    if total_steps <= 0:
        return beta_high
    t = total_steps / 3.0
    s = float(step)
    if s < t:
        return beta_high
    if s < 2.0 * t:
        return beta_high * (1.0 - (s - t) / t)
    return 0.0
