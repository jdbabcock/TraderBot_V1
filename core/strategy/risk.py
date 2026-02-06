def compute_position_size(capital, size_fraction, perf_score=0.0, min_frac=0.05, max_frac=0.5):
    """
    Compute dynamic position size based on capital, base size fraction, and performance score.
    Positive perf_score increases size_fraction, negative decreases it.

    Args:
        capital (float): current capital
        size_fraction (float): base fraction suggested by LLM
        perf_score (float): recent performance score (-inf, +inf)
        min_frac (float): minimum size fraction
        max_frac (float): maximum size fraction

    Returns:
        float: position size
    """
    # Normalize perf_score to a moderate adjustment factor (-0.2 to +0.2)
    adjustment = max(-0.2, min(0.2, perf_score / 1000))  # tweak denominator for sensitivity
    dynamic_fraction = size_fraction * (1 + adjustment)

    # Clamp
    dynamic_fraction = max(min_frac, min(max_frac, dynamic_fraction))

    return capital * dynamic_fraction
