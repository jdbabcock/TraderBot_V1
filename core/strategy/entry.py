def should_enter_trade(price_signal, sentiment_score, velocity, ml_signal):
    """
    Decide whether to enter a trade. Returns True or False.
    ml_signal is optional for AI-enhanced decision-making.
    """
    # must have price signal and sentiment aligned
    if not price_signal:
        return False
    if sentiment_score > 0 and velocity > 0:
        # combine with ML signal if available
        if ml_signal is None:
            return True
        return ml_signal
    return False
"""
Entry strategy helpers (placeholder for future logic).
"""
