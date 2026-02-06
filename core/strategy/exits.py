def should_exit(entry_price, current_price, sentiment_score, profit_target=0.01, stop_loss=0.005):
    """
    Determines whether to exit a position.
    - entry_price: price at which the position was entered
    - current_price: latest price
    - sentiment_score: current sentiment
    - profit_target: fraction of gain to take profit
    - stop_loss: fraction of loss to exit
    Returns True if exit conditions are met.
    """
    # Take profit
    if current_price >= entry_price * (1 + profit_target):
        return True

    # Stop loss
    if current_price <= entry_price * (1 - stop_loss):
        return True

    # Sentiment-based exit
    if sentiment_score < -0.3:
        return True

    return False
"""
Exit strategy helpers (placeholder for future logic).
"""
