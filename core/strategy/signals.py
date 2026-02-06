import ta

def compute_signals(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"]
    ).volume_weighted_average_price()

    latest = df.iloc[-1]

    score = 0.0

    # --- Trend component ---
    if latest["close"] > latest["ema_20"]:
        score += 0.4

    # --- VWAP positioning ---
    if latest["close"] > latest["vwap"]:
        score += 0.3

    # --- RSI momentum ---
    if latest["rsi"] > 55:
        score += 0.3
    elif latest["rsi"] > 45:
        score += 0.15

    # --- Debug output ---
    print(
        f"[SIGNAL SCORE] "
        f"Close={latest['close']:.2f} | "
        f"EMA20={latest['ema_20']:.2f} | "
        f"VWAP={latest['vwap']:.2f} | "
        f"RSI={latest['rsi']:.2f} | "
        f"Score={score:.2f}"
    )

    return score
"""
Signal generation helpers (placeholder for future logic).
"""
