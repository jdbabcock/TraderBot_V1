"""
Market data access with caching.

Responsibilities:
- Fetch OHLCV from Binance.US
- Cache per symbol/timeframe to reduce rate usage
"""
import pandas as pd
import time
import requests

# --- CACHE SETTINGS ---
_ohlcv_cache = {}
CACHE_EXPIRE_SECONDS = 30  # Only hit the API once every 20s per symbol/timeframe

def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 100, client=None, exchange: str = "binance") -> pd.DataFrame:
    """
    Fetch OHLCV candles with caching.
    
    :param symbol: BTCUSD, ETHUSD, etc.
    :param timeframe: '1m', '5m', '1h', '1d'
    :param limit: Number of candles (lower = better for rate limits)
    :param client: The active exchange client instance from main.py
    :param exchange: "binance" or "kraken"
    """
    global _ohlcv_cache
 
    
    # Ensure we have a client to work with (binance) or use public Kraken API
    if exchange == "binance" and client is None:
        print("⚠️ No Binance client provided to fetch_ohlcv")
        return pd.DataFrame(), time.time()

    # Normalize symbol
    clean_symbol = symbol.replace("/", "")
    if exchange == "kraken" and clean_symbol.startswith("BTC"):
        clean_symbol = clean_symbol.replace("BTC", "XBT", 1)
    cache_key = f"{clean_symbol}_{timeframe}"
    now = time.time()

    # 1. Check Cache Safeguard
    if cache_key in _ohlcv_cache:
        cached_df, timestamp = _ohlcv_cache[cache_key]
        if now - timestamp < CACHE_EXPIRE_SECONDS:
            return cached_df, timestamp

    try:
        # 2. Fetch Data (Optimized limit)
        if exchange == "kraken":
            tf_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
            interval = tf_map.get(timeframe, 5)
            res = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": clean_symbol, "interval": interval},
                timeout=15
            ).json()
            result = res.get("result", {})
            key = next((k for k in result.keys() if k != "last"), None)
            klines = result.get(key, []) if key else []
        else:
            # python-binance uses 'get_klines' for OHLCV
            klines = client.get_klines(
                symbol=clean_symbol,
                interval=timeframe,
                limit=limit
            )

        if not klines:
            return pd.DataFrame(), time.time()

        # 3. Process Data
        if exchange == "kraken":
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
        else:
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"
            ])

        # Convert to numeric and datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s" if exchange == "kraken" else "ms")
        cols_to_fix = ["open", "high", "low", "close", "volume"]
        df[cols_to_fix] = df[cols_to_fix].astype(float)

        # 4. Update Cache and Return
        _ohlcv_cache[cache_key] = (df, now)
        return df, now

    except Exception as e:
        print(f"⚠️ Error fetching OHLCV for {symbol}: {e}")
        if cache_key in _ohlcv_cache:
            return _ohlcv_cache[cache_key][0], _ohlcv_cache[cache_key][1] # Return cached time too
        return pd.DataFrame(), time.time() # Return current time if total fail
