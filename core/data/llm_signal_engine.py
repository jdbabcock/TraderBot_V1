"""
LLM + sentiment pipeline.

Responsibilities:
- Fetch RSS sentiment and cache it
- Build LLM prompts and parse model output
- Normalize action schema and risk outputs
- Expose helper accessors used by the main loop/UI
"""
import random
import time
import threading
import os
import re
from textblob import TextBlob
import feedparser
import openai
import json
import pandas as pd

# -----------------------------
# Feature toggles
# -----------------------------
USE_LLM = True
TOKEN_FREE_MODE = False

# -----------------------------
# Prompt tracking (debug/UI)
# -----------------------------
LAST_PROMPT = {}

def get_last_prompt(symbol=None):
    if symbol:
        return LAST_PROMPT.get(symbol, "")
    return dict(LAST_PROMPT)

# -----------------------------
# OpenAI key (prefer env var)
# -----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY # Sets the key globally for the openai library

# -----------------------------
# LLM guardrails
# -----------------------------
LLM_SCHEMA_VERSION = 1
LLM_DEFAULTS = {
    "confidence": 0.5,
    "size_fraction": 0.1,
    "stop_loss_pct": 0.01,
    "take_profit_pct": 0.02,
    "trailing_stop_pct": 0.02,
    "exit": False,
    "reason": "",
    "pattern_reason": "",
    "order_action": None,
    "fill_rate_scale": 1.0,
    "fill_spread_sensitivity": 1.0,
    "fill_vol_sensitivity": 1.0
}
LLM_BOUNDS = {
    "confidence": (0.0, 1.0),
    "size_fraction": (0.01, 0.5),
    "stop_loss_pct": (0.005, 0.05),
    "take_profit_pct": (0.01, 0.1),
    "trailing_stop_pct": (0.005, 0.08),
    "fill_rate_scale": (0.5, 2.0),
    "fill_spread_sensitivity": (0.5, 2.0),
    "fill_vol_sensitivity": (0.5, 2.0)
}
LLM_LOG_PATH = "llm_activity.log"

# -----------------------------
# RSS feeds for crypto news
# -----------------------------
CRYPTO_RSS_FEEDS = [
    "https://cryptonews.com/news/feed",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/"
]

# -----------------------------
# Cache & background fetching
# -----------------------------
SENTIMENT_CACHE = {}
CACHE_INTERVAL = 86400  # 24 hours
FETCH_INTERVAL = 600
LAST_RSS_FETCH_TIME = 0.0
RSS_ACTIVE = False
SYMBOLS_TO_TRACK = []

# -----------------------------
# Crypto-specific sentiment keywords
# -----------------------------
BULLISH_WORDS = ["moon", "pump", "bullish", "rally", "surge", "soars", "breakout", "gain"]
BEARISH_WORDS = ["dump", "drop", "bearish", "plunge", "crash", "rekt", "sell-off", "slump"]

# -----------------------------
# Internal function to fetch sentiment & volume context
# -----------------------------
def _fetch_sentiment(symbol, ohlcv_df=None):
    global LAST_RSS_FETCH_TIME
    now = time.time()
    keywords = symbol.split("/")[0].lower()
    total_score = 0
    count = 0

    for feed_url in CRYPTO_RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            title = entry.title.lower()
            summary = entry.get("summary", "").lower()
            published = getattr(entry, "published_parsed", None)

            if published:
                published_ts = time.mktime(published)
                if now - published_ts > CACHE_INTERVAL:
                    continue

            text = title + " " + summary
            if keywords not in text:
                continue

            polarity = TextBlob(text).sentiment.polarity
            for word in BULLISH_WORDS:
                if word in text:
                    polarity += 0.2
            for word in BEARISH_WORDS:
                if word in text:
                    polarity -= 0.2
            polarity = max(min(polarity, 1.0), -1.0)

            total_score += polarity
            count += 1

    sentiment = total_score / count if count > 0 else 0.0

    # Volume context
    current_vol = avg_vol = vol_ratio = None
    if ohlcv_df is not None and not ohlcv_df.empty:
        current_vol = ohlcv_df["volume"].iloc[-1]
        avg_vol = ohlcv_df["volume"].rolling(window=20).mean().iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol else 1.0

    SENTIMENT_CACHE[symbol] = {
        "score": sentiment,
        "timestamp": now,
        "current_vol": current_vol,
        "avg_vol": avg_vol,
        "vol_ratio": vol_ratio
    }

    print(f"[{time.strftime('%H:%M:%S')}] 📰 RSS updated for {symbol}: Sentiment={sentiment:.3f} | Vol={current_vol} | AvgVol={avg_vol} | Ratio={vol_ratio}")
    LAST_RSS_FETCH_TIME = time.time()
    return sentiment, current_vol, avg_vol, vol_ratio

# -----------------------------
# Public fetch function
# -----------------------------
def fetch_sentiment_score(symbol, ohlcv_df=None):
    now = time.time()

    if TOKEN_FREE_MODE:
        score = random.uniform(-1, 1)
        vol_info = (random.uniform(100, 1000), random.uniform(100, 1000), random.uniform(0.5, 1.5))
        print(f"[{time.strftime('%H:%M:%S')}] 🔹 TOKEN-FREE sentiment for {symbol}: {score:.3f}")
        return score, *vol_info

    cached = SENTIMENT_CACHE.get(symbol)
    if cached and now - cached["timestamp"] < CACHE_INTERVAL:
        print(f"[{time.strftime('%H:%M:%S')}] 🔹 Using cached sentiment for {symbol}: {cached['score']:.3f}")
        return cached["score"], cached.get("current_vol"), cached.get("avg_vol"), cached.get("vol_ratio")

    return _fetch_sentiment(symbol, ohlcv_df)

# -----------------------------
# Background RSS fetch
# -----------------------------
def _background_fetch_loop():
    global LAST_RSS_FETCH_TIME, RSS_ACTIVE
    while True:
        if SYMBOLS_TO_TRACK:
            print(f"[{time.strftime('%H:%M:%S')}] 🔹 Background RSS fetch starting...")
            for symbol in SYMBOLS_TO_TRACK:
                try:
                    _fetch_sentiment(symbol)
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Error fetching {symbol}: {e}")
            print(f"[{time.strftime('%H:%M:%S')}] 🔹 Background RSS fetch complete")
            LAST_RSS_FETCH_TIME = time.time()
            RSS_ACTIVE = True
        time.sleep(FETCH_INTERVAL)

def start_background_fetch(symbols):
    global SYMBOLS_TO_TRACK
    SYMBOLS_TO_TRACK = symbols
    threading.Thread(target=_background_fetch_loop, daemon=True).start()

def get_last_rss_fetch_time():
    return LAST_RSS_FETCH_TIME

def get_rss_active():
    return RSS_ACTIVE

# -----------------------------
# 🔥 LLM DECISION FUNCTION
# -----------------------------
def llm_decision(
    symbol,
    last_price,
    sentiment,
    velocity,
    positions,
    price_score=0.0,
    rsi=None,
    ema_20=None,
    vwap=None,
    atr_pct=None,
    recent_closes=None,
    recent_ohlcv=None,
    candle_patterns=None,
    current_vol=None,
    avg_vol=None,
    vol_ratio=None,
    perf_score=None,
    perf_summary=None,
    execution_context=None,
    risk_state=None,
    perf_snapshot=None,
    wallet_state=None
):
    """
    Returns a JSON object with LLM-driven trading signals:
    - confidence (0–1)
    - size_fraction (0–1)
    - stop_loss_pct
    - take_profit_pct
    - exit (bool)
    
    Technical indicators, recent price history, sentiment keywords, and volume context
    are passed for context. The LLM fully decides entries, exits, and sizing.
    """
    def _extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

    def _safe_float(val, default):
        try:
            if val is None:
                return float(default)
            if isinstance(val, str) and not val.strip():
                return float(default)
            return float(val)
        except Exception:
            return float(default)

    def _sanitize_llm_output(output):
        if not isinstance(output, dict):
            output = {}
        cleaned = {}
        cleaned["confidence"] = _safe_float(output.get("confidence", LLM_DEFAULTS["confidence"]), LLM_DEFAULTS["confidence"])
        cleaned["exit"] = bool(output.get("exit", LLM_DEFAULTS["exit"]))
        cleaned["reason"] = str(output.get("reason", LLM_DEFAULTS["reason"]))
        cleaned["pattern_reason"] = str(output.get("pattern_reason", LLM_DEFAULTS["pattern_reason"]))

        if cleaned["exit"]:
            cleaned["size_fraction"] = 0.0
        else:
            cleaned["size_fraction"] = _safe_float(output.get("size_fraction", LLM_DEFAULTS["size_fraction"]), LLM_DEFAULTS["size_fraction"])
            cleaned["size_fraction"] = _clamp(cleaned["size_fraction"], *LLM_BOUNDS["size_fraction"])

        cleaned["confidence"] = _clamp(cleaned["confidence"], *LLM_BOUNDS["confidence"])
        cleaned["stop_loss_pct"] = _safe_float(output.get("stop_loss_pct", LLM_DEFAULTS["stop_loss_pct"]), LLM_DEFAULTS["stop_loss_pct"])
        cleaned["take_profit_pct"] = _safe_float(output.get("take_profit_pct", LLM_DEFAULTS["take_profit_pct"]), LLM_DEFAULTS["take_profit_pct"])
        cleaned["trailing_stop_pct"] = _safe_float(output.get("trailing_stop_pct", LLM_DEFAULTS["trailing_stop_pct"]), LLM_DEFAULTS["trailing_stop_pct"])
        cleaned["stop_loss_pct"] = _clamp(cleaned["stop_loss_pct"], *LLM_BOUNDS["stop_loss_pct"])
        cleaned["take_profit_pct"] = _clamp(cleaned["take_profit_pct"], *LLM_BOUNDS["take_profit_pct"])
        cleaned["trailing_stop_pct"] = _clamp(cleaned["trailing_stop_pct"], *LLM_BOUNDS["trailing_stop_pct"])
        cleaned["fill_rate_scale"] = _safe_float(output.get("fill_rate_scale", LLM_DEFAULTS["fill_rate_scale"]), LLM_DEFAULTS["fill_rate_scale"])
        cleaned["fill_rate_scale"] = _clamp(cleaned["fill_rate_scale"], *LLM_BOUNDS["fill_rate_scale"])
        cleaned["fill_spread_sensitivity"] = _safe_float(output.get("fill_spread_sensitivity", LLM_DEFAULTS["fill_spread_sensitivity"]), LLM_DEFAULTS["fill_spread_sensitivity"])
        cleaned["fill_spread_sensitivity"] = _clamp(cleaned["fill_spread_sensitivity"], *LLM_BOUNDS["fill_spread_sensitivity"])
        cleaned["fill_vol_sensitivity"] = _safe_float(output.get("fill_vol_sensitivity", LLM_DEFAULTS["fill_vol_sensitivity"]), LLM_DEFAULTS["fill_vol_sensitivity"])
        cleaned["fill_vol_sensitivity"] = _clamp(cleaned["fill_vol_sensitivity"], *LLM_BOUNDS["fill_vol_sensitivity"])
        order_action = output.get("order_action")
        # Normalize LLM action schema variants
        if isinstance(order_action, dict):
            if "type" not in order_action and order_action.get("order_type"):
                order_action["type"] = order_action.get("order_type")
            if "action" not in order_action and order_action.get("side"):
                order_action["action"] = order_action.get("side")
            if "quantity" not in order_action:
                if order_action.get("size") is not None:
                    order_action["quantity"] = order_action.get("size")
                elif order_action.get("qty") is not None:
                    order_action["quantity"] = order_action.get("qty")
            if "size_fraction" not in order_action and order_action.get("quantity_fraction"):
                order_action["size_fraction"] = order_action.get("quantity_fraction")
        cleaned["order_action"] = order_action
        cleaned["schema_version"] = LLM_SCHEMA_VERSION
        return cleaned

    def _log_llm_output(symbol, raw_text, cleaned):
        try:
            record = {
                "timestamp": time.time(),
                "symbol": symbol,
                "raw": raw_text,
                "cleaned": cleaned
            }
            with open(LLM_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    if TOKEN_FREE_MODE or not USE_LLM or not OPENAI_KEY:
        return _sanitize_llm_output({
            "confidence": random.uniform(0.3, 0.6),
            "size_fraction": random.uniform(0.05, 0.3),
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "exit": False,
            "reason": "token_free_or_disabled"
        })

    try:
        if recent_closes is None:
            recent_closes = []

        recent_prices_str = ", ".join([f"{c:.2f}" for c in recent_closes[-20:]])
        recent_ohlcv_str = json.dumps(recent_ohlcv[-20:]) if recent_ohlcv else "N/A"
        candle_patterns_str = ", ".join(candle_patterns) if candle_patterns else "none"

        prompt = (
            "You are an expert crypto trading assistant. "
            "Make profitable trades while managing risk.\n\n"
            f"Symbol: {symbol}\n"
            f"Last price: {last_price}\n"
            f"Recent closes: [{recent_prices_str}]\n"
            f"Price velocity (% change): {velocity:.2f}\n"
            f"Sentiment score (-1 to 1): {sentiment:.2f}\n"
            f"Current open positions: {json.dumps(positions)}\n"
            f"Technical price score (0–1, bullish): {price_score:.2f}\n"
            f"RSI: {rsi}\nEMA-20: {ema_20}\nVWAP: {vwap}\n"
            f"ATR (% of price): {atr_pct}\n"
            f"Current volume: {current_vol}\nAverage volume: {avg_vol}\nVolume ratio: {vol_ratio}\n\n"
            f"Recent OHLCV candles (last 20): {recent_ohlcv_str}\n"
            f"Detected candle patterns: {candle_patterns_str}\n\n"
            "Use the recent OHLCV and candle patterns as part of your learning feedback loop when forming decisions.\n"
            f"Performance summary: {json.dumps(perf_summary) if perf_summary else 'N/A'}\n"
            f"Forward-test snapshot: {json.dumps(perf_snapshot) if perf_snapshot else 'N/A'}\n"
            f"Wallet state: {json.dumps(wallet_state) if wallet_state else 'N/A'}\n"
            f"Execution context: {json.dumps(execution_context) if execution_context else 'N/A'}\n"
            f"Risk state: {json.dumps(risk_state) if risk_state else 'N/A'}\n"
            "If confidence >= 0.4 and exit is false, include a concrete order_action.\n"
            "Return ONLY a JSON object with keys:\n"
            "confidence (0–1), size_fraction (0–1), stop_loss_pct, take_profit_pct, trailing_stop_pct, exit (bool), reason (short string), "
            "pattern_reason (short string explaining any candle pattern usage), order_action (object or null), fill_rate_scale (0.5-2.0), "
            "fill_spread_sensitivity (0.5-2.0), fill_vol_sensitivity (0.5-2.0)"
        )
        LAST_PROMPT[symbol] = prompt
        if perf_score is not None:
          prompt += f"Past trade performance score: {perf_score:.2f}\n"
        try:
            pattern_msg = candle_patterns_str if candle_patterns_str else "none"
            print(f"[LLM_INPUT] {symbol} patterns={pattern_msg} candles={len(recent_ohlcv) if recent_ohlcv else 0}")
        except Exception:
            pass

        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        llm_output = _extract_json(answer)
        cleaned = _sanitize_llm_output(llm_output)
        if cleaned.get("pattern_reason"):
            print(f"[LLM_PATTERN] {symbol} {cleaned.get('pattern_reason')}")
        _log_llm_output(symbol, answer, cleaned)
        return cleaned

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ LLM API error for {symbol}: {e}")
        return _sanitize_llm_output({
            "confidence": random.uniform(0.3, 0.6),
            "size_fraction": random.uniform(0.05, 0.3),
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "exit": False,
            "reason": "fallback"
        })

# -----------------------------
# Autopilot tuning
# -----------------------------
def llm_autopilot_tune(current_settings, trade_stats=None, execution_context=None):
    def _extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    if TOKEN_FREE_MODE or not USE_LLM or not OPENAI_KEY:
        return {}

    allowed_keys = [
        "min_confidence_to_order",
        "size_fraction",
        "stop_loss_pct_default",
        "take_profit_pct_default",
        "trailing_stop_pct_default",
        "cooldown_seconds",
        "max_trades_per_day",
        "daily_loss_limit_pct",
        "max_drawdown_pct",
        "max_total_exposure_pct",
        "min_exposure_resume_pct",
        "max_symbol_exposure_pct",
        "max_open_positions"
    ]

    prompt = (
        "You are tuning trading style parameters for an autopilot crypto trader. "
        "Return ONLY a JSON object containing any of these keys:\n"
        f"{', '.join(allowed_keys)}\n"
        "Do NOT include llm_check_interval. "
        "Only return keys you want to change. If no change, return {}.\n\n"
        f"Current settings: {json.dumps(current_settings)}\n"
        f"Trade stats: {json.dumps(trade_stats) if trade_stats else 'N/A'}\n"
        f"Execution context: {json.dumps(execution_context) if execution_context else 'N/A'}\n"
        "Make small, conservative adjustments based on recent performance."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        parsed = _extract_json(answer) or {}
        if not isinstance(parsed, dict):
            return {}
        filtered = {}
        for key in allowed_keys:
            if key in parsed:
                filtered[key] = parsed.get(key)
        return filtered
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] âš ï¸ Autopilot tune error: {e}")
        return {}

# -----------------------------
# Rebalance advisory
# -----------------------------
def llm_rebalance_advice(target_symbol, target_score, candidate_symbol, candidate_score, candidate_pnl_pct, execution_context=None):
    def _extract_json(text):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    if TOKEN_FREE_MODE or not USE_LLM or not OPENAI_KEY:
        return {"rebalance": False, "reason": "token_free_or_disabled"}

    prompt = (
        "You are advising whether to rebalance between crypto positions. "
        "Return ONLY JSON with keys: rebalance (true/false), reason (short string).\n\n"
        f"Target symbol (want to buy): {target_symbol}\n"
        f"Target score: {target_score:.3f}\n"
        f"Candidate symbol (would sell): {candidate_symbol}\n"
        f"Candidate score: {candidate_score:.3f}\n"
        f"Candidate PnL %: {candidate_pnl_pct:.2f}\n"
        f"Execution context: {json.dumps(execution_context) if execution_context else 'N/A'}\n"
        "Recommend rebalancing only if the target is materially stronger and it makes sense to rotate."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        parsed = _extract_json(answer) or {}
        rebalance = bool(parsed.get("rebalance", False))
        reason = str(parsed.get("reason", ""))[:200]
        return {"rebalance": rebalance, "reason": reason}
    except Exception as e:
        return {"rebalance": False, "reason": f"advisory_error:{e}"}
