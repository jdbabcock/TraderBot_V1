"""
Run entrypoint for the trading bot.

Responsibilities:
- Load config and environment variables
- Start background data feeds (RSS + websocket)
- Orchestrate LLM decision loop
- Route execution to live or sim clients
- Update the GUI dashboard
"""

print("🚀 Bot starting...")

import os
import time
import json
import math
import random
import threading
import traceback
import csv
import sys
import importlib
import re
import ast
import pprint
import pandas as pd
import ta
import openai
from dotenv import load_dotenv

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if getattr(sys, "frozen", False):
    ROOT_DIR = os.path.dirname(sys.executable)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)


# Internal Project Imports
from core.data.llm_signal_engine import fetch_sentiment_score, llm_decision, llm_autopilot_tune, llm_rebalance_advice, start_background_fetch, get_last_prompt, get_last_rss_fetch_time, get_rss_active, FETCH_INTERVAL, SENTIMENT_CACHE
from core.strategy.risk import compute_position_size
from execution.sim.sim_executor import AccountingAgent
from execution.sim.mock_wallet_store import MockWallet
from execution.live.account_sync import BinanceUSAccount
from execution.live.kraken_account import KrakenAccount
from execution.sim.sim_exchange import SimBinanceClient
from execution.live.live_exchange import LiveBinanceClient
from execution.live.kraken_exchange import LiveKrakenClient
from execution.live.live_positions import LivePortfolio
from rl.agent import execute_llm_action
from ui.dashboard import RealTimeEquityPlot
from ui.startup_screen import run_startup_ui
from core.data.binance_streams import set_ws_debug

def _backfill_order_actions_logs():
    try:
        from utils.backfill_order_actions import main as _bf_main
        _bf_main()
    except Exception:
        pass

def panic_sell_all(execution_client, portfolio, live_prices):
    """
    Emergency liquidation: Cancels all orders and sells all held assets.
    """
    print("\n🚨 [PANIC] EMERGENCY LIQUIDATION TRIGGERED!")
    
    # 1. Cancel all open orders on exchange
    try:
        open_orders = execution_client.get_open_orders()
        if isinstance(open_orders, dict):
            open_orders = list(open_orders.values())
        for order in open_orders or []:
            symbol = order.get("symbol") or order.get("pair")
            order_id = order.get("orderId") or order.get("id") or order.get("txid")
            if order_id:
                execution_client.cancel_order(orderId=order_id, symbol=symbol)
                print(f"🛑 Cancelled Order: {symbol}")
    except Exception as e:
        print(f"⚠️ Error cancelling orders: {e}")

    # 2. Sell everything we can see in balances (more reliable than positions)
    try:
        balances = execution_client.get_balance() if execution_client else {}
        if isinstance(balances, dict) and "result" in balances:
            balances = balances.get("result", {})
    except Exception:
        balances = {}

    assets_to_sell = []
    if isinstance(balances, dict):
        for asset, info in balances.items():
            if isinstance(info, dict):
                qty = float(info.get("free", 0.0) or 0.0)
            else:
                try:
                    qty = float(info or 0.0)
                except Exception:
                    qty = 0.0
            if qty <= 0:
                continue
            norm_asset = _normalize_asset_code(asset)
            if norm_asset in ("USD", "USDT", "ZUSD"):
                continue
            assets_to_sell.append((norm_asset, qty))

    # Fallback to portfolio positions if balances are empty
    if not assets_to_sell:
        for symbol in list(portfolio.positions.keys()):
            asset = symbol.replace("USD", "").replace("/", "")
            assets_to_sell.append((asset, float(portfolio.positions.get(symbol, {}).get("size", 0.0) or 0.0)))

    for asset, qty in assets_to_sell:
        if qty <= 0:
            continue
        symbol = f"{asset}/USD"
        try:
            v_qty, _ = execution_client.validate_order(symbol, qty)
            execution_client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=v_qty
            )
            print(f"🔥 Market Sold: {symbol} | Qty: {v_qty}")
        except Exception as e:
            print(f"❌ Failed to liquidate {symbol}: {e}")

        # Clear it from the Paper Trader UI
        try:
            if hasattr(portfolio, "exit"):
                portfolio.exit(symbol, price=live_prices.get(symbol, 0))
        except Exception:
            pass

    if not hasattr(portfolio, "exit"):
        portfolio.positions = {}
    print("✅ Liquidation Complete. Bot is now flat.\n")


def emergency_stop(live_client, portfolio, live_prices):
    """
    Emergency kill switch: close all trades and stop the bot loop.
    """
    try:
        if live_client is not None and EXCHANGE == "binance":
            import kill_switch
            kill_switch.stop_everything()
        else:
            panic_sell_all(live_client, portfolio, live_prices)
    except Exception:
        try:
            panic_sell_all(live_client, portfolio, live_prices)
        except Exception:
            pass
    global _panic_requested
    _panic_requested = True
    
def _resolve_config_base_dir(default_dir):
    if not getattr(sys, "frozen", False):
        return default_dir
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    temp_markers = ("\\appdata\\local\\temp\\", "/tmp/", "_mei")
    if any(marker in exe_dir.lower() for marker in temp_markers):
        local_appdata = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(local_appdata, "TraderBot")
    return exe_dir

def _prepare_runtime_config(base_dir):
    config_dir = os.path.join(base_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    target_config = os.path.join(config_dir, "config.py")
    target_env = os.path.join(config_dir, ".env")
    if not os.path.exists(target_config):
        # Copy bundled config if running from a PyInstaller bundle
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundled_config = os.path.join(sys._MEIPASS, "config", "config.py")
            if os.path.exists(bundled_config):
                with open(bundled_config, "rb") as fsrc, open(target_config, "wb") as fdst:
                    fdst.write(fsrc.read())
        else:
            # Dev mode fallback: use repo config
            repo_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.py")
            if os.path.exists(repo_config):
                with open(repo_config, "rb") as fsrc, open(target_config, "wb") as fdst:
                    fdst.write(fsrc.read())
    if not os.path.exists(target_env):
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundled_env = os.path.join(sys._MEIPASS, "config", ".env")
            if os.path.exists(bundled_env):
                with open(bundled_env, "rb") as fsrc, open(target_env, "wb") as fdst:
                    fdst.write(fsrc.read())
        else:
            repo_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env")
            if os.path.exists(repo_env):
                with open(repo_env, "rb") as fsrc, open(target_env, "wb") as fdst:
                    fdst.write(fsrc.read())
    return target_config, target_env

# -----------------------------
# 1. Startup Screen (pre-config)
# -----------------------------
CONFIG_BASE_DIR = _resolve_config_base_dir(ROOT_DIR)
CONFIG_PATH, ENV_PATH = _prepare_runtime_config(CONFIG_BASE_DIR)
run_startup_ui(CONFIG_PATH)

# -----------------------------
# 2. Bot Configuration
# -----------------------------
def _is_real_key(value):
    if not value:
        return False
    v = str(value).strip()
    if not v:
        return False
    placeholder_tokens = (
        "YOUR_",
        "_HERE",
        "CHANGEME",
        "REPLACE_ME",
        "PLACEHOLDER",
        "NOT_SET",
        "NOT SET",
    )
    v_upper = v.upper()
    return not any(token in v_upper for token in placeholder_tokens)


load_dotenv(ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"🔐 OpenAI key set: {_is_real_key(os.getenv('OPENAI_API_KEY'))}")
try:
    import core.data.llm_signal_engine as llm_signal_engine
    llm_signal_engine.OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    llm_signal_engine.openai.api_key = os.getenv("OPENAI_API_KEY")
    if bot_config:
        llm_signal_engine.CONTRARIAN_SENTIMENT_ENABLED = bool(getattr(bot_config, "CONTRARIAN_SENTIMENT_ENABLED", llm_signal_engine.CONTRARIAN_SENTIMENT_ENABLED))
        llm_signal_engine.CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD = float(getattr(bot_config, "CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD", llm_signal_engine.CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD))
        llm_signal_engine.CONTRARIAN_SENTIMENT_MAX_WEIGHT = float(getattr(bot_config, "CONTRARIAN_SENTIMENT_MAX_WEIGHT", llm_signal_engine.CONTRARIAN_SENTIMENT_MAX_WEIGHT))
        llm_signal_engine.CONTRARIAN_SENTIMENT_LOG = bool(getattr(bot_config, "CONTRARIAN_SENTIMENT_LOG", llm_signal_engine.CONTRARIAN_SENTIMENT_LOG))
except Exception:
    pass

bot_config = None
config_load_error = None
try:
    from config import config as bot_config
except Exception as e:
    config_load_error = e

# Env vars are the source of truth for API keys (.env).
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    import core.data.llm_signal_engine as llm_signal_engine
    llm_signal_engine.OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    llm_signal_engine.openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    pass

CAPITAL = getattr(bot_config, "CAPITAL", 10000) if bot_config else 10000
USE_LLM = getattr(bot_config, "USE_LLM", True) if bot_config else True
MIN_NOTIONAL = getattr(bot_config, "MIN_NOTIONAL", 10.0) if bot_config else 10.0
KRAKEN_COST_MIN_USD = float(getattr(bot_config, "KRAKEN_COST_MIN_USD", 0.5)) if bot_config else 0.5
ALLOW_MIN_UPSIZE = bool(getattr(bot_config, "ALLOW_MIN_UPSIZE", True)) if bot_config else True
QTY_STEP = getattr(bot_config, "QTY_STEP", 0.0001) if bot_config else 0.0001
ENABLE_REBALANCE = bool(getattr(bot_config, "ENABLE_REBALANCE", True)) if bot_config else True
REBALANCE_SELL_FRACTION = float(getattr(bot_config, "REBALANCE_SELL_FRACTION", 0.25)) if bot_config else 0.25
REBALANCE_MIN_SCORE_DELTA = float(getattr(bot_config, "REBALANCE_MIN_SCORE_DELTA", 0.25)) if bot_config else 0.25
REBALANCE_MIN_HOLD_SECONDS = int(getattr(bot_config, "REBALANCE_MIN_HOLD_SECONDS", 600)) if bot_config else 600
REBALANCE_COOLDOWN_SECONDS = int(getattr(bot_config, "REBALANCE_COOLDOWN_SECONDS", 600)) if bot_config else 600
REBALANCE_PREFER_LOSERS = bool(getattr(bot_config, "REBALANCE_PREFER_LOSERS", True)) if bot_config else True
REBALANCE_ADVISORY_MODE = bool(getattr(bot_config, "REBALANCE_ADVISORY_MODE", True)) if bot_config else True
TARGET_ALLOCATION = getattr(bot_config, "TARGET_ALLOCATION", {"BTC": (0.4, 0.6), "ETH": (0.2, 0.4), "SOL": (0.1, 0.2)}) if bot_config else {"BTC": (0.4, 0.6), "ETH": (0.2, 0.4), "SOL": (0.1, 0.2)}
PNL_EXIT_MAX_DRAWDOWN_PCT = float(getattr(bot_config, "PNL_EXIT_MAX_DRAWDOWN_PCT", 0.08)) if bot_config else 0.08
PNL_EXIT_LOSER_THRESHOLD_PCT = float(getattr(bot_config, "PNL_EXIT_LOSER_THRESHOLD_PCT", -0.05)) if bot_config else -0.05
RESET_SIM_WALLET_ON_START = getattr(bot_config, "RESET_SIM_WALLET_ON_START", False) if bot_config else False
RUN_MODE = getattr(bot_config, "RUN_MODE", "live") if bot_config else "live"
EXCHANGE = getattr(bot_config, "EXCHANGE", "binance") if bot_config else "binance"
TRADING_STYLE = getattr(bot_config, "TRADING_STYLE", "balanced") if bot_config else "balanced"
SYMBOLS = getattr(bot_config, "SYMBOLS", ["SOL/USD", "BTC/USD", "ETH/USD"]) if bot_config else ["SOL/USD", "BTC/USD", "ETH/USD"]
TIMEFRAME = getattr(bot_config, "TIMEFRAME", "5m") if bot_config else "5m"
LLM_CHECK_INTERVAL = getattr(bot_config, "LLM_CHECK_INTERVAL", 300) if bot_config else 300
ACCOUNT_INFO_REFRESH_SECONDS = getattr(bot_config, "ACCOUNT_INFO_REFRESH_SECONDS", 60) if bot_config else 60
LIVE_TRADES_REFRESH_SECONDS = getattr(bot_config, "LIVE_TRADES_REFRESH_SECONDS", 15) if bot_config else 15
UI_REFRESH_SECONDS = getattr(bot_config, "UI_REFRESH_SECONDS", 1) if bot_config else 1
DEBUG_STATUS = getattr(bot_config, "DEBUG_STATUS", False) if bot_config else False
DEBUG_LOG_ATTEMPTS = getattr(bot_config, "DEBUG_LOG_ATTEMPTS", False) if bot_config else False
RESET_DAILY_RISK_ON_START = getattr(bot_config, "RESET_DAILY_RISK_ON_START", True) if bot_config else True
PERF_GUARD_PERSIST = bool(getattr(bot_config, "PERF_GUARD_PERSIST", True)) if bot_config else True
PERF_GUARD_MODE = str(getattr(bot_config, "PERF_GUARD_MODE", "rolling")) if bot_config else "rolling"
PERF_GUARD_LOOKBACK_TRADES = int(getattr(bot_config, "PERF_GUARD_LOOKBACK_TRADES", 200)) if bot_config else 200
STALE_PRICE_SECONDS = getattr(bot_config, "STALE_PRICE_SECONDS", 15) if bot_config else 15
STALE_WARN_INTERVAL_SECONDS = getattr(bot_config, "STALE_WARN_INTERVAL_SECONDS", 60) if bot_config else 60
STALE_GRACE_SECONDS = getattr(bot_config, "STALE_GRACE_SECONDS", 20) if bot_config else 20
ORDER_RETRY_SECONDS = getattr(bot_config, "ORDER_RETRY_SECONDS", 30) if bot_config else 30
BLOCK_ON_STALE_PRICE = bool(getattr(bot_config, "BLOCK_ON_STALE_PRICE", True)) if bot_config else True
REJECT_BACKOFF_SECONDS = int(getattr(bot_config, "REJECT_BACKOFF_SECONDS", 60)) if bot_config else 60
MAX_API_WEIGHT_1M = int(getattr(bot_config, "MAX_API_WEIGHT_1M", 1000)) if bot_config else 1000
MAX_API_WEIGHT_1M_KRAKEN = int(getattr(bot_config, "MAX_API_WEIGHT_1M_KRAKEN", 120)) if bot_config else 120
MAX_ORDER_COUNT_10S = int(getattr(bot_config, "MAX_ORDER_COUNT_10S", 8)) if bot_config else 8
ATTEMPT_LOG_COOLDOWN_SECONDS = int(getattr(bot_config, "ATTEMPT_LOG_COOLDOWN_SECONDS", 20)) if bot_config else 20
ATTEMPT_LOG_DEDUP_BY_REASON = bool(getattr(bot_config, "ATTEMPT_LOG_DEDUP_BY_REASON", True)) if bot_config else True
KRAKEN_MAKER_FEE_PCT = float(getattr(bot_config, "KRAKEN_MAKER_FEE_PCT", 0.0025)) if bot_config else 0.0025
KRAKEN_TAKER_FEE_PCT = float(getattr(bot_config, "KRAKEN_TAKER_FEE_PCT", 0.004)) if bot_config else 0.004
ESTIMATED_SLIPPAGE_PCT = float(getattr(bot_config, "ESTIMATED_SLIPPAGE_PCT", 0.0075)) if bot_config else 0.0075
STYLE_PRESETS = getattr(bot_config, "STYLE_PRESETS", {}) if bot_config else {}
DAILY_LOSS_LIMIT_PCT = float(getattr(bot_config, "DAILY_LOSS_LIMIT_PCT", 0.02)) if bot_config else 0.02
MAX_DRAWDOWN_PCT = float(getattr(bot_config, "MAX_DRAWDOWN_PCT", 0.05)) if bot_config else 0.05
MAX_TOTAL_EXPOSURE_PCT = float(getattr(bot_config, "MAX_TOTAL_EXPOSURE_PCT", 0.5)) if bot_config else 0.5
MIN_EXPOSURE_RESUME_PCT = float(getattr(bot_config, "MIN_EXPOSURE_RESUME_PCT", 0.2)) if bot_config else 0.2
MAX_SYMBOL_EXPOSURE_PCT = float(getattr(bot_config, "MAX_SYMBOL_EXPOSURE_PCT", 0.2)) if bot_config else 0.2
MAX_OPEN_POSITIONS = int(getattr(bot_config, "MAX_OPEN_POSITIONS", 3)) if bot_config else 3
set_ws_debug(getattr(bot_config, "DEBUG_STATUS", False) if bot_config else False)
# Track config file changes for live UI updates
_config_mtime = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else 0.0
DISPLAY_MODE = RUN_MODE
DISPLAY_STYLE = TRADING_STYLE

if config_load_error is not None:
    print(f"⚠️ Config load error: {config_load_error} (using defaults where needed)")

if EXCHANGE == "kraken":
    MAX_API_WEIGHT_1M = MAX_API_WEIGHT_1M_KRAKEN
    MIN_NOTIONAL = KRAKEN_COST_MIN_USD

print(f"🔧 Config: EXCHANGE={EXCHANGE} RUN_MODE={RUN_MODE}")
print(f"🧭 Trading Mode: {RUN_MODE.upper()}")
print(f"🎯 Trading Style: {TRADING_STYLE}")

_backfill_order_actions_logs()

# Sanity check for Kraken keys (presence only, never values)
if EXCHANGE == "kraken":
    kraken_key_ok = _is_real_key(os.getenv("KRAKEN_API_KEY"))
    kraken_secret_ok = _is_real_key(os.getenv("KRAKEN_API_SECRET"))
    print(f"🔑 Kraken keys set: key={kraken_key_ok} secret={kraken_secret_ok}")

style_settings = STYLE_PRESETS.get(TRADING_STYLE, {})
AUTOPILOT_ENABLED = TRADING_STYLE == "autopilot"
AUTOPILOT_SETTINGS = None
if AUTOPILOT_ENABLED:
    autopilot_seed = STYLE_PRESETS.get("balanced", style_settings) or {}
    AUTOPILOT_SETTINGS = dict(autopilot_seed)
    if "llm_check_interval" in style_settings:
        AUTOPILOT_SETTINGS["llm_check_interval"] = style_settings.get("llm_check_interval")
    style_settings = AUTOPILOT_SETTINGS
MIN_CONFIDENCE_TO_ORDER = float(style_settings.get("min_confidence_to_order", 0.4))
SIZE_FRACTION_DEFAULT = float(style_settings.get("size_fraction", 0.1))
LLM_CHECK_INTERVAL = int(style_settings.get("llm_check_interval", LLM_CHECK_INTERVAL))
STOP_LOSS_PCT_DEFAULT = float(style_settings.get("stop_loss_pct_default", 0.01))
TAKE_PROFIT_PCT_DEFAULT = float(style_settings.get("take_profit_pct_default", 0.02))
TRAILING_STOP_PCT_DEFAULT = float(style_settings.get("trailing_stop_pct_default", 0.015))
COOLDOWN_SECONDS = int(style_settings.get("cooldown_seconds", 60))
SYMBOL_COOLDOWN_SECONDS = int(style_settings.get("symbol_cooldown_seconds", 0))
MAX_TRADES_PER_DAY = int(style_settings.get("max_trades_per_day", 25))
DAILY_LOSS_LIMIT_PCT = float(style_settings.get("daily_loss_limit_pct", DAILY_LOSS_LIMIT_PCT))
MAX_DRAWDOWN_PCT = float(style_settings.get("max_drawdown_pct", MAX_DRAWDOWN_PCT))
MAX_TOTAL_EXPOSURE_PCT = float(style_settings.get("max_total_exposure_pct", MAX_TOTAL_EXPOSURE_PCT))
MIN_EXPOSURE_RESUME_PCT = float(style_settings.get("min_exposure_resume_pct", MIN_EXPOSURE_RESUME_PCT))
MAX_SYMBOL_EXPOSURE_PCT = float(style_settings.get("max_symbol_exposure_pct", MAX_SYMBOL_EXPOSURE_PCT))
MAX_OPEN_POSITIONS = int(style_settings.get("max_open_positions", MAX_OPEN_POSITIONS))
AUTOPILOT_TUNE_INTERVAL_SECONDS = int(getattr(bot_config, "AUTOPILOT_TUNE_INTERVAL_SECONDS", 900)) if bot_config else 900
AUTOPILOT_MIN_TRADES_FOR_TUNE = int(getattr(bot_config, "AUTOPILOT_MIN_TRADES_FOR_TUNE", 5)) if bot_config else 5
AUTOPILOT_MAX_RECENT_TRADES = int(getattr(bot_config, "AUTOPILOT_MAX_RECENT_TRADES", 40)) if bot_config else 40

EXECUTION_MODE = RUN_MODE

# Exchange-specific websocket feeds
if EXCHANGE == "kraken":
    from core.data.kraken_streams import start_ws_thread, prices as live_prices, price_timestamps
    recent_prices = {}
    order_books = {}
else:
    from core.data.binance_streams import start_ws_thread, prices as live_prices, recent_prices, order_books, price_timestamps

# -----------------------------
# 2. Initialize Clients
# -----------------------------
prices = {}
account_info = {}
last_account_fetch = 0.0
last_llm_call = {sym: 0.0 for sym in SYMBOLS}
llm_outputs = {}
sentiments = {}
indicators = {sym: {} for sym in SYMBOLS}
trade_history = {sym: [] for sym in SYMBOLS}
last_trades_fetch = 0.0
trades_by_symbol = {sym: [] for sym in SYMBOLS}
realized_pnl_snapshot = {}
_trading_ready = False
_trading_ready_last_notice = 0.0


# Initialize Clients
try:
    if EXCHANGE == "kraken":
        live_execution_client = LiveKrakenClient()
        live_execution_client.weight_limit_1m = MAX_API_WEIGHT_1M
        print("✅ Kraken Client Initialized")
    else:
        live_execution_client = LiveBinanceClient()
        print("✅ Binance.US Client Initialized")
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    exit()

account_client = None
wallet = None
if RUN_MODE == "live":
    account_client = KrakenAccount() if EXCHANGE == "kraken" else BinanceUSAccount()
    execution_client = live_execution_client
    portfolio = LivePortfolio(SYMBOLS)
else:
    if RESET_SIM_WALLET_ON_START:
        try:
            import os
            os.makedirs("data", exist_ok=True)
            with open("data/mock_wallet.json", "w") as f:
                f.write("")
        except Exception:
            pass
    wallet = MockWallet(live_prices=live_prices, initial_usd=CAPITAL)
    execution_client = SimBinanceClient(wallet, live_prices, order_books)
    portfolio = AccountingAgent(CAPITAL, csv_file="trades.csv", wallet=wallet)

trader = portfolio

# Optional rebuild flag for live positions (Kraken)
rebuild_flag = os.path.join("data", "rebuild_positions.flag")
if RUN_MODE == "live" and os.path.exists(rebuild_flag):
    try:
        if hasattr(trader, "state_path") and trader.state_path and os.path.exists(trader.state_path):
            os.remove(trader.state_path)
        if hasattr(trader, "positions"):
            trader.positions = {}
        os.remove(rebuild_flag)
        print("🔁 Rebuild flag detected: cleared saved positions state.")
    except Exception:
        pass
plotter = RealTimeEquityPlot(
    portfolio,
    csv_file="trades.csv",
    show_mock_wallet=(RUN_MODE == "sim"),
    orders_title="Live Trades" if RUN_MODE == "live" else "Mock Orders"
)

# Link the panic button to the emergency stop (close trades + stop)
if RUN_MODE == "live":
    plotter.panic_callback = lambda: emergency_stop(live_execution_client, trader, live_prices)
else:
    plotter.panic_callback = lambda: emergency_stop(None, trader, live_prices)

# -----------------------------
# 3. Background Services
# -----------------------------
start_background_fetch(SYMBOLS)
start_ws_thread(SYMBOLS)
ws_start_time = time.time()

# -----------------------------
# 4. Helper Functions
# -----------------------------
def _round_to_step(value, step):
    return (value // step) * step if step > 0 else value

def _to_slash_symbol(symbol):
    if "/" in symbol:
        return symbol
    if isinstance(symbol, str) and len(symbol) > 3:
        return f"{symbol[:-3]}/{symbol[-3:]}"
    return symbol

def _normalize_asset_code(asset):
    norm = asset or ""
    if not isinstance(norm, str):
        return norm
    # Drop Kraken suffixes like .F or .S
    if "." in norm:
        norm = norm.split(".", 1)[0]
    # Strip leading X/Z multiple times (Kraken uses XXBT, ZUSD, etc.)
    while len(norm) > 3 and norm[0] in ("X", "Z"):
        norm = norm[1:]
    if norm == "XBT":
        norm = "BTC"
    return norm

def _summarize_account(balances, live_prices):
    cash = 0.0
    total = 0.0
    for asset, info in balances.items():
        free = float(info.get("free", 0.0))
        norm_asset = _normalize_asset_code(asset)
        if norm_asset in ("USD", "USDT", "ZUSD"):
            cash += free
            total += free
        else:
            pair = f"{norm_asset}/USD"
            price = live_prices.get(pair)
            if price is None and "/" in pair:
                price = live_prices.get(pair.replace("/", ""))
            if price is not None:
                total += free * float(price)
    return {"cash_usd": cash, "total_usd": total}

def _positions_from_balances(balances, live_prices):
    positions = {}
    now = time.time()
    for asset, info in balances.items():
        qty = float(info.get("free", 0.0) or 0.0)
        if qty <= 0:
            continue
        norm_asset = _normalize_asset_code(asset)
        if norm_asset in ("USD", "USDT", "ZUSD"):
            continue
        sym = f"{norm_asset}/USD"
        if sym not in SYMBOLS:
            continue
        price = live_prices.get(sym)
        if price is None and "/" in sym:
            price = live_prices.get(sym.replace("/", ""))
        if price is None:
            continue
        positions[sym] = {
            "entry_price": float(price),
            "fill_price": float(price),
            "size": qty,
            "notional_entry": float(price) * qty,
            "symbol": sym,
            "timestamp": now,
            "stop_loss": None,
            "take_profit": None,
            "trailing_stop_pct": 0.0,
            "trailing_stop": None,
            "llm_decision": None,
            "entry_source": "balance"
        }
    return positions

def _compute_equity(trader, prices, account_info):
    try:
        if hasattr(trader, "equity"):
            return float(trader.equity(prices) or 0.0)
    except Exception:
        pass
    if isinstance(account_info, dict):
        total = account_info.get("total_usd")
        if total is not None:
            return float(total)
        cash = account_info.get("cash_usd")
        if cash is not None:
            return float(cash)
    return float(getattr(trader, "capital", 0.0) or 0.0)

def _compute_open_notional(trader, prices):
    total = 0.0
    per_symbol = {}
    positions = getattr(trader, "positions", {}) or {}
    for sym, pos in positions.items():
        sym_key = sym
        if isinstance(sym, str) and "/" not in sym and len(sym) > 3:
            sym_key = f"{sym[:-3]}/{sym[-3:]}"
        size = float(pos.get("size", 0.0) or 0.0)
        if size <= 0:
            continue
        price = prices.get(sym)
        if price is None and isinstance(sym, str) and "/" not in sym:
            slash_sym = f"{sym[:-3]}/{sym[-3:]}" if len(sym) > 3 else sym
            price = prices.get(slash_sym)
        if price is None:
            price = float(pos.get("entry_price", 0.0) or 0.0)
        notional = size * float(price or 0.0)
        total += notional
        per_symbol[sym_key] = per_symbol.get(sym_key, 0.0) + notional
    return total, per_symbol

def _normalize_action(action, symbol, last_price, trader, size_frac, perf_score):
    if not action or not isinstance(action, dict): return None, "None"
    
    act_name = str(action.get("action", "")).strip().upper()
    side = str(action.get("side", "")).strip().upper()
    order_type = str(action.get("type") or action.get("order_type") or "").strip().upper()
    if not act_name and not side:
        type_hint = str(action.get("type", "")).strip().upper()
        if type_hint in ("BUY", "SELL"):
            act_name = type_hint
        elif type_hint in ("CLOSE_POSITION", "CLOSE", "EXIT"):
            act_name = "CLOSE_POSITION"

    if act_name in ("HOLD", "NONE", "NOOP"): return None, "None"
    if act_name in ("CLOSE_POSITION", "CLOSE", "EXIT"):
        act_name = "SELL"
    if act_name in ("BUY", "SELL"): side, act_name = act_name, "PLACE_ORDER"
    
    qty = action.get("quantity")
    if qty is None:
        qty = action.get("size")
    if (qty is None or qty <= 0) and act_name == "PLACE_ORDER":
        # If selling and we have a live position, use full position size
        if side == "SELL" and hasattr(trader, "positions"):
            pos = trader.positions.get(symbol) or trader.positions.get(symbol.replace("/", "")) or None
            if pos and float(pos.get("size", 0.0) or 0.0) > 0:
                qty = float(pos.get("size"))
        # Calculate size based on risk module
        if qty is None or qty <= 0:
            raw_notional = compute_position_size(trader.capital, size_frac, perf_score=perf_score)
            if last_price:
                qty = raw_notional / last_price
            
    if not qty or qty <= 0: return None, "None"
    
    if order_type not in ("LIMIT", "MARKET"):
        order_type = "MARKET"
    action.update({
        "action": act_name,
        "side": side,
        "symbol": symbol.replace("/", ""),
        "quantity": qty,
        "type": order_type
    })
    if order_type == "LIMIT" and action.get("price") in (None, "", "None") and last_price:
        action["price"] = float(last_price)
    summary = f"{act_name} {side} {symbol} qty={qty:.4f}"
    if action.get("type") == "LIMIT" and action.get("price"):
        summary += f" price={action.get('price')}"
    return action, summary

def _estimate_realized_pnl(symbol, qty, price, trader):
    try:
        if qty is None or price is None:
            return None
        qty = float(qty)
        price = float(price)
        pos = None
        if hasattr(trader, "positions") and isinstance(trader.positions, dict):
            pos = trader.positions.get(symbol) or trader.positions.get(_to_slash_symbol(symbol))
        if not pos:
            return None
        entry = float(pos.get("entry_price") or pos.get("fill_price") or 0.0)
        if entry <= 0 or qty <= 0:
            return None
        return (price - entry) * qty
    except Exception:
        return None

def _fee_rate_for_order(order_type):
    if EXCHANGE != "kraken":
        return 0.0
    t = str(order_type or "").upper()
    if t == "LIMIT":
        return KRAKEN_MAKER_FEE_PCT
    return KRAKEN_TAKER_FEE_PCT

def _get_entry_info(symbol):
    norm = _to_slash_symbol(symbol)
    tracker = _trade_trackers.get(norm) or _trade_trackers.get(symbol)
    entry_price = None
    entry_order_type = None
    if tracker:
        entry_price = tracker.get("entry_price")
        entry_order_type = tracker.get("entry_order_type")
    if entry_price is None and hasattr(trader, "positions") and isinstance(trader.positions, dict):
        pos = trader.positions.get(norm) or trader.positions.get(symbol)
        if isinstance(pos, dict):
            entry_price = pos.get("entry_price") or pos.get("fill_price")
            entry_order_type = pos.get("entry_order_type")
    try:
        entry_price = float(entry_price) if entry_price is not None else None
    except Exception:
        entry_price = None
    return entry_price, entry_order_type

def _estimate_round_trip_costs(symbol, qty, exit_price, exit_order_type=None, entry_price=None, entry_order_type=None):
    try:
        qty = float(qty)
        exit_price = float(exit_price)
        if qty <= 0 or exit_price <= 0:
            return None, None
        if entry_price is None:
            entry_price, entry_order_type = _get_entry_info(symbol)
        if entry_price is None or entry_price <= 0:
            return None, None
        entry_notional = entry_price * qty
        exit_notional = exit_price * qty
        fee_total = (
            entry_notional * _fee_rate_for_order(entry_order_type)
            + exit_notional * _fee_rate_for_order(exit_order_type)
        )
        slippage_est = (entry_notional + exit_notional) * ESTIMATED_SLIPPAGE_PCT
        return fee_total, slippage_est
    except Exception:
        return None, None


def _record_win_loss(realized_pnl, symbol=None, realized_pnl_net=None, fee_total=None):
    global _win_count, _loss_count, _realized_total, _realized_total_gross, _realized_total_fees
    if realized_pnl is None:
        return
    try:
        gross_val = float(realized_pnl)
    except Exception:
        return
    fee_val = 0.0
    if fee_total is not None:
        try:
            fee_val = float(fee_total)
        except Exception:
            fee_val = 0.0
    if realized_pnl_net is None:
        pnl_val = gross_val - fee_val
    else:
        try:
            pnl_val = float(realized_pnl_net)
        except Exception:
            pnl_val = gross_val - fee_val
    _realized_total += pnl_val
    _realized_total_gross += gross_val
    _realized_total_fees += fee_val
    if pnl_val > 0:
        _win_count += 1
    elif pnl_val < 0:
        _loss_count += 1
    print(
        f"[TRADE_CLOSE] {symbol or ''} gross={gross_val:.4f} fees={fee_val:.4f} net={pnl_val:.4f} "
        f"wins={_win_count} losses={_loss_count} total_realized={_realized_total:.4f}"
    )

def _load_perf_guard_history():
    global _win_count, _loss_count, _realized_total, _realized_total_gross, _realized_total_fees, _autopilot_trade_pnls
    if not PERF_GUARD_PERSIST or PERF_GUARD_MODE.lower() == "session":
        return
    state_total = None
    state_path = os.path.join("data", "live_positions.json")
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("realized_pnl_total") is not None:
                state_total = float(data.get("realized_pnl_total") or 0.0)
    except Exception:
        state_total = None
    trades_csv = os.path.join("data", "live_trades.csv")
    trade_pnls = None
    trade_wins = 0
    trade_losses = 0
    trade_total_net = 0.0
    trade_total_gross = 0.0
    trade_total_fees = 0.0
    if os.path.exists(trades_csv):
        try:
            per_symbol = {}
            with open(trades_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get("symbol")
                    if not symbol:
                        continue
                    try:
                        qty = float(row.get("qty", 0.0) or 0.0)
                        price = float(row.get("price", 0.0) or 0.0)
                    except Exception:
                        continue
                    side = str(row.get("side", "")).upper()
                    if qty <= 0 or price <= 0 or side not in ("BUY", "SELL"):
                        continue
                    per_symbol.setdefault(symbol, []).append((row.get("timestamp") or 0, side, qty, price))
            pnl_list = []
            for symbol, trades in per_symbol.items():
                trades.sort(key=lambda x: x[0])
                qty = 0.0
                cost = 0.0
                for _, side, t_qty, t_price in trades:
                    if side == "BUY":
                        cost += t_qty * t_price
                        qty += t_qty
                    else:
                        if qty <= 0:
                            continue
                        avg_cost = cost / qty if qty else 0.0
                        reduce_qty = min(qty, t_qty)
                        realized = (t_price - avg_cost) * reduce_qty
                        cost -= avg_cost * reduce_qty
                        qty -= reduce_qty
                        fee_rate = KRAKEN_TAKER_FEE_PCT if EXCHANGE == "kraken" else 0.0
                        fee_est = (avg_cost * reduce_qty + t_price * reduce_qty) * fee_rate
                        realized_net = realized - fee_est
                        trade_total_net += realized_net
                        trade_total_gross += realized
                        trade_total_fees += fee_est
                        pnl_list.append(realized_net)
                        if realized_net > 0:
                            trade_wins += 1
                        elif realized_net < 0:
                            trade_losses += 1
            if pnl_list:
                trade_pnls = pnl_list
                _win_count = trade_wins
                _loss_count = trade_losses
                _autopilot_trade_pnls = pnl_list[-AUTOPILOT_MAX_RECENT_TRADES:]
        except Exception:
            trade_pnls = None

    if trade_pnls:
        if state_total is not None:
            _realized_total = state_total
            _realized_total_gross = state_total
            _realized_total_fees = 0.0
        else:
            _realized_total = trade_total_net
            _realized_total_gross = trade_total_gross
            _realized_total_fees = trade_total_fees
        print(f"[PERF_GUARD] Loaded {len(trade_pnls)} trades from live_trades.csv (wins={trade_wins} losses={trade_losses} realized_total={_realized_total:.2f})")
        return

    path = os.path.join("logs", "order_actions.jsonl")
    if not os.path.exists(path):
        if state_total is not None:
            _realized_total = state_total
            _realized_total_gross = state_total
            _realized_total_fees = 0.0
            print(f"[PERF_GUARD] Loaded realized_total from live_positions.json: {_realized_total:.2f}")
        return
    try:
        max_trades = max(1, int(PERF_GUARD_LOOKBACK_TRADES or 0))
    except Exception:
        max_trades = 200
    if max_trades <= 0:
        return
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                realized = payload.get("realized_pnl")
                realized_net = payload.get("realized_pnl_net")
                fee_est = payload.get("fee_total_est")
                if realized is None or realized == "":
                    continue
                outcome = payload.get("outcome")
                mode = payload.get("mode")
                if outcome not in ("W", "L", "B") and mode not in ("pnl_exit", "rebalance"):
                    continue
                try:
                    realized = float(realized)
                except Exception:
                    continue
                try:
                    fee_est = float(fee_est) if fee_est is not None and fee_est != "" else 0.0
                except Exception:
                    fee_est = 0.0
                if realized_net is None or realized_net == "":
                    realized_net = realized - fee_est
                else:
                    try:
                        realized_net = float(realized_net)
                    except Exception:
                        realized_net = realized - fee_est
                entries.append((realized_net, realized, fee_est, outcome))
        if not entries:
            return
        entries = entries[-max_trades:]
        wins = 0
        losses = 0
        total = 0.0
        total_gross = 0.0
        total_fees = 0.0
        pnl_list = []
        for realized_net, realized_gross, fee_est, outcome in entries:
            total += realized_net
            total_gross += realized_gross
            total_fees += fee_est
            pnl_list.append(realized_net)
            if isinstance(outcome, str):
                o = outcome.upper()
                if o == "W":
                    wins += 1
                    continue
                if o == "L":
                    losses += 1
                    continue
            if realized_net > 0:
                wins += 1
            elif realized_net < 0:
                losses += 1
        _win_count = wins
        _loss_count = losses
        _realized_total = total
        _realized_total_gross = total_gross
        _realized_total_fees = total_fees
        if pnl_list:
            _autopilot_trade_pnls = pnl_list[-AUTOPILOT_MAX_RECENT_TRADES:]
        if state_total is not None:
            _realized_total = state_total
            _realized_total_gross = state_total
            _realized_total_fees = 0.0
            print(f"[PERF_GUARD] Loaded {len(entries)} trades from history (wins={wins} losses={losses} realized_total={total:.2f}); using live_positions total={state_total:.2f}")
        else:
            print(f"[PERF_GUARD] Loaded {len(entries)} trades from history (wins={wins} losses={losses} realized_total={total:.2f})")
    except Exception:
        return

def _detect_candle_patterns(df):
    patterns = []
    if df is None or df.empty or len(df) < 2:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    try:
        o = float(last["open"])
        c = float(last["close"])
        h = float(last["high"])
        l = float(last["low"])
        body = abs(c - o)
        rng = max(h - l, 1e-8)
        upper = h - max(c, o)
        lower = min(c, o) - l
        if body / rng < 0.1:
            patterns.append("doji")
        if lower > body * 2 and upper < body:
            patterns.append("hammer")
        if upper > body * 2 and lower < body:
            patterns.append("shooting_star")
        # Engulfing vs previous
        po = float(prev["open"])
        pc = float(prev["close"])
        if c > o and pc < po and c >= po and o <= pc:
            patterns.append("bullish_engulfing")
        if c < o and pc > po and o >= pc and c <= po:
            patterns.append("bearish_engulfing")
    except Exception:
        return patterns
    return patterns

def _score_symbol(sym, sentiments, indicators):
    score = 0.0
    sent = sentiments.get(sym)
    if isinstance(sent, (list, tuple)) and sent:
        sent = sent[0]
    if sent is not None:
        try:
            score += float(sent)
        except Exception:
            pass
    rsi = indicators.get(sym, {}).get("rsi")
    if rsi is not None:
        try:
            score += (float(rsi) - 50.0) / 100.0
        except Exception:
            pass
    return score

def _portfolio_snapshot(trader, prices, account_info, total_notional, per_symbol_notional, equity):
    snapshot = {}
    try:
        cash = account_info.get("cash_usd") if isinstance(account_info, dict) else None
        total = account_info.get("total_usd") if isinstance(account_info, dict) else None
        snapshot["cash_usd"] = float(cash) if cash is not None else None
        snapshot["total_usd"] = float(total) if total is not None else None
        snapshot["equity_usd"] = float(equity) if equity is not None else None
        snapshot["cash_pct"] = (float(cash) / float(total)) if cash is not None and total else None
    except Exception:
        snapshot["cash_usd"] = None
        snapshot["total_usd"] = None
        snapshot["equity_usd"] = float(equity) if equity is not None else None
        snapshot["cash_pct"] = None

    try:
        positions = getattr(trader, "positions", {}) or {}
        unrealized_by_symbol = {}
        exposure_by_symbol = {}
        llm_prediction_by_symbol = {}
        for sym, pos in positions.items():
            price = prices.get(sym) or prices.get(sym.replace("/", "")) or pos.get("entry_price")
            if price is None:
                continue
            entry = float(pos.get("entry_price", price) or price)
            size = float(pos.get("size", 0.0) or 0.0)
            unrealized = (float(price) - entry) * size
            unrealized_by_symbol[sym] = unrealized
            if isinstance(pos, dict):
                dec = pos.get("llm_decision") or {}
                if isinstance(dec, dict):
                    pred = dec.get("price_prediction")
                    horizon = dec.get("prediction_horizon_min")
                    conviction = dec.get("conviction")
                    preds = dec.get("predictions")
                    if pred is not None or horizon is not None or conviction is not None:
                        llm_prediction_by_symbol[sym] = {
                            "price_prediction": pred,
                            "prediction_horizon_min": horizon,
                            "conviction": conviction,
                            "predictions": preds if isinstance(preds, list) else None,
                            "ts": dec.get("_ts")
                        }
        if per_symbol_notional:
            for sym, notional in per_symbol_notional.items():
                exposure_by_symbol[sym] = (float(notional) / float(equity)) if equity else None
        snapshot["unrealized_by_symbol"] = unrealized_by_symbol
        snapshot["exposure_by_symbol"] = exposure_by_symbol
        snapshot["total_exposure_pct"] = (float(total_notional) / float(equity)) if equity else None
        snapshot["llm_prediction_by_symbol"] = llm_prediction_by_symbol
        # Allocation deviation vs targets
        allocation_deviation = {}
        if TARGET_ALLOCATION:
            for asset, band in TARGET_ALLOCATION.items():
                sym = f"{asset}/USD"
                exposure = exposure_by_symbol.get(sym)
                if exposure is None:
                    continue
                try:
                    lo, hi = band
                    allocation_deviation[asset] = {
                        "exposure": exposure,
                        "target_min": lo,
                        "target_max": hi,
                        "over_min": exposure - lo,
                        "over_max": exposure - hi
                    }
                except Exception:
                    pass
        snapshot["allocation_deviation"] = allocation_deviation
    except Exception:
        snapshot["unrealized_by_symbol"] = {}
        snapshot["exposure_by_symbol"] = {}
        snapshot["total_exposure_pct"] = None
        snapshot["allocation_deviation"] = {}
    return snapshot

def _build_ta_context(symbol, indicators, recent_ohlcv, candle_patterns):
    ind = indicators.get(symbol, {}) if isinstance(indicators, dict) else {}
    return {
        "rsi": ind.get("rsi"),
        "ema_20": ind.get("ema20") if ind.get("ema20") is not None else ind.get("ema"),
        "vwap": ind.get("vwap"),
        "atr_pct": ind.get("atr_pct"),
        "vol_ratio": ind.get("vol_ratio"),
        "current_vol": ind.get("current_vol"),
        "avg_vol": ind.get("avg_vol"),
        "recent_ohlcv": recent_ohlcv,
        "candle_patterns": candle_patterns or []
    }

def _build_perf_summary(win_count, loss_count, realized_total, trade_count, win_rate, underperforming, equity, realized_total_gross=None, realized_total_fees=None):
    summary = {
        "wins": win_count,
        "losses": loss_count,
        "realized_total": realized_total,
        "realized_total_gross": realized_total_gross if realized_total_gross is not None else realized_total,
        "realized_total_fees": realized_total_fees if realized_total_fees is not None else 0.0,
        "win_rate": (win_rate if trade_count else 0.0),
        "underperforming": underperforming,
        "trade_count": trade_count,
        "daily_opportunity_summary": _daily_opportunity_summary or {},
        "weekly_opportunity_summary": _weekly_opportunity_summary or {},
        "monthly_opportunity_summary": _monthly_opportunity_summary or {},
        "yearly_opportunity_summary": _yearly_opportunity_summary or {}
    }
    hist_baseline = _get_historical_baseline()
    if hist_baseline is not None and hist_baseline > 0 and equity is not None:
        hist_pnl = float(equity) - float(hist_baseline)
        hist_pct = (hist_pnl / float(hist_baseline)) * 100.0
        summary["historical_baseline"] = float(hist_baseline)
        summary["historical_pnl"] = float(hist_pnl)
        summary["historical_pnl_pct"] = float(hist_pct)
    return summary

def _build_risk_state(equity, drawdown_pct, daily_loss_pct, total_notional, per_symbol_notional, open_positions, confidence_gate):
    exposure_ratio = (total_notional / equity) if equity and equity > 0 else 0.0
    return {
        "equity": float(equity) if equity is not None else None,
        "drawdown_pct": float(drawdown_pct),
        "daily_loss_pct": float(daily_loss_pct),
        "exposure_ratio": float(exposure_ratio),
        "total_notional": float(total_notional),
        "per_symbol_notional": per_symbol_notional,
        "open_positions": int(open_positions),
        "max_drawdown_pct": float(MAX_DRAWDOWN_PCT),
        "confidence_gate": float(confidence_gate) if confidence_gate is not None else None
    }

def _build_wallet_state(account_info, trader, equity):
    state = {
        "equity": float(equity) if equity is not None else None
    }
    if isinstance(account_info, dict):
        state["cash_usd"] = account_info.get("cash_usd")
        state["total_usd"] = account_info.get("total_usd")
        state["has_non_cash_balances"] = account_info.get("has_non_cash_balances")
    try:
        state["positions"] = list(getattr(trader, "positions", {}) or {})
    except Exception:
        state["positions"] = []
    return state

def _build_execution_context(symbol, trades_by_symbol, portfolio_context):
    return {
        "symbol": symbol,
        "mode": EXECUTION_MODE,
        "style": TRADING_STYLE,
        "llm_interval": LLM_CHECK_INTERVAL,
        "recent_trades": trades_by_symbol.get(symbol, [])[-5:],
        "portfolio": portfolio_context
    }

def _ensure_position_risk_defaults(trader):
    if not hasattr(trader, "positions"):
        return
    positions = trader.positions if isinstance(trader.positions, dict) else {}
    for sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        entry = pos.get("entry_price") or pos.get("fill_price")
        if not entry:
            continue
        if pos.get("stop_loss") is None and STOP_LOSS_PCT_DEFAULT:
            pos["stop_loss"] = float(entry) * (1 - float(STOP_LOSS_PCT_DEFAULT))
        if pos.get("take_profit") is None and TAKE_PROFIT_PCT_DEFAULT:
            pos["take_profit"] = float(entry) * (1 + float(TAKE_PROFIT_PCT_DEFAULT))
        trailing_pct = float(pos.get("trailing_stop_pct") or 0.0)
        if trailing_pct <= 0 and TRAILING_STOP_PCT_DEFAULT:
            trailing_pct = float(TRAILING_STOP_PCT_DEFAULT)
            pos["trailing_stop_pct"] = trailing_pct
        if pos.get("trailing_stop") is None and trailing_pct > 0:
            pos["trailing_stop"] = float(entry) * (1 - float(trailing_pct))

def _log_portfolio_snapshot(snapshot, tag, symbol=None):
    try:
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", "portfolio_snapshots.jsonl")
        payload = {
            "timestamp": time.time(),
            "tag": tag,
            "symbol": symbol,
            "snapshot": snapshot
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _log_order_action(symbol, action, result, mode=None, outcome=None, realized_pnl=None, realized_pnl_net=None, fee_total_est=None, slippage_est=None, reward_score=None, llm_payload=None, last_price=None):
    try:
        os.makedirs("logs", exist_ok=True)
        csv_path = os.path.join("logs", "order_actions.csv")
        jsonl_path = os.path.join("logs", "order_actions.jsonl")
        rebalance_path = os.path.join("logs", "rebalance_actions.jsonl")

        action_name = str(action.get("action", "")) if isinstance(action, dict) else ""
        order_type = str(action.get("type", "")) if isinstance(action, dict) else ""
        side = str(action.get("side", "")) if isinstance(action, dict) else ""
        qty = action.get("quantity") if isinstance(action, dict) else ""
        price = action.get("price") if isinstance(action, dict) else ""
        status = ""
        if isinstance(result, dict):
            status = str(result.get("status", result.get("status", "")))
        llm_data = llm_payload if isinstance(llm_payload, dict) else {}
        stop_loss_pct = llm_data.get("stop_loss_pct")
        take_profit_pct = llm_data.get("take_profit_pct")
        trailing_stop_pct = llm_data.get("trailing_stop_pct")
        price_base = last_price if last_price is not None else price
        stop_loss_price = None
        take_profit_price = None
        trailing_stop_price = None
        if price_base:
            try:
                price_base = float(price_base)
                if stop_loss_pct is not None:
                    stop_loss_price = price_base * (1 - float(stop_loss_pct))
                if take_profit_pct is not None:
                    take_profit_price = price_base * (1 + float(take_profit_pct))
                if trailing_stop_pct is not None:
                    trailing_stop_price = price_base * (1 - float(trailing_stop_pct))
            except Exception:
                stop_loss_price = None
                take_profit_price = None
                trailing_stop_price = None

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "action", "order_type", "side", "qty", "price", "status", "mode",
                    "outcome", "realized_pnl", "realized_pnl_net", "fee_total_est", "slippage_est", "reward_score",
                    "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
                    "stop_loss_price", "take_profit_price", "trailing_stop_price"
                ])
            writer.writerow([
                time.time(),
                symbol,
                action_name,
                order_type,
                side,
                qty,
                price,
                status,
                mode or "",
                outcome or "",
                realized_pnl if realized_pnl is not None else "",
                realized_pnl_net if realized_pnl_net is not None else "",
                fee_total_est if fee_total_est is not None else "",
                slippage_est if slippage_est is not None else "",
                reward_score if reward_score is not None else "",
                stop_loss_pct if stop_loss_pct is not None else "",
                take_profit_pct if take_profit_pct is not None else "",
                trailing_stop_pct if trailing_stop_pct is not None else "",
                stop_loss_price if stop_loss_price is not None else "",
                take_profit_price if take_profit_price is not None else "",
                trailing_stop_price if trailing_stop_price is not None else ""
            ])

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "symbol": symbol,
                "action": action,
                "result": result,
                "mode": mode,
                "outcome": outcome,
                "realized_pnl": realized_pnl,
                "realized_pnl_net": realized_pnl_net,
                "fee_total_est": fee_total_est,
                "slippage_est": slippage_est,
                "reward_score": reward_score,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_stop_pct": trailing_stop_pct,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "trailing_stop_price": trailing_stop_price
            }, ensure_ascii=False) + "\n")
        if mode == "rebalance":
            with open(rebalance_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "action": action,
                    "result": result,
                    "outcome": outcome,
                    "llm_payload": llm_payload,
                    "last_price": last_price
                }, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _autopilot_summary():
    trades = len(_autopilot_trade_pnls)
    if trades <= 0:
        return None
    wins = sum(1 for p in _autopilot_trade_pnls if p > 0)
    losses = sum(1 for p in _autopilot_trade_pnls if p < 0)
    breakeven = trades - wins - losses
    pnl_total = sum(_autopilot_trade_pnls)
    avg_pnl = pnl_total / trades if trades else 0.0
    win_rate = wins / trades if trades else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "pnl_total": pnl_total,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate
    }

def _persist_autopilot_settings():
    global _autopilot_last_persist_ts
    if not AUTOPILOT_ENABLED or not isinstance(AUTOPILOT_SETTINGS, dict):
        return
    now = time.time()
    if (now - _autopilot_last_persist_ts) < 60:
        return
    _autopilot_last_persist_ts = now
    try:
        if not os.path.exists(CONFIG_PATH):
            return
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        block_pattern = re.compile(r"# BEGIN STYLE_PRESETS.*?# END STYLE_PRESETS", re.DOTALL)
        match = block_pattern.search(content)
        if not match:
            return
        block = match.group(0)
        style_text = block.split("# BEGIN STYLE_PRESETS", 1)[-1].split("# END STYLE_PRESETS", 1)[0].strip()
        if style_text.startswith("STYLE_PRESETS"):
            style_text = style_text.split("=", 1)[-1].strip()
        try:
            presets = ast.literal_eval(style_text)
        except Exception:
            presets = dict(STYLE_PRESETS) if isinstance(STYLE_PRESETS, dict) else {}
        if not isinstance(presets, dict):
            presets = {}
        presets["autopilot"] = dict(AUTOPILOT_SETTINGS)
        style_text = "STYLE_PRESETS = " + pprint.pformat(presets, width=120)
        new_block = f"# BEGIN STYLE_PRESETS\n{style_text}\n# END STYLE_PRESETS"
        content = block_pattern.sub(new_block, content)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print("[AUTOPILOT] Persisted tuned settings to config.")
    except Exception as e:
        if DEBUG_STATUS:
            print(f"[AUTOPILOT] Persist failed: {e}")

def _log_autopilot_tuning(updates, reason=None):
    try:
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", "autopilot_tuning.jsonl")
        payload = {
            "timestamp": time.time(),
            "reason": reason,
            "updates": updates
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _load_autopilot_tune_message():
    global _autopilot_tune_mtime, _autopilot_tune_last
    try:
        path = os.path.join("logs", "autopilot_tuning.jsonl")
        if not os.path.exists(path):
            return ""
        mtime = os.path.getmtime(path)
        if mtime == _autopilot_tune_mtime:
            return _autopilot_tune_last
        last_line = ""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if not last_line:
            return ""
        payload = json.loads(last_line)
        ts = float(payload.get("timestamp", 0.0) or 0.0)
        reason = payload.get("reason") or "update"
        updates = payload.get("updates") or {}
        updates_str = ", ".join([f"{k}={v}" for k, v in updates.items()]) if isinstance(updates, dict) else ""
        updates_str = updates_str[:120] + ("…" if len(updates_str) > 120 else "")
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "unknown"
        message = f"Autopilot tuned {tstr} ({reason}) {updates_str}"
        _autopilot_tune_mtime = mtime
        _autopilot_tune_last = message
        return message
    except Exception:
        return _autopilot_tune_last

def _apply_autopilot_overrides(overrides, reason=None):
    global MIN_CONFIDENCE_TO_ORDER
    global SIZE_FRACTION_DEFAULT
    global STOP_LOSS_PCT_DEFAULT
    global TAKE_PROFIT_PCT_DEFAULT
    global TRAILING_STOP_PCT_DEFAULT
    global COOLDOWN_SECONDS
    global MAX_TRADES_PER_DAY
    global DAILY_LOSS_LIMIT_PCT
    global MAX_DRAWDOWN_PCT
    global MAX_TOTAL_EXPOSURE_PCT
    global MIN_EXPOSURE_RESUME_PCT
    global MAX_SYMBOL_EXPOSURE_PCT
    global MAX_OPEN_POSITIONS

    if not AUTOPILOT_ENABLED or not isinstance(overrides, dict) or not overrides:
        return

    bounds = {
        "min_confidence_to_order": (0.1, 0.9, float),
        "size_fraction": (0.02, 0.35, float),
        "stop_loss_pct_default": (0.005, 0.05, float),
        "take_profit_pct_default": (0.01, 0.1, float),
        "trailing_stop_pct_default": (0.005, 0.08, float),
        "cooldown_seconds": (0, 600, int),
        "max_trades_per_day": (1, 5000, int),
        "daily_loss_limit_pct": (0.005, 0.2, float),
        "max_drawdown_pct": (0.01, 0.5, float),
        "max_total_exposure_pct": (0.2, 1.0, float),
        "min_exposure_resume_pct": (0.05, 0.8, float),
        "max_symbol_exposure_pct": (0.05, 0.5, float),
        "max_open_positions": (1, 25, int)
    }

    updates = {}
    for key, value in overrides.items():
        if key not in bounds:
            continue
        lo, hi, cast = bounds[key]
        try:
            val = float(value)
        except Exception:
            continue
        if cast is int:
            val = int(round(val))
        val = max(lo, min(hi, val))
        updates[key] = val

    if not updates:
        return

    if "min_confidence_to_order" in updates:
        MIN_CONFIDENCE_TO_ORDER = float(updates["min_confidence_to_order"])
    if "size_fraction" in updates:
        SIZE_FRACTION_DEFAULT = float(updates["size_fraction"])
    if "stop_loss_pct_default" in updates:
        STOP_LOSS_PCT_DEFAULT = float(updates["stop_loss_pct_default"])
    if "take_profit_pct_default" in updates:
        TAKE_PROFIT_PCT_DEFAULT = float(updates["take_profit_pct_default"])
    if "trailing_stop_pct_default" in updates:
        TRAILING_STOP_PCT_DEFAULT = float(updates["trailing_stop_pct_default"])
    if "cooldown_seconds" in updates:
        COOLDOWN_SECONDS = int(updates["cooldown_seconds"])
    if "max_trades_per_day" in updates:
        MAX_TRADES_PER_DAY = int(updates["max_trades_per_day"])
    if "daily_loss_limit_pct" in updates:
        DAILY_LOSS_LIMIT_PCT = float(updates["daily_loss_limit_pct"])
    if "max_drawdown_pct" in updates:
        MAX_DRAWDOWN_PCT = float(updates["max_drawdown_pct"])
    if "max_total_exposure_pct" in updates:
        MAX_TOTAL_EXPOSURE_PCT = float(updates["max_total_exposure_pct"])
    if "min_exposure_resume_pct" in updates:
        MIN_EXPOSURE_RESUME_PCT = float(updates["min_exposure_resume_pct"])
    if "max_symbol_exposure_pct" in updates:
        MAX_SYMBOL_EXPOSURE_PCT = float(updates["max_symbol_exposure_pct"])
    if "max_open_positions" in updates:
        MAX_OPEN_POSITIONS = int(updates["max_open_positions"])

    if isinstance(AUTOPILOT_SETTINGS, dict):
        AUTOPILOT_SETTINGS.update(updates)

    tag = f" reason={reason}" if reason else ""
    print(f"[AUTOPILOT] Applied tuning{tag}: {json.dumps(updates)}")
    _log_autopilot_tuning(updates, reason=reason)
    _persist_autopilot_settings()

def _maybe_autopilot_tune(symbol=None):
    global _autopilot_last_tune_ts
    if not AUTOPILOT_ENABLED:
        return
    if len(_autopilot_trade_pnls) < AUTOPILOT_MIN_TRADES_FOR_TUNE:
        return
    now = time.time()
    if (now - _autopilot_last_tune_ts) < AUTOPILOT_TUNE_INTERVAL_SECONDS:
        return
    summary = _autopilot_summary()
    if not summary:
        return
    current_settings = {
        "min_confidence_to_order": MIN_CONFIDENCE_TO_ORDER,
        "size_fraction": SIZE_FRACTION_DEFAULT,
        "stop_loss_pct_default": STOP_LOSS_PCT_DEFAULT,
        "take_profit_pct_default": TAKE_PROFIT_PCT_DEFAULT,
        "trailing_stop_pct_default": TRAILING_STOP_PCT_DEFAULT,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "daily_loss_limit_pct": DAILY_LOSS_LIMIT_PCT,
        "max_drawdown_pct": MAX_DRAWDOWN_PCT,
        "max_total_exposure_pct": MAX_TOTAL_EXPOSURE_PCT,
        "min_exposure_resume_pct": MIN_EXPOSURE_RESUME_PCT,
        "max_symbol_exposure_pct": MAX_SYMBOL_EXPOSURE_PCT,
        "max_open_positions": MAX_OPEN_POSITIONS
    }
    overrides = llm_autopilot_tune(
        current_settings,
        trade_stats=summary,
        execution_context={"symbol": symbol, "mode": EXECUTION_MODE, "style": TRADING_STYLE}
    )
    if overrides:
        _apply_autopilot_overrides(overrides, reason="llm_tune")
        _autopilot_last_tune_ts = now

def _compute_performance_guard():
    total = _win_count + _loss_count
    win_rate = (_win_count / total) if total else 0.0
    underperforming = False
    if total >= 5:
        underperforming = (win_rate < 0.60) or (_realized_total < 0)
    elif total > 0:
        underperforming = (_realized_total < 0)
    return underperforming, win_rate, total

def _autopilot_on_realized_pnl(realized_pnl, symbol=None):
    if not AUTOPILOT_ENABLED:
        return
    if realized_pnl is None:
        return
    try:
        pnl_val = float(realized_pnl)
    except Exception:
        return
    _autopilot_trade_pnls.append(pnl_val)
    if len(_autopilot_trade_pnls) > AUTOPILOT_MAX_RECENT_TRADES:
        _autopilot_trade_pnls[:] = _autopilot_trade_pnls[-AUTOPILOT_MAX_RECENT_TRADES:]
    _maybe_autopilot_tune(symbol=symbol)

# -----------------------------
# 5. Main Loop
# -----------------------------
print("🚀 Trading loop active...")
iteration_count = 0
orders_this_min = 0
stale_warned = {}
last_trade_time = 0.0
last_symbol_trade_time = {}
trades_today = 0
trades_day_key = time.strftime("%Y-%m-%d", time.localtime())
last_status_log = 0.0
last_action_attempt = {}
daily_start_equity = None
peak_equity = None
block_new_trades = False
block_reasons = []
reject_backoff_until = {}
attempted_orders = []
_attempt_log_state = {}
risk_guard_log_state = {}
_panic_requested = False
_risk_reset_done = False
_equity_debug_last = 0.0
_initial_sync_done = False
_ui_ready = False
_cooldown_log_state = {}
COOLDOWN_LOG_INTERVAL_SECONDS = 15
_symbol_cooldown_log_state = {}
SYMBOL_COOLDOWN_LOG_INTERVAL_SECONDS = 30
_symbol_exposure_log_state = {}
SYMBOL_EXPOSURE_LOG_INTERVAL_SECONDS = 30
_reject_backoff_log_state = {}
REJECT_BACKOFF_LOG_INTERVAL_SECONDS = 20
_auto_exit_log_state = {}
AUTO_EXIT_LOG_INTERVAL_SECONDS = 30
_rebalance_log_state = {}
_rebalance_advice_cache = {}
_llm_action_log = []
_llm_action_log_max = 200
latest_llm_symbol = None
latest_llm_summary = {}
latest_llm_perf = {}
latest_llm_risk = {}
latest_llm_ta = {}
latest_llm_sentiment = None
latest_llm_wallet = {}
latest_llm_ts = None
TRADE_OUTCOMES_PATH = os.path.join("logs", "trade_outcomes.jsonl")
PRICE_HISTORY_PATH = os.path.join("logs", "price_history.jsonl")
LLM_PREDICTIONS_PATH = os.path.join("logs", "llm_predictions.jsonl")
_trade_trackers = {}
_last_price_history_ts = 0.0
_daily_opportunity_summary = {}
_last_opportunity_day = None
_last_candle_close_ts = {}
_weekly_opportunity_summary = {}
_last_opportunity_week = None
_monthly_opportunity_summary = {}
_last_opportunity_month = None
_yearly_opportunity_summary = {}
_last_opportunity_year = None

def _timeframe_to_seconds(timeframe):
    tf = str(timeframe or "").strip().lower()
    if tf.endswith("m"):
        try:
            return int(tf[:-1]) * 60
        except Exception:
            return 300
    if tf.endswith("h"):
        try:
            return int(tf[:-1]) * 3600
        except Exception:
            return 3600
    if tf.endswith("d"):
        try:
            return int(tf[:-1]) * 86400
        except Exception:
            return 86400
    return 300

def _append_trade_outcome(record):
    try:
        os.makedirs(os.path.dirname(TRADE_OUTCOMES_PATH), exist_ok=True)
        with open(TRADE_OUTCOMES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def _append_llm_prediction(record):
    try:
        os.makedirs(os.path.dirname(LLM_PREDICTIONS_PATH), exist_ok=True)
        with open(LLM_PREDICTIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def _log_price_history(prices, now_ts):
    global _last_price_history_ts
    if (now_ts - _last_price_history_ts) < 60:
        return
    _last_price_history_ts = now_ts
    try:
        os.makedirs(os.path.dirname(PRICE_HISTORY_PATH), exist_ok=True)
        with open(PRICE_HISTORY_PATH, "a", encoding="utf-8") as f:
            for sym, px in (prices or {}).items():
                if px is None:
                    continue
                f.write(json.dumps({"ts": now_ts, "symbol": sym, "price": float(px)}) + "\n")
    except Exception:
        pass

def _update_trade_tracker(symbol, last_price):
    if symbol not in _trade_trackers:
        return
    try:
        tracker = _trade_trackers[symbol]
        px = float(last_price) if last_price is not None else None
        if px is None:
            return
        if tracker.get("max_price") is None or px > tracker["max_price"]:
            tracker["max_price"] = px
        if tracker.get("min_price") is None or px < tracker["min_price"]:
            tracker["min_price"] = px
    except Exception:
        pass

def _on_trade_open(symbol, entry_price, qty=None, order_type=None):
    try:
        px = float(entry_price) if entry_price is not None else None
        if px is None:
            return
        tracker = _trade_trackers.get(symbol)
        if tracker is None:
            _trade_trackers[symbol] = {
                "entry_ts": time.time(),
                "entry_price": px,
                "qty": float(qty or 0.0),
                "max_price": px,
                "min_price": px,
                "entry_order_type": order_type
            }
        else:
            tracker["qty"] = float(tracker.get("qty", 0.0) or 0.0) + float(qty or 0.0)
            tracker["max_price"] = max(tracker.get("max_price", px), px)
            tracker["min_price"] = min(tracker.get("min_price", px), px)
    except Exception:
        pass

def _on_trade_close(symbol, exit_price, reason=None):
    tracker = _trade_trackers.pop(symbol, None)
    if not tracker:
        return
    try:
        entry_price = float(tracker.get("entry_price") or 0.0)
        exit_price = float(exit_price or 0.0)
        if entry_price <= 0 or exit_price <= 0:
            return
        max_price = float(tracker.get("max_price") or entry_price)
        min_price = float(tracker.get("min_price") or entry_price)
        mfe_pct = (max_price - entry_price) / entry_price if entry_price else 0.0
        mae_pct = (entry_price - min_price) / entry_price if entry_price else 0.0
        record = {
            "ts": time.time(),
            "symbol": symbol,
            "entry_ts": tracker.get("entry_ts"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": tracker.get("qty"),
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "exit_reason": reason or "close"
        }
        _append_trade_outcome(record)
    except Exception:
        pass

def _compute_daily_opportunity_summary(day_key, symbol_prices, trade_records, predictions):
    if not trade_records:
        trade_records = []
    mfe_list = []
    mae_list = []
    post_list = []
    post_gt_1 = 0
    for rec in trade_records:
        mfe_list.append(rec.get("mfe_pct") or 0.0)
        mae_list.append(rec.get("mae_pct") or 0.0)
        sym = rec.get("symbol")
        exit_ts = rec.get("ts")
        exit_price = rec.get("exit_price")
        prices = symbol_prices.get(sym, [])
        if not prices or not exit_ts or not exit_price:
            continue
        max_after = None
        for ts, px in prices:
            if ts <= exit_ts:
                continue
            if max_after is None or px > max_after:
                max_after = px
        if max_after is None:
            continue
        post_fav = (max_after - exit_price) / exit_price
        post_list.append(post_fav)
        if post_fav >= 0.01:
            post_gt_1 += 1
    total = len(trade_records)
    summary = {
        "day": day_key,
        "trades": total,
        "avg_mfe_pct": float(sum(mfe_list) / len(mfe_list)) if mfe_list else 0.0,
        "avg_mae_pct": float(sum(mae_list) / len(mae_list)) if mae_list else 0.0,
        "avg_post_exit_fav_pct": float(sum(post_list) / len(post_list)) if post_list else 0.0,
        "pct_post_exit_fav_gt_1pct": float(post_gt_1 / total) if total else 0.0
    }
    pred_count, pred_hits, pred_conv, pred_stats = _compute_prediction_stats(
        symbol_prices,
        predictions,
        end_ts=time.mktime(time.strptime(day_key + " 23:59:59", "%Y-%m-%d %H:%M:%S"))
    )
    summary["prediction_count"] = pred_count
    summary["prediction_hit_rate"] = float(pred_hits / pred_count) if pred_count else 0.0
    summary["avg_prediction_conviction"] = float(sum(pred_conv) / len(pred_conv)) if pred_conv else 0.0
    summary["prediction_horizon_stats"] = pred_stats
    return summary

def _compute_prediction_stats(symbol_prices, predictions, end_ts):
    pred_count = 0
    pred_hits = 0
    pred_conv = []
    horizon_stats = {}
    if not predictions:
        return pred_count, pred_hits, pred_conv, horizon_stats
    for pred in predictions:
        sym = pred.get("symbol")
        ts = pred.get("ts")
        last_price = pred.get("last_price")
        if sym is None or ts is None or last_price is None:
            continue
        try:
            ts = float(ts)
            last_price = float(last_price)
        except Exception:
            continue
        series = symbol_prices.get(sym, [])
        if not series:
            continue
        pred_items = pred.get("predictions")
        if not isinstance(pred_items, list) or not pred_items:
            pred_items = [pred]
        for item in pred_items:
            if not isinstance(item, dict):
                continue
            target = item.get("price_prediction") or pred.get("price_prediction")
            horizon = item.get("prediction_horizon_min") or pred.get("prediction_horizon_min") or 60
            conviction = item.get("conviction")
            if conviction is None:
                conviction = pred.get("conviction")
            if target is None:
                continue
            try:
                target = float(target)
                horizon = int(horizon)
            except Exception:
                continue
            key = str(horizon)
            bucket = horizon_stats.setdefault(key, {"count": 0, "hits": 0, "conv": []})
            if conviction is not None:
                try:
                    conv_val = float(conviction)
                    pred_conv.append(conv_val)
                    bucket["conv"].append(conv_val)
                except Exception:
                    pass
            end_window = ts + (horizon * 60)
            if end_ts and end_window > end_ts:
                end_window = end_ts
            window = [px for t, px in series if t >= ts and t <= end_window]
            if not window:
                continue
            pred_count += 1
            bucket["count"] += 1
            if target >= last_price:
                hit = max(window) >= target
            else:
                hit = min(window) <= target
            if hit:
                pred_hits += 1
                bucket["hits"] += 1
    for key, bucket in list(horizon_stats.items()):
        count = bucket.get("count") or 0
        hits = bucket.get("hits") or 0
        convs = bucket.get("conv") or []
        horizon_stats[key] = {
            "count": int(count),
            "hit_rate": float(hits / count) if count else 0.0,
            "avg_conv": float(sum(convs) / len(convs)) if convs else 0.0
        }
    return pred_count, pred_hits, pred_conv, horizon_stats

def _maybe_daily_opportunity_reflection(now_ts):
    global _daily_opportunity_summary, _last_opportunity_day
    local = time.localtime(now_ts)
    day_key = time.strftime("%Y-%m-%d", local)
    if _last_opportunity_day == day_key:
        return
    if local.tm_hour < 9:
        return
    prev_day_ts = now_ts - 86400
    prev_day_key = time.strftime("%Y-%m-%d", time.localtime(prev_day_ts))
    trades = []
    try:
        if os.path.exists(TRADE_OUTCOMES_PATH):
            with open(TRADE_OUTCOMES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    rec_day = time.strftime("%Y-%m-%d", time.localtime(float(ts)))
                    if rec_day == prev_day_key:
                        trades.append(rec)
    except Exception:
        trades = []
    symbol_prices = {}
    try:
        if os.path.exists(PRICE_HISTORY_PATH):
            with open(PRICE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = row.get("ts")
                    sym = row.get("symbol")
                    px = row.get("price")
                    if ts is None or sym is None or px is None:
                        continue
                    rec_day = time.strftime("%Y-%m-%d", time.localtime(float(ts)))
                    if rec_day != prev_day_key:
                        continue
                    symbol_prices.setdefault(sym, []).append((float(ts), float(px)))
    except Exception:
        symbol_prices = {}
    for sym in symbol_prices:
        symbol_prices[sym].sort(key=lambda x: x[0])
    predictions = []
    try:
        if os.path.exists(LLM_PREDICTIONS_PATH):
            with open(LLM_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    rec_day = time.strftime("%Y-%m-%d", time.localtime(float(ts)))
                    if rec_day == prev_day_key:
                        predictions.append(rec)
    except Exception:
        predictions = []
    summary = _compute_daily_opportunity_summary(prev_day_key, symbol_prices, trades, predictions)
    _daily_opportunity_summary = summary
    _last_opportunity_day = day_key
    try:
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "daily_opportunity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

def _maybe_weekly_opportunity_reflection(now_ts):
    global _weekly_opportunity_summary, _last_opportunity_week
    local = time.localtime(now_ts)
    if local.tm_wday != 0 or local.tm_hour < 9:
        return
    week_key = time.strftime("%Y-%W", local)
    if _last_opportunity_week == week_key:
        return
    day_start = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, local.tm_wday, local.tm_yday, local.tm_isdst))
    week_start = day_start - (local.tm_wday * 86400)
    prev_start = week_start - (7 * 86400)
    prev_end = week_start - 1
    week_label = time.strftime("%Y-%W", time.localtime(prev_start))

    trades = []
    try:
        if os.path.exists(TRADE_OUTCOMES_PATH):
            with open(TRADE_OUTCOMES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        trades.append(rec)
    except Exception:
        trades = []

    symbol_prices = {}
    try:
        if os.path.exists(PRICE_HISTORY_PATH):
            with open(PRICE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = row.get("ts")
                    sym = row.get("symbol")
                    px = row.get("price")
                    if ts is None or sym is None or px is None:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        symbol_prices.setdefault(sym, []).append((ts, float(px)))
    except Exception:
        symbol_prices = {}
    for sym in symbol_prices:
        symbol_prices[sym].sort(key=lambda x: x[0])

    predictions = []
    try:
        if os.path.exists(LLM_PREDICTIONS_PATH):
            with open(LLM_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        predictions.append(rec)
    except Exception:
        predictions = []

    summary = _compute_daily_opportunity_summary(
        week_label,
        symbol_prices,
        trades,
        predictions
    )
    summary["week"] = week_label
    _weekly_opportunity_summary = summary
    _last_opportunity_week = week_key
    try:
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "weekly_opportunity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

def _maybe_monthly_opportunity_reflection(now_ts):
    global _monthly_opportunity_summary, _last_opportunity_month
    local = time.localtime(now_ts)
    if local.tm_mday != 1 or local.tm_hour < 9:
        return
    month_key = time.strftime("%Y-%m", local)
    if _last_opportunity_month == month_key:
        return
    current_start = time.mktime((local.tm_year, local.tm_mon, 1, 0, 0, 0, 0, 0, -1))
    prev_year = local.tm_year
    prev_month = local.tm_mon - 1
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
    prev_start = time.mktime((prev_year, prev_month, 1, 0, 0, 0, 0, 0, -1))
    prev_end = current_start - 1
    month_label = time.strftime("%Y-%m", time.localtime(prev_start))

    trades = []
    try:
        if os.path.exists(TRADE_OUTCOMES_PATH):
            with open(TRADE_OUTCOMES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        trades.append(rec)
    except Exception:
        trades = []

    symbol_prices = {}
    try:
        if os.path.exists(PRICE_HISTORY_PATH):
            with open(PRICE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = row.get("ts")
                    sym = row.get("symbol")
                    px = row.get("price")
                    if ts is None or sym is None or px is None:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        symbol_prices.setdefault(sym, []).append((ts, float(px)))
    except Exception:
        symbol_prices = {}
    for sym in symbol_prices:
        symbol_prices[sym].sort(key=lambda x: x[0])

    predictions = []
    try:
        if os.path.exists(LLM_PREDICTIONS_PATH):
            with open(LLM_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        predictions.append(rec)
    except Exception:
        predictions = []

    summary = _compute_daily_opportunity_summary(
        month_label,
        symbol_prices,
        trades,
        predictions
    )
    summary["month"] = month_label
    _monthly_opportunity_summary = summary
    _last_opportunity_month = month_key
    try:
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "monthly_opportunity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

def _maybe_yearly_opportunity_reflection(now_ts):
    global _yearly_opportunity_summary, _last_opportunity_year
    local = time.localtime(now_ts)
    if local.tm_mon != 1 or local.tm_mday != 1 or local.tm_hour < 9:
        return
    year_key = time.strftime("%Y", local)
    if _last_opportunity_year == year_key:
        return
    current_start = time.mktime((local.tm_year, 1, 1, 0, 0, 0, 0, 0, -1))
    prev_year = local.tm_year - 1
    prev_start = time.mktime((prev_year, 1, 1, 0, 0, 0, 0, 0, -1))
    prev_end = current_start - 1
    year_label = str(prev_year)

    trades = []
    try:
        if os.path.exists(TRADE_OUTCOMES_PATH):
            with open(TRADE_OUTCOMES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        trades.append(rec)
    except Exception:
        trades = []

    symbol_prices = {}
    try:
        if os.path.exists(PRICE_HISTORY_PATH):
            with open(PRICE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = row.get("ts")
                    sym = row.get("symbol")
                    px = row.get("price")
                    if ts is None or sym is None or px is None:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        symbol_prices.setdefault(sym, []).append((ts, float(px)))
    except Exception:
        symbol_prices = {}
    for sym in symbol_prices:
        symbol_prices[sym].sort(key=lambda x: x[0])

    predictions = []
    try:
        if os.path.exists(LLM_PREDICTIONS_PATH):
            with open(LLM_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = rec.get("ts")
                    if not ts:
                        continue
                    ts = float(ts)
                    if prev_start <= ts <= prev_end:
                        predictions.append(rec)
    except Exception:
        predictions = []

    summary = _compute_daily_opportunity_summary(
        year_label,
        symbol_prices,
        trades,
        predictions
    )
    summary["year"] = year_label
    _yearly_opportunity_summary = summary
    _last_opportunity_year = year_key
    try:
        os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", "yearly_opportunity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

def _append_llm_action_log(message):
    if not message:
        return
    try:
        _llm_action_log.append(message)
        if len(_llm_action_log) > _llm_action_log_max:
            del _llm_action_log[0:len(_llm_action_log) - _llm_action_log_max]
    except Exception:
        pass
_loading_spin_index = 0
_loading_spin_last = 0.0
_win_count = 0
_loss_count = 0
_realized_total = 0.0
_realized_total_gross = 0.0
_realized_total_fees = 0.0
_autopilot_trade_pnls = []
_autopilot_last_tune_ts = 0.0
_autopilot_tune_mtime = 0.0
_autopilot_tune_last = ""
_perf_guard_state = "normal"
_perf_guard_last_log = 0.0
_autopilot_last_persist_ts = 0.0
_snapshot_stats_mtime = 0.0
_snapshot_first_value = None
_snapshot_first_ts = None
_equity_history_stats_mtime = 0.0
_equity_history_first_value = None
_equity_history_first_ts = None
_load_perf_guard_history()

def _load_portfolio_snapshot_first():
    global _snapshot_stats_mtime, _snapshot_first_value, _snapshot_first_ts
    path = os.path.join("logs", "portfolio_snapshots.jsonl")
    if not os.path.exists(path):
        return (None, None)
    try:
        mtime = os.path.getmtime(path)
        if mtime == _snapshot_stats_mtime:
            return (_snapshot_first_value, _snapshot_first_ts)
        first_val = None
        first_ts = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("timestamp")
                try:
                    ts = float(ts) if ts is not None else None
                except Exception:
                    ts = None
                snap = row.get("snapshot") or {}
                val = None
                for key in ("equity_usd", "total_usd", "cash_usd"):
                    if snap.get(key) not in (None, "", "None"):
                        try:
                            val = float(snap.get(key))
                        except Exception:
                            val = None
                        break
                if val is None or not math.isfinite(val) or val <= 0:
                    continue
                if ts is not None and ts > 0:
                    if first_ts is None or first_ts <= 0 or ts < first_ts:
                        first_ts = ts
                        first_val = val
                elif first_val is None:
                    first_val = val
        _snapshot_stats_mtime = mtime
        _snapshot_first_value = first_val
        _snapshot_first_ts = first_ts
        return (first_val, first_ts)
    except Exception:
        return (_snapshot_first_value, _snapshot_first_ts)

def _load_equity_history_first():
    global _equity_history_stats_mtime, _equity_history_first_value, _equity_history_first_ts
    path = os.path.join("data", "equity_history.json")
    if not os.path.exists(path):
        return (None, None)
    try:
        mtime = os.path.getmtime(path)
        if mtime == _equity_history_stats_mtime:
            return (_equity_history_first_value, _equity_history_first_ts)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        values = []
        timestamps = []
        if isinstance(data, dict):
            values = data.get("equity") or []
            timestamps = data.get("timestamps") or []
        elif isinstance(data, list):
            values = data
        first_val = None
        first_ts = None
        if values:
            try:
                first_val = float(values[0])
            except Exception:
                first_val = None
        if timestamps and len(timestamps) == len(values):
            min_ts = None
            min_idx = None
            for idx, raw_ts in enumerate(timestamps):
                try:
                    ts = float(raw_ts)
                except Exception:
                    continue
                if not math.isfinite(ts) or ts <= 0:
                    continue
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                    min_idx = idx
            if min_idx is not None:
                try:
                    first_val = float(values[min_idx])
                except Exception:
                    first_val = first_val
                first_ts = min_ts
        _equity_history_stats_mtime = mtime
        _equity_history_first_value = first_val
        _equity_history_first_ts = first_ts
        return (first_val, first_ts)
    except Exception:
        return (_equity_history_first_value, _equity_history_first_ts)

def _load_locked_baseline():
    path = os.path.join("data", "historical_baseline.json")
    if not os.path.exists(path):
        return (None, None)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val = data.get("baseline")
        ts = data.get("timestamp")
        try:
            val = float(val)
        except Exception:
            val = None
        try:
            ts = float(ts) if ts is not None else None
        except Exception:
            ts = None
        if val is None or not math.isfinite(val) or val <= 0:
            return (None, None)
        return (val, ts)
    except Exception:
        return (None, None)

def _maybe_lock_baseline(value, ts=None, source=None):
    try:
        if value is None:
            return None
        val = float(value)
        if not math.isfinite(val) or val <= 0:
            return None
        path = os.path.join("data", "historical_baseline.json")
        if os.path.exists(path):
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "baseline": val,
            "timestamp": float(ts) if ts is not None else time.time(),
            "source": source or "account_sync"
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return val
    except Exception:
        return None

def _get_historical_baseline():
    locked_val, locked_ts = _load_locked_baseline()
    if locked_val is not None and locked_val > 0:
        return float(locked_val)
    candidates = []
    snap_val, snap_ts = _load_portfolio_snapshot_first()
    if snap_val is not None and snap_val > 0:
        candidates.append(("snap", snap_ts, snap_val))
    hist_val, hist_ts = _load_equity_history_first()
    if hist_val is not None and hist_val > 0:
        candidates.append(("hist", hist_ts, hist_val))
    ts_candidates = [c for c in candidates if c[1] is not None and c[1] > 0]
    if ts_candidates:
        return float(min(ts_candidates, key=lambda c: c[1])[2])
    if candidates:
        return float(candidates[0][2])
    return None

def _initial_account_sync():
    try:
        if RUN_MODE != "live":
            return
        if account_client is None:
            return
        raw = account_client.get_balance()
        summary = _summarize_account(raw, live_prices)
        try:
            _maybe_lock_baseline(summary.get("total_usd"), ts=time.time(), source="kraken_account_sync")
        except Exception:
            pass
        live_cash = float(summary.get("cash_usd", 0.0))
        if hasattr(trader, "set_cash"):
            trader.set_cash(live_cash)
        else:
            trader.capital = live_cash
        if EXCHANGE == "kraken" and hasattr(trader, "positions"):
            inferred = _positions_from_balances(raw, live_prices)
            if inferred:
                merged = {}
                for sym, pos in inferred.items():
                    existing = trader.positions.get(sym) if isinstance(trader.positions, dict) else None
                    if existing:
                        pos["entry_price"] = existing.get("entry_price", pos["entry_price"])
                        pos["fill_price"] = existing.get("fill_price", pos["fill_price"])
                        pos["stop_loss"] = existing.get("stop_loss")
                        pos["take_profit"] = existing.get("take_profit")
                        pos["trailing_stop_pct"] = existing.get("trailing_stop_pct", 0.0)
                        pos["trailing_stop"] = existing.get("trailing_stop")
                        pos["llm_decision"] = existing.get("llm_decision")
                        pos["entry_source"] = existing.get("entry_source", pos.get("entry_source"))
                    merged[sym] = pos
                trader.positions = merged
        return summary
    except Exception:
        return None

def _record_attempt(symbol, side, status, reason=None, qty=None, price=None):
    attempted_orders.append({
        "timestamp": time.time(),
        "symbol": symbol,
        "side": side,
        "status": status,
        "reason": reason,
        "qty": qty,
        "price": price
    })
    if len(attempted_orders) > 200:
        attempted_orders[:] = attempted_orders[-200:]
    now = time.time()
    key = symbol or "UNKNOWN"
    last = _attempt_log_state.get(key, {})
    last_status = last.get("status")
    last_reason = last.get("reason")
    last_ts = last.get("ts", 0.0)
    if not DEBUG_LOG_ATTEMPTS:
        should_log = True
        if ATTEMPT_LOG_DEDUP_BY_REASON:
            if status == last_status and reason == last_reason and (now - last_ts) < ATTEMPT_LOG_COOLDOWN_SECONDS:
                should_log = False
        if (now - last_ts) < ATTEMPT_LOG_COOLDOWN_SECONDS and not should_log:
            return
    _attempt_log_state[key] = {"status": status, "reason": reason, "ts": now}
    if reason:
        print(f"[ATTEMPT] {symbol} {side} status={status} reason={reason}")
    else:
        print(f"[ATTEMPT] {symbol} {side} status={status}")

while True:
    try:
        now = time.time()
        if not _initial_sync_done:
            _initial_sync_done = True
            _initial_account_sync()
        if os.path.exists(CONFIG_PATH):
            new_mtime = os.path.getmtime(CONFIG_PATH)
            if new_mtime != _config_mtime:
                _config_mtime = new_mtime
                try:
                    from config import config as bot_config
                    importlib.reload(bot_config)
                    DISPLAY_MODE = getattr(bot_config, "RUN_MODE", DISPLAY_MODE)
                    DISPLAY_STYLE = getattr(bot_config, "TRADING_STYLE", DISPLAY_STYLE)
                    try:
                        import core.data.llm_signal_engine as llm_signal_engine
                        llm_signal_engine.CONTRARIAN_SENTIMENT_ENABLED = bool(getattr(bot_config, "CONTRARIAN_SENTIMENT_ENABLED", llm_signal_engine.CONTRARIAN_SENTIMENT_ENABLED))
                        llm_signal_engine.CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD = float(getattr(bot_config, "CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD", llm_signal_engine.CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD))
                        llm_signal_engine.CONTRARIAN_SENTIMENT_MAX_WEIGHT = float(getattr(bot_config, "CONTRARIAN_SENTIMENT_MAX_WEIGHT", llm_signal_engine.CONTRARIAN_SENTIMENT_MAX_WEIGHT))
                        llm_signal_engine.CONTRARIAN_SENTIMENT_LOG = bool(getattr(bot_config, "CONTRARIAN_SENTIMENT_LOG", llm_signal_engine.CONTRARIAN_SENTIMENT_LOG))
                    except Exception:
                        pass
                except Exception:
                    pass
        
        # A. Sync Account Balance
        wallet_snapshot = None
        if (now - last_account_fetch) >= ACCOUNT_INFO_REFRESH_SECONDS:
            if RUN_MODE == "live":
                account_info_raw = account_client.get_balance() if account_client else {}
                account_info = _summarize_account(account_info_raw, live_prices)
                # Detect non-cash balances to help UI gating
                has_non_cash = False
                try:
                    for asset, info in account_info_raw.items():
                        qty = float(info.get("free", 0.0) or 0.0)
                        if qty <= 0:
                            continue
                        norm_asset = _normalize_asset_code(asset)
                        if norm_asset in ("USD", "USDT", "ZUSD"):
                            continue
                        has_non_cash = True
                        break
                except Exception:
                    has_non_cash = False
                live_cash = float(account_info.get("cash_usd", 0.0))
                if hasattr(trader, "set_cash"):
                    trader.set_cash(live_cash)
                else:
                    trader.capital = live_cash
                if EXCHANGE == "kraken" and hasattr(trader, "positions"):
                    try:
                        inferred = _positions_from_balances(account_info_raw, live_prices)
                        if inferred is not None:
                            merged = {}
                            for sym, pos in inferred.items():
                                existing = trader.positions.get(sym) if isinstance(trader.positions, dict) else None
                                if existing:
                                    pos["entry_price"] = existing.get("entry_price", pos["entry_price"])
                                    pos["fill_price"] = existing.get("fill_price", pos["fill_price"])
                                    pos["stop_loss"] = existing.get("stop_loss")
                                    pos["take_profit"] = existing.get("take_profit")
                                    pos["trailing_stop_pct"] = existing.get("trailing_stop_pct", 0.0)
                                    pos["trailing_stop"] = existing.get("trailing_stop")
                                    pos["llm_decision"] = existing.get("llm_decision")
                                    pos["entry_source"] = existing.get("entry_source", pos.get("entry_source"))
                                merged[sym] = pos
                            # For Kraken, balances are source of truth: drop stale positions not in balances
                            trader.positions = merged
                    except Exception:
                        pass
                account_info["positions_ready"] = True
                account_info["has_non_cash_balances"] = has_non_cash
                try:
                    account_info["positions_count"] = len(getattr(trader, "positions", {}) or {})
                except Exception:
                    account_info["positions_count"] = None
                last_account_fetch = now
                print(f"🧾 Live Balance USD={live_cash:.2f} | EXECUTION_MODE={EXECUTION_MODE}")
                if DEBUG_STATUS:
                    try:
                        if (now - _equity_debug_last) > 15:
                            print(f"[DEBUG] Raw balances: {account_info_raw}")
                            print(f"[DEBUG] Account summary: cash_usd={account_info.get('cash_usd')} total_usd={account_info.get('total_usd')}")
                            print(f"[DEBUG] Inferred positions: {list(getattr(trader, 'positions', {}).keys())}")
                            _equity_debug_last = now
                    except Exception:
                        pass
            else:
                if wallet is not None:
                    wallet_snapshot = wallet.snapshot()
                    has_non_cash = False
                    try:
                        for asset, qty in (wallet_snapshot.get("balances", {}) or {}).items():
                            if asset in ("USD", "USDT"):
                                continue
                            if float(qty or 0.0) > 0:
                                has_non_cash = True
                                break
                    except Exception:
                        has_non_cash = False
                    account_info = {
                        "cash_usd": float(wallet_snapshot.get("balances", {}).get("USD", 0.0)),
                        "total_usd": float(wallet.wallet_state().get("total_usd", 0.0)),
                        "positions_ready": True,
                        "positions_count": len(getattr(trader, "positions", {}) or {}),
                        "has_non_cash_balances": has_non_cash
                    }
                last_account_fetch = now
        if RUN_MODE == "sim" and wallet is not None and wallet_snapshot is None:
            wallet_snapshot = wallet.snapshot()

        # Risk guardrails (block new trades only)
        equity = _compute_equity(trader, prices, account_info)
        day_key = time.strftime("%Y-%m-%d", time.localtime())
        if day_key != trades_day_key:
            trades_day_key = day_key
            trades_today = 0
            if equity > 0:
                daily_start_equity = equity
                peak_equity = equity
                print(f"[RISK_RESET] Daily reset to equity={equity:.2f} ts={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        if equity > 0 and daily_start_equity is None:
            daily_start_equity = equity
        if equity > 0 and (peak_equity is None or equity > peak_equity):
            peak_equity = equity
        if RESET_DAILY_RISK_ON_START and not _risk_reset_done and equity > 0:
            daily_start_equity = equity
            peak_equity = equity
            _risk_reset_done = True
            print(f"[RISK_RESET] Baselines reset to equity={equity:.2f} ts={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        total_notional, per_symbol_notional = _compute_open_notional(trader, prices)
        open_positions = len(getattr(trader, "positions", {}) or {})
        daily_loss_pct = 0.0
        drawdown_pct = 0.0
        if daily_start_equity:
            daily_loss_pct = (equity - daily_start_equity) / daily_start_equity
        if peak_equity:
            drawdown_pct = (equity - peak_equity) / peak_equity

        new_block_reasons = []
        if MAX_DRAWDOWN_PCT > 0 and drawdown_pct <= -MAX_DRAWDOWN_PCT:
            new_block_reasons.append("MAX_DRAWDOWN")

        if new_block_reasons != block_reasons:
            if new_block_reasons:
                print(f"[RISK_GUARD] New trades paused: {', '.join(new_block_reasons)}")
            else:
                print("[RISK_GUARD] New trades resumed.")
        block_reasons = new_block_reasons
        block_new_trades = bool(block_reasons)

        # Performance guard: tighten risk when underperforming
        underperforming, win_rate, trade_count = _compute_performance_guard()
        if underperforming:
            perf_state = "recover"
        else:
            perf_state = "normal"
        if perf_state != _perf_guard_state or (now - _perf_guard_last_log) > 120:
            _perf_guard_state = perf_state
            _perf_guard_last_log = now
            status = "UNDERPERFORMING" if underperforming else "NORMAL"
            print(f"[PERF_GUARD] {status} win_rate={win_rate:.2%} trades={trade_count} realized_total={_realized_total:.2f}")

        effective_min_conf = 0.0
        effective_size_fraction = SIZE_FRACTION_DEFAULT
        effective_cooldown = COOLDOWN_SECONDS
        effective_symbol_cooldown = SYMBOL_COOLDOWN_SECONDS
        effective_stop_loss = STOP_LOSS_PCT_DEFAULT
        effective_take_profit = TAKE_PROFIT_PCT_DEFAULT
        effective_trailing = TRAILING_STOP_PCT_DEFAULT
        if underperforming:
            # LLM/RL controls risk parameters; do not auto-tighten here.
            pass

        # B. Refresh live trades -> positions (live mode)
        if RUN_MODE == "live" and (now - last_trades_fetch) >= LIVE_TRADES_REFRESH_SECONDS:
            for symbol in SYMBOLS:
                try:
                    trades_by_symbol[symbol] = execution_client.get_my_trades(symbol=symbol)
                except Exception as e:
                    print(f"❌ Trade fetch error for {symbol}: {e}")
            if hasattr(trader, "update_from_trades"):
                trader.update_from_trades(trades_by_symbol)
            last_trades_fetch = now

        # Gate trading until balances are fully reflected
        trading_ready_now = False
        if RUN_MODE == "live":
            try:
                acc_total = float(account_info.get("total_usd") or 0.0)
                acc_cash = float(account_info.get("cash_usd") or 0.0)
                has_non_cash = bool(account_info.get("has_non_cash_balances"))
                positions_ready = bool(account_info.get("positions_ready"))
                if positions_ready and acc_total > 0 and account_info.get("cash_usd") is not None:
                    if has_non_cash:
                        min_delta = max(1.0, 0.005 * max(acc_total, 1.0))
                        non_cash_delta = (acc_total - acc_cash)
                        positions_count = int(account_info.get("positions_count") or 0)
                        dust_notional = total_notional < max(1.0, MIN_NOTIONAL)
                        if non_cash_delta >= min_delta:
                            trading_ready_now = True
                        elif positions_count == 0 or dust_notional:
                            trading_ready_now = True
                        else:
                            trading_ready_now = False
                    else:
                        trading_ready_now = True
            except Exception:
                trading_ready_now = False
        else:
            trading_ready_now = True

        if trading_ready_now and not _trading_ready:
            _trading_ready = True
            print("[START] Trading enabled after account sync.")
        elif not _trading_ready:
            if (now - _trading_ready_last_notice) > 5:
                print("[SYNC] Waiting for balances before trading...")
                _trading_ready_last_notice = now

        for symbol in SYMBOLS:
            if symbol not in live_prices:
                continue
            
            last_price = live_prices[symbol]
            _update_trade_tracker(symbol, last_price)
            # --- SAFEGUARD: Request Staggering ---
            # Wait 200ms between symbols to spread out API weight
            time.sleep(0.2)
            prices[symbol] = last_price
            last_ts = price_timestamps.get(symbol)
            if last_ts and (now - last_ts) > STALE_PRICE_SECONDS:
                if (now - ws_start_time) > STALE_GRACE_SECONDS:
                    last_warn = stale_warned.get(symbol, 0)
                    if (now - last_warn) > STALE_WARN_INTERVAL_SECONDS:
                        print(f"WARNING [STALE_DATA] {symbol} price update >{STALE_PRICE_SECONDS}s old")
                        if DEBUG_STATUS:
                            age = now - last_ts
                            print(f"[STALE_DEBUG] {symbol} age={age:.2f}s last_ts={last_ts:.3f} now={now:.3f}")
                        stale_warned[symbol] = now
                    # Kraken fallback: refresh price via REST if WS stalls
                    if EXCHANGE == "kraken" and hasattr(live_execution_client, "get_ticker"):
                        try:
                            ticker = live_execution_client.get_ticker(symbol)
                            result = ticker.get("result", {}) if isinstance(ticker, dict) else {}
                            key = next((k for k in result.keys() if k != "last"), None)
                            data = result.get(key, {}) if key else {}
                            last = data.get("c")
                            if isinstance(last, (list, tuple)) and last:
                                last = last[0]
                            if last is not None:
                                live_prices[symbol] = float(last)
                                price_timestamps[symbol] = time.time()
                                if DEBUG_STATUS:
                                    print(f"[STALE_FALLBACK] {symbol} refreshed via REST: {last}")
                        except Exception as e:
                            if DEBUG_STATUS:
                                print(f"[STALE_FALLBACK] {symbol} REST refresh failed: {e}")

            # B. Technical Analysis
            from core.data.market_data_provider import fetch_ohlcv
            df, last_fetch_time = fetch_ohlcv(
                symbol,
                TIMEFRAME,
                limit=100,
                client=getattr(live_execution_client, "client", None),
                exchange=EXCHANGE
            )
            # Format the time for the UI (e.g., "12:05:01")
            formatted_time = time.strftime("%H:%M:%S", time.localtime(last_fetch_time))


            recent_ohlcv = None
            recent_closes = None
            candle_patterns = None
            last_candle_close_ts = None
            if not df.empty:
                tf_seconds = _timeframe_to_seconds(TIMEFRAME)
                closed_df = df
                try:
                    last_open_ts = df["timestamp"].iloc[-1].timestamp()
                    if now < (last_open_ts + tf_seconds) and len(df) > 1:
                        closed_df = df.iloc[:-1]
                        last_open_ts = closed_df["timestamp"].iloc[-1].timestamp()
                    last_candle_close_ts = last_open_ts + tf_seconds
                except Exception:
                    closed_df = df
                if closed_df.empty:
                    recent_ohlcv = None
                    recent_closes = None
                    candle_patterns = None
                else:
                    recent_ohlcv = (
                        closed_df[["open", "high", "low", "close", "volume"]]
                    .tail(30)
                    .to_dict(orient="records")
                    )
                    recent_closes = closed_df["close"].tail(30).tolist()
                    candle_patterns = _detect_candle_patterns(closed_df)
                    rsi = ta.momentum.RSIIndicator(closed_df["close"]).rsi().iloc[-1]
                    ema20 = ta.trend.EMAIndicator(closed_df["close"], window=20).ema_indicator().iloc[-1]
                    atr = ta.volatility.AverageTrueRange(closed_df["high"], closed_df["low"], closed_df["close"]).average_true_range().iloc[-1]
                    vwap = ta.volume.VolumeWeightedAveragePrice(
                        closed_df["high"], closed_df["low"], closed_df["close"], closed_df["volume"]
                    ).volume_weighted_average_price().iloc[-1]
                    avg_vol = None
                    current_vol = None
                    vol_ratio = None
                    try:
                        avg_vol = closed_df["volume"].rolling(window=20).mean().iloc[-1]
                        current_vol = closed_df["volume"].iloc[-1]
                        if avg_vol and avg_vol > 0:
                            vol_ratio = float(current_vol) / float(avg_vol)
                    except Exception:
                        avg_vol = None
                        current_vol = None
                        vol_ratio = None
                    indicators[symbol] = {
                        "rsi": rsi,
                        "ema": ema20,
                        "ema20": ema20,
                        "vwap": vwap,
                        "vol_ratio": vol_ratio,
                        "current_vol": current_vol,
                        "avg_vol": avg_vol,
                        "atr_pct": (atr/last_price),
                        "last_update": formatted_time
                    }

            if not _trading_ready:
                continue

            # C. LLM Decision (Throttled)
            if USE_LLM and (now - last_llm_call[symbol] >= LLM_CHECK_INTERVAL):
                if last_candle_close_ts is None or now < last_candle_close_ts:
                    continue
                last_seen_candle = _last_candle_close_ts.get(symbol)
                if last_seen_candle == last_candle_close_ts:
                    continue
                _last_candle_close_ts[symbol] = last_candle_close_ts
                sentiment_payload = fetch_sentiment_score(symbol)
                if isinstance(sentiment_payload, (list, tuple)) and len(sentiment_payload) > 0:
                    sentiment_score = sentiment_payload[0]
                    vol_ratio = sentiment_payload[3] if len(sentiment_payload) > 3 else None
                else:
                    sentiment_score = sentiment_payload
                    vol_ratio = None
                sentiments[symbol] = sentiment_score
                if vol_ratio is not None:
                    indicators.setdefault(symbol, {})
                    indicators[symbol]["vol_ratio"] = vol_ratio
                
                portfolio_context = _portfolio_snapshot(
                    trader,
                    prices,
                    account_info,
                    total_notional,
                    per_symbol_notional,
                    equity
                )
                portfolio_context["target_allocation"] = TARGET_ALLOCATION
                portfolio_context["risk_overrides"] = {
                    "drawdown_pct": drawdown_pct,
                    "daily_loss_pct": daily_loss_pct
                }
                ta_context = _build_ta_context(symbol, indicators, recent_ohlcv, candle_patterns)
                multi_tf = {}
                try:
                    for tf in ("15m", "1h", "4h", "1d"):
                        df_tf, _ = fetch_ohlcv(
                            symbol,
                            tf,
                            limit=200,
                            client=getattr(live_execution_client, "client", None),
                            exchange=EXCHANGE
                        )
                        if df_tf is None or df_tf.empty:
                            continue
                        tf_seconds = _timeframe_to_seconds(tf)
                        try:
                            last_open_tf = df_tf["timestamp"].iloc[-1].timestamp()
                            if now < (last_open_tf + tf_seconds) and len(df_tf) > 1:
                                df_tf = df_tf.iloc[:-1]
                        except Exception:
                            pass
                        if df_tf.empty:
                            continue
                        patterns_tf = _detect_candle_patterns(df_tf)
                        rsi_tf = ta.momentum.RSIIndicator(df_tf["close"]).rsi().iloc[-1]
                        ema_tf = ta.trend.EMAIndicator(df_tf["close"], window=20).ema_indicator().iloc[-1]
                        atr_tf = ta.volatility.AverageTrueRange(df_tf["high"], df_tf["low"], df_tf["close"]).average_true_range().iloc[-1]
                        last_close_tf = df_tf["close"].iloc[-1]
                        trend_tf = "bullish" if last_close_tf >= ema_tf else "bearish"
                        multi_tf[tf] = {
                            "trend": trend_tf,
                            "rsi": float(rsi_tf) if rsi_tf is not None else None,
                            "ema20": float(ema_tf) if ema_tf is not None else None,
                            "atr_pct": float(atr_tf / last_close_tf) if last_close_tf else None,
                            "patterns": patterns_tf
                        }
                except Exception:
                    multi_tf = {}
                perf_summary = _build_perf_summary(
                    _win_count,
                    _loss_count,
                    _realized_total,
                    trade_count,
                    win_rate,
                    underperforming,
                    equity,
                    realized_total_gross=_realized_total_gross,
                    realized_total_fees=_realized_total_fees
                )
                risk_state = _build_risk_state(
                    equity,
                    drawdown_pct,
                    daily_loss_pct,
                    total_notional,
                    per_symbol_notional,
                    open_positions,
                    effective_min_conf
                )
                execution_context = _build_execution_context(symbol, trades_by_symbol, portfolio_context)
                execution_context["decision_candle_close_ts"] = last_candle_close_ts
                execution_context["fee_model"] = {
                    "maker_pct": KRAKEN_MAKER_FEE_PCT,
                    "taker_pct": KRAKEN_TAKER_FEE_PCT,
                    "slippage_pct": ESTIMATED_SLIPPAGE_PCT
                }
                if multi_tf:
                    execution_context["multi_timeframe"] = multi_tf
                wallet_state = _build_wallet_state(account_info, trader, equity)
                llm_result = llm_decision(
                    symbol, last_price, sentiment_score, 0.0, trader.positions,
                    rsi=ta_context.get("rsi", 50),
                    ema_20=ta_context.get("ema_20", last_price),
                    vwap=ta_context.get("vwap"),
                    atr_pct=ta_context.get("atr_pct"),
                    recent_closes=recent_closes,
                    recent_ohlcv=ta_context.get("recent_ohlcv"),
                    candle_patterns=ta_context.get("candle_patterns"),
                    current_vol=ta_context.get("current_vol"),
                    avg_vol=ta_context.get("avg_vol"),
                    vol_ratio=ta_context.get("vol_ratio"),
                    perf_summary=perf_summary,
                    execution_context=execution_context if RUN_MODE == "live" else None,
                    risk_state=risk_state,
                    wallet_state=wallet_state
                )
                latest_llm_symbol = symbol
                latest_llm_ts = time.strftime("%H:%M:%S")
                latest_llm_sentiment = sentiment_score
                latest_llm_ta = ta_context
                latest_llm_perf = perf_summary
                latest_llm_risk = risk_state
                latest_llm_wallet = wallet_state
                if isinstance(llm_result, dict):
                    latest_llm_summary = {
                        "confidence": llm_result.get("confidence"),
                        "size_fraction": llm_result.get("size_fraction"),
                        "stop_loss_pct": llm_result.get("stop_loss_pct"),
                        "take_profit_pct": llm_result.get("take_profit_pct"),
                        "trailing_stop_pct": llm_result.get("trailing_stop_pct"),
                        "price_prediction": llm_result.get("price_prediction"),
                        "prediction_horizon_min": llm_result.get("prediction_horizon_min"),
                        "conviction": llm_result.get("conviction"),
                        "predictions": llm_result.get("predictions"),
                        "exit": llm_result.get("exit"),
                        "order_action": llm_result.get("order_action"),
                        "reason": llm_result.get("reason"),
                        "pattern_reason": llm_result.get("pattern_reason")
                    }
                    if llm_result.get("price_prediction") is not None or llm_result.get("conviction") is not None:
                        _append_llm_prediction({
                            "ts": now,
                            "symbol": symbol,
                            "last_price": last_price,
                            "price_prediction": llm_result.get("price_prediction"),
                            "prediction_horizon_min": llm_result.get("prediction_horizon_min"),
                            "conviction": llm_result.get("conviction"),
                            "predictions": llm_result.get("predictions")
                        })
                else:
                    latest_llm_summary = {"decision": "no_result"}
                if isinstance(llm_result, dict):
                    try:
                        if hasattr(trader, "positions") and isinstance(trader.positions, dict):
                            pos = trader.positions.get(symbol) or trader.positions.get(symbol.replace("/", ""))
                            if isinstance(pos, dict):
                                pos["llm_decision"] = llm_result
                    except Exception:
                        pass
                llm_outputs[symbol] = llm_result
                if isinstance(llm_result, dict):
                    llm_result["_ts"] = now
                    try:
                        ts = time.strftime("%H:%M:%S")
                        action = llm_result.get("order_action") or "none"
                        conf = float(llm_result.get("confidence", 0.0) or 0.0)
                        size = float(llm_result.get("size_fraction", 0.0) or 0.0)
                        exit_flag = bool(llm_result.get("exit", False))
                        reason = (llm_result.get("reason") or "").strip()[:140]
                        pattern_reason = (llm_result.get("pattern_reason") or "").strip()[:140]
                        line = f"[{ts}] {symbol} | conf={conf:.2f} size={size:.2f} exit={exit_flag} action={action}"
                        if reason:
                            line += f" | reason={reason}"
                        if pattern_reason:
                            line += f" | pattern={pattern_reason}"
                        _append_llm_action_log(line)
                    except Exception:
                        pass
                if isinstance(llm_result, dict) and llm_result.get("order_action"):
                    print(f"🧠 LLM candidate for {symbol}: {llm_result.get('order_action')}")
                else:
                    if isinstance(llm_result, dict):
                        conf = llm_result.get("confidence")
                        exit_flag = llm_result.get("exit")
                        reason = llm_result.get("reason") or "no_action"
                    else:
                        conf = "n/a"
                        exit_flag = "n/a"
                        reason = "no_action"
                    if DEBUG_LOG_ATTEMPTS or DEBUG_STATUS:
                        print(f"[LLM] {symbol} no order_action | confidence={conf} min_required={effective_min_conf} exit={exit_flag} reason={reason}")
                    if DEBUG_LOG_ATTEMPTS:
                        _record_attempt(symbol, None, "SKIPPED", reason.upper())
                if hasattr(trader, "apply_llm_risk"):
                    trader.apply_llm_risk(symbol, llm_result)
                last_llm_call[symbol] = now
            
            current_llm = llm_outputs.get(symbol, {})
            
            # D. Execution Routing
            raw_action = current_llm.get("order_action")
            confidence = float(current_llm.get("confidence", 0.0) or 0.0)
            if raw_action and confidence < effective_min_conf:
                if DEBUG_LOG_ATTEMPTS or DEBUG_STATUS:
                    print(f"[LLM] {symbol} action below confidence threshold {confidence:.3f} < {effective_min_conf:.3f} | action={raw_action}")
                if DEBUG_LOG_ATTEMPTS:
                    _record_attempt(symbol, None, "SKIPPED", "LOW_CONFIDENCE")
            if raw_action and EXECUTION_MODE in ("live", "sim") and confidence >= effective_min_conf:
                # Allow exits even when risk guard blocks new entries
                action_intent = None
                if isinstance(raw_action, dict):
                    action_intent = (
                        raw_action.get("action")
                        or raw_action.get("side")
                        or raw_action.get("type")
                        or raw_action.get("order_type")
                    )
                action_intent = str(action_intent).upper() if action_intent else None
                if action_intent in ("CLOSE_POSITION", "CLOSE", "EXIT"):
                    action_intent = "SELL"
                if BLOCK_ON_STALE_PRICE:
                    if not last_ts or (now - last_ts) > STALE_PRICE_SECONDS:
                        _record_attempt(symbol, None, "BLOCKED", "STALE_PRICE")
                        continue
                if block_new_trades:
                    if action_intent != "SELL":
                        # Optional rebalance: sell weaker positions to free exposure
                        if ENABLE_REBALANCE and "MAX_TOTAL_EXPOSURE" in block_reasons and action_intent == "BUY":
                            if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                print(f"[REBALANCE] Considered for {symbol}: exposure={total_notional:.2f} equity={equity:.2f} target_free={MIN_EXPOSURE_RESUME_PCT:.2f}")
                            target_notional = MIN_EXPOSURE_RESUME_PCT * equity if equity > 0 else 0.0
                            need_reduce = max(0.0, total_notional - target_notional)
                            candidates = []
                            for pos_sym, pos in (trader.positions or {}).items():
                                if pos_sym == symbol:
                                    continue
                                price = live_prices.get(pos_sym) or pos.get("entry_price")
                                if not price:
                                    continue
                                held_for = now - float(pos.get("timestamp", now))
                                if held_for < REBALANCE_MIN_HOLD_SECONDS:
                                    continue
                                last_reb = _rebalance_log_state.get(pos_sym, 0.0)
                                if (now - last_reb) < REBALANCE_COOLDOWN_SECONDS:
                                    continue
                                entry = float(pos.get("entry_price", 0.0) or 0.0)
                                pnl_pct = (float(price) - entry) / entry if entry else 0.0
                                score = _score_symbol(pos_sym, sentiments, indicators)
                                candidates.append((score, pnl_pct, pos_sym, pos, float(price)))
                            if need_reduce > 0 and candidates:
                                if REBALANCE_PREFER_LOSERS:
                                    candidates.sort(key=lambda x: (x[1], x[0]))
                                else:
                                    candidates.sort(key=lambda x: x[0])
                                weakest_score = candidates[0][0]
                                target_score = _score_symbol(symbol, sentiments, indicators)
                                if target_score < (weakest_score + REBALANCE_MIN_SCORE_DELTA):
                                    if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                        print(f"[REBALANCE] Skipped: target score {target_score:.3f} < weakest {weakest_score:.3f}+{REBALANCE_MIN_SCORE_DELTA}")
                                else:
                                    _, _, sell_sym, pos, price = candidates[0]
                                    if REBALANCE_ADVISORY_MODE:
                                        advice_key = f"{symbol}->{sell_sym}"
                                        last_advice_ts = _rebalance_advice_cache.get(advice_key, 0.0)
                                        if (now - last_advice_ts) > 120:
                                            advice = llm_rebalance_advice(
                                                target_symbol=symbol,
                                                target_score=target_score,
                                                candidate_symbol=sell_sym,
                                                candidate_score=weakest_score,
                                                candidate_pnl_pct=(candidates[0][1] * 100.0),
                                                execution_context={"reason": "max_total_exposure"}
                                            )
                                            _rebalance_advice_cache[advice_key] = now
                                        else:
                                            advice = {"rebalance": True, "reason": "cached"}
                                        if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                            print(f"[REBALANCE] Advisory for {symbol}->{sell_sym}: {advice}")
                                        if not advice.get("rebalance", False):
                                            if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                                print(f"[REBALANCE] Advisory blocked: {advice.get('reason')}")
                                            continue
                                    pos_qty = float(pos.get("size", 0.0) or 0.0)
                                    max_sell_qty = pos_qty * REBALANCE_SELL_FRACTION
                                    desired_qty = min(max_sell_qty, need_reduce / float(price))
                                    if desired_qty > 0:
                                        sell_action = {
                                            "action": "PLACE_ORDER",
                                            "side": "SELL",
                                            "symbol": sell_sym.replace("/", ""),
                                            "quantity": desired_qty,
                                            "type": "MARKET"
                                        }
                                        if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                            print(f"[REBALANCE] Selling {sell_sym} qty={desired_qty:.6f} to free exposure")
                                        try:
                                            v_qty, _ = live_execution_client.validate_order(sell_sym, desired_qty)
                                            sell_action["quantity"] = v_qty
                                            res = execute_llm_action(execution_client, sell_action)
                                            _record_attempt(sell_sym, "SELL", res.get("status", "SUCCESS"), res.get("info"), qty=v_qty, price=price)
                                            _log_order_action(
                                                sell_sym,
                                                sell_action,
                                                res,
                                                mode="rebalance",
                                                outcome="REBALANCE",
                                                realized_pnl=None,
                                                reward_score=None,
                                                llm_payload=current_llm,
                                                last_price=price
                                            )
                                            _log_portfolio_snapshot(
                                                _portfolio_snapshot(trader, prices, account_info, total_notional, per_symbol_notional, equity),
                                                tag="rebalance",
                                                symbol=sell_sym
                                            )
                                            if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                                print(f"[REBALANCE] Logged rebalance sell for {sell_sym} qty={v_qty}")
                                            _rebalance_log_state[sell_sym] = now
                                            # Skip blocking this loop to allow subsequent ticks to act on freed exposure
                                            continue
                                        except Exception as e:
                                            if DEBUG_STATUS:
                                                print(f"[REBALANCE] Failed to sell {sell_sym}: {e}")
                            else:
                                if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                    reason = "no_reduction_needed" if need_reduce <= 0 else "no_candidates"
                                    print(f"[REBALANCE] Skipped: {reason}")
                        llm_ts = current_llm.get("_ts") if isinstance(current_llm, dict) else None
                        last_guard_ts = risk_guard_log_state.get(symbol)
                        if llm_ts is None or last_guard_ts != llm_ts:
                            _record_attempt(symbol, None, "BLOCKED", "RISK_GUARD")
                            if DEBUG_LOG_ATTEMPTS or DEBUG_STATUS:
                                print(f"[RISK_GUARD] Blocked {symbol} action={action_intent} reasons={', '.join(block_reasons)}")
                            risk_guard_log_state[symbol] = llm_ts
                        continue
                # Cooldown prevents rapid-fire trades across symbols
                if (now - last_trade_time) < effective_cooldown:
                    last_cd = _cooldown_log_state.get(symbol, 0.0)
                    if (now - last_cd) >= COOLDOWN_LOG_INTERVAL_SECONDS:
                        _record_attempt(symbol, None, "BLOCKED", "COOLDOWN")
                        _cooldown_log_state[symbol] = now
                    continue
                # Per-symbol cooldown
                if effective_symbol_cooldown > 0:
                    last_sym_trade = last_symbol_trade_time.get(symbol, 0.0)
                    if (now - last_sym_trade) < effective_symbol_cooldown:
                        last_sym_cd = _symbol_cooldown_log_state.get(symbol, 0.0)
                        if (now - last_sym_cd) >= SYMBOL_COOLDOWN_LOG_INTERVAL_SECONDS:
                            _record_attempt(symbol, None, "BLOCKED", "SYMBOL_COOLDOWN")
                            _symbol_cooldown_log_state[symbol] = now
                        continue
                if trades_today >= MAX_TRADES_PER_DAY:
                    _record_attempt(symbol, None, "BLOCKED", "MAX_TRADES_PER_DAY")
                    continue
                if now < reject_backoff_until.get(symbol, 0.0):
                    last_rej = _reject_backoff_log_state.get(symbol, 0.0)
                    if (now - last_rej) >= REJECT_BACKOFF_LOG_INTERVAL_SECONDS:
                        _record_attempt(symbol, None, "BLOCKED", "REJECT_BACKOFF")
                        _reject_backoff_log_state[symbol] = now
                    continue
                if EXECUTION_MODE == "live":
                    used_weight = getattr(live_execution_client, "last_used_weight", None)
                    order_count = getattr(live_execution_client, "last_order_count", None)
                    if used_weight is not None and used_weight >= MAX_API_WEIGHT_1M:
                        _record_attempt(symbol, None, "BLOCKED", "API_WEIGHT")
                        continue
                    if order_count is not None and order_count >= MAX_ORDER_COUNT_10S:
                        _record_attempt(symbol, None, "BLOCKED", "ORDER_COUNT")
                        continue
                if isinstance(raw_action, dict):
                    # Normalize LLM action schema variations
                    if "action" not in raw_action and raw_action.get("side"):
                        raw_action["action"] = raw_action.get("side")
                    if "size_fraction" not in raw_action and raw_action.get("quantity_fraction"):
                        raw_action["size_fraction"] = raw_action.get("quantity_fraction")
                # Retry throttle for failed actions
                last_attempt = last_action_attempt.get(symbol, 0.0)
                if (now - last_attempt) < ORDER_RETRY_SECONDS:
                    continue
                size_fraction = effective_size_fraction
                if isinstance(raw_action, dict):
                    try:
                        size_fraction = float(raw_action.get("size_fraction") or effective_size_fraction)
                    except Exception:
                        size_fraction = effective_size_fraction
                action, summary = _normalize_action(raw_action, symbol, last_price, trader, size_fraction, 0.0)
                if action is None and raw_action:
                    if DEBUG_LOG_ATTEMPTS or DEBUG_STATUS:
                        print(f"[LLM] {symbol} invalid order_action schema: {raw_action}")
                    if DEBUG_LOG_ATTEMPTS:
                        _record_attempt(symbol, None, "SKIPPED", "INVALID_ACTION")
                if action:
                    try:
                        last_action_attempt[symbol] = now
                        if EXECUTION_MODE == "live":
                            order_price = action.get("price") or last_price
                            available_cash = None
                            if isinstance(account_info, dict):
                                available_cash = account_info.get("cash_usd")
                            if available_cash is None:
                                available_cash = getattr(trader, "capital", 0.0)
                            if action.get("side", "").upper() == "BUY" and order_price:
                                cash_buffer = 0.98
                                max_notional = max(0.0, float(available_cash or 0.0) * cash_buffer)
                                notional = float(order_price or 0.0) * float(action.get("quantity") or 0.0)
                                if max_notional > 0 and notional > max_notional:
                                    new_qty = max_notional / float(order_price)
                                    if new_qty <= 0:
                                        reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                        _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "INSUFFICIENT_CASH", qty=action.get("quantity"), price=order_price)
                                        continue
                                    action["quantity"] = new_qty
                                    if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                        print(f"[CASH_CLAMP] {action.get('symbol', symbol)} qty adjusted to {new_qty:.8f} based on cash={available_cash}")
                            # Finalize qty with exchange filters
                            v_qty, _ = live_execution_client.validate_order(action['symbol'], action['quantity'])
                            action['quantity'] = v_qty
                            order_price = action.get("price") or last_price
                            if EXCHANGE == "kraken":
                                min_qty = None
                                if hasattr(live_execution_client, "get_min_order_qty"):
                                    min_qty = live_execution_client.get_min_order_qty(action.get("symbol", symbol))
                                if min_qty is not None and v_qty < min_qty:
                                    if str(action.get("side", "")).upper() == "BUY":
                                        available_cash = float(getattr(trader, "capital", 0.0) or 0.0)
                                        needed_notional = float(order_price or 0.0) * float(min_qty or 0.0)
                                        if ALLOW_MIN_UPSIZE and available_cash >= needed_notional:
                                            action['quantity'] = min_qty
                                            v_qty = min_qty
                                            if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                                print(f"[KRAKEN_MIN] Upsized to min qty {min_qty} for {action.get('symbol', symbol)}")
                                        else:
                                            reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                            _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "KRAKEN_MIN_SIZE", qty=v_qty, price=order_price)
                                            continue
                                    else:
                                        reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                        _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "KRAKEN_MIN_SIZE", qty=v_qty, price=order_price)
                                        continue
                            notional = float(order_price or 0.0) * float(v_qty or 0.0)
                            if action.get("side", "").upper() == "BUY" and available_cash is not None:
                                max_notional = float(available_cash) * 0.98
                                if max_notional > 0 and notional > max_notional:
                                    reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                    _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "INSUFFICIENT_CASH", qty=v_qty, price=order_price)
                                    continue
                            if notional < MIN_NOTIONAL or v_qty <= 0:
                                reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "LOCAL_MIN_NOTIONAL", qty=v_qty, price=order_price)
                                _log_order_action(
                                    action.get("symbol", symbol),
                                    action,
                                    {"status": "REJECTED", "info": "LOCAL_MIN_NOTIONAL"},
                                    mode=EXECUTION_MODE,
                                    outcome="LOCAL_MIN_NOTIONAL",
                                    realized_pnl=None,
                                    reward_score=None,
                                    llm_payload=current_llm,
                                    last_price=last_price
                                )
                                continue
                            print(f"🧮 Live order sizing {action['symbol']} qty={v_qty} price={last_price}")
                        else:
                            order_price = action.get("price") or last_price
                            available_cash = getattr(trader, "capital", 0.0)
                            if action.get("side", "").upper() == "BUY" and order_price:
                                cash_buffer = 0.98
                                max_notional = max(0.0, float(available_cash or 0.0) * cash_buffer)
                                notional = float(order_price or 0.0) * float(action.get("quantity") or 0.0)
                                if max_notional > 0 and notional > max_notional:
                                    new_qty = max_notional / float(order_price)
                                    if new_qty <= 0:
                                        reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                        _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "INSUFFICIENT_CASH", qty=action.get("quantity"), price=order_price)
                                        continue
                                    action["quantity"] = new_qty
                                    if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                        print(f"[CASH_CLAMP] {action.get('symbol', symbol)} qty adjusted to {new_qty:.8f} based on cash={available_cash}")
                            notional = float(order_price or 0.0) * float(action.get("quantity") or 0.0)
                            if notional < MIN_NOTIONAL or float(action.get("quantity") or 0.0) <= 0:
                                reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                                _record_attempt(action.get("symbol", symbol), action.get("side"), "REJECTED", "LOCAL_MIN_NOTIONAL", qty=action.get("quantity"), price=order_price)
                                _log_order_action(
                                    action.get("symbol", symbol),
                                    action,
                                    {"status": "REJECTED", "info": "LOCAL_MIN_NOTIONAL"},
                                    mode=EXECUTION_MODE,
                                    outcome="LOCAL_MIN_NOTIONAL",
                                    realized_pnl=None,
                                    reward_score=None,
                                    llm_payload=current_llm,
                                    last_price=last_price
                                )
                                continue
                        # EXECUTE
                        log_symbol = action.get("symbol", symbol)
                        res = execute_llm_action(execution_client, action)
                        print(f"🔥 [{EXECUTION_MODE.upper()}] {summary} | Status: {res.get('status', 'SUCCESS')}")
                        _record_attempt(log_symbol, action.get("side"), res.get("status", "SUCCESS"), res.get("info"), qty=action.get("quantity"), price=order_price)
                        entry_price_for_fee = None
                        entry_order_type_for_fee = None
                        if res.get("status") not in ("REJECTED", "ERROR"):
                            side = str(action.get("side", "")).upper()
                            if side == "BUY":
                                _on_trade_open(
                                    _to_slash_symbol(log_symbol),
                                    order_price,
                                    qty=action.get("quantity"),
                                    order_type=action.get("type")
                                )
                            elif side == "SELL":
                                entry_price_for_fee, entry_order_type_for_fee = _get_entry_info(log_symbol)
                                _on_trade_close(_to_slash_symbol(log_symbol), order_price, reason="sell")
                        outcome = None
                        realized_pnl = None
                        if EXECUTION_MODE == "live" and hasattr(trader, "realized_pnl_by_symbol"):
                            side = action.get("side") if isinstance(action, dict) else None
                            norm_symbol = _to_slash_symbol(log_symbol)
                            current_realized = trader.realized_pnl_by_symbol.get(norm_symbol, 0.0)
                            prev_realized = realized_pnl_snapshot.get(norm_symbol, 0.0)
                            delta = current_realized - prev_realized
                            realized_pnl_snapshot[norm_symbol] = current_realized
                            if side and str(side).upper() == "SELL":
                                if delta != 0:
                                    realized_pnl = delta
                                    outcome = "W" if delta > 0 else "L"
                                else:
                                    outcome = "B"
                                # Fallback: estimate outcome if trade history hasn't caught up yet
                                if realized_pnl in (None, 0) and outcome == "B":
                                    try:
                                        pos = trader.positions.get(log_symbol) if isinstance(trader.positions, dict) else None
                                        if pos is None and isinstance(trader.positions, dict):
                                            pos = trader.positions.get(norm_symbol)
                                        entry = float(pos.get("entry_price")) if pos else None
                                        qty = float(action.get("quantity") or 0.0)
                                        if entry and qty and order_price:
                                            est = (float(order_price) - entry) * qty
                                            realized_pnl = est
                                            outcome = "W" if est > 0 else ("L" if est < 0 else "B")
                                    except Exception:
                                        pass
                        fee_total_est = None
                        slippage_est = None
                        realized_pnl_net = None
                        if realized_pnl is not None:
                            try:
                                qty_for_fee = action.get("quantity") if isinstance(action, dict) else None
                                exit_px_for_fee = order_price or last_price or (action.get("price") if isinstance(action, dict) else None)
                                fee_total_est, slippage_est = _estimate_round_trip_costs(
                                    log_symbol,
                                    qty_for_fee,
                                    exit_px_for_fee,
                                    exit_order_type=(action.get("type") if isinstance(action, dict) else None),
                                    entry_price=entry_price_for_fee,
                                    entry_order_type=entry_order_type_for_fee
                                )
                                if fee_total_est is not None:
                                    realized_pnl_net = float(realized_pnl) - float(fee_total_est)
                            except Exception:
                                fee_total_est = None
                                slippage_est = None
                        if realized_pnl_net is not None:
                            outcome = "W" if realized_pnl_net > 0 else ("L" if realized_pnl_net < 0 else "B")
                        if realized_pnl is not None:
                            _record_win_loss(realized_pnl, symbol=log_symbol, realized_pnl_net=realized_pnl_net, fee_total=fee_total_est)
                        reward_score = realized_pnl_net if realized_pnl_net is not None else (realized_pnl if realized_pnl is not None else 0.0)
                        _log_order_action(
                            log_symbol,
                            action,
                            res,
                            mode=EXECUTION_MODE,
                            outcome=outcome,
                            realized_pnl=realized_pnl,
                            realized_pnl_net=realized_pnl_net,
                            fee_total_est=fee_total_est,
                            slippage_est=slippage_est,
                            reward_score=reward_score,
                            llm_payload=current_llm,
                            last_price=last_price
                        )
                        _log_portfolio_snapshot(portfolio_context, tag="trade_execute", symbol=log_symbol)
                        _autopilot_on_realized_pnl(realized_pnl_net if realized_pnl_net is not None else realized_pnl, symbol=log_symbol)
                        if res.get("status") in ("REJECTED", "ERROR"):
                            reject_backoff_until[symbol] = now + REJECT_BACKOFF_SECONDS
                        if res.get("status") not in ("REJECTED", "ERROR"):
                            # Clear cached action only after a successful submit
                            if isinstance(current_llm, dict):
                                current_llm["order_action"] = None
                            orders_this_min += 1
                            trades_today += 1
                            last_trade_time = now
                            last_symbol_trade_time[symbol] = now
                        
                        # Sync Paper Trader (sim only)
                        if EXECUTION_MODE == "sim":
                            if action['side'] == 'BUY':
                                trader.enter(last_price, (action['quantity'] * last_price), symbol, size_in_dollars=False)
                            else:
                                realized_pnl = trader.exit(symbol, price=last_price)
                                _autopilot_on_realized_pnl(realized_pnl_net if realized_pnl_net is not None else realized_pnl, symbol=log_symbol)
                                fee_total_est, slippage_est = _estimate_round_trip_costs(
                                    log_symbol,
                                    action.get("quantity"),
                                    last_price,
                                    exit_order_type=(action.get("type") if isinstance(action, dict) else None)
                                )
                                realized_pnl_net = None
                                if realized_pnl is not None and fee_total_est is not None:
                                    try:
                                        realized_pnl_net = float(realized_pnl) - float(fee_total_est)
                                    except Exception:
                                        realized_pnl_net = None
                                _record_win_loss(realized_pnl, symbol=log_symbol, realized_pnl_net=realized_pnl_net, fee_total=fee_total_est)
                    except Exception as e:
                        print(f"❌ Execution Fail: {e}")

        # E. SL/TP Protection
        if EXECUTION_MODE in ("live", "sim"):
            exits = trader.check_sl_tp(prices)
            for exit_item in exits:
                s = exit_item[0] if isinstance(exit_item, (list, tuple)) else exit_item
                reason = None
                if isinstance(exit_item, (list, tuple)) and len(exit_item) > 1:
                    reason = exit_item[1]
                last_auto = _auto_exit_log_state.get(s, 0.0)
                if (now - last_auto) >= AUTO_EXIT_LOG_INTERVAL_SECONDS:
                    print(f"🛑 [AUTO-EXIT] {s} hit Stop/Target.")
                    _auto_exit_log_state[s] = now
                if EXECUTION_MODE == "sim":
                    pos = trader.positions.get(s, {}) if hasattr(trader, "positions") else {}
                    qty = float(pos.get("size", 0.0) or 0.0)
                    realized_pnl = _estimate_realized_pnl(s, qty, prices.get(s), trader)
                    realized_pnl_net = None
                    fee_total_est = None
                    slippage_est = None
                    outcome = None
                    if realized_pnl is not None:
                        fee_total_est, slippage_est = _estimate_round_trip_costs(
                            s,
                            qty,
                            prices.get(s),
                            exit_order_type="MARKET"
                        )
                        if fee_total_est is not None:
                            try:
                                realized_pnl_net = float(realized_pnl) - float(fee_total_est)
                            except Exception:
                                realized_pnl_net = None
                        outcome = "W" if (realized_pnl_net if realized_pnl_net is not None else realized_pnl) > 0 else (
                            "L" if (realized_pnl_net if realized_pnl_net is not None else realized_pnl) < 0 else "B"
                        )
                        _record_win_loss(realized_pnl, symbol=s, realized_pnl_net=realized_pnl_net, fee_total=fee_total_est)
                    _log_order_action(
                        s,
                        {"action": "AUTO_EXIT", "side": "SELL", "symbol": s.replace("/", ""), "type": "MARKET"},
                        {"status": "AUTO_EXIT"},
                        mode=EXECUTION_MODE,
                        outcome=outcome or reason,
                        realized_pnl=realized_pnl,
                        realized_pnl_net=realized_pnl_net,
                        fee_total_est=fee_total_est,
                        slippage_est=slippage_est,
                        reward_score=None,
                        llm_payload=llm_outputs.get(s, {}),
                        last_price=prices.get(s)
                    )
                    _on_trade_close(_to_slash_symbol(s), prices.get(s), reason="auto_exit")
                    continue
                try:
                    pos = trader.positions.get(s, {})
                    qty = float(pos.get("size", 0.0))
                    if qty > 0:
                        v_qty, _ = live_execution_client.validate_order(s, qty)
                        live_execution_client.create_order(symbol=s, side="SELL", type="MARKET", quantity=v_qty)
                        realized_pnl = _estimate_realized_pnl(s, v_qty, prices.get(s), trader)
                        realized_pnl_net = None
                        fee_total_est = None
                        slippage_est = None
                        outcome = None
                        if realized_pnl is not None:
                            fee_total_est, slippage_est = _estimate_round_trip_costs(
                                s,
                                v_qty,
                                prices.get(s),
                                exit_order_type="MARKET"
                            )
                            if fee_total_est is not None:
                                try:
                                    realized_pnl_net = float(realized_pnl) - float(fee_total_est)
                                except Exception:
                                    realized_pnl_net = None
                            outcome = "W" if (realized_pnl_net if realized_pnl_net is not None else realized_pnl) > 0 else (
                                "L" if (realized_pnl_net if realized_pnl_net is not None else realized_pnl) < 0 else "B"
                            )
                            _record_win_loss(realized_pnl, symbol=s, realized_pnl_net=realized_pnl_net, fee_total=fee_total_est)
                        _log_order_action(
                            s,
                            {"action": "AUTO_EXIT", "side": "SELL", "symbol": s.replace("/", ""), "type": "MARKET"},
                            {"status": "AUTO_EXIT"},
                            mode=EXECUTION_MODE,
                            outcome=outcome or reason,
                            realized_pnl=realized_pnl,
                            realized_pnl_net=realized_pnl_net,
                            fee_total_est=fee_total_est,
                            slippage_est=slippage_est,
                            reward_score=None,
                            llm_payload=llm_outputs.get(s, {}),
                            last_price=prices.get(s)
                        )
                except Exception as e:
                    print(f"❌ Exit Error: {e}")

        # F. Shutdown Check
        if _panic_requested:
            print("🛑 Panic requested. Exiting...")
            break

        orders_payload = None
        if RUN_MODE == "live" and hasattr(trader, "recent_trades"):
            orders_payload = trader.recent_trades(limit=50)
            if (not orders_payload) and attempted_orders:
                # Fallback: show successful attempts as live trades until trade history catches up
                orders_payload = []
                for o in attempted_orders[-80:]:
                    if o.get("status") != "SUCCESS":
                        continue
                    orders_payload.append({
                        "timestamp": o.get("timestamp"),
                        "symbol": o.get("symbol"),
                        "side": o.get("side"),
                        "type": "MARKET",
                        "status": "FILLED",
                        "origQty": o.get("qty"),
                        "price": o.get("price")
                    })
        elif RUN_MODE == "sim":
            if hasattr(execution_client, "get_all_orders"):
                orders_payload = execution_client.get_all_orders()
            else:
                orders_payload = getattr(trader, "orders", None)

        prompt_lines = []
        if latest_llm_symbol:
            prompt_lines.append(f"LLM Decision Trace @ {latest_llm_ts} | {latest_llm_symbol}")
            try:
                prompt_lines.append(f"Decision: {json.dumps(latest_llm_summary)}")
            except Exception:
                prompt_lines.append(f"Decision: {latest_llm_summary}")
            try:
                prompt_lines.append(f"Sentiment: {float(latest_llm_sentiment):.3f}")
            except Exception:
                prompt_lines.append(f"Sentiment: {latest_llm_sentiment}")
            try:
                pred = latest_llm_summary.get("price_prediction")
                horizon = latest_llm_summary.get("prediction_horizon_min")
                conv = latest_llm_summary.get("conviction")
                if pred is not None or horizon is not None or conv is not None:
                    prompt_lines.append(
                        f"Prediction: price={pred} horizon_min={horizon} conviction={conv}"
                    )
                preds = latest_llm_summary.get("predictions") or []
                if preds:
                    def _find_pred(target_h):
                        for p in preds:
                            try:
                                h = p.get("prediction_horizon_min")
                                if h is None:
                                    continue
                                if abs(int(h) - target_h) <= 5:
                                    return p
                            except Exception:
                                continue
                        return None
                    day_pred = _find_pred(1440)
                    week_pred = _find_pred(10080)
                    lt_bits = []
                    if day_pred:
                        lt_bits.append(f"1d={day_pred.get('price_prediction')}")
                    if week_pred:
                        lt_bits.append(f"1w={week_pred.get('price_prediction')}")
                    if lt_bits:
                        prompt_lines.append("Long-term: " + " | ".join(lt_bits))
            except Exception:
                pass
            try:
                dd = latest_llm_risk.get("drawdown_pct")
                max_dd = latest_llm_risk.get("max_drawdown_pct")
                exp = latest_llm_risk.get("exposure_ratio")
                opn = latest_llm_risk.get("open_positions")
                gate = latest_llm_risk.get("confidence_gate")
                prompt_lines.append(
                    f"Risk: drawdown={dd:.2%} max_dd={max_dd:.2%} exposure={exp:.2f} open_positions={opn} conf_gate={gate:.2f}"
                )
            except Exception:
                prompt_lines.append(f"Risk: {latest_llm_risk}")
            try:
                wins = latest_llm_perf.get("wins")
                losses = latest_llm_perf.get("losses")
                wr = latest_llm_perf.get("win_rate")
                rt = latest_llm_perf.get("realized_total")
                rt_gross = latest_llm_perf.get("realized_total_gross")
                rt_fees = latest_llm_perf.get("realized_total_fees")
                prompt_lines.append(
                    f"Perf: wins={wins} losses={losses} win_rate={wr:.2%} net={rt} gross={rt_gross} fees={rt_fees}"
                )
            except Exception:
                prompt_lines.append(f"Perf: {latest_llm_perf}")
            try:
                tune_msg = _load_autopilot_tune_message()
                if tune_msg:
                    prompt_lines.append(tune_msg)
            except Exception:
                pass
            try:
                opp = latest_llm_perf.get("daily_opportunity_summary") or {}
                if opp:
                    stats = opp.get("prediction_horizon_stats") or {}
                    def _hr(key):
                        item = stats.get(str(key)) or {}
                        return item.get("hit_rate", 0.0)
                    prompt_lines.append(
                        "Opportunity: "
                        f"avg_mfe={opp.get('avg_mfe_pct', 0.0):.2%} "
                        f"avg_mae={opp.get('avg_mae_pct', 0.0):.2%} "
                        f"avg_post_exit={opp.get('avg_post_exit_fav_pct', 0.0):.2%} "
                        f"post_exit>1%={opp.get('pct_post_exit_fav_gt_1pct', 0.0):.0%} "
                        f"pred_hit={opp.get('prediction_hit_rate', 0.0):.0%} "
                        f"avg_conv={opp.get('avg_prediction_conviction', 0.0):.2f} "
                        f"hit_60m={_hr(60):.0%} hit_1d={_hr(1440):.0%} hit_1w={_hr(10080):.0%}"
                    )
            except Exception:
                pass
            try:
                wk = latest_llm_perf.get("weekly_opportunity_summary") or {}
                if wk:
                    stats = wk.get("prediction_horizon_stats") or {}
                    def _hr(key):
                        item = stats.get(str(key)) or {}
                        return item.get("hit_rate", 0.0)
                    prompt_lines.append(
                        "Weekly: "
                        f"pred_hit={wk.get('prediction_hit_rate', 0.0):.0%} "
                        f"avg_conv={wk.get('avg_prediction_conviction', 0.0):.2f} "
                        f"hit_60m={_hr(60):.0%} hit_1d={_hr(1440):.0%} hit_1w={_hr(10080):.0%}"
                    )
            except Exception:
                pass
            try:
                mo = latest_llm_perf.get("monthly_opportunity_summary") or {}
                if mo:
                    stats = mo.get("prediction_horizon_stats") or {}
                    def _hr(key):
                        item = stats.get(str(key)) or {}
                        return item.get("hit_rate", 0.0)
                    prompt_lines.append(
                        "Monthly: "
                        f"pred_hit={mo.get('prediction_hit_rate', 0.0):.0%} "
                        f"avg_conv={mo.get('avg_prediction_conviction', 0.0):.2f} "
                        f"hit_60m={_hr(60):.0%} hit_1d={_hr(1440):.0%} hit_1w={_hr(10080):.0%}"
                    )
            except Exception:
                pass
            try:
                yr = latest_llm_perf.get("yearly_opportunity_summary") or {}
                if yr:
                    stats = yr.get("prediction_horizon_stats") or {}
                    def _hr(key):
                        item = stats.get(str(key)) or {}
                        return item.get("hit_rate", 0.0)
                    prompt_lines.append(
                        "Yearly: "
                        f"pred_hit={yr.get('prediction_hit_rate', 0.0):.0%} "
                        f"avg_conv={yr.get('avg_prediction_conviction', 0.0):.2f} "
                        f"hit_60m={_hr(60):.0%} hit_1d={_hr(1440):.0%} hit_1w={_hr(10080):.0%}"
                    )
            except Exception:
                pass
            try:
                patterns = ", ".join(latest_llm_ta.get("candle_patterns") or [])
                prompt_lines.append(
                    "TA: "
                    f"RSI={latest_llm_ta.get('rsi')}, "
                    f"EMA20={latest_llm_ta.get('ema_20')}, "
                    f"VWAP={latest_llm_ta.get('vwap')}, "
                    f"ATR%={latest_llm_ta.get('atr_pct')}, "
                    f"vol_ratio={latest_llm_ta.get('vol_ratio')}, "
                    f"patterns={patterns or 'none'}"
                )
            except Exception:
                prompt_lines.append(f"TA: {latest_llm_ta}")
            prompt_lines.append("")
            prompt_lines.append("Recent LLM actions:")
        prompt_lines.extend(_llm_action_log[-20:])
        prompt_text = prompt_lines if prompt_lines else list(_llm_action_log)
        _log_price_history(live_prices, now)
        _maybe_daily_opportunity_reflection(now)
        _maybe_weekly_opportunity_reflection(now)
        _maybe_monthly_opportunity_reflection(now)
        _maybe_yearly_opportunity_reflection(now)
        last_rss = get_last_rss_fetch_time()
        rss_active = bool(
            get_rss_active()
            or (last_rss and (now - last_rss) < (FETCH_INTERVAL * 2))
            or (sentiments is not None and len(sentiments) > 0)
            or (SENTIMENT_CACHE is not None and len(SENTIMENT_CACHE) > 0)
        )
        exchange_active = bool(
            RUN_MODE == "live" and (
                getattr(live_execution_client, "last_used_weight", None) is not None
                or (account_info is not None and len(account_info) > 0)
                or (live_prices is not None and len(live_prices) > 0)
            )
        )
        llm_active = bool(USE_LLM and _trading_ready)
        bot_active = True
        status_flags = {
            "Bot": bot_active,
            "LLM": llm_active,
            "Exchange": exchange_active,
            "RSS": rss_active,
            "TradingReady": _trading_ready
        }
        if DEBUG_STATUS and (now - last_status_log) > 10:
            print(f"[STATUS] Bot={bot_active} LLM={llm_active} ExchangeAPI={exchange_active} RSS={rss_active} | prices={len(live_prices)} sentiments={len(sentiments)} last_rss={last_rss}")
            last_status_log = now

        display_mode = DISPLAY_MODE or RUN_MODE
        display_style = DISPLAY_STYLE or TRADING_STYLE
        plotter.set_status(
            status_flags=status_flags,
            status_mode=display_mode,
            status_style=display_style,
            rss_last_ts=last_rss
        )
        if rss_active:
            try:
                plotter.set_indicator("RSS", True)
            except Exception:
                pass
        if not _ui_ready:
            if account_info and account_info.get("total_usd"):
                _ui_ready = True
            elif RUN_MODE != "live":
                _ui_ready = True
        if not _ui_ready:
            now = time.time()
            if (now - _loading_spin_last) >= 0.5:
                spinner = ["|", "/", "-", "\\"][int(_loading_spin_index % 4)]
                print(f"\r[UI] Loading dashboard {spinner}", end="")
                _loading_spin_index += 1
                _loading_spin_last = now
        else:
            if _loading_spin_last:
                print("\r[UI] Loading dashboard... done.   ")
        _ensure_position_risk_defaults(trader)
        plotter.update(
            prices,
            sentiments,
            indicators,
            account_info=account_info,
            wallet_snapshot=wallet_snapshot,
            orders=orders_payload,
            attempted_orders=attempted_orders,
            prompt_text=prompt_text,
            status_mode=display_mode,
            status_style=display_style,
            status_flags=status_flags,
            rss_last_ts=last_rss,
            price_timestamps=price_timestamps
        )
        if getattr(plotter, "restart_requested", False):
            print("🔁 Restart requested. Exiting to startup...")
            break
        iteration_count += 1
        if iteration_count % 60 == 0:
            current_weight = getattr(live_execution_client, "last_used_weight", None)
            current_weight_display = current_weight if current_weight is not None else "N/A"
            print(f"[MINUTE REPORT] Total Weight Used: {current_weight_display} | Orders Placed: {orders_this_min}")
            orders_this_min = 0
        time.sleep(UI_REFRESH_SECONDS)
        # main.py


    except Exception:
        print(traceback.format_exc())
        time.sleep(1)

if getattr(plotter, "restart_requested", False):
    os.execv(sys.executable, [sys.executable] + sys.argv)
