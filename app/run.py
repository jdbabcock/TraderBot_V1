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
load_dotenv(ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"🔐 OpenAI key set: {bool(os.getenv('OPENAI_API_KEY'))}")
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
    kraken_key_ok = bool(os.getenv("KRAKEN_API_KEY"))
    kraken_secret_ok = bool(os.getenv("KRAKEN_API_SECRET"))
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
    
    action.update({
        "action": act_name, 
        "side": side, 
        "symbol": symbol.replace("/", ""), 
        "quantity": qty, 
        "type": "MARKET"
    })
    return action, f"{act_name} {side} {symbol} qty={qty:.4f}"

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

def _record_win_loss(realized_pnl, symbol=None):
    global _win_count, _loss_count, _realized_total
    if realized_pnl is None:
        return
    try:
        pnl_val = float(realized_pnl)
    except Exception:
        return
    _realized_total += pnl_val
    if pnl_val > 0:
        _win_count += 1
    elif pnl_val < 0:
        _loss_count += 1
    print(f"[TRADE_CLOSE] {symbol or ''} realized_pnl={pnl_val:.4f} wins={_win_count} losses={_loss_count} total_realized={_realized_total:.4f}")

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
        for sym, pos in positions.items():
            price = prices.get(sym) or prices.get(sym.replace("/", "")) or pos.get("entry_price")
            if price is None:
                continue
            entry = float(pos.get("entry_price", price) or price)
            size = float(pos.get("size", 0.0) or 0.0)
            unrealized = (float(price) - entry) * size
            unrealized_by_symbol[sym] = unrealized
        if per_symbol_notional:
            for sym, notional in per_symbol_notional.items():
                exposure_by_symbol[sym] = (float(notional) / float(equity)) if equity else None
        snapshot["unrealized_by_symbol"] = unrealized_by_symbol
        snapshot["exposure_by_symbol"] = exposure_by_symbol
        snapshot["total_exposure_pct"] = (float(total_notional) / float(equity)) if equity else None
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

def _log_order_action(symbol, action, result, mode=None, outcome=None, realized_pnl=None, reward_score=None, llm_payload=None, last_price=None):
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
                    "outcome", "realized_pnl", "reward_score",
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
_rebalance_log_state = {}
_rebalance_advice_cache = {}
_llm_action_log = []
_llm_action_log_max = 200

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
_autopilot_trade_pnls = []
_autopilot_last_tune_ts = 0.0
_perf_guard_state = "normal"
_perf_guard_last_log = 0.0
_autopilot_last_persist_ts = 0.0

def _initial_account_sync():
    try:
        if RUN_MODE != "live":
            return
        if account_client is None:
            return
        raw = account_client.get_balance()
        summary = _summarize_account(raw, live_prices)
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
        if equity > 0 and daily_start_equity is None:
            daily_start_equity = equity
        if equity > 0 and (peak_equity is None or equity > peak_equity):
            peak_equity = equity
        if RESET_DAILY_RISK_ON_START and not _risk_reset_done and equity > 0:
            daily_start_equity = equity
            peak_equity = equity
            _risk_reset_done = True
            print(f"[RISK_RESET] Baselines reset to equity={equity:.2f}")

        total_notional, per_symbol_notional = _compute_open_notional(trader, prices)
        open_positions = len(getattr(trader, "positions", {}) or {})
        daily_loss_pct = 0.0
        drawdown_pct = 0.0
        if daily_start_equity:
            daily_loss_pct = (equity - daily_start_equity) / daily_start_equity
        if peak_equity:
            drawdown_pct = (equity - peak_equity) / peak_equity

        new_block_reasons = []
        exposure_ratio = (total_notional / equity) if equity > 0 else 0.0
        if DAILY_LOSS_LIMIT_PCT > 0 and daily_loss_pct <= -DAILY_LOSS_LIMIT_PCT:
            new_block_reasons.append("DAILY_LOSS_LIMIT")
        if MAX_DRAWDOWN_PCT > 0 and drawdown_pct <= -MAX_DRAWDOWN_PCT:
            new_block_reasons.append("MAX_DRAWDOWN")
        if MAX_OPEN_POSITIONS > 0 and open_positions >= MAX_OPEN_POSITIONS:
            new_block_reasons.append("MAX_OPEN_POSITIONS")
        if MAX_TOTAL_EXPOSURE_PCT > 0 and equity > 0:
            if exposure_ratio >= MAX_TOTAL_EXPOSURE_PCT:
                new_block_reasons.append("MAX_TOTAL_EXPOSURE")
            elif "MAX_TOTAL_EXPOSURE" in block_reasons and exposure_ratio > MIN_EXPOSURE_RESUME_PCT:
                new_block_reasons.append("MAX_TOTAL_EXPOSURE")

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

        effective_min_conf = MIN_CONFIDENCE_TO_ORDER
        effective_size_fraction = SIZE_FRACTION_DEFAULT
        effective_cooldown = COOLDOWN_SECONDS
        effective_symbol_cooldown = SYMBOL_COOLDOWN_SECONDS
        effective_stop_loss = STOP_LOSS_PCT_DEFAULT
        effective_take_profit = TAKE_PROFIT_PCT_DEFAULT
        effective_trailing = TRAILING_STOP_PCT_DEFAULT
        if underperforming:
            effective_min_conf = min(0.9, MIN_CONFIDENCE_TO_ORDER + 0.15)
            effective_size_fraction = max(0.02, SIZE_FRACTION_DEFAULT * 0.6)
            effective_cooldown = max(COOLDOWN_SECONDS, 60) + 30
            effective_symbol_cooldown = max(SYMBOL_COOLDOWN_SECONDS, 60)
            effective_stop_loss = max(0.005, STOP_LOSS_PCT_DEFAULT * 0.8)
            effective_take_profit = max(0.01, TAKE_PROFIT_PCT_DEFAULT * 0.8)
            effective_trailing = max(0.005, TRAILING_STOP_PCT_DEFAULT * 0.8)

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
                        trading_ready_now = (acc_total - acc_cash) >= min_delta
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
            candle_patterns = None
            if not df.empty:
                recent_ohlcv = (
                    df[["open", "high", "low", "close", "volume"]]
                    .tail(30)
                    .to_dict(orient="records")
                )
                candle_patterns = _detect_candle_patterns(df)
                rsi = ta.momentum.RSIIndicator(df["close"]).rsi().iloc[-1]
                ema20 = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator().iloc[-1]
                atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range().iloc[-1]
                vwap = ta.volume.VolumeWeightedAveragePrice(
                    df["high"], df["low"], df["close"], df["volume"]
                ).volume_weighted_average_price().iloc[-1]
                vol_ratio = None
                try:
                    avg_vol = df["volume"].rolling(window=20).mean().iloc[-1]
                    current_vol = df["volume"].iloc[-1]
                    if avg_vol and avg_vol > 0:
                        vol_ratio = float(current_vol) / float(avg_vol)
                except Exception:
                    vol_ratio = None
                indicators[symbol] = {
                    "rsi": rsi,
                    "ema": ema20,
                    "ema20": ema20,
                    "vwap": vwap,
                    "vol_ratio": vol_ratio,
                    "atr_pct": (atr/last_price),
                    "last_update": formatted_time
                }

            if not _trading_ready:
                continue

            # C. LLM Decision (Throttled)
            if USE_LLM and (now - last_llm_call[symbol] >= LLM_CHECK_INTERVAL):
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
                perf_summary = {
                    "wins": _win_count,
                    "losses": _loss_count,
                    "realized_total": _realized_total,
                    "win_rate": (win_rate if trade_count else 0.0),
                    "underperforming": underperforming
                }
                llm_result = llm_decision(
                    symbol, last_price, sentiment_score, 0.0, trader.positions,
                    rsi=indicators[symbol].get("rsi", 50),
                    ema_20=indicators[symbol].get("ema", last_price),
                    recent_ohlcv=recent_ohlcv,
                    candle_patterns=candle_patterns,
                    perf_summary=perf_summary,
                    execution_context={
                        "recent_trades": trades_by_symbol.get(symbol, [])[-5:],
                        "portfolio": portfolio_context
                    } if RUN_MODE == "live" else None
                )
                if isinstance(llm_result, dict):
                    if not llm_result.get("stop_loss_pct"):
                        llm_result["stop_loss_pct"] = effective_stop_loss
                    if not llm_result.get("take_profit_pct"):
                        llm_result["take_profit_pct"] = effective_take_profit
                    if not llm_result.get("trailing_stop_pct"):
                        llm_result["trailing_stop_pct"] = effective_trailing
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

            # Portfolio-aware PnL exits: trim losers when drawdown exceeds threshold
            if EXECUTION_MODE in ("live", "sim") and drawdown_pct <= -PNL_EXIT_MAX_DRAWDOWN_PCT:
                pos = trader.positions.get(symbol) if isinstance(trader.positions, dict) else None
                if pos:
                    entry = float(pos.get("entry_price", last_price) or last_price)
                    size = float(pos.get("size", 0.0) or 0.0)
                    if entry and size:
                        pnl_pct = (float(last_price) - entry) / entry
                        if pnl_pct <= PNL_EXIT_LOSER_THRESHOLD_PCT:
                            action = {
                                "action": "PLACE_ORDER",
                                "side": "SELL",
                                "symbol": symbol.replace("/", ""),
                                "quantity": size,
                                "type": "MARKET"
                            }
                            if DEBUG_STATUS or DEBUG_LOG_ATTEMPTS:
                                print(f"[PNL_EXIT] Selling loser {symbol} pnl_pct={pnl_pct:.3f} drawdown={drawdown_pct:.3f}")
                            try:
                                v_qty, _ = live_execution_client.validate_order(symbol, size)
                                action["quantity"] = v_qty
                                res = execute_llm_action(execution_client, action)
                                _record_attempt(symbol, "SELL", res.get("status", "SUCCESS"), "PNL_EXIT", qty=v_qty, price=last_price)
                                _log_order_action(symbol, action, res, mode="pnl_exit", outcome="PNL_EXIT", llm_payload=current_llm, last_price=last_price)
                                _log_portfolio_snapshot(
                                    _portfolio_snapshot(trader, prices, account_info, total_notional, per_symbol_notional, equity),
                                    tag="pnl_exit",
                                    symbol=symbol
                                )
                                continue
                            except Exception as e:
                                if DEBUG_STATUS:
                                    print(f"[PNL_EXIT] Failed {symbol}: {e}")
            
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
                    _record_attempt(symbol, None, "BLOCKED", "REJECT_BACKOFF")
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
                    if MAX_SYMBOL_EXPOSURE_PCT > 0 and equity > 0:
                        sym_key = _to_slash_symbol(symbol)
                        current_sym_notional = per_symbol_notional.get(sym_key, 0.0)
                        if current_sym_notional > 0 and (current_sym_notional / equity) >= MAX_SYMBOL_EXPOSURE_PCT:
                            last_exposure_log = _symbol_exposure_log_state.get(sym_key, 0.0)
                            if (now - last_exposure_log) >= SYMBOL_EXPOSURE_LOG_INTERVAL_SECONDS:
                                _record_attempt(symbol, action.get("side"), "BLOCKED", "MAX_SYMBOL_EXPOSURE")
                                _symbol_exposure_log_state[sym_key] = now
                            continue
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
                        if realized_pnl is not None:
                            _record_win_loss(realized_pnl, symbol=log_symbol)
                        reward_score = realized_pnl if realized_pnl is not None else 0.0
                        _log_order_action(
                            log_symbol,
                            action,
                            res,
                            mode=EXECUTION_MODE,
                            outcome=outcome,
                            realized_pnl=realized_pnl,
                            reward_score=reward_score,
                            llm_payload=current_llm,
                            last_price=last_price
                        )
                        _log_portfolio_snapshot(portfolio_context, tag="trade_execute", symbol=log_symbol)
                        _autopilot_on_realized_pnl(realized_pnl, symbol=log_symbol)
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
                                _autopilot_on_realized_pnl(realized_pnl, symbol=log_symbol)
                                _record_win_loss(realized_pnl, symbol=log_symbol)
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
                print(f"🛑 [AUTO-EXIT] {s} hit Stop/Target.")
                if EXECUTION_MODE == "sim":
                    pos = trader.positions.get(s, {}) if hasattr(trader, "positions") else {}
                    qty = float(pos.get("size", 0.0) or 0.0)
                    realized_pnl = _estimate_realized_pnl(s, qty, prices.get(s), trader)
                    outcome = None
                    if realized_pnl is not None:
                        outcome = "W" if realized_pnl > 0 else ("L" if realized_pnl < 0 else "B")
                        _record_win_loss(realized_pnl, symbol=s)
                    _log_order_action(
                        s,
                        {"action": "AUTO_EXIT", "side": "SELL", "symbol": s.replace("/", ""), "type": "MARKET"},
                        {"status": "AUTO_EXIT"},
                        mode=EXECUTION_MODE,
                        outcome=outcome or reason,
                        realized_pnl=realized_pnl,
                        reward_score=None,
                        llm_payload=llm_outputs.get(s, {}),
                        last_price=prices.get(s)
                    )
                    continue
                try:
                    pos = trader.positions.get(s, {})
                    qty = float(pos.get("size", 0.0))
                    if qty > 0:
                        v_qty, _ = live_execution_client.validate_order(s, qty)
                        live_execution_client.create_order(symbol=s, side="SELL", type="MARKET", quantity=v_qty)
                        realized_pnl = _estimate_realized_pnl(s, v_qty, prices.get(s), trader)
                        outcome = None
                        if realized_pnl is not None:
                            outcome = "W" if realized_pnl > 0 else ("L" if realized_pnl < 0 else "B")
                            _record_win_loss(realized_pnl, symbol=s)
                        _log_order_action(
                            s,
                            {"action": "AUTO_EXIT", "side": "SELL", "symbol": s.replace("/", ""), "type": "MARKET"},
                            {"status": "AUTO_EXIT"},
                            mode=EXECUTION_MODE,
                            outcome=outcome or reason,
                            realized_pnl=realized_pnl,
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

        prompt_text = list(_llm_action_log)
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
