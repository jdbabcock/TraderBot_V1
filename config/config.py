"""
Central control room for the bot.

Most users should only edit values in this file. The main application reads
these settings on startup and applies the selected trading style presets.
"""

# Execution mode
# - "live": place real orders on exchange
# - "sim": paper trade with a persistent mock wallet
RUN_MODE = "live"

# Exchange selection (live mode only)
# - "binance": Binance.US
# - "kraken": Kraken spot
EXCHANGE = "kraken"

# Trading style presets (see STYLE_PRESETS below)
# - conservative: lower risk, fewer trades
# - balanced: moderate risk/reward
# - aggressive: higher risk, more frequent trades
# - autopilot: LLM may adapt settings over time (except llm_check_interval)
TRADING_STYLE = "autopilot"

# Core config
# - CAPITAL: starting balance for sim mode only (mock wallet seed on first run)
# - USE_LLM: enable/disable LLM decisions
# - MIN_NOTIONAL: exchange minimum order size (USD)
# - QTY_STEP: default step size; live client refines with exchange filters
CAPITAL = 10000.0
USE_LLM = True
MIN_NOTIONAL = 10.0
KRAKEN_COST_MIN_USD = 0.5
ALLOW_MIN_UPSIZE = True
QTY_STEP = 0.0001
ENABLE_REBALANCE = True
REBALANCE_SELL_FRACTION = 0.25
REBALANCE_MIN_SCORE_DELTA = 0.25
REBALANCE_MIN_HOLD_SECONDS = 600
REBALANCE_COOLDOWN_SECONDS = 600
REBALANCE_PREFER_LOSERS = True
REBALANCE_ADVISORY_MODE = True
TARGET_ALLOCATION = {'BTC': [0.4, 0.6], 'ETH': [0.2, 0.4], 'SOL': [0.1, 0.2]}
PNL_EXIT_MAX_DRAWDOWN_PCT = 0.08
PNL_EXIT_LOSER_THRESHOLD_PCT = -0.05


# Sim wallet persistence
# - False: keep balances/positions between runs (recommended)
# - True: wipe mock wallet on startup and reseed with CAPITAL
RESET_SIM_WALLET_ON_START = False

# Refresh and timing
# - ACCOUNT_INFO_REFRESH_SECONDS: live account balance poll interval
# - LIVE_TRADES_REFRESH_SECONDS: live trade history poll interval (positions built from trades)
# - UI_REFRESH_SECONDS: UI update cadence
ACCOUNT_INFO_REFRESH_SECONDS = 60
LIVE_TRADES_REFRESH_SECONDS = 15
UI_REFRESH_SECONDS = 1
DEBUG_STATUS = True
DEBUG_LOG_ATTEMPTS = True
RESET_DAILY_RISK_ON_START = True
PERF_GUARD_PERSIST = True
PERF_GUARD_MODE = "rolling"  # "rolling" or "session"
PERF_GUARD_LOOKBACK_TRADES = 200

# Stale price detection (warn when the websocket feed pauses)
# - STALE_PRICE_SECONDS: age threshold for a price to be considered stale
# - STALE_WARN_INTERVAL_SECONDS: minimum time between warnings per symbol
# - STALE_GRACE_SECONDS: startup grace period before warnings begin
STALE_PRICE_SECONDS = 15
STALE_WARN_INTERVAL_SECONDS = 60
STALE_GRACE_SECONDS = 20
ORDER_RETRY_SECONDS = 5

# Execution guardrails
# - BLOCK_ON_STALE_PRICE: block new orders if prices are stale
# - REJECT_BACKOFF_SECONDS: per-symbol cooloff after rejected order
# - MAX_API_WEIGHT_1M: pause new orders when Binance API weight is too high
# - MAX_API_WEIGHT_1M_KRAKEN: pause new orders when Kraken requests/1m are too high
# - MAX_ORDER_COUNT_10S: pause new orders when order count is too high
BLOCK_ON_STALE_PRICE = True
REJECT_BACKOFF_SECONDS = 60
MAX_API_WEIGHT_1M = 1000
MAX_API_WEIGHT_1M_KRAKEN = 120
MAX_ORDER_COUNT_10S = 8
ATTEMPT_LOG_COOLDOWN_SECONDS = 20
ATTEMPT_LOG_DEDUP_BY_REASON = True

# Fee + slippage estimates (used for net PnL and win/loss)
# Kraken tier 1: Maker 0.25%, Taker 0.40%
KRAKEN_MAKER_FEE_PCT = 0.0025
KRAKEN_TAKER_FEE_PCT = 0.004
ESTIMATED_SLIPPAGE_PCT = 0.0075


# Symbols and strategy
# - SYMBOLS: list of markets (exchange pairs)
# - TIMEFRAME: OHLCV timeframe for indicators
SYMBOLS = ['SOL/USD', 'BTC/USD', 'ETH/USD']
TIMEFRAME = "5m"

# LLM cadence (style presets can override this)
LLM_CHECK_INTERVAL = 60

# Sentiment contrarian settings
# - CONTRARIAN_SENTIMENT_ENABLED: apply contrarian adjustment at extremes
# - CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD: abs(sentiment) level to start contrarian weighting
# - CONTRARIAN_SENTIMENT_MAX_WEIGHT: max inversion weight applied at |sentiment|=1.0
# - CONTRARIAN_SENTIMENT_LOG: log raw vs adjusted sentiment
CONTRARIAN_SENTIMENT_ENABLED = True
CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD = 0.7
CONTRARIAN_SENTIMENT_MAX_WEIGHT = 0.6
CONTRARIAN_SENTIMENT_LOG = False

# Risk limits (percent-based, halt new trades only)
# - DAILY_LOSS_LIMIT_PCT: max loss from day start equity before pausing new trades
# - MAX_DRAWDOWN_PCT: max drawdown from peak equity before pausing new trades
# - MAX_TOTAL_EXPOSURE_PCT: cap total open notional vs. equity
# - MAX_SYMBOL_EXPOSURE_PCT: cap per-symbol open notional vs. equity
# - MAX_OPEN_POSITIONS: cap number of simultaneous open positions
DAILY_LOSS_LIMIT_PCT = 0.02
MAX_DRAWDOWN_PCT = 0.05
MAX_TOTAL_EXPOSURE_PCT = 1.0
MIN_EXPOSURE_RESUME_PCT = 0.2
MAX_SYMBOL_EXPOSURE_PCT = 0.2
MAX_OPEN_POSITIONS = 3

# Style presets (autopilot only)
# Notes:
# - min_confidence_to_order: minimum LLM confidence to place a trade
# - size_fraction: fraction of capital per trade (risk sizing)
# - llm_check_interval: seconds between LLM calls per symbol
# - stop_loss_pct_default / take_profit_pct_default / trailing_stop_pct_default:
#   default risk values if the LLM omits them
# - cooldown_seconds: minimum time between trades
# - symbol_cooldown_seconds: minimum time between trades per symbol
# - max_trades_per_day: hard cap on daily trades
# BEGIN STYLE_PRESETS
STYLE_PRESETS = {
    "autopilot": {
        "min_confidence_to_order": 0.10,
        "size_fraction": 0.1,
        "llm_check_interval": 120,
        "stop_loss_pct_default": 0.05,
        "take_profit_pct_default": 0.05,
        "trailing_stop_pct_default": 0.03,
        "cooldown_seconds": 0,
        "symbol_cooldown_seconds": 0,
        "max_trades_per_day": 999,
        "daily_loss_limit_pct": 0.02,
        "max_drawdown_pct": 0.5,
        "max_total_exposure_pct": 1.0,
        "min_exposure_resume_pct": 0.4,
        "max_symbol_exposure_pct": 0.5,
        "max_open_positions": 6
    }
}
# END STYLE_PRESETS

# Optional UI descriptions (used by the startup/settings screen)
CONFIG_DOC = {
    "RUN_MODE": "Execution mode: live (real trades) or sim (paper trades).",
    "EXCHANGE": "Exchange selection for live mode (binance or kraken).",
    "TRADING_STYLE": "Preset for trading aggressiveness.",
    "CAPITAL": "Starting balance for sim mode only.",
    "USE_LLM": "Enable/disable LLM decision making.",
    "MIN_NOTIONAL": "Exchange minimum order size (USD).",
    "KRAKEN_COST_MIN_USD": "Kraken cost minimum for USD-quoted pairs (USD).",
    "ALLOW_MIN_UPSIZE": "If true, auto-upsize Kraken orders to meet minimums when cash allows.",
    "ENABLE_REBALANCE": "If true, allow the bot to sell other positions to rebalance into stronger signals.",
    "REBALANCE_SELL_FRACTION": "Max fraction of a position to sell during a rebalance step.",
    "REBALANCE_MIN_SCORE_DELTA": "Min score edge required to rebalance into target symbol.",
    "REBALANCE_MIN_HOLD_SECONDS": "Minimum hold time before a position can be rebalanced (seconds).",
    "REBALANCE_COOLDOWN_SECONDS": "Minimum seconds between rebalances per symbol.",
    "REBALANCE_PREFER_LOSERS": "If true, rebalance out of losers before winners.",
    "REBALANCE_ADVISORY_MODE": "If true, request LLM advisory approval before rebalancing.",
    "TARGET_ALLOCATION": "Target allocation bands by asset, e.g. {'BTC': (0.4, 0.6)}.",
    "PNL_EXIT_MAX_DRAWDOWN_PCT": "Portfolio drawdown threshold to prioritize trimming losers.",
    "PNL_EXIT_LOSER_THRESHOLD_PCT": "Per-position loss threshold to consider trimming losers.",
    "QTY_STEP": "Default quantity step; live client refines with exchange filters.",
    "RESET_SIM_WALLET_ON_START": "If true, reset mock wallet on startup.",
    "ACCOUNT_INFO_REFRESH_SECONDS": "Live account balance poll interval.",
    "LIVE_TRADES_REFRESH_SECONDS": "Live trade history poll interval.",
    "UI_REFRESH_SECONDS": "UI update cadence.",
    "DEBUG_STATUS": "If true, print periodic status diagnostics to the console.",
    "DEBUG_LOG_ATTEMPTS": "If true, log every trade attempt (no dedup/cooldown).",
    "RESET_DAILY_RISK_ON_START": "If true, reset daily loss/drawdown baselines on startup.",
    "PERF_GUARD_PERSIST": "If true, load recent trade performance from logs on startup.",
    "PERF_GUARD_MODE": "Performance guard mode: 'rolling' to use recent history, 'session' to reset each run.",
    "PERF_GUARD_LOOKBACK_TRADES": "Number of most recent trades to load for performance guard.",
    "STALE_PRICE_SECONDS": "Seconds before a price is considered stale.",
    "STALE_WARN_INTERVAL_SECONDS": "Minimum seconds between stale warnings per symbol.",
    "STALE_GRACE_SECONDS": "Startup grace period before stale warnings start.",
    "ORDER_RETRY_SECONDS": "Seconds to wait before retrying a failed order action.",
    "BLOCK_ON_STALE_PRICE": "If true, block new orders when price data is stale.",
    "REJECT_BACKOFF_SECONDS": "Seconds to pause a symbol after a rejected order.",
    "MAX_API_WEIGHT_1M": "Pause new orders when Binance API weight is above this.",
    "MAX_API_WEIGHT_1M_KRAKEN": "Pause new orders when Kraken requests/1m exceed this.",
    "MAX_ORDER_COUNT_10S": "Pause new orders when Binance order count exceeds this.",
    "ATTEMPT_LOG_COOLDOWN_SECONDS": "Minimum seconds between repeated attempt logs per symbol.",
    "ATTEMPT_LOG_DEDUP_BY_REASON": "If true, only log attempt changes when reason/status changes.",
    "SYMBOLS": "List of market symbols to trade.",
    "TIMEFRAME": "OHLCV timeframe used for indicators.",
    "LLM_CHECK_INTERVAL": "Seconds between LLM calls per symbol.",
    "CONTRARIAN_SENTIMENT_ENABLED": "If true, apply contrarian weighting to extreme sentiment scores.",
    "CONTRARIAN_SENTIMENT_EXTREME_THRESHOLD": "Absolute sentiment level where contrarian weighting begins (0-1).",
    "CONTRARIAN_SENTIMENT_MAX_WEIGHT": "Max inversion weight at |sentiment|=1.0 (0-1).",
    "CONTRARIAN_SENTIMENT_LOG": "If true, log raw vs adjusted sentiment values.",
    "DAILY_LOSS_LIMIT_PCT": "Max daily loss (fraction of day-start equity) before pausing new trades.",
    "MAX_DRAWDOWN_PCT": "Max drawdown (fraction from equity peak) before pausing new trades.",
    "MAX_TOTAL_EXPOSURE_PCT": "Max total open notional as a fraction of equity.",
    "MIN_EXPOSURE_RESUME_PCT": "Resume new trades once total exposure drops to this fraction.",
    "MAX_SYMBOL_EXPOSURE_PCT": "Max per-symbol open notional as a fraction of equity.",
    "MAX_OPEN_POSITIONS": "Max number of simultaneous open positions.",
    "STYLE_PRESETS": "Dictionary of trading style presets."
}
