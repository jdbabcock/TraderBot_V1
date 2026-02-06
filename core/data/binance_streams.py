"""
Binance.US websocket streams.

Responsibilities:
- Maintain shared in-memory price/orderbook stores
- Update timestamps for staleness detection
- Provide a background thread starter
"""
import json
import time
import websocket
import threading
from collections import deque

# Shared price store (imported by main.py)
prices = {}
order_books = {}
price_timestamps = {}

# Keep last N prices for velocity computation
recent_prices = {}

# Debug controls
DEBUG_WS = False
_ws_counts = {}
_ws_last_log = 0.0
_ws_symbol_last = {}

# Number of ticks to track
MAX_TICKS = 20

def ccxt_to_binance_ws(symbol: str, stream: str) -> str:
    # "SOL/USD" -> "solusd@ticker" or "solusd@depth5@100ms"
    return symbol.replace("/", "").lower() + f"@{stream}"

def start_binance_ws(symbols):
    streams = []
    for sym in symbols:
        streams.append(ccxt_to_binance_ws(sym, "ticker"))
        streams.append(ccxt_to_binance_ws(sym, "depth5@100ms"))
    stream_url = "/".join(streams)
    ws_url = f"wss://stream.binance.us:9443/stream?streams={stream_url}"

    print("🔹 Subscribed to Binance ticker streams:", streams)
    print("🔹 WS URL:", ws_url)

    def on_message(ws, message):
        global _ws_counts, _ws_last_log
        try:
            msg = json.loads(message)
            # Combined streams send {"stream": "...", "data": {...}}
            data = msg.get("data", msg)
            stream_name = msg.get("stream", "")
            if DEBUG_WS and stream_name:
                _ws_counts[stream_name] = _ws_counts.get(stream_name, 0) + 1
                now_ts = time.time()
                if (now_ts - _ws_last_log) > 5:
                    total = sum(_ws_counts.values())
                    top = ", ".join(list(_ws_counts.keys())[:3])
                    ages = ", ".join(
                        f"{sym}:{now_ts - ts:.1f}s" for sym, ts in list(_ws_symbol_last.items())[:3]
                    )
                    print(f"[WS_DEBUG] msgs/5s={total} streams={len(_ws_counts)} sample={top} ages={ages}")
                    _ws_counts = {}
                    _ws_last_log = now_ts

            if "s" in data and "c" in data:
                raw_symbol = data["s"]  # e.g. SOLUSD
                price = float(data["c"])

                base = raw_symbol[:-3]        # SOL
                symbol = f"{base}/USD"        # SOL/USD

                prices[symbol] = price
                price_timestamps[symbol] = time.time()
                _ws_symbol_last[symbol] = price_timestamps[symbol]

                # Track recent prices for velocity
                if symbol not in recent_prices:
                    recent_prices[symbol] = deque(maxlen=MAX_TICKS)
                recent_prices[symbol].append(price)
            elif "depth" in stream_name and "s" in data:
                raw_symbol = data["s"]
                base = raw_symbol[:-3]
                symbol = f"{base}/USD"
                bids = data.get("bids") or data.get("b") or []
                asks = data.get("asks") or data.get("a") or []
                best_bid = float(bids[0][0]) if bids else None
                best_ask = float(asks[0][0]) if asks else None
                spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
                # Depth updates indicate the stream is alive; keep timestamps fresh even if bids/asks are empty.
                price_timestamps[symbol] = time.time()
                _ws_symbol_last[symbol] = price_timestamps[symbol]
                if bids or asks:
                    order_books[symbol] = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "bids": bids,
                        "asks": asks
                    }

        except Exception as e:
            print("⚠️ Error parsing WS message:", e)

    def on_error(ws, error):
        print("⚠️ Binance WS error:", error)

    def on_close(ws, *args):
        print("⚠️ Binance WS closed:", args)

    def on_open(ws):
        print("✅ Binance WebSocket connected")

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    ws.run_forever(ping_interval=20, ping_timeout=10)

def start_ws_thread(symbols):
    thread = threading.Thread(
        target=start_binance_ws,
        args=(symbols,),
        daemon=True
    )
    thread.start()

def set_ws_debug(enabled: bool):
    global DEBUG_WS
    DEBUG_WS = bool(enabled)
