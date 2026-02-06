"""
Kraken websocket streams.
Keeps live_prices and price_timestamps in the same shape as Binance.
"""
import json
import threading
import time
import websocket

prices = {}
price_timestamps = {}
_debug_payload_logged = False
_debug_symbols_logged = set()
_expected_symbols = set()
_ws_ref = {"ws": None}


def _normalize_symbol(sym):
    if sym == "XBT/USD":
        return "BTC/USD"
    return sym


def _denormalize_symbol(sym):
    if sym == "BTC/USD":
        return "XBT/USD"
    return sym


def _subscription_candidates(symbol):
    sym = symbol
    if sym in ("BTC/USD", "XBT/USD"):
        return ["BTC/USD"]
    return [sym]


def start_kraken_ws(symbols):
    kraken_symbols = []
    for s in symbols:
        for cand in _subscription_candidates(s):
            if cand not in kraken_symbols:
                kraken_symbols.append(cand)
    global _expected_symbols
    _expected_symbols = set(_normalize_symbol(s) for s in symbols)
    ws_url = "wss://ws.kraken.com/v2"
    print("🔹 Subscribed to Kraken ticker streams:", kraken_symbols)

    def _extract_price(last_field):
        if last_field is None:
            return None
        if isinstance(last_field, (list, tuple)) and last_field:
            return last_field[0]
        if isinstance(last_field, dict):
            return last_field.get("price") or last_field.get("p") or last_field.get("c")
        return last_field

    def on_open(ws):
        subscribe = {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": kraken_symbols
            }
        }
        _ws_ref["ws"] = ws
        ws.send(json.dumps(subscribe))

    def on_message(ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if isinstance(data, dict):
            if data.get("method") == "subscribe":
                if not data.get("success", False):
                    print(f"⚠️ Kraken subscribe error: {data}")
                return
            if data.get("channel") == "ticker":
                global _debug_payload_logged
                if not _debug_payload_logged:
                    print(f"[DEBUG] Kraken ticker payload: {data}")
                    _debug_payload_logged = True
                for item in data.get("data", []):
                    sym = _normalize_symbol(item.get("symbol", ""))
                    price = _extract_price(item.get("last"))
                    if price is None:
                        continue
                    try:
                        prices[sym] = float(price)
                        price_timestamps[sym] = time.time()
                        if sym not in _debug_symbols_logged:
                            print(f"[DEBUG] Kraken ticker update: {sym} last={price}")
                            _debug_symbols_logged.add(sym)
                    except Exception:
                        pass

    def on_error(ws, error):
        print("⚠️ Kraken WS error:", error)

    def on_close(ws, *args):
        print("⚠️ Kraken WS closed:", args)

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)


def start_ws_thread(symbols):
    thread = threading.Thread(target=start_kraken_ws, args=(symbols,), daemon=True)
    thread.start()
    def _missing_watchdog():
        time.sleep(10)
        missing = sorted(_expected_symbols - set(prices.keys()))
        if missing:
            print(f"⚠️ Kraken ticker missing symbols after 10s: {missing}")
            # Attempt a resubscribe with BTC aliases if BTC is missing
            if "BTC/USD" in missing:
                try:
                    ws = _ws_ref.get("ws")
                    if ws is not None and getattr(ws, "sock", None) and ws.sock.connected:
                        resub = {
                            "method": "subscribe",
                            "params": {
                                "channel": "ticker",
                                "symbol": ["BTC/USD"]
                            }
                        }
                        ws.send(json.dumps(resub))
                        print("🔁 Kraken resubscribe sent for BTC aliases")
                except Exception as e:
                    print(f"⚠️ Kraken resubscribe failed: {e}")
    threading.Thread(target=_missing_watchdog, daemon=True).start()
    return thread
