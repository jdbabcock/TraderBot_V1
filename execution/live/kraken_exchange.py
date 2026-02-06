"""
Live Kraken execution client.

Responsibilities:
- Place/cancel live orders
- Validate against exchange filters
- Fetch trades/open orders
"""
import base64
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import requests


class LiveKrakenClient:
    def __init__(self):
        self.api_key = os.getenv("KRAKEN_API_KEY") or ""
        self.api_secret = os.getenv("KRAKEN_API_SECRET") or ""
        if not self.api_key or not self.api_secret:
            from config import config as cfg
            self.api_key = self.api_key or getattr(cfg, "KRAKEN_API_KEY", "")
            self.api_secret = self.api_secret or getattr(cfg, "KRAKEN_API_SECRET", "")
        self.base_url = "https://api.kraken.com"
        self._asset_pairs = None
        self.last_used_weight = None
        self.last_order_count = None
        self._request_timestamps = []
        self._order_timestamps = []
        self._last_weight_log_ts = 0.0
        self.weight_limit_1m = 120

    # -----------------------------
    # Helpers
    # -----------------------------
    def _nonce(self):
        return str(int(time.time() * 1000))

    def _sign(self, url_path, data):
        postdata = urlencode(data)
        encoded = (data["nonce"] + postdata).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _private(self, path, data):
        url_path = f"/0/private/{path}"
        url = self.base_url + url_path
        data = dict(data)
        data["nonce"] = self._nonce()
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, data),
        }
        self._record_request()
        resp = requests.post(url, headers=headers, data=data, timeout=15)
        return resp.json()

    def _public(self, path, params=None):
        url = f"{self.base_url}/0/public/{path}"
        self._record_request()
        resp = requests.get(url, params=params or {}, timeout=15)
        return resp.json()

    def _load_asset_pairs(self):
        if self._asset_pairs is None:
            res = self._public("AssetPairs")
            self._asset_pairs = res.get("result", {}) if isinstance(res, dict) else {}
        return self._asset_pairs

    def _find_pair_info(self, symbol):
        pair = self._normalize_pair(symbol)
        pairs = self._load_asset_pairs()
        pair_key = pair.replace("/", "")
        if pair_key in pairs:
            return pairs[pair_key]
        if pair in pairs:
            return pairs[pair]
        # Search by wsname / altname
        for key, info in pairs.items():
            wsname = info.get("wsname")
            altname = info.get("altname")
            if wsname == pair or altname == pair_key or altname == pair.replace("/", ""):
                return info
        return None

    def _normalize_pair(self, symbol):
        sym = symbol.replace("/", "")
        if sym.endswith("USD") and len(sym) > 3:
            base = sym[:-3]
            quote = "USD"
            if base == "BTC":
                base = "XBT"
            return f"{base}/{quote}"
        if symbol == "BTC/USD":
            return "XBT/USD"
        return symbol

    def _denormalize_pair(self, symbol):
        if not symbol:
            return symbol
        sym = str(symbol).replace("/", "")
        # Kraken often returns pairs like XETHZUSD, XXBTZUSD, etc.
        if sym == "XBTUSD":
            return "BTC/USD"
        if symbol == "XBT/USD":
            return "BTC/USD"
        quote_map = {
            "ZUSD": "USD",
            "USD": "USD",
            "ZUSDT": "USDT",
            "USDT": "USDT",
            "ZEUR": "EUR",
            "EUR": "EUR",
            "ZGBP": "GBP",
            "GBP": "GBP"
        }
        for suffix, quote in quote_map.items():
            if sym.endswith(suffix):
                base = sym[: -len(suffix)]
                if base in ("XBT", "XXBT"):
                    base = "BTC"
                elif base.startswith(("X", "Z")) and len(base) > 3:
                    base = base[1:]
                return f"{base}/{quote}"
        # Fallback: naive split for non-standard pairs
        if len(sym) > 3:
            return f"{sym[:-3]}/{sym[-3:]}"
        return symbol

    def _record_request(self):
        now = time.time()
        self._request_timestamps.append(now)
        cutoff = now - 60
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.pop(0)
        self.last_used_weight = len(self._request_timestamps)
        if (now - self._last_weight_log_ts) >= 60:
            limit = self.weight_limit_1m or 0
            suffix = f"/{limit}" if limit else ""
            print(f"?? [WEIGHT_UPDATE] Used Weight: {self.last_used_weight}{suffix}")
            self._last_weight_log_ts = now

    def _record_order(self):
        now = time.time()
        self._order_timestamps.append(now)
        cutoff = now - 10
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.pop(0)
        self.last_order_count = len(self._order_timestamps)

    def validate_order(self, symbol, quantity, price=None):
        pair = self._normalize_pair(symbol)
        info = self._find_pair_info(symbol)
        if not info:
            return quantity, price
        lot_decimals = int(info.get("lot_decimals", 8))
        pair_decimals = int(info.get("pair_decimals", 5))
        q = round(float(quantity), lot_decimals)
        p = round(float(price), pair_decimals) if price is not None else None
        return q, p

    def get_min_order_qty(self, symbol):
        info = self._find_pair_info(symbol)
        if not info:
            return None
        try:
            return float(info.get("ordermin")) if info.get("ordermin") is not None else None
        except Exception:
            return None

    # -----------------------------
    # Public
    # -----------------------------
    def get_ohlcv(self, symbol, interval=5, since=None):
        pair = self._normalize_pair(symbol).replace("/", "")
        params = {"pair": pair, "interval": interval}
        if since is not None:
            params["since"] = since
        res = self._public("OHLC", params=params)
        return res

    def get_ticker(self, symbol):
        pair = self._normalize_pair(symbol).replace("/", "")
        return self._public("Ticker", params={"pair": pair})

    # -----------------------------
    # Private
    # -----------------------------
    def get_balance(self):
        return self._private("Balance", {})

    def get_asset_balance(self, asset):
        res = self.get_balance()
        if isinstance(res, dict) and res.get("error"):
            return None
        balances = res.get("result", {}) if isinstance(res, dict) else {}
        return {"asset": asset, "free": balances.get(asset, "0.0"), "locked": "0.0"}

    def create_order(self, symbol, side, type, quantity, price=None, **kwargs):
        pair = self._normalize_pair(symbol).replace("/", "")
        ordertype = type.lower()
        data = {
            "pair": pair,
            "type": side.lower(),
            "ordertype": ordertype,
            "volume": quantity,
        }
        if ordertype == "limit" and price is not None:
            data["price"] = price
        self._record_order()
        res = self._private("AddOrder", data)
        if res.get("error"):
            return {"status": "REJECTED", "info": ";".join(res.get("error", []))}
        return {"status": "SUCCESS", "info": res.get("result")}

    def cancel_order(self, orderId, symbol=None):
        res = self._private("CancelOrder", {"txid": orderId})
        if res.get("error"):
            return {"status": "REJECTED", "info": ";".join(res.get("error", []))}
        return res.get("result", {})

    def get_open_orders(self, symbol=None):
        res = self._private("OpenOrders", {})
        return res.get("result", {}).get("open", {})

    def get_my_trades(self, symbol=None, limit=500):
        res = self._private("TradesHistory", {"trades": True})
        trades = res.get("result", {}).get("trades", {})
        out = []
        for _txid, t in trades.items():
            pair = self._denormalize_pair(t.get("pair", ""))
            if symbol and pair != symbol:
                continue
            out.append({
                "id": _txid,
                "symbol": pair,
                "price": t.get("price"),
                "qty": t.get("vol"),
                "side": "BUY" if t.get("type") == "buy" else "SELL",
                "time": t.get("time"),
            })
            if len(out) >= limit:
                break
        return out
