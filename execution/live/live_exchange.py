"""
Live Binance.US execution client.

Responsibilities:
- Place/cancel live orders
- Validate against exchange filters
"""
import math
import os
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException

class LiveBinanceClient:
    """
    Live adapter that mimics SimBinanceClient's interface.
    Integrates precision filtering and error handling for live execution.
    """
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")

        # Use tld='us' for Binance.US, remove for Binance.com
        self.client = Client(api_key, api_secret, tld='us')

        # Fetch exchange info once to cache decimal/lot filters
        print("Fetching exchange filters...")
        self.info = self.client.get_exchange_info()
        self.filters = {s['symbol']: s['filters'] for s in self.info['symbols']}

        # Internal state tracking
        self._trades = []
        self.last_used_weight = None
        self.last_order_count = None
        self.log_api_health(getattr(self.client, "response", None))

    def validate_order(self, symbol, quantity, price=None):
        """Adjusts quantity and price to meet Binance's stepSize and tickSize."""
        clean_symbol = symbol.replace("/", "")
        sym_filters = self.filters.get(clean_symbol)

        if not sym_filters:
            return quantity, price

        lot_filter = next(f for f in sym_filters if f['filterType'] == 'LOT_SIZE')
        price_filter = next(f for f in sym_filters if f['filterType'] == 'PRICE_FILTER')

        step_size = float(lot_filter['stepSize'])
        tick_size = float(price_filter['tickSize'])

        q_precision = int(round(-math.log(step_size, 10), 0))
        valid_qty = math.floor(float(quantity) * (10 ** q_precision)) / (10 ** q_precision)

        valid_price = None
        if price:
            p_precision = int(round(-math.log(tick_size, 10), 0))
            valid_price = round(float(price), p_precision)

        return valid_qty, valid_price

    def get_asset_balance(self, asset):
        """Fetches real-time balance for a specific asset (e.g., 'USDT')."""
        try:
            res = self.client.get_asset_balance(asset=asset)
            self.log_api_health(getattr(self.client, "response", None))
            if res:
                return {
                    "asset": asset,
                    "free": res['free'],
                    "locked": res['locked']
                }
            return {"asset": asset, "free": "0.0", "locked": "0.0"}
        except Exception as e:
            print(f"Error fetching balance for {asset}: {e}")
            return None

    def create_order(self, symbol, side, type, quantity, price=None, **kwargs):
        """Executes a live order on Binance."""
        clean_symbol = symbol.replace("/", "")

        v_qty, v_price = self.validate_order(clean_symbol, quantity, price)

        try:
            params = {
                "symbol": clean_symbol,
                "side": side.upper(),
                "type": type.upper(),
                "quantity": v_qty
            }
            if type.upper() == "LIMIT" and v_price:
                params["price"] = v_price
                params["timeInForce"] = "GTC"

            order = self.client.create_order(**params)
            self.log_api_health(getattr(self.client, "response", None))

            formatted_order = {
                "orderId": order.get("orderId"),
                "status": order.get("status"),
                "symbol": symbol,
                "executedQty": order.get("executedQty", "0.0"),
                "price": order.get("price", str(v_price))
            }
            self._trades.append(formatted_order)
            return formatted_order

        except BinanceAPIException as e:
            self._log_rate_limit_error(e)
            print(f"Binance API Error: {e.message}")
            return {"status": "REJECTED", "info": e.message}
        except Exception as e:
            print(f"Connection Error: {e}")
            return {"status": "ERROR", "info": str(e)}

    def get_open_orders(self, symbol=None):
        clean_symbol = symbol.replace("/", "") if symbol else None
        res = self.client.get_open_orders(symbol=clean_symbol)
        self.log_api_health(getattr(self.client, "response", None))
        return res

    def get_my_trades(self, symbol=None, limit=500):
        clean_symbol = symbol.replace("/", "") if symbol else None
        if clean_symbol:
            res = self.client.get_my_trades(symbol=clean_symbol, limit=limit)
        else:
            res = self.client.get_my_trades(limit=limit)
        self.log_api_health(getattr(self.client, "response", None))
        return res

    def cancel_order(self, symbol, orderId):
        clean_symbol = symbol.replace("/", "")
        res = self.client.cancel_order(symbol=clean_symbol, orderId=orderId)
        self.log_api_health(getattr(self.client, "response", None))
        return res

    def log_api_health(self, response):
        """
        Extracts and logs the current API weight usage.
        """
        headers = getattr(response, "headers", None) if response is not None else None
        if not headers:
            return
        used_weight = headers.get('x-mbx-used-weight-1m')
        order_count = headers.get('x-mbx-order-count-10s')

        if used_weight:
            weight_int = int(used_weight)
            self.last_used_weight = weight_int
            color = "??" if weight_int < 600 else "??" if weight_int < 900 else "??"
            print(f"{color} [WEIGHT_UPDATE] Used Weight: {weight_int}/1200")
            if weight_int > 1000:
                print("?? [WEIGHT_UPDATE] WARNING: API Weight critical. Sleeping for 10 seconds...")
                time.sleep(10)

        if order_count:
            self.last_order_count = int(order_count)
            print(f"?? [ORDER_COUNT] {self.last_order_count}/10s")

    def _log_rate_limit_error(self, error):
        try:
            if getattr(error, "status_code", None) == 429 or "429" in str(error):
                print("?? [429_ERROR] Rate Limit Exceeded. Bot should halt to avoid ban.")
        except Exception:
            pass
