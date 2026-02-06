"""
Simulated Binance exchange client.

Responsibilities:
- Validate order sizes against min notional/step rules
- Simulate order placement with fills
"""
import time
import uuid


class SimBinanceClient:
    """
    Simulation-only client that mimics a tiny subset of python-binance.
    It never places real orders; it updates the MockWallet instead.
    """
    def __init__(self, wallet, prices, order_books=None, fee_bps=0.0):
        self.wallet = wallet
        self.prices = prices
        self.order_books = order_books or {}
        self.fee_bps = fee_bps
        self._orders = []
        self._trades = []
        self._order_lists = []
        self._pending_fills = {}
        self._fills = []

    def _record_fill(self, symbol, side, qty, price, order_id):
        self._fills.append({
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "price": float(price),
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        })

    def consume_fills(self):
        fills = list(self._fills)
        self._fills.clear()
        return fills

    def _simulate_fill_from_book(self, side, quantity, book, fallback_price, limit_price=None):
        qty_remaining = float(quantity)
        if qty_remaining <= 0:
            return 0.0, None
        levels = []
        if side.upper() == "BUY":
            levels = book.get("asks", []) if book else []
        else:
            levels = book.get("bids", []) if book else []

        filled_qty = 0.0
        notional = 0.0

        for price, qty in levels:
            px = float(price)
            if limit_price is not None:
                if side.upper() == "BUY" and px > limit_price:
                    break
                if side.upper() == "SELL" and px < limit_price:
                    break
            lvl_qty = float(qty)
            if lvl_qty <= 0:
                continue
            take = min(qty_remaining, lvl_qty)
            notional += take * px
            filled_qty += take
            qty_remaining -= take
            if qty_remaining <= 0:
                break

        if filled_qty > 0:
            avg_price = notional / filled_qty
            return filled_qty, avg_price

        # Fallback to last price if no book depth
        if fallback_price is not None:
            return float(quantity), float(fallback_price)
        return 0.0, None

    def _parse_symbol(self, symbol):
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
        else:
            # Assume USD quote for simple symbols like BTCUSD
            base = symbol[:-3]
            quote = symbol[-3:]
        return base, quote

    def get_symbol_ticker(self, symbol):
        sym = symbol.replace("", "")
        if "/" not in sym:
            sym = f"{sym[:-3]}/{sym[-3:]}"
        price = self.prices.get(sym)
        return {"symbol": symbol, "price": str(price) if price is not None else None}

    def get_account(self):
        balances = []
        for asset, amt in self.wallet.all_balances().items():
            balances.append({"asset": asset, "free": str(amt), "locked": "0"})
        return {"balances": balances}

    def get_asset_balance(self, asset):
        amt = self.wallet.get_balance(asset)
        return {"asset": asset, "free": str(amt), "locked": "0"}

    def create_order(self, symbol, side, type, quantity, price=None, timeInForce="GTC", signature=None, order_id=None, timestamp=None, apiKey=None):
        base, quote = self._parse_symbol(symbol)
        side = side.upper()
        type = type.upper()

        # Use live price for MARKET, price param for LIMIT
        book = self.order_books.get(f"{base}/{quote}", {})
        best_bid = book.get("best_bid")
        best_ask = book.get("best_ask")
        market_px = self.prices.get(f"{base}/{quote}")
        if type == "MARKET":
            px = best_ask if side == "BUY" and best_ask is not None else (
                best_bid if side == "SELL" and best_bid is not None else market_px
            )
            if px is None:
                raise ValueError(f"No live price for {base}/{quote}")
        else:
            if price is None:
                raise ValueError("LIMIT order requires price")
            px = float(price)

        qty = float(quantity)
        notional = qty * px
        fee = notional * (self.fee_bps / 10000.0)

        order_id = order_id or str(uuid.uuid4())
        status = "FILLED"

        if type == "LIMIT" and market_px is not None:
            if side == "BUY" and px < market_px:
                status = "NEW"
            if side == "SELL" and px > market_px:
                status = "NEW"

        executed_qty = qty
        executed_notional = notional
        executed_price = px
        if status == "FILLED":
            # Simulate fill from order book for market or crossing limit orders
            limit_px = px if type == "LIMIT" else None
            fill_qty, avg_price = self._simulate_fill_from_book(side, qty, book, px, limit_price=limit_px)
            if avg_price is None or fill_qty <= 0:
                status = "REJECTED"
            else:
                executed_qty = fill_qty
                executed_price = avg_price
                executed_notional = executed_qty * executed_price
                if executed_qty < qty:
                    status = "PARTIALLY_FILLED"
                    # Track remaining quantity for time-sliced fills
                    self._pending_fills[order_id] = {
                        "symbol": symbol,
                        "side": side,
                        "type": type,
                        "qty_remaining": qty - executed_qty,
                        "limit_price": limit_px
                    }
            if side == "BUY":
                if self.wallet.get_balance(quote) < (executed_notional + fee):
                    status = "REJECTED"
                else:
                    self.wallet.adjust_balance(quote, -(executed_notional + fee))
                    self.wallet.adjust_balance(base, executed_qty)
            elif side == "SELL":
                if self.wallet.get_balance(base) < executed_qty:
                    status = "REJECTED"
                else:
                    self.wallet.adjust_balance(base, -executed_qty)
                    self.wallet.adjust_balance(quote, (executed_notional - fee))
        else:
            raise ValueError("side must be BUY or SELL")

        order = {
            "id": order_id,
            "timestamp": int(timestamp) if timestamp is not None else int(time.time() * 1000),
            "symbol": symbol,
            "orderId": order_id,
            "transactTime": int(time.time() * 1000),
            "price": str(executed_price),
            "origQty": str(qty),
            "executedQty": str(executed_qty if status in ("FILLED", "PARTIALLY_FILLED") else 0.0),
            "cummulativeQuoteQty": str(executed_notional if status in ("FILLED", "PARTIALLY_FILLED") else 0.0),
            "status": status,
            "type": type,
            "side": side,
            "timeInForce": timeInForce,
            "signature": signature,
            "apiKey": apiKey
        }
        self._orders.append(order)
        if status in ("FILLED", "PARTIALLY_FILLED"):
            self._trades.append(order)
            self._record_fill(symbol, side, executed_qty, executed_price, order_id)
        return order

    def process_pending_fills(self, max_fill_fraction=0.5, volatility_pct=0.0, spread_sensitivity=1.0, vol_sensitivity=1.0):
        """
        Time-slice fills for partially filled orders. Call this once per tick.
        """
        completed = []
        for order_id, pf in list(self._pending_fills.items()):
            symbol = pf["symbol"]
            base, quote = self._parse_symbol(symbol)
            book = self.order_books.get(f"{base}/{quote}", {})
            side = pf["side"]
            limit_px = pf.get("limit_price")
            qty_remaining = pf["qty_remaining"]

            spread = book.get("spread")
            best_bid = book.get("best_bid")
            best_ask = book.get("best_ask")
            mid = (best_bid + best_ask) / 2 if (best_bid is not None and best_ask is not None) else None
            spread_pct = (spread / mid) if (spread is not None and mid) else 0.0
            # Smaller fill fraction when spread/volatility are high
            spread_factor = max(0.2, min(1.0, 1.0 - min(spread_pct * 10 * spread_sensitivity, 0.8)))
            vol_factor = max(0.2, min(1.0, 1.0 - min(volatility_pct * 5 * vol_sensitivity, 0.8)))
            adaptive_fraction = max_fill_fraction * spread_factor * vol_factor
            adaptive_fraction = max(0.05, min(0.8, adaptive_fraction))

            slice_qty = qty_remaining * adaptive_fraction
            if slice_qty <= 0:
                completed.append(order_id)
                continue

            fill_qty, avg_price = self._simulate_fill_from_book(side, slice_qty, book, None, limit_price=limit_px)
            if avg_price is None or fill_qty <= 0:
                continue

            notional = fill_qty * avg_price
            fee = notional * (self.fee_bps / 10000.0)

            if side == "BUY":
                if self.wallet.get_balance(quote) < (notional + fee):
                    continue
                self.wallet.adjust_balance(quote, -(notional + fee))
                self.wallet.adjust_balance(base, fill_qty)
            else:
                if self.wallet.get_balance(base) < fill_qty:
                    continue
                self.wallet.adjust_balance(base, -fill_qty)
                self.wallet.adjust_balance(quote, (notional - fee))

            pf["qty_remaining"] -= fill_qty
            self._record_fill(symbol, side, fill_qty, avg_price, order_id)
            # Update order record
            for o in self._orders:
                if o["orderId"] == order_id:
                    prev_exec = float(o["executedQty"])
                    new_exec = prev_exec + fill_qty
                    o["executedQty"] = str(new_exec)
                    o["cummulativeQuoteQty"] = str(float(o["cummulativeQuoteQty"]) + notional)
                    o["status"] = "FILLED" if pf["qty_remaining"] <= 0 else "PARTIALLY_FILLED"
                    break

            if pf["qty_remaining"] <= 0:
                completed.append(order_id)

        for oid in completed:
            self._pending_fills.pop(oid, None)

    def get_open_orders(self, symbol=None):
        orders = [o for o in self._orders if o["status"] in ("NEW", "PARTIALLY_FILLED")]
        if symbol:
            return [o for o in orders if o["symbol"] == symbol]
        return orders

    def get_all_orders(self, symbol=None):
        if symbol:
            return [o for o in self._orders if o["symbol"] == symbol]
        return list(self._orders)

    def get_order(self, symbol, orderId):
        for o in self._orders:
            if o["orderId"] == orderId and o["symbol"] == symbol:
                return o
        return {"symbol": symbol, "orderId": orderId, "status": "NOT_FOUND"}

    def get_my_trades(self, symbol=None):
        if symbol:
            return [t for t in self._trades if t["symbol"] == symbol]
        return list(self._trades)

    def cancel_order(self, symbol, orderId):
        for o in self._orders:
            if o["orderId"] == orderId and o["symbol"] == symbol:
                if o["status"] in ("NEW", "PARTIALLY_FILLED"):
                    o["status"] = "CANCELED"
                return o
        return {
            "symbol": symbol,
            "orderId": orderId,
            "status": "NOT_FOUND",
            "id": orderId,
            "timestamp": int(time.time() * 1000)
        }

    def cancel_open_orders(self, symbol=None):
        canceled = []
        for o in self._orders:
            if o["status"] in ("NEW", "PARTIALLY_FILLED") and (symbol is None or o["symbol"] == symbol):
                o["status"] = "CANCELED"
                canceled.append(o)
        return canceled

    def cancel_and_replace_order(self, symbol, orderId, side, type, quantity, price=None):
        self.cancel_order(symbol, orderId)
        return self.create_order(symbol=symbol, side=side, type=type, quantity=quantity, price=price)

    def amend_keep_priority(self, symbol, orderId, quantity=None, price=None):
        for o in self._orders:
            if o["orderId"] == orderId and o["symbol"] == symbol:
                if o["status"] in ("NEW", "PARTIALLY_FILLED"):
                    if quantity is not None:
                        o["origQty"] = str(quantity)
                    if price is not None:
                        o["price"] = str(price)
                return o
        return {"symbol": symbol, "orderId": orderId, "status": "NOT_FOUND"}

    def create_order_list(self, symbol, orders):
        list_id = str(uuid.uuid4())
        results = []
        for item in orders:
            results.append(self.create_order(symbol=symbol, **item))
        entry = {
            "listId": list_id,
            "id": list_id,
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "orders": results
        }
        self._order_lists.append(entry)
        return entry

    def create_sor_order(self, symbol, side, quantity):
        # Simulated smart order routing: apply a small price improvement
        base, quote = self._parse_symbol(symbol)
        market_px = self.prices.get(f"{base}/{quote}")
        if market_px is None:
            raise ValueError(f"No live price for {base}/{quote}")
        improvement = 0.0001  # 1 bp
        price = market_px * (1 - improvement) if side.upper() == "BUY" else market_px * (1 + improvement)
        return self.create_order(symbol=symbol, side=side, type="MARKET", quantity=quantity, price=price)
