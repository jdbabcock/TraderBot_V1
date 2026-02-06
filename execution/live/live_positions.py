import csv
import json
import os
"""
Live portfolio state builder.

Responsibilities:
- Build open positions from live trades
- Track realized PnL by symbol
- Maintain risk settings per symbol
"""
import time


def _to_slash_symbol(symbol):
    if "/" in symbol:
        return symbol
    if len(symbol) > 3:
        return f"{symbol[:-3]}/{symbol[-3:]}"
    return symbol


def _trade_timestamp(trade):
    return trade.get("time") or trade.get("timestamp") or trade.get("transactTime") or 0


class LivePortfolio:
    def __init__(self, symbols, trades_csv_path="data/live_trades.csv", state_path="data/live_positions.json"):
        self.symbols = symbols
        self.positions = {}
        self.risk_settings = {}
        self.starting_capital = 0.0
        self.capital = 0.0
        self._recent_trades = {}
        self.realized_pnl_by_symbol = {}
        self.realized_pnl_total = 0.0
        self._seen_trade_ids = set()
        self.trades_csv_path = trades_csv_path
        self.state_path = state_path
        self._ensure_trades_log()
        self._load_state()

    def _ensure_trades_log(self):
        try:
            os.makedirs(os.path.dirname(self.trades_csv_path), exist_ok=True)
            if os.path.exists(self.trades_csv_path):
                return
            with open(self.trades_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "side",
                    "qty",
                    "price",
                    "quote_qty",
                    "trade_id",
                    "order_id",
                    "is_buyer",
                    "is_maker"
                ])
        except Exception:
            pass

    def _load_state(self):
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                positions = data.get("positions", {})
                if isinstance(positions, dict):
                    self.positions = positions
                realized = data.get("realized_pnl_by_symbol", {})
                if isinstance(realized, dict):
                    self.realized_pnl_by_symbol = realized
                self.realized_pnl_total = float(data.get("realized_pnl_total", 0.0) or 0.0)
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            payload = {
                "positions": self.positions,
                "realized_pnl_by_symbol": self.realized_pnl_by_symbol,
                "realized_pnl_total": self.realized_pnl_total,
                "timestamp": time.time()
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _trade_key(self, trade, symbol):
        trade_id = trade.get("id") or trade.get("tradeId")
        if trade_id is not None:
            return (symbol, str(trade_id))
        return (
            symbol,
            str(trade.get("time") or trade.get("timestamp") or trade.get("transactTime")),
            str(trade.get("qty", trade.get("executedQty", ""))),
            str(trade.get("price", "")),
            str(trade.get("isBuyer"))
        )

    def _append_trades_log(self, symbol, trades):
        if not trades:
            return
        try:
            with open(self.trades_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for t in trades:
                    key = self._trade_key(t, symbol)
                    if key in self._seen_trade_ids:
                        continue
                    self._seen_trade_ids.add(key)
                    qty = float(t.get("qty", t.get("executedQty", 0.0)) or 0.0)
                    price = float(t.get("price", 0.0) or 0.0)
                    quote_qty = float(t.get("quoteQty", t.get("cummulativeQuoteQty", 0.0)) or 0.0)
                    is_buyer = t.get("isBuyer")
                    side = t.get("side")
                    if is_buyer is None and side:
                        is_buyer = str(side).upper() == "BUY"
                    writer.writerow([
                        _trade_timestamp(t),
                        _to_slash_symbol(symbol),
                        "BUY" if is_buyer else "SELL",
                        qty,
                        price,
                        quote_qty,
                        t.get("id") or t.get("tradeId"),
                        t.get("orderId"),
                        is_buyer,
                        t.get("isMaker")
                    ])
        except Exception:
            pass

    def set_cash(self, cash):
        cash = float(cash or 0.0)
        if self.starting_capital <= 0 and cash > 0:
            self.starting_capital = cash
        self.capital = cash

    def apply_llm_risk(self, symbol, llm_output):
        if not llm_output:
            return
        sym = _to_slash_symbol(symbol)
        self.risk_settings[sym] = {
            "stop_loss_pct": float(llm_output.get("stop_loss_pct", 0.0)),
            "take_profit_pct": float(llm_output.get("take_profit_pct", 0.0)),
            "trailing_stop_pct": float(llm_output.get("trailing_stop_pct", 0.0)),
            "trailing_stop": self.risk_settings.get(sym, {}).get("trailing_stop"),
            "updated_at": time.time()
        }

    def _normalize_trade(self, trade, symbol):
        qty = float(trade.get("qty", trade.get("executedQty", 0.0)) or 0.0)
        price = float(trade.get("price", 0.0) or 0.0)
        is_buyer = trade.get("isBuyer")
        side = trade.get("side")
        if is_buyer is None and side:
            is_buyer = str(side).upper() == "BUY"
        side_label = "BUY" if is_buyer else "SELL"
        ts = _trade_timestamp(trade)
        return {
            "symbol": _to_slash_symbol(symbol),
            "side": side_label,
            "type": "TRADE",
            "status": "FILLED",
            "origQty": str(qty),
            "price": str(price),
            "transactTime": ts
        }

    def update_from_trades(self, trades_by_symbol):
        new_positions = {}
        normalized_trades = {}
        realized_by_symbol = {}
        realized_total = 0.0
        prev_realized_total = float(self.realized_pnl_total or 0.0)
        any_trades = False
        for raw_symbol, trades in trades_by_symbol.items():
            if not trades:
                continue
            any_trades = True
            symbol = _to_slash_symbol(raw_symbol)
            trades_sorted = sorted(trades, key=_trade_timestamp)
            self._append_trades_log(symbol, trades_sorted)
            qty = 0.0
            cost = 0.0
            realized_pnl = 0.0
            last_trade_ts = None
            for t in trades_sorted:
                trade_qty = float(t.get("qty", t.get("executedQty", 0.0)) or 0.0)
                trade_price = float(t.get("price", 0.0) or 0.0)
                if trade_qty <= 0 or trade_price <= 0:
                    try:
                        print(f"[POSITIONS] Skipped trade with invalid qty/price: qty={trade_qty} price={trade_price}")
                    except Exception:
                        pass
                    continue
                is_buyer = t.get("isBuyer")
                side = t.get("side")
                if is_buyer is None and side:
                    is_buyer = str(side).upper() == "BUY"
                if is_buyer:
                    cost += trade_qty * trade_price
                    qty += trade_qty
                else:
                    if qty > 0:
                        avg_cost = cost / qty if qty else 0.0
                        reduce_qty = min(qty, trade_qty)
                        realized_pnl += (trade_price - avg_cost) * reduce_qty
                        cost -= avg_cost * reduce_qty
                        qty -= reduce_qty
                    else:
                        qty = 0.0
                        cost = 0.0
                last_trade_ts = _trade_timestamp(t)

            norm = [self._normalize_trade(t, symbol) for t in trades_sorted]
            normalized_trades[symbol] = norm
            realized_by_symbol[symbol] = realized_pnl
            realized_total += realized_pnl

            if qty > 0:
                avg_entry = cost / qty if qty else 0.0
                existing = self.positions.get(symbol, {}) if isinstance(self.positions, dict) else {}
                risk = self.risk_settings.get(symbol, {})
                sl_pct = float(risk.get("stop_loss_pct", 0.0) or 0.0)
                tp_pct = float(risk.get("take_profit_pct", 0.0) or 0.0)
                trailing_pct = float(risk.get("trailing_stop_pct", 0.0) or 0.0)
                trailing_stop = risk.get("trailing_stop")
                if trailing_pct and trailing_stop is None:
                    trailing_stop = avg_entry * (1 - trailing_pct)
                    risk["trailing_stop"] = trailing_stop
                    self.risk_settings[symbol] = risk
                new_positions[symbol] = {
                    "entry_price": avg_entry,
                    "fill_price": avg_entry,
                    "size": qty,
                    "notional_entry": avg_entry * qty,
                    "symbol": symbol,
                    "timestamp": last_trade_ts or time.time(),
                    "stop_loss": avg_entry * (1 - sl_pct) if sl_pct else None,
                    "take_profit": avg_entry * (1 + tp_pct) if tp_pct else None,
                    "trailing_stop_pct": trailing_pct,
                    "trailing_stop": trailing_stop,
                    "llm_decision": existing.get("llm_decision"),
                    "entry_source": "trades"
                }

        for sym in list(self.risk_settings.keys()):
            if sym not in new_positions:
                self.risk_settings.pop(sym, None)

        # Preserve existing positions when trade history is missing/partial
        if new_positions:
            merged = dict(self.positions) if isinstance(self.positions, dict) else {}
            merged.update(new_positions)
            self.positions = merged
        self._save_state()
        self._recent_trades = normalized_trades
        if any_trades:
            self.realized_pnl_by_symbol = realized_by_symbol
            self.realized_pnl_total = realized_total
            if realized_total != prev_realized_total:
                delta = realized_total - prev_realized_total
                print(f"[PNL] Realized updated: total=${realized_total:.2f} (delta=${delta:.2f})")

    def recent_trades(self, symbol=None, limit=50):
        if symbol:
            sym = _to_slash_symbol(symbol)
            trades = self._recent_trades.get(sym, [])
            return trades[-limit:]
        all_trades = []
        for trades in self._recent_trades.values():
            all_trades.extend(trades)
        all_trades.sort(key=lambda t: t.get("transactTime", 0))
        return all_trades[-limit:]

    def unrealized_pnl(self, prices):
        pnl = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if price is None and isinstance(sym, str) and "/" not in sym:
                slash_sym = f"{sym[:-3]}/{sym[-3:]}" if len(sym) > 3 else sym
                price = prices.get(slash_sym)
            if price is not None:
                pnl += (price - pos["entry_price"]) * pos["size"]
        return pnl

    def equity(self, prices):
        total = float(self.capital or 0.0)
        for pos in self.positions.values():
            sym = pos.get("symbol")
            current_price = prices.get(sym)
            if current_price is None and isinstance(sym, str) and "/" not in sym:
                slash_sym = f"{sym[:-3]}/{sym[-3:]}" if len(sym) > 3 else sym
                current_price = prices.get(slash_sym)
            if current_price is None:
                current_price = pos["entry_price"]
            total += pos["size"] * current_price
        return total

    def check_sl_tp(self, prices):
        exits = []
        for sym, pos in self.positions.items():
            last_price = prices.get(sym)
            if last_price is None:
                continue
            trailing_pct = pos.get("trailing_stop_pct", 0.0) or 0.0
            if trailing_pct:
                new_trailing = last_price * (1 - trailing_pct)
                if pos.get("trailing_stop") is None or new_trailing > pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trailing
                    if sym in self.risk_settings:
                        self.risk_settings[sym]["trailing_stop"] = new_trailing
                if last_price <= pos["trailing_stop"]:
                    exits.append((sym, "TRAILING_STOP"))
                    continue
            stop_loss = pos.get("stop_loss")
            take_profit = pos.get("take_profit")
            if stop_loss is not None and last_price <= stop_loss:
                exits.append((sym, "STOP_LOSS"))
            elif take_profit is not None and last_price >= take_profit:
                exits.append((sym, "TAKE_PROFIT"))
        return exits
