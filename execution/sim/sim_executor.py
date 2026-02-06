"""
Simulated execution engine.

Responsibilities:
- Manage simulated positions and cash
- Enforce SL/TP/trailing stops
- Log trades to CSV for analysis/LLM feedback
"""
import csv
import time
import random


@property
def available_capital(self):
    allocated = sum(
        pos["size"] * pos["entry_price"]
        for pos in self.positions.values()
    )
    return max(0.0, self.capital - allocated)

class AccountingAgent:
    def __init__(self, starting_capital, csv_file="trades.csv", max_position_fraction=0.1, wallet=None):
        self.starting_capital = starting_capital
        self.capital = starting_capital  # realized cash
        self.available_capital = starting_capital  # cash available to open new positions
        self.positions = {}  # symbol -> position dict
        self.orders = []  # list of order dicts for lifecycle tracking
        self.csv_file = csv_file
        self.max_position_fraction = max_position_fraction  # fraction of capital allowed per trade
        self.wallet = wallet

        # Create CSV file with headers
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "side", "price", "size",
                "pnl", "equity", "reason", "sentiment_score",
                "velocity", "llm_decision",
                "stop_loss_price", "take_profit_price", "trailing_stop_price",
                "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
                "fill_price", "status", "notional", "slippage_bps", "latency_ms",
                "result"
            ])

    # -----------------------------
    # Enter a new position
    # -----------------------------
    def enter(self, price, size, symbol, stop_loss_pct=0.01, take_profit_pct=0.02,
              sentiment_score=None, velocity=None, llm_decision=None, size_in_dollars=False,
              simulate_slippage_bps=0.0, latency_ms=0, reject_rate=0.0,
              partial_fill_rate=0.0, partial_fill_fraction=0.5, trailing_stop_pct=0.0):
        if symbol in self.positions:
            return  # already in position

        # Interpret size as dollars (notional) or units
        if size_in_dollars:
            notional = size
            units = notional / price if price else 0.0
        else:
            units = size
            notional = units * price

        # Enforce max position fraction and available capital (dollars)
        max_allowed = self.starting_capital * self.max_position_fraction
        notional = min(notional, max_allowed, self.available_capital)
        if notional <= 0:
            print(f"⚠️ Cannot enter {symbol}: insufficient available capital")
            return
        units = notional / price if price else 0.0
        # Simulate order lifecycle
        order = {
            "timestamp": time.time(),
            "symbol": symbol,
            "requested_price": price,
            "requested_units": units,
            "requested_notional": notional,
            "status": "SUBMITTED",
            "latency_ms": latency_ms,
        }
        self.orders.append(order)

        if reject_rate > 0 and random.random() < reject_rate:
            order["status"] = "REJECTED"
            self._log(symbol, "ENTER", price, 0, 0, reason="REJECTED",
                      sentiment_score=sentiment_score, velocity=velocity, llm_decision=llm_decision,
                      fill_price=None, status="REJECTED", notional=0,
                      slippage_bps=simulate_slippage_bps, latency_ms=latency_ms)
            print(f"⚠️ ENTER REJECTED | {symbol} price={price:.2f}")
            return

        fill_fraction = 1.0
        if partial_fill_rate > 0 and random.random() < partial_fill_rate:
            fill_fraction = max(0.0, min(1.0, partial_fill_fraction))
            order["status"] = "PARTIAL_FILLED"
        else:
            order["status"] = "FILLED"

        fill_price = price * (1 + (simulate_slippage_bps / 10000.0))
        fill_units = units * fill_fraction
        fill_notional = fill_units * fill_price

        self.positions[symbol] = {
            "entry_price": price,
            "fill_price": fill_price,
            "size": fill_units,
            "notional_entry": fill_notional,
            "symbol": symbol,
            "timestamp": time.time(),
            "stop_loss": price * (1 - stop_loss_pct),
            "take_profit": price * (1 + take_profit_pct),
            "trailing_stop_pct": trailing_stop_pct,
            "trailing_stop": (price * (1 - trailing_stop_pct)) if trailing_stop_pct else None,
            "llm_decision": llm_decision
        }

        # Deduct used capital
        self.available_capital -= fill_notional
        if self.wallet:
            base = symbol.split("/")[0]
            self.wallet.adjust_balance("USD", -fill_notional)
            self.wallet.adjust_balance(base, fill_units)
            # Persist positions in mock wallet
            positions = list(self.wallet.get_positions())
            positions = [p for p in positions if p.get("symbol") != symbol]
            positions.append({
                "symbol": symbol,
                "entry_price": price,
                "fill_price": fill_price,
                "size": fill_units,
                "notional_entry": fill_notional,
                "timestamp": time.time(),
                "stop_loss": self.positions[symbol]["stop_loss"],
                "take_profit": self.positions[symbol]["take_profit"],
                "trailing_stop_pct": trailing_stop_pct,
                "trailing_stop": self.positions[symbol].get("trailing_stop"),
                "llm_decision": llm_decision
            })
            self.wallet.set_positions(positions)

        print(f"📈 ENTER | {symbol} price={price:.2f} size={fill_units:.4f} "
              f"SL={self.positions[symbol]['stop_loss']:.2f} TP={self.positions[symbol]['take_profit']:.2f}")

        self._log(symbol, "ENTER", price, fill_units, 0, reason="ENTRY",
                  sentiment_score=sentiment_score, velocity=velocity, llm_decision=llm_decision,
                  stop_loss=self.positions[symbol].get("stop_loss"),
                  take_profit=self.positions[symbol].get("take_profit"),
                  trailing_stop=self.positions[symbol].get("trailing_stop"),
                  stop_loss_pct=stop_loss_pct,
                  take_profit_pct=take_profit_pct,
                  trailing_stop_pct=trailing_stop_pct,
                  fill_price=fill_price, status=order["status"], notional=fill_notional,
                  slippage_bps=simulate_slippage_bps, latency_ms=latency_ms)

    # -----------------------------
    # Exit an open position
    # -----------------------------
    def exit(self, symbol, price=None, reason="EXIT", sentiment_score=None, velocity=None, llm_decision=None):
        if symbol not in self.positions:
            return 0.0

        pos = self.positions.pop(symbol)
        entry_price = pos.get("fill_price", pos["entry_price"])
        exit_price = price if price is not None else entry_price
        pnl = (exit_price - entry_price) * pos["size"]
        self.capital += pnl
        self.available_capital += pos["size"] * exit_price  # release capital
        if self.wallet:
            base = symbol.split("/")[0]
            self.wallet.adjust_balance(base, -pos["size"])
            self.wallet.adjust_balance("USD", pos["size"] * exit_price)
            positions = list(self.wallet.get_positions())
            positions = [p for p in positions if p.get("symbol") != symbol]
            self.wallet.set_positions(positions)

        print(f"📉 EXIT  | {symbol} price={exit_price:.2f} "
              f"PnL={pnl:.2f} Equity={self.capital:.2f} | Reason: {reason}")

        self._log(symbol, "EXIT", exit_price, pos["size"], pnl, reason=reason,
                  sentiment_score=sentiment_score, velocity=velocity, llm_decision=llm_decision,
                  stop_loss=pos.get("stop_loss"),
                  take_profit=pos.get("take_profit"),
                  trailing_stop=pos.get("trailing_stop"),
                  stop_loss_pct=llm_decision.get("stop_loss_pct") if isinstance(llm_decision, dict) else None,
                  take_profit_pct=llm_decision.get("take_profit_pct") if isinstance(llm_decision, dict) else None,
                  trailing_stop_pct=pos.get("trailing_stop_pct"))

        return pnl

    # -----------------------------
    # Check stop-loss / take-profit
    # -----------------------------
    def check_sl_tp(self, prices: dict, sentiment_dict=None, velocity_dict=None, llm_dict=None):
        to_exit = []
        reasons = {}
        results = []
        for sym, pos in self.positions.items():
            if sym not in prices:
                continue
            last_price = prices[sym]
            trailing_pct = pos.get("trailing_stop_pct", 0.0)
            if trailing_pct:
                new_trailing = last_price * (1 - trailing_pct)
                if pos.get("trailing_stop") is None or new_trailing > pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trailing
                if last_price <= pos["trailing_stop"]:
                    to_exit.append(sym)
                    reasons[sym] = "TRAILING_STOP"
                    continue
            if last_price <= pos["stop_loss"]:
                to_exit.append(sym)
                reasons[sym] = "STOP_LOSS"
            elif last_price >= pos["take_profit"]:
                to_exit.append(sym)
                reasons[sym] = "TAKE_PROFIT"
        for sym in to_exit:
            pnl = self.exit(
                sym,
                price=prices[sym],
                reason=reasons.get(sym, "EXIT"),
                sentiment_score=sentiment_dict.get(sym) if sentiment_dict else None,
                velocity=velocity_dict.get(sym) if velocity_dict else None,
                llm_decision=llm_dict.get(sym) if llm_dict else None
            )
            results.append((sym, pnl))
        return results

    # -----------------------------
    # Signal-only SL/TP checks (no execution)
    # -----------------------------
    def check_sl_tp_signals(self, prices: dict):
        signals = []
        for sym, pos in self.positions.items():
            if sym not in prices:
                continue
            last_price = prices[sym]
            trailing_pct = pos.get("trailing_stop_pct", 0.0)
            if trailing_pct:
                new_trailing = last_price * (1 - trailing_pct)
                if pos.get("trailing_stop") is None or new_trailing > pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trailing
                if last_price <= pos["trailing_stop"]:
                    signals.append((sym, "TRAILING_STOP"))
                    continue
            if last_price <= pos["stop_loss"]:
                signals.append((sym, "STOP_LOSS"))
            elif last_price >= pos["take_profit"]:
                signals.append((sym, "TAKE_PROFIT"))
        return signals

    # -----------------------------
    # Apply a simulated fill to ledger
    # -----------------------------
    def apply_fill(self, symbol, side, price, qty, llm_decision=None):
        side = side.upper()
        qty = float(qty)
        price = float(price)
        if qty <= 0:
            return

        if side == "BUY":
            if symbol not in self.positions:
                sl_pct = float((llm_decision or {}).get("stop_loss_pct_final", 0.01))
                tp_pct = float((llm_decision or {}).get("take_profit_pct_final", 0.02))
                trailing_pct = float((llm_decision or {}).get("trailing_stop_pct_final", 0.0))
                self.positions[symbol] = {
                    "entry_price": price,
                    "fill_price": price,
                    "size": qty,
                    "notional_entry": price * qty,
                    "symbol": symbol,
                    "timestamp": time.time(),
                    "stop_loss": price * (1 - sl_pct),
                    "take_profit": price * (1 + tp_pct),
                    "trailing_stop_pct": trailing_pct,
                    "trailing_stop": (price * (1 - trailing_pct)) if trailing_pct else None,
                    "llm_decision": llm_decision,
                    "llm_size_fraction": (llm_decision or {}).get("size_fraction", 0.0),
                    "llm_confidence": (llm_decision or {}).get("confidence", 0.0)
                }
            else:
                pos = self.positions[symbol]
                old_size = pos["size"]
                new_size = old_size + qty
                avg_price = ((pos["entry_price"] * old_size) + (price * qty)) / new_size
                pos["entry_price"] = avg_price
                pos["fill_price"] = avg_price
                pos["size"] = new_size
        elif side == "SELL":
            if symbol not in self.positions:
                return
            pos = self.positions[symbol]
            sell_qty = min(qty, pos["size"])
            pnl = (price - pos["entry_price"]) * sell_qty
            self.capital += pnl
            self.available_capital += sell_qty * price
            pos["size"] -= sell_qty
            if pos["size"] <= 0:
                self.positions.pop(symbol, None)
            # Log partial or full exit
            reason = "SIM_FILL_SELL"
            side_label = "EXIT" if pos.get("size", 0) <= 0 else "EXIT_PARTIAL"
            self._log(symbol, side_label, price, sell_qty, pnl, reason=reason, llm_decision=llm_decision)

    # -----------------------------
    # Compute unrealized PnL
    # -----------------------------
    def unrealized_pnl(self, prices: dict):
        pnl = 0
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if price is None and isinstance(sym, str) and "/" not in sym:
                slash_sym = f"{sym[:-3]}/{sym[-3:]}" if len(sym) > 3 else sym
                price = prices.get(slash_sym)
            if price is not None:
                pnl += (price - pos["entry_price"]) * pos["size"]
        return pnl

    # -----------------------------
    # Total equity = available capital + unrealized positions
    # -----------------------------
    def equity(self, prices: dict):
        total = self.available_capital
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

    # -----------------------------
    # Get recent closed PnL for reinforcement
    # -----------------------------
    def recent_pnls(self, symbol, n=5):
        # This could read from CSV or keep in memory
        try:
            with open(self.csv_file, "r") as f:
                lines = list(csv.reader(f))[1:]  # skip header
            pnls = [float(line[5]) for line in lines if line[1] == symbol and line[2] == "EXIT"]
            return pnls[-n:]
        except Exception:
            return []

    # -----------------------------
    # Logging
    # -----------------------------
    def _log(self, symbol, side, price, size, pnl, reason="",
             sentiment_score=None, velocity=None, llm_decision=None,
             stop_loss=None, take_profit=None, trailing_stop=None,
             stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None,
             fill_price=None, status=None, notional=None, slippage_bps=None, latency_ms=None):
        result = ""
        try:
            pnl_val = float(pnl)
            if side in ("EXIT", "EXIT_PARTIAL", "SIM_FILL_SELL"):
                if pnl_val > 0:
                    result = "W"
                elif pnl_val < 0:
                    result = "L"
                else:
                    result = "B"
        except Exception:
            result = ""
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), symbol, side, price, size,
                pnl, self.equity({sym: price for sym in self.positions.keys()}),
                reason,
                sentiment_score if sentiment_score is not None else "",
                velocity if velocity is not None else "",
                llm_decision if llm_decision is not None else "",
                stop_loss if stop_loss is not None else "",
                take_profit if take_profit is not None else "",
                trailing_stop if trailing_stop is not None else "",
                stop_loss_pct if stop_loss_pct is not None else "",
                take_profit_pct if take_profit_pct is not None else "",
                trailing_stop_pct if trailing_stop_pct is not None else "",
                fill_price if fill_price is not None else "",
                status if status is not None else "",
                notional if notional is not None else "",
                slippage_bps if slippage_bps is not None else "",
                latency_ms if latency_ms is not None else "",
                result
            ])

    # -----------------------------
    # Performance summary for LLM feedback
    # -----------------------------
    def performance_summary(self, symbol=None, lookback=50, objective_weights=None):
        try:
            with open(self.csv_file, "r") as f:
                lines = list(csv.reader(f))[1:]  # skip header
            if symbol:
                lines = [line for line in lines if line[1] == symbol]
            exits = [line for line in lines if line[2] == "EXIT"]
            if lookback:
                exits = exits[-lookback:]
            pnls = [float(line[5]) for line in exits if line[5] not in ("", None)]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            total = len(pnls)
            win_rate = (wins / total) if total else 0.0
            avg_pnl = (sum(pnls) / total) if total else 0.0
            # Objective: reward profitability and consistency, penalize drawdown
            weights = objective_weights or {
                "avg_pnl": 1.0,
                "win_rate": 1.0,
                "max_drawdown": 1.0
            }
            equity_series = [float(line[6]) for line in exits if line[6] not in ("", None)]
            peak = None
            max_dd = 0.0
            for e in equity_series:
                if peak is None or e > peak:
                    peak = e
                dd = (peak - e) / peak if peak else 0.0
                if dd > max_dd:
                    max_dd = dd
            # Simple objective score
            objective = (
                (avg_pnl * weights.get("avg_pnl", 1.0)) * (1 + win_rate * weights.get("win_rate", 1.0))
                - (max_dd * weights.get("max_drawdown", 1.0) * (abs(avg_pnl) + 1))
            )

            return {
                "trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "max_drawdown": max_dd,
                "objective_score": objective,
                "objective_weights": weights,
                "recent_pnls": pnls[-5:] if pnls else [],
                "recent_results": [
                    (line[17] if len(line) > 17 and line[17] else ("W" if float(line[5]) > 0 else "L" if float(line[5]) < 0 else "B"))
                    for line in exits[-5:]
                    if line[5] not in ("", None)
                ],
                "avg_trailing_stop_pct": (
                    sum(
                        float(line[11]) for line in exits
                        if len(line) > 11 and line[11] not in ("", None)
                    ) / len(exits)
                ) if exits else 0.0
            }
        except Exception:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
                "objective_score": 0.0,
                "recent_pnls": []
            }

# Backward-compatible alias
PaperTrader = AccountingAgent


