"""
Backfill missing realized_pnl / outcome fields in logs/order_actions.jsonl.

Usage:
  python utils/backfill_order_actions.py
"""
import json
import os


def _to_slash_symbol(symbol):
    if not symbol:
        return symbol
    if "/" in symbol:
        return symbol
    sym = str(symbol)
    if sym.endswith("USD") and len(sym) > 3:
        return f"{sym[:-3]}/USD"
    return sym


def _extract_side(record):
    action = record.get("action") or {}
    side = action.get("side") or action.get("action") or ""
    side = str(side).upper()
    if side in ("BUY", "SELL"):
        return side
    # Fallback to result side if present
    result = record.get("result") or {}
    side = str(result.get("side") or "").upper()
    if side in ("BUY", "SELL"):
        return side
    return ""


def _extract_qty_price(record):
    action = record.get("action") or {}
    qty = action.get("quantity") or action.get("qty") or action.get("size")
    price = action.get("price")
    result = record.get("result") or {}
    if qty is None:
        qty = result.get("executedQty") or result.get("origQty")
    if price is None:
        price = result.get("price")
    try:
        qty = float(qty)
    except Exception:
        qty = None
    try:
        price = float(price)
    except Exception:
        price = None
    return qty, price


def main():
    path = os.path.join("logs", "order_actions.jsonl")
    if not os.path.exists(path):
        print("No logs/order_actions.jsonl found.")
        return

    out_path = path + ".backfill"
    positions = {}

    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except Exception:
                continue

            symbol = _to_slash_symbol(record.get("symbol") or "")
            side = _extract_side(record)
            qty, price = _extract_qty_price(record)

            held = positions.get(symbol, {"qty": 0.0, "cost": 0.0})
            if side == "BUY" and qty and price:
                held["cost"] += qty * price
                held["qty"] += qty
            elif side == "SELL" and qty and price:
                if held["qty"] > 0:
                    avg_cost = held["cost"] / held["qty"] if held["qty"] else 0.0
                    reduce_qty = min(held["qty"], qty)
                    realized = (price - avg_cost) * reduce_qty
                    if record.get("realized_pnl") in (None, "", "None"):
                        record["realized_pnl"] = realized
                    if not record.get("outcome"):
                        if realized > 0:
                            record["outcome"] = "W"
                        elif realized < 0:
                            record["outcome"] = "L"
                        else:
                            record["outcome"] = "B"
                    held["cost"] -= avg_cost * reduce_qty
                    held["qty"] -= reduce_qty
            positions[symbol] = held

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(out_path, path)
    print("Backfill complete: logs/order_actions.jsonl updated.")


if __name__ == "__main__":
    main()
