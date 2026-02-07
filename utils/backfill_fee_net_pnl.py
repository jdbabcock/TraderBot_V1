"""
Backfill fee-estimated net PnL in logs/order_actions.jsonl and logs/order_actions.csv.

Usage:
  python utils/backfill_fee_net_pnl.py
"""
import csv
import json
import os
import shutil
import time

try:
    from config import config as bot_config
except Exception:
    bot_config = None

KRAKEN_MAKER_FEE_PCT = float(getattr(bot_config, "KRAKEN_MAKER_FEE_PCT", 0.0025)) if bot_config else 0.0025
KRAKEN_TAKER_FEE_PCT = float(getattr(bot_config, "KRAKEN_TAKER_FEE_PCT", 0.004)) if bot_config else 0.004
ESTIMATED_SLIPPAGE_PCT = float(getattr(bot_config, "ESTIMATED_SLIPPAGE_PCT", 0.0075)) if bot_config else 0.0075


def _fee_rate_for_type(order_type):
    t = str(order_type or "").upper()
    if t == "LIMIT":
        return KRAKEN_MAKER_FEE_PCT
    return KRAKEN_TAKER_FEE_PCT


def _extract_side(record):
    action = record.get("action") or {}
    side = action.get("side") or action.get("action") or ""
    side = str(side).upper()
    if side in ("BUY", "SELL"):
        return side
    result = record.get("result") or {}
    side = str(result.get("side") or "").upper()
    if side in ("BUY", "SELL"):
        return side
    return ""


def _extract_qty_price(record):
    action = record.get("action") or {}
    qty = action.get("quantity") or action.get("qty") or action.get("size")
    price = action.get("price")
    order_type = action.get("type") or action.get("order_type")
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
    return qty, price, order_type


def _compute_fee_net(qty, exit_price, realized_pnl, entry_order_type=None):
    if qty is None or exit_price is None or realized_pnl is None:
        return None, None, None
    try:
        qty = float(qty)
        exit_price = float(exit_price)
        realized_pnl = float(realized_pnl)
    except Exception:
        return None, None, None
    if qty <= 0 or exit_price <= 0:
        return None, None, None
    entry_price = exit_price - (realized_pnl / qty)
    if entry_price <= 0:
        return None, None, None
    entry_notional = entry_price * qty
    exit_notional = exit_price * qty
    fee_total = (
        entry_notional * _fee_rate_for_type(entry_order_type)
        + exit_notional * _fee_rate_for_type("MARKET")
    )
    slippage_est = (entry_notional + exit_notional) * ESTIMATED_SLIPPAGE_PCT
    realized_net = realized_pnl - fee_total
    return realized_net, fee_total, slippage_est


def _backup(path):
    if not os.path.exists(path):
        return None
    ts = time.strftime("%Y%m%d%H%M%S")
    backup = path + ".bak"
    if os.path.exists(backup):
        backup = path + f".bak.{ts}"
    shutil.copy2(path, backup)
    return backup


def _backfill_jsonl(path):
    if not os.path.exists(path):
        print("No logs/order_actions.jsonl found.")
        return 0
    _backup(path)
    updated = 0
    out_path = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except Exception:
                continue
            side = _extract_side(record)
            realized = record.get("realized_pnl")
            if side == "SELL" and realized not in (None, "", "None"):
                if record.get("realized_pnl_net") in (None, "", "None"):
                    qty, price, order_type = _extract_qty_price(record)
                    realized_net, fee_total, slippage_est = _compute_fee_net(qty, price, realized, order_type)
                    if realized_net is not None:
                        record["realized_pnl_net"] = realized_net
                        record["fee_total_est"] = fee_total
                        record["slippage_est"] = slippage_est
                        record["outcome"] = "W" if realized_net > 0 else ("L" if realized_net < 0 else "B")
                        updated += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(out_path, path)
    return updated


def _backfill_csv(path):
    if not os.path.exists(path):
        print("No logs/order_actions.csv found.")
        return 0
    _backup(path)
    updated = 0
    out_path = path + ".tmp"
    with open(path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = [f for f in (reader.fieldnames or []) if f]
        for col in ("realized_pnl_net", "fee_total_est", "slippage_est"):
            if col not in fieldnames:
                fieldnames.append(col)
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                # Guard against malformed rows with None keys
                row = {k: v for k, v in row.items() if k}
                side = (row.get("side") or row.get("action") or "").upper()
                realized = row.get("realized_pnl")
                if side == "SELL" and realized not in (None, "", "None"):
                    if row.get("realized_pnl_net") in (None, "", "None"):
                        try:
                            qty = float(row.get("qty") or row.get("quantity") or 0.0)
                        except Exception:
                            qty = None
                        try:
                            price = float(row.get("price") or 0.0)
                        except Exception:
                            price = None
                        order_type = row.get("type") or row.get("order_type")
                        realized_net, fee_total, slippage_est = _compute_fee_net(qty, price, realized, order_type)
                        if realized_net is not None:
                            row["realized_pnl_net"] = f"{realized_net:.8f}"
                            row["fee_total_est"] = f"{fee_total:.8f}"
                            row["slippage_est"] = f"{slippage_est:.8f}"
                            row["outcome"] = "W" if realized_net > 0 else ("L" if realized_net < 0 else "B")
                            updated += 1
                writer.writerow(row)
    os.replace(out_path, path)
    return updated


def main():
    jsonl_path = os.path.join("logs", "order_actions.jsonl")
    csv_path = os.path.join("logs", "order_actions.csv")
    updated_jsonl = _backfill_jsonl(jsonl_path)
    updated_csv = _backfill_csv(csv_path)
    print(f"Backfill complete: jsonl_updated={updated_jsonl} csv_updated={updated_csv}")


if __name__ == "__main__":
    main()
