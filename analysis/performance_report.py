# analysis/performance_report.py
import argparse
import csv
import json
import os
from datetime import datetime


def _read_trades(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trades file not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_llm_log(log_path):
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _max_drawdown(equity_series):
    peak = None
    max_dd = 0.0
    for e in equity_series:
        if peak is None or e > peak:
            peak = e
        dd = (peak - e) / peak if peak else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _match_llm(llm_logs, symbol, trade_ts, max_age_seconds=600):
    best = None
    best_dt = None
    for rec in llm_logs:
        if rec.get("symbol") != symbol:
            continue
        ts = _safe_float(rec.get("timestamp"), 0.0)
        if ts <= trade_ts and (trade_ts - ts) <= max_age_seconds:
            dt = trade_ts - ts
            if best_dt is None or dt < best_dt:
                best = rec
                best_dt = dt
    return best


def build_report(trades_path, llm_log_path, lookback=None):
    trades = _read_trades(trades_path)
    llm_logs = _read_llm_log(llm_log_path)

    exits = [t for t in trades if t.get("side") == "EXIT"]
    if lookback:
        exits = exits[-lookback:]

    pnls = [_safe_float(t.get("pnl")) for t in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = len(pnls)

    equity_series = [_safe_float(t.get("equity")) for t in exits if t.get("equity")]
    max_dd = _max_drawdown(equity_series) if equity_series else 0.0

    # Per-symbol breakdown
    per_symbol = {}
    for t in exits:
        sym = t.get("symbol")
        per_symbol.setdefault(sym, [])
        per_symbol[sym].append(_safe_float(t.get("pnl")))
    per_symbol_stats = {
        sym: {
            "trades": len(vals),
            "wins": sum(1 for v in vals if v > 0),
            "losses": sum(1 for v in vals if v < 0),
            "win_rate": (sum(1 for v in vals if v > 0) / len(vals)) if vals else 0.0,
            "avg_pnl": (sum(vals) / len(vals)) if vals else 0.0,
        }
        for sym, vals in per_symbol.items()
    }

    # LLM decision alignment (best-effort)
    aligned = []
    for t in exits:
        sym = t.get("symbol")
        ts = _safe_float(t.get("timestamp"))
        rec = _match_llm(llm_logs, sym, ts)
        if not rec:
            continue
        cleaned = rec.get("cleaned", {})
        aligned.append({
            "symbol": sym,
            "pnl": _safe_float(t.get("pnl")),
            "confidence": _safe_float(cleaned.get("confidence")),
            "size_fraction": _safe_float(cleaned.get("size_fraction")),
            "reason": cleaned.get("reason", "")
        })

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trades_file": trades_path,
        "llm_log_file": llm_log_path,
        "summary": {
            "trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / total) if total else 0.0,
            "avg_pnl": (sum(pnls) / total) if total else 0.0,
            "total_pnl": sum(pnls),
            "profit_factor": (sum(wins) / abs(sum(losses))) if losses else None,
            "max_drawdown": max_dd,
        },
        "per_symbol": per_symbol_stats,
        "llm_alignment_samples": aligned[-100:],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate performance report from trades/logs.")
    parser.add_argument("--trades", default="trades.csv")
    parser.add_argument("--llm-log", default="llm_activity.log")
    parser.add_argument("--lookback", type=int, default=0)
    parser.add_argument("--out", default="reports/perf_report.json")
    args = parser.parse_args()

    report = build_report(
        trades_path=args.trades,
        llm_log_path=args.llm_log,
        lookback=args.lookback if args.lookback > 0 else None
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.out}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()
