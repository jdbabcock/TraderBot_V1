# analysis/regime_report.py
import argparse
import csv
import json
import os
from datetime import datetime

import pandas as pd


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _read_trades(csv_path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _load_ohlcv(path):
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ts"] = df["timestamp"].astype("int64") // 10**9
    elif "ts" in df.columns:
        df["ts"] = df["ts"].astype(int)
    else:
        raise ValueError("OHLCV file must include 'timestamp' or 'ts' column.")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def _compute_regimes(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["vol"] = df["returns"].rolling(20).std().fillna(0.0)
    df["ema_fast"] = df["close"].ewm(span=20).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()
    df["trend"] = (df["ema_fast"] > df["ema_slow"]).map({True: "up", False: "down"})

    vol_q = df["vol"].quantile([0.33, 0.66]).values
    low_q, high_q = vol_q[0], vol_q[1]

    def vol_bucket(v):
        if v <= low_q:
            return "low"
        if v >= high_q:
            return "high"
        return "mid"

    df["vol_regime"] = df["vol"].apply(vol_bucket)
    df["regime"] = df["trend"] + "_" + df["vol_regime"]
    return df[["ts", "regime"]]


def _map_trades_to_regime(trades, regimes):
    regimes = regimes.sort_values("ts")
    r_ts = regimes["ts"].values
    r_reg = regimes["regime"].values

    def find_regime(ts):
        idx = r_ts.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return r_reg[idx]

    mapped = []
    for t in trades:
        ts = _safe_float(t.get("timestamp"))
        regime = find_regime(ts)
        mapped.append((t, regime))
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Regime performance report.")
    parser.add_argument("--trades", default="trades.csv")
    parser.add_argument("--history-dir", default="data/history")
    parser.add_argument("--out", default="reports/regime_report.json")
    args = parser.parse_args()

    trades = _read_trades(args.trades)
    exits = [t for t in trades if t.get("side") == "EXIT"]
    if not exits:
        print("No EXIT trades found.")
        return

    per_symbol = {}
    for t in exits:
        sym = t.get("symbol")
        per_symbol.setdefault(sym, [])
        per_symbol[sym].append(t)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "per_symbol": {}
    }

    for sym, sym_trades in per_symbol.items():
        fname = sym.replace("/", "_") + ".csv"
        path = os.path.join(args.history_dir, fname)
        if not os.path.exists(path):
            report["per_symbol"][sym] = {"error": f"Missing OHLCV file: {path}"}
            continue
        df = _load_ohlcv(path)
        regimes = _compute_regimes(df)
        mapped = _map_trades_to_regime(sym_trades, regimes)

        by_regime = {}
        for t, reg in mapped:
            if not reg:
                continue
            by_regime.setdefault(reg, []).append(_safe_float(t.get("pnl")))

        report["per_symbol"][sym] = {
            "trades": len(sym_trades),
            "regimes": {
                reg: {
                    "trades": len(pnls),
                    "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
                    "avg_pnl": (sum(pnls) / len(pnls)) if pnls else 0.0,
                    "total_pnl": sum(pnls)
                }
                for reg, pnls in by_regime.items()
            }
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
