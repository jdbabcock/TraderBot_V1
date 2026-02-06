# analysis/forward_test.py
import argparse
import json
import os
import time

from analysis.performance_report import build_report


def main():
    parser = argparse.ArgumentParser(description="Forward-test snapshot report.")
    parser.add_argument("--trades", default="trades.csv")
    parser.add_argument("--llm-log", default="llm_activity.log")
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--out", default="reports/forward_snapshot.json")
    args = parser.parse_args()

    report = build_report(args.trades, args.llm_log, lookback=args.lookback)
    report["snapshot_at"] = time.time()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote snapshot to {args.out}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()
