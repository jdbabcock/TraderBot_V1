"""
Mock wallet persistence for sim mode.

Responsibilities:
- Persist balances/positions to disk
- Emit history/totals CSVs for analytics
"""
import json
import os
import time
import csv


class MockWallet:
    def __init__(self, path="data/mock_wallet.json", snapshot_interval_seconds=10, live_prices=None, initial_usd=10000.0):
        self.path = path
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.live_prices = live_prices or {}
        self._last_snapshot = 0.0
        self.data = {
            "updated_at": time.time(),
            "balances": {
                "USD": float(initial_usd)
            }
        }
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self._save()
            return
        try:
            with open(self.path, "r") as f:
                self.data = json.load(f)
        except Exception:
            self._save()

    def _save(self):
        self.data["updated_at"] = time.time()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        self._append_snapshot()

    def _append_snapshot(self):
        try:
            csv_path = os.path.join(os.path.dirname(self.path), "mock_wallet_history.csv")
            totals_path = os.path.join(os.path.dirname(self.path), "mock_wallet_totals.csv")
            file_exists = os.path.exists(csv_path)
            totals_exists = os.path.exists(totals_path)
            now = self.data.get("updated_at", time.time())
            if (now - self._last_snapshot) < self.snapshot_interval_seconds:
                return
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "asset", "balance", "usd_value", "price_ts"])
                ts = now
                total_usd = 0.0
                for asset, bal in self.data.get("balances", {}).items():
                    usd_value = ""
                    price_ts = ""
                    if asset == "USD":
                        usd_value = float(bal)
                        total_usd += float(bal)
                    else:
                        pair = f"{asset}/USD"
                        price = self.live_prices.get(pair)
                        if price is not None:
                            usd_value = float(bal) * float(price)
                            total_usd += float(usd_value)
                            price_ts = self.live_prices.get("_timestamp")
                    writer.writerow([ts, asset, bal, usd_value, price_ts])
                writer.writerow([ts, "TOTAL_USD", "", total_usd, self.live_prices.get("_timestamp")])
            with open(totals_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not totals_exists:
                    writer.writerow(["timestamp", "total_usd", "price_ts"])
                writer.writerow([ts, total_usd, self.live_prices.get("_timestamp")])
            self._last_snapshot = now
        except Exception:
            pass

    def backfill_usd_values(self):
        csv_path = os.path.join(os.path.dirname(self.path), "mock_wallet_history.csv")
        if not os.path.exists(csv_path):
            return
        try:
            rows = []
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return
            header = rows[0]
            if "usd_value" not in header:
                return
            ts_idx = header.index("timestamp")
            asset_idx = header.index("asset")
            bal_idx = header.index("balance")
            usd_idx = header.index("usd_value")
            price_ts_idx = header.index("price_ts") if "price_ts" in header else None
            updated = [header]
            for row in rows[1:]:
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                asset = row[asset_idx]
                if row[usd_idx]:
                    updated.append(row)
                    continue
                if asset == "USD":
                    row[usd_idx] = row[bal_idx]
                elif asset == "TOTAL_USD":
                    updated.append(row)
                    continue
                else:
                    pair = f"{asset}/USD"
                    price = self.live_prices.get(pair)
                    if price is not None:
                        row[usd_idx] = str(float(row[bal_idx]) * float(price))
                        if price_ts_idx is not None:
                            row[price_ts_idx] = str(self.live_prices.get("_timestamp", ""))
                updated.append(row)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(updated)
        except Exception:
            pass

    def get_balance(self, asset):
        return float(self.data.get("balances", {}).get(asset, 0.0))

    def set_balance(self, asset, amount):
        self.data.setdefault("balances", {})
        self.data["balances"][asset] = float(amount)
        self._save()

    def adjust_balance(self, asset, delta):
        current = self.get_balance(asset)
        self.set_balance(asset, current + float(delta))

    def snapshot(self):
        return dict(self.data)

    def all_balances(self):
        return dict(self.data.get("balances", {}))

    def get_positions(self):
        return list(self.data.get("positions", []))

    def set_positions(self, positions):
        self.data["positions"] = positions
        self._save()

    def ensure_positions_key(self):
        if "positions" not in self.data:
            self.data["positions"] = []
            self._save()

    def wallet_state(self, lookback=50):
        balances = self.all_balances()
        total_usd = 0.0
        allocations = {}
        for asset, bal in balances.items():
            if asset == "USD":
                usd_val = float(bal)
            else:
                pair = f"{asset}/USD"
                price = self.live_prices.get(pair)
                usd_val = float(bal) * float(price) if price is not None else 0.0
            allocations[asset] = usd_val
            total_usd += usd_val
        alloc_pct = {k: (v / total_usd if total_usd else 0.0) for k, v in allocations.items()}

        # Recent totals from history
        totals_path = os.path.join(os.path.dirname(self.path), "mock_wallet_totals.csv")
        totals = []
        if os.path.exists(totals_path):
            try:
                with open(totals_path, "r", newline="") as f:
                    rows = list(csv.DictReader(f))
                for row in rows[-lookback:]:
                    try:
                        totals.append(float(row.get("total_usd", 0.0)))
                    except Exception:
                        pass
            except Exception:
                pass

        change_pct = 0.0
        max_dd = 0.0
        if len(totals) >= 2 and totals[-2] > 0:
            change_pct = (totals[-1] - totals[-2]) / totals[-2]
        if totals:
            peak = None
            for v in totals:
                if peak is None or v > peak:
                    peak = v
                dd = (peak - v) / peak if peak else 0.0
                if dd > max_dd:
                    max_dd = dd

        return {
            "total_usd": total_usd,
            "alloc_usd": allocations,
            "alloc_pct": alloc_pct,
            "change_pct": change_pct,
            "max_drawdown": max_dd
        }
