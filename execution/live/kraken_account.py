"""
Account balance sync for Kraken.
"""
from execution.live.kraken_exchange import LiveKrakenClient


class KrakenAccount:
    def __init__(self):
        self.client = LiveKrakenClient()

    def get_balance(self):
        res = self.client.get_balance()
        if isinstance(res, dict) and res.get("error"):
            return {}
        balances = res.get("result", {}) if isinstance(res, dict) else {}
        # Normalize to {asset: {free: , locked:}}
        out = {}
        for asset, amt in balances.items():
            out[asset] = {"free": amt, "locked": "0.0"}
        return out
