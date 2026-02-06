"""
Account balance sync for Binance.US.

Responsibilities:
- Fetch balances and convert to USD totals
- Provide simplified cash/total snapshot
"""
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException

class BinanceUSAccount:
    def __init__(self):
        # Securely load keys from .env
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise ValueError("API Keys not found in environment variables.")

        # Standardizing on python-binance
        self.client = Client(api_key, api_secret, tld='us')

    def get_balance(self):
        """
        Fetches all non-zero balances from the live account.
        Returns a dictionary formatted for your main loop.
        """
        try:
            account = self.client.get_account()
            balances = {}
            for item in account['balances']:
                free = float(item['free'])
                locked = float(item['locked'])
                if free > 0 or locked > 0:
                    balances[item['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        except Exception as e:
            print(f"❌ Failed to fetch live balances: {e}")
            return {}

    def get_asset_balance(self, asset):
        """Helper for a single asset."""
        try:
            res = self.client.get_asset_balance(asset=asset)
            return float(res['free']) if res else 0.0
        except:
            return 0.0
