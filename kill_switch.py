import os
from dotenv import load_dotenv
from binance.client import Client

ROOT_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(ROOT_DIR, "config", ".env")
load_dotenv(ENV_PATH)
# Force US TLD here too!
client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"), tld='us')

def stop_everything():
    print("🛑 SHUTTING DOWN ALL POSITIONS...")
    
    # 1. Cancel Open Orders
    orders = client.get_open_orders()
    for o in orders:
        client.cancel_order(symbol=o['symbol'], orderId=o['orderId'])
        print(f"Cancelled: {o['symbol']}")

    # 2. Market Sell everything to USD
    acc = client.get_account()
    for b in acc['balances']:
        if float(b['free']) > 0 and b['asset'] not in ['USD', 'USDT']:
            try:
                # Simple Market Sell
                print(f"Selling {b['asset']}...")
                client.create_order(symbol=f"{b['asset']}USD", side='SELL', type='MARKET', quantity=b['free'])
            except: pass

if __name__ == "__main__":
    stop_everything()
