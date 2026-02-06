import os
from dotenv import load_dotenv
from binance.client import Client

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(ROOT_DIR, "config", ".env")
load_dotenv(ENV_PATH)
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

def test_connection():
    # Masking keys for the printout
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key else "None"
    print(f"Checking Key: {masked_key}")

    try:
        # TRY GLOBAL FIRST (No TLD)
        print("Attempting Global (binance.com) connection...")
        client = Client(api_key, api_secret)
        client.get_account()
        print("✅ Success! You are using a Global account.")
        return
    except Exception as e:
        print(f"❌ Global failed: {e}")

    try:
        # TRY US SECOND
        print("Attempting US (binance.us) connection...")
        client = Client(api_key, api_secret, tld='us')
        client.get_account()
        print("✅ Success! You are using a Binance.US account.")
        return
    except Exception as e:
        print(f"❌ US failed: {e}")

if __name__ == "__main__":
    test_connection()
