# execution/mock_wallet_cli.py
import argparse
from execution.sim.mock_wallet_store import MockWallet


def main():
    parser = argparse.ArgumentParser(description="Mock wallet utility")
    parser.add_argument("--path", default="data/mock_wallet.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="Show balances")

    setb = sub.add_parser("set", help="Set balance")
    setb.add_argument("asset")
    setb.add_argument("amount", type=float)

    adj = sub.add_parser("adjust", help="Adjust balance")
    adj.add_argument("asset")
    adj.add_argument("delta", type=float)

    args = parser.parse_args()
    wallet = MockWallet(args.path)

    if args.cmd == "show":
        print(wallet.all_balances())
    elif args.cmd == "set":
        wallet.set_balance(args.asset, args.amount)
        print(wallet.all_balances())
    elif args.cmd == "adjust":
        wallet.adjust_balance(args.asset, args.delta)
        print(wallet.all_balances())


if __name__ == "__main__":
    main()
