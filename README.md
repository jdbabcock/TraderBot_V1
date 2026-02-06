# Trader Bot

## Getting Started
1. Install dependencies:
```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
2. Add your API keys in `config/.env` (placeholders are already provided).
3. Start the bot:
```
.\run.ps1
```

Notes:
- `config/.env` must contain your OpenAI and exchange keys before starting.
- Runtime data is written to `data/` and `logs/`.

## Run
Setup:
```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Recommended (PowerShell):
```
.\run.ps1
```

Recommended (Git Bash):
```
./run.sh
```

Direct (venv Python):
```
.\.venv\Scripts\python.exe app\run.py
```


## Project Layout
- `app/` entrypoint and runtime orchestration (`run.py`)
- `config/` configuration and environment
- `core/data/` data feeds, sentiment, and market data
- `core/strategy/` risk and signal logic
- `execution/live/` live exchange clients and portfolio logic
- `execution/sim/` paper trading and mock wallet
- `ui/` dashboard + startup/settings UI
- `data/` runtime state and logs (wallets, trades, equity history)
- `logs/` order actions logs
- `tests/` utilities and pre-flight checks
