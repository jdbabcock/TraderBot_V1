# Trader Bot

## Quick Start (Windows PowerShell)
1. Open PowerShell in the project folder.
2. Create a virtual environment:
```powershell
python -m venv .venv
```
3. Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```
4. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```
5. Add your API keys in `config/.env` (placeholders are provided).
6. Start the bot:
```powershell
.\run.ps1
```

## Alternative (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
./run.sh
```

## Direct Run (without run.ps1)
```powershell
.\.venv\Scripts\python.exe app\run.py
```

## Notes
1. If PowerShell blocks scripts, run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
2. `config/.env` must contain your OpenAI and exchange keys before starting.
3. Runtime data is written to `data/` and `logs/`.

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
