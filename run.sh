#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_python="$here/.venv/Scripts/python.exe"
if [[ ! -f "$venv_python" ]]; then
  echo "Venv Python not found at $venv_python. Create the venv first." >&2
  exit 1
fi
exec "$venv_python" "$here/app/run.py"
