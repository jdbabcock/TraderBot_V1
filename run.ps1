$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Venv Python not found at $venvPython. Create the venv first."
    exit 1
}
& $venvPython (Join-Path $here "app\run.py")
