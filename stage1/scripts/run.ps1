param([string]$Target="help")

$PY = "C:\aqf311\.venv\Scripts\python.exe"

function Help {
@"
Available commands:
  help            Show this help
  check-python    Show the Python interpreter being used
  bootstrap       Upgrade pip/setuptools/wheel and install deps; copy .env.example -> .env if missing
  smoke           Run the local smoke test (prints 'Smoke test OK.' on success)
"@ | Write-Host
}

switch ($Target.ToLower()) {
  "help"         { Help }
  "check-python" { & $PY -V }
  "bootstrap"    {
    & $PY -m pip install --upgrade pip setuptools wheel
    & $PY -m pip install -r requirements.txt
    if (-not (Test-Path ".\.env")) {
      Copy-Item "config\env\.env.example" ".\.env" -Force
    }
    Write-Host "bootstrap: done"
  }
  "smoke"        {
    & $PY scripts\smoke_test.py
  }
  default        {
    Write-Host "Unknown target '$Target'." -ForegroundColor Yellow
    Help
    exit 1
  }
}
