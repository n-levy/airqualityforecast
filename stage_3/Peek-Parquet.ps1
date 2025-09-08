param(
  [Parameter(Mandatory=$true)][string]$Path,
  [int]$Rows=5
)

$ErrorActionPreference = "Stop"

# Resolve Python exe (prefer Stage 2 venv if available)
$Stage3Root = $PSScriptRoot
$PyStage2   = Join-Path (Join-Path $Stage3Root "..") "stage_2\.venv_stage2\Scripts\python.exe"
$PythonExe  = $null
if (Test-Path $PyStage2) {
  $PythonExe = $PyStage2
} else {
  $PythonExe = Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe"
}

# Make sure path is absolute (so it's robust regardless of where this is called)
if (-not (Split-Path $Path -IsAbsolute)) {
  $Path = Join-Path $Stage3Root $Path
}

$code = @"
import pandas as pd
p = r'$Path'
df = pd.read_parquet(p)
print(df.head($Rows).to_string(index=False))
"@

$code | & $PythonExe -
if ($LASTEXITCODE -ne 0) { throw "Peek-Parquet failed for $Path" }
