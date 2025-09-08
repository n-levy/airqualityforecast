param(
  [Parameter(Mandatory=$true)][string]$Path,
  [int]$Rows=5
)
$Stage3Root = $PSScriptRoot
$PyStage2 = Join-Path (Join-Path $Stage3Root "..") "stage_2\.venv_stage2\Scripts\python.exe"
$py = (Test-Path $PyStage2) ? $PyStage2 : (Join-Path $Stage3Root ".venv_stage3\Scripts\python.exe")
$code = @"
import pandas as pd, sys
p = r'$Path'
df = pd.read_parquet(p)
print(df.head($Rows).to_string(index=False))
"@
$code | & $py -
Run:
```powershell
powershell -ExecutionPolicy Bypass -File .\Peek-Parquet.ps1 -Path .\data\providers_processed\cams_forecast.parquet -Rows 10
