param(
  [Parameter(Mandatory=$false, ValueFromRemainingArguments=$true)]
  [string[]]$Args
)

# Find stage_1 (repo root or when called from subfolders)
$stage1 = Resolve-Path "stage_1" -ErrorAction SilentlyContinue
if (-not $stage1) { $stage1 = Resolve-Path "..\stage_1" -ErrorAction SilentlyContinue }

# Put stage_1 on PYTHONPATH for imports like: from stage1_forecast import ...
if ($stage1) {
  $stage1Path = $stage1.Path
  if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$stage1Path;$env:PYTHONPATH"
  } else {
    $env:PYTHONPATH = $stage1Path
  }
}

& "C:\aqf311\.venv\Scripts\python.exe" @Args
