<#
  Stage 2 Orchestrator with Progress Bar
  --------------------------------------
  - One progress bar for the whole pipeline (Fetch -> Validate -> Process)
  - Runs each Python step as a background job so progress updates smoothly
  - Streams stdout/stderr to the stage log via Tee-Object
  - Stops on first failure with a clear error
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('INFO','WARN','ERROR')][string]$Level = 'INFO',
        [string]$LogFile
    )
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "$ts | $Level | $Message"
    Write-Host $line
    if ($LogFile) { Add-Content -Path $LogFile -Value $line -Encoding UTF8 }
}

function Invoke-StepWithProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$LogFile,
        [Parameter(Mandatory = $true)][int]$OverallStartPercent,
        [Parameter(Mandatory = $true)][int]$OverallEndPercent
    )

    Write-Log -Message ("--- " + $Title + " ---`nCommand: " + $PythonExe + " " + ($Args -join ' ')) -LogFile $LogFile

    $exitFile = Join-Path $env:TEMP ("stage2_exit_" + ([guid]::NewGuid().ToString("N")) + ".txt")
    if (Test-Path $exitFile) { Remove-Item $exitFile -Force -ErrorAction SilentlyContinue }

    $scriptBlock = {
        param($py, $pyArgs, $logFilePath, $exitFilePath)
        & $py @pyArgs 2>&1 | Tee-Object -FilePath $logFilePath -Append
        Set-Content -Path $exitFilePath -Value $LASTEXITCODE -Encoding ASCII
    }

    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList $PythonExe, $Args, $LogFile, $exitFile

    $spinner = @('|','/','-','\')
    $tick = 0
    $stageWidth = [Math]::Max(1, ($OverallEndPercent - $OverallStartPercent))
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    while ($job.State -eq 'Running') {
        $frame = $spinner[$tick % $spinner.Count]
        $soft = ($tick % $stageWidth)
        $percent = [Math]::Min($OverallStartPercent + $soft, $OverallEndPercent - 1)
        $elapsed = $stopwatch.Elapsed.ToString("hh\:mm\:ss")

        Write-Progress -Activity "Stage 2 Pipeline" -Status ($Title + "  " + $frame + "  (elapsed: " + $elapsed + ")") -PercentComplete $percent
        Start-Sleep -Milliseconds 300
        $tick = $tick + 1
    }

    Receive-Job $job | Out-Host
    Remove-Job $job -Force | Out-Null

    $exitCode = 1
    if (Test-Path $exitFile) {
        $raw = Get-Content -Path $exitFile -ErrorAction SilentlyContinue
        $parsed = 0
        if ([int]::TryParse($raw, [ref]$parsed)) { $exitCode = $parsed }
        Remove-Item $exitFile -Force -ErrorAction SilentlyContinue
    }

    Write-Progress -Activity "Stage 2 Pipeline" -Status ($Title + "  ✔") -PercentComplete $OverallEndPercent

    if ($exitCode -ne 0) {
        Write-Log -Message ($Title + " failed with exit code " + $exitCode) -Level ERROR -LogFile $LogFile
        throw ($Title + " failed. See log: " + $LogFile)
    } else {
        Write-Log -Message ($Title + " completed successfully.") -LogFile $LogFile
    }
}

# --------------------- Main ---------------------

# $PSScriptRoot is the folder of this running script
$root = $PSScriptRoot
if (-not $root) { throw "PSScriptRoot is empty. Save and run this script from a .ps1 file." }
Set-Location $root

$logDir       = Join-Path $root "logs"
$dataDir      = Join-Path $root "data"
$rawDir       = Join-Path $dataDir "raw"
$interimDir   = Join-Path $dataDir "interim"
$processedDir = Join-Path $dataDir "processed"

# Ensure directories exist
[void](New-Item -ItemType Directory -Path $logDir       -Force)
[void](New-Item -ItemType Directory -Path $rawDir       -Force)
[void](New-Item -ItemType Directory -Path $interimDir   -Force)
[void](New-Item -ItemType Directory -Path $processedDir -Force)

$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$logFile = Join-Path $logDir ("stage2_" + $ts + ".log")

$pythonExe = Join-Path $root ".venv_stage2\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw ("Python venv not found: " + $pythonExe + ". Run setup_stage2.ps1 first.")
}

$configPath = Join-Path $root "config\data_sources.yaml"
if (-not (Test-Path $configPath)) {
    throw ("Config not found: " + $configPath)
}

Write-Log -Message "=== Stage 2: Starting pipeline (fetch -> validate -> process) ===" -LogFile $logFile

# Progress allocation across steps (0..100)
$fetchStart = 0;  $fetchEnd = 60
$valStart   = 60; $valEnd   = 80
$procStart  = 80; $procEnd  = 100

# --- Fetch ---
Invoke-StepWithProgress -Title "Fetch raw data" `
    -PythonExe $pythonExe `
    -Args @("$root\scripts\fetch_data.py", "--config", $configPath) `
    -LogFile $logFile `
    -OverallStartPercent $fetchStart `
    -OverallEndPercent $fetchEnd

# --- Validate ---
Invoke-StepWithProgress -Title "Validate raw data" `
    -PythonExe $pythonExe `
    -Args @("$root\scripts\validate_data.py", "--config", $configPath) `
    -LogFile $logFile `
    -OverallStartPercent $valStart `
    -OverallEndPercent $valEnd

# Confirm validated file exists before processing
$validatedCsv = Join-Path $interimDir "validated_air_quality.csv"
if (-not (Test-Path $validatedCsv)) {
    Write-Log -Message ("Validated file not found: " + $validatedCsv) -Level ERROR -LogFile $logFile
    throw "Validated CSV missing. Validation step may have failed. See: $logFile"
}

# --- Process ---
Invoke-StepWithProgress -Title "Process data" `
    -PythonExe $pythonExe `
    -Args @("$root\scripts\process_data.py", "--config", $configPath) `
    -LogFile $logFile `
    -OverallStartPercent $procStart `
    -OverallEndPercent $procEnd

Write-Progress -Activity "Stage 2 Pipeline" -Completed -Status "All steps completed"
Write-Log -Message "=== Stage 2: Pipeline finished successfully ===" -LogFile $logFile
Write-Log -Message ("Log: " + $logFile) -LogFile $logFile
