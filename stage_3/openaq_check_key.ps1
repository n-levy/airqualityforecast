# openaq_check_key.ps1
$ErrorActionPreference = "Stop"

if (-not $env:OPENAQ_API_KEY) {
  Write-Error "OPENAQ_API_KEY is not set in THIS PowerShell session."
  exit 2
}

# 1) Show length and head/tail (do NOT print the full key)
$tok  = $env:OPENAQ_API_KEY
$len  = $tok.Length
$head = if ($len -ge 4) { $tok.Substring(0,4) } else { $tok }
$tail = if ($len -ge 4) { $tok.Substring($len-4) } else { $tok }
Write-Host ("Key length: {0}, head: {1}... tail: ...{2}" -f $len, $head, $tail)

# 2) Clean common copy/paste artifacts
$clean = $tok.Trim().Trim('"').Trim("'")
if ($clean -ne $tok) {
  Write-Host "Stripped quotes/whitespace from the token for this check."
}

# 3) Call a lightweight v3 endpoint that requires a valid key
$uri = "https://api.openaq.org/v3/locations?limit=1"
$hdr = @{ "X-API-Key" = $clean }

try {
  $resp = Invoke-WebRequest -Uri $uri -Headers $hdr -TimeoutSec 60 -ErrorAction Stop
  if ($resp.StatusCode -eq 200) {
    Write-Host "OK -- OpenAQ v3 key works with X-API-Key header."
    exit 0
  } else {
    Write-Error ("Unexpected status: {0}" -f $resp.StatusCode)
    exit 3
  }
}
catch {
  # Try to extract status code and response body safely
  $code = $null
  $body = $null
  try {
    if ($_.Exception.Response) {
      $code = [int]$_.Exception.Response.StatusCode
      $stream = $_.Exception.Response.GetResponseStream()
      if ($stream) {
        $reader = New-Object System.IO.StreamReader($stream)
        $body = $reader.ReadToEnd()
        $reader.Close()
        $stream.Close()
      }
    }
  } catch {}
  if (-not $body) {
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
      $body = $_.ErrorDetails.Message
    } else {
      $body = $_.Exception.Message
    }
  }
  if (-not $code) { $code = "unknown" }
  Write-Error ("HTTP {0} -- {1}" -f $code, $body)
  exit 1
}
