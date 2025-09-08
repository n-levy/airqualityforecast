# tests\test_openaq_auth.ps1
# Verifies that OPENAQ_API_KEY works against /v3/locations with a geo query.
# Uses UriBuilder + HttpUtility to avoid special-character parsing issues.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- 0) Pre-flight checks ---
if (-not $env:OPENAQ_API_KEY) {
  throw "OPENAQ_API_KEY is not set in this shell. Run:`n  `$env:OPENAQ_API_KEY = '<YOUR_TOKEN>'"
}

# Optional: trim accidental quotes/spaces from the value
$env:OPENAQ_API_KEY = $env:OPENAQ_API_KEY.Trim('"').Trim()

# --- 1) Query parameters ---
$coords = '52.5200,13.4050'   # Berlin
$radius = 15000               # meters
$limit  = 5
$page   = 1

# --- 2) Build a safe URI (no raw '&' in the string) ---
Add-Type -AssemblyName System.Web
$qb = [System.Web.HttpUtility]::ParseQueryString([string]::Empty)
$qb['coordinates'] = $coords
$qb['radius']      = $radius
$qb['limit']       = $limit
$qb['page']        = $page

$ub = [System.UriBuilder]::new('https://api.openaq.org/v3/locations')
$ub.Query = $qb.ToString()
$uri = $ub.Uri.AbsoluteUri

# --- 3) Call the API with the auth header ---
$headers = @{ 'X-API-Key' = $env:OPENAQ_API_KEY }

try {
  $resp = Invoke-RestMethod -Method GET -Uri $uri -Headers $headers -TimeoutSec 60
  "Status: OK"
  "Results returned: $($resp.results.Count)"
  if ($resp.results.Count -gt 0) {
    "First location id: $($resp.results[0].id)"
  } else {
    "No locations found (try increasing radius)."
  }
}
catch {
  "Request failed:"
  $_ | Out-String
  if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__) {
    "HTTP code: $([int]$_.Exception.Response.StatusCode)"
  }
  throw "OpenAQ auth smoke test failed."
}
