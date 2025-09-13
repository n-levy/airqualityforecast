# AUDIT\_TASK — Prove 100% Real, Complete Data (100 Cities)

Role: Senior ML infra engineer. Produce verifiable proof that data collection is complete, authentic (no synthetic), and covers ALL 100 cities for ALL required pollutants (PM2.5, PM10, NO2, SO2, CO, O3) over last two years for sources: CAMS, NOAA GEFS-Aerosol, ground truth, local features.

## Step 1 — Coverage Matrix \& Manifests

* Create ${DATA\_ROOT}/audit/coverage/coverage\_matrix.parquet and CSV with: city, pollutant, source (cams|noaa|truth|features), timestamp\_utc (final frequency), has\_data(bool), rows, first\_ts, last\_ts. Include a summary sheet with % completeness per (city,pollutant,source).
* Raw download manifests: ${DATA\_ROOT}/audit/manifests/{cams,noaa,truth}/downloads.csv with columns: url,http\_status,bytes,sha256,sha256\_ok,saved\_path,ts\_utc.
* Extract manifests: ${DATA\_ROOT}/audit/manifests/{cams,noaa,truth}/extracts.csv with columns: path\_in,path\_out,pollutants\_found,rows\_out,bbox,sha256\_in\_verified,ts\_utc,status,error.

Acceptance: coverage must show 100% TRUE has\_data for all (city×pollutant×source); otherwise fail with a gap report.

## Step 2 — Authenticity (No Synthetic)

* Recompute sha256 for all raw files. Re-download a random ≥3% sample (min 50 files) via HTTPS; hashes must match.
* Reject any pipeline branch that fabricates rows (no imputation/fill). If fabrication detected → fail.
* Record exact variable IDs: GRIB shortNames per pollutant per source. Save to ${DATA\_ROOT}/audit/dictionaries/variables\_map.json.

Acceptance: AUTHENTICITY\_REPORT.md summarizing sample re-download results (100% match) and listing shortNames per pollutant per source.

## Step 3 — Completeness (No Skips)

* Build ${DATA\_ROOT}/audit/gaps/gap\_report.parquet and CSV: city,pollutant,source,missing\_from\_ts,missing\_to\_ts,expected\_count,actual\_count. If any row exists → non-zero exit.
* Expected vs actual counts computed from final time grid (highest frequency of ground truth).
* Random 100-key spot checks per source: for each sampled (city,timestamp) compare curated Parquet value to raw file decode; write ${DATA\_ROOT}/audit/spots/spot\_diffs.csv (must be empty).

Acceptance: gap\_report.csv empty, spot\_diffs.csv empty.

## Step 4 — Frequency, Units, Schema

* Final unified dataset frequency = highest frequency of ground truth. No interpolation; align forecasts to this grid.
* Units: PM in µg/m³; gases in ppb (or project standard). Log original vs final in ${DATA\_ROOT}/audit/units/units\_audit.csv.
* Validate final and curated layers against config/schemas/\*.json. Log schema validation pass.

Acceptance: schema pass, no unknown units,
requency\_log.md documenting cadence/origins.

## Step 5 — Reproducibility

* Create scripts/verify\_all.py (cross-platform). It re-builds coverage matrix, gap report, spot checks, checksum sample; exits non-zero on any failure; prints OK summary.
* Add Makefile: make verify → runs verifier; make audit-artifacts → zips ${DATA\_ROOT}/audit.
* Add CI (GitHub Actions) nightly job that runs a small randomized subset: make verify SUBSET=1.

Acceptance: make verify returns 0 and prints success counts.

## Step 6 — Human Reports

* AUTHENTICITY\_REPORT.md — checksums, re-download sample stats, shortName map.
* COMPLETENESS\_REPORT.md — coverage heatmaps/tables, totals, zero gaps confirmation.
* SUMMARY.md — storage size, date range, frequency, pollutants, cities, raw file counts, curated rows.

## Step 7 — Docs \& Push

* Update docs (PROVIDERS.md, CONTEXT.md, ROADMAP.md, NFRs.md) with “Data Authenticity \& Completeness” and links to audit artifacts.
* Commit and push.

## Non-Negotiable Constraints

* No synthetic/simulated data. If a source is missing, log and fail.
* Cross-platform only (Linux/macOS/Windows). Avoid PowerShell-only features.
* All outputs Parquet via fsspec; manifests/audit under ${DATA\_ROOT}/audit/\*\*.
* If data exists, don’t overwrite; validate and add missing pieces only.

## Acceptance Criteria (ALL must pass)

1. Coverage matrix = 100% complete for all (city×pollutant×source) across two years.
2. gap\_report.csv empty.
3. Spot checks: 0 mismatches (spot\_diffs.csv empty).
4. Authenticity re-downloads: 100% hash matches.
5. Schema validation pass; units audit has no unknowns.
6. make verify exit code 0 with success summary.
7. Reports present and committed; CI nightly spot-verify added.
