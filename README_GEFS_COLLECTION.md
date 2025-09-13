# NOAA GEFS-Aerosols Data Collection System

Universal Python system for downloading and processing two years of NOAA GEFS-Aerosols pollutant data (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃) via HTTPS only. Runs on any laptop, server, or cloud instance with Python 3.8+.

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Or check environment and install interactively
python scripts/setup_environment.py
```

### 2. Smoke Test (1 day, single cycle)
```bash
python scripts/orchestrate_gefs_https.py \
  --start-date 2025-09-12 \
  --end-date 2025-09-12 \
  --cycles 00 \
  --fhours 0:6:6 \
  --bbox="-180,-60,180,85"
```

**Note**: Use `--bbox="coordinates"` format with equals sign for proper argument parsing with negative coordinates.

### 3. Full Collection (2 years)
```bash
python scripts/orchestrate_gefs_https.py \
  --start-date 2023-09-01 \
  --end-date 2025-09-01 \
  --cycles 00,12 \
  --fhours 0:6:48 \
  --bbox="-180,-60,180,85" \
  --pollutants PM25,PM10,NO2,SO2,CO,O3 \
  --workers 4 \
  --force
```

### 4. Using the 100-City Collection Script
```bash
python scripts/collect_2year_gefs_data.py \
  --start-date 2023-09-01 \
  --end-date 2025-09-01 \
  --data-root /path/to/data \
  --chunk-months 3
```

## Recent Updates (September 2025)

### ✅ **Fixed NOAA GEFS Data Collection**
The bbox argument parsing issue has been resolved. The system now successfully:
- Downloads real GEFS-Aerosol GRIB2 files from NOAA's S3 bucket
- Handles negative longitude coordinates properly using `--bbox="coordinates"` format
- Supports global coverage with bbox `"-180,-60,180,85"`
- Provides comprehensive error handling and retry logic

### ✅ **Verified Real Data Access**
- **NOAA GEFS-PDS S3 Bucket**: Confirmed accessible via HTTPS
- **File Sizes**: 24+ MB GRIB2 files download successfully
- **Data Content**: Contains real atmospheric pollutant forecasts (PMTF=PM2.5, PMTC=PM10)
- **Global Coverage**: 721×1440 grid points at 0.25° resolution
- **Temporal Range**: Current operational forecasts available

### 🔧 **Known Issues**
- GRIB extraction may encounter ECCODES decoding issues on some systems
- Download functionality is fully operational and tested
- Alternative GRIB processing approaches may be needed for some environments

## System Architecture

### Components
- **`orchestrate_gefs_https.py`** - Main orchestrator script
- **`download_gefs_https.py`** - HTTPS downloader with resume capability
- **`extract_gefs_pollutants.py`** - GRIB2 to Parquet converter
- **`setup_environment.py`** - Environment validator and setup helper
- **`requirements.txt`** - Python dependencies

### Data Flow
```
NOAA S3 Bucket → Raw GRIB2 → Extraction → Partitioned Parquet
```

### Directory Structure
```
{DATA_ROOT}/
├── raw/gefs_chem/
│   ├── YYYYMMDD/HH/gefs.chem.tHHz.a2d_0p25.fFFF.grib2
│   └── _manifests/download_manifest.csv
└── curated/gefs_chem/parquet/
    ├── run_date=YYYY-MM-DD/run_hour=HH/f_hour=FFF/part-*.parquet
    └── extract_manifest.csv
```

## Parameters

### Geographic
- `--bbox`: Longitude/latitude bounds in format `--bbox="lon_min,lat_min,lon_max,lat_max"`
  - Use equals sign format for negative coordinates: `--bbox="-180,-60,180,85"`
  - Default: `5,16,47,56` for Germany, Global: `-180,-60,180,85`
- `--pollutants`: Comma-separated list (default: `PM25,PM10,NO2,SO2,CO,O3`)

### Temporal
- `--start-date`: Start date `YYYY-MM-DD`
- `--end-date`: End date `YYYY-MM-DD`
- `--cycles`: Forecast cycles `00,06,12,18` (default: `00,12`)
- `--fhours`: Forecast hours `start:step:end` or comma-separated (default: `0:6:120`)

### System
- `--data-root`: Data directory (default: `$DATA_ROOT` or `~/gefs_data`)
- `--workers`: Parallel extraction threads (default: 4)
- `--max-planned-files`: Safety limit (default: 50000)
- `--force`: Override safety limits
- `--dry-run`: Show what would be done
- `--verbose`: Detailed output

## Data Sources

**HTTPS URL Pattern:**
```
https://noaa-gefs-pds.s3.amazonaws.com/gefs.YYYYMMDD/HH/chem/pgrb2ap25/gefs.chem.tHHz.a2d_0p25.fFFF.grib2
```

**Available Data:**
- **Cycles**: 00, 06, 12, 18 UTC
- **Forecast Hours**: 000-384 (every 3 hours)
- **Pollutants**: PM2.5, PM10, NO2, SO2, CO, O3
- **Resolution**: 0.25° global grid
- **Coverage**: Global with geographic subsetting

## Output Format

### Parquet Schema
```
run_date: string     # YYYY-MM-DD
run_hour: string     # HH
f_hour: string       # FFF
lat: float64         # Latitude
lon: float64         # Longitude
pollutant: string    # PM25, PM10, etc.
value: float64       # Concentration (μg/m³ or ppm)
```

### Partitioning
Files are partitioned by `run_date=YYYY-MM-DD/run_hour=HH/f_hour=FFF/` for efficient querying.

## Features

### Robust Download
- **Resume capability**: Partial downloads continue from where they left off
- **Retry logic**: Automatic retry with exponential backoff
- **404 handling**: Missing files logged but not fatal
- **Manifest tracking**: Complete download history with HTTP codes and file sizes

### GRIB Processing
- **Fuzzy variable matching**: Handles different GRIB variable naming conventions
- **Surface level preference**: Automatically selects surface-level data
- **Geographic subsetting**: Clips to specified bounding box
- **Parallel processing**: Configurable worker threads

### Universal Compatibility
- **Pure Python**: No OS-specific dependencies or shell tools required
- **Cross-platform paths**: Uses pathlib for universal file operations
- **Console safe**: ASCII-only output for all terminal types
- **Tested on**: Linux, macOS, Windows servers and laptops

## Troubleshooting

### Missing Dependencies
```bash
# Install individual packages
pip install requests pandas xarray cfgrib pyarrow

# Or use requirements file
pip install -r requirements.txt
```

### ECCODES Issues
On some systems, eccodes may need special installation:
```bash
# conda (if available)
conda install -c conda-forge eccodes
pip install cfgrib xarray pandas pyarrow

# Or use pre-compiled wheels
pip install --only-binary=all eccodes cfgrib
```

### Permission/Network Issues
```bash
# Install to user directory (any OS)
pip install --user -r requirements.txt

# Behind corporate firewall
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Server without internet access
# 1. Download wheels on connected machine: pip download -r requirements.txt
# 2. Transfer wheel files to server
# 3. Install offline: pip install --no-index --find-links . -r requirements.txt
```

### Disk Space
- **Two years of data**: ~50-100 GB (raw GRIB2)
- **Processed Parquet**: ~10-20 GB (compressed)
- **Temporary space**: ~5-10 GB during processing

## Examples

### Custom Region (Japan)
```bash
python scripts/orchestrate_gefs_https.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --bbox 129,146,30,46 \
  --cycles 00,12
```

### Single Pollutant
```bash
python scripts/orchestrate_gefs_https.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --pollutants PM25 \
  --fhours 0:12:48
```

### High-Frequency Collection
```bash
python scripts/orchestrate_gefs_https.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-01 \
  --cycles 00,06,12,18 \
  --fhours 0:3:72 \
  --workers 8
```

## System Requirements

- **Python**: 3.8+ (any distribution: CPython, conda, system package)
- **Memory**: 4GB+ recommended for processing
- **Disk**: 100GB+ for full two-year collection
- **Network**: Stable internet for downloads (resumable if interrupted)
- **OS**: Any - Linux servers, macOS laptops, Windows workstations, cloud instances
- **Privileges**: No admin/root required (installs to user space)

## Environment Variables

- `DATA_ROOT`: Override default data directory
- `HTTP_PROXY`/`HTTPS_PROXY`: For corporate proxies (requests library will use automatically)

## License

This system accesses public NOAA data. Follow NOAA data usage policies.
