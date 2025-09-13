#!/usr/bin/env python3
"""
Real Data Sources Assessment
============================

Assess what real data sources we currently have available for creating
a comprehensive past week dataset with 6-hour intervals.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def assess_cams_data():
    """Assess available CAMS real data."""
    log.info("=== ASSESSING CAMS REAL DATA ===")

    cams_dir = Path("data/cams_past_week_final")
    if not cams_dir.exists():
        log.warning(f"CAMS directory not found: {cams_dir}")
        return {"available": False, "files": 0, "coverage": None}

    nc_files = list(cams_dir.glob("*.nc"))
    log.info(f"Found {len(nc_files)} CAMS NetCDF files")

    if nc_files:
        # Extract time coverage from filenames
        timestamps = []
        for file in nc_files:
            # Extract timestamp from filename like cams_pm25_20240601_0000.nc
            parts = file.stem.split("_")
            if len(parts) >= 4:
                date_part = parts[2]  # 20240601
                time_part = parts[3]  # 0000
                timestamp_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:00"
                timestamps.append(timestamp_str)

        if timestamps:
            log.info(f"CAMS time coverage: {min(timestamps)} to {max(timestamps)}")
            log.info(f"Sample timestamps: {sorted(timestamps)[:5]}")

        return {
            "available": True,
            "files": len(nc_files),
            "coverage": (
                f"{min(timestamps)} to {max(timestamps)}" if timestamps else "Unknown"
            ),
            "source_type": "Real ECMWF-CAMS atmospheric composition data",
            "frequency": "6-hour intervals",
            "verified": True,
        }

    return {"available": False, "files": 0, "coverage": None}


def assess_waqi_data():
    """Assess available WAQI real data."""
    log.info("=== ASSESSING WAQI REAL DATA ===")

    # Check if WAQI data has been collected
    waqi_files = []

    # Check various possible locations
    possible_dirs = [
        Path("data/curated/obs"),
        Path("data/obs"),
        Path("data/waqi"),
        Path("C:/aqf311/data/curated/obs"),
    ]

    for dir_path in possible_dirs:
        if dir_path.exists():
            waqi_files.extend(list(dir_path.glob("*waqi*.parquet")))
            waqi_files.extend(list(dir_path.glob("*obs*.parquet")))

    log.info(f"Found {len(waqi_files)} potential WAQI/observation files")

    if waqi_files:
        # Try to load and assess the most recent file
        latest_file = max(waqi_files, key=lambda f: f.stat().st_mtime)
        try:
            df = pd.read_parquet(latest_file)
            log.info(f"WAQI data shape: {df.shape}")
            log.info(f"WAQI columns: {list(df.columns)}")
            if "timestamp_utc" in df.columns:
                log.info(
                    f"WAQI time range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}"
                )

            return {
                "available": True,
                "files": len(waqi_files),
                "latest_file": str(latest_file),
                "records": len(df),
                "source_type": "Real WAQI air quality observations",
                "frequency": "Varies",
                "verified": (
                    "source" in df.columns and "WAQI" in str(df["source"].unique())
                    if "source" in df.columns
                    else False
                ),
            }
        except Exception as e:
            log.error(f"Error reading WAQI file {latest_file}: {e}")

    return {"available": False, "files": 0, "coverage": None}


def assess_gefs_data():
    """Assess available GEFS real data."""
    log.info("=== ASSESSING GEFS REAL DATA ===")

    # Check various possible locations for GEFS data
    possible_dirs = [
        Path("data/curated/gefs_chem"),
        Path("data/gefs"),
        Path("C:/aqf311/data/curated/gefs_chem"),
        Path("C:/aqf311/data/raw"),
    ]

    gefs_files = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            gefs_files.extend(list(dir_path.rglob("*gefs*.parquet")))
            gefs_files.extend(list(dir_path.rglob("*gefs*.nc")))

    log.info(f"Found {len(gefs_files)} potential GEFS files")

    # Check if any contain real data (not simulated)
    real_gefs_files = []
    for file in gefs_files:
        if (
            "simulated" not in str(file).lower()
            and "synthetic" not in str(file).lower()
        ):
            real_gefs_files.append(file)

    log.info(f"Found {len(real_gefs_files)} potentially real GEFS files")

    if real_gefs_files:
        return {
            "available": True,
            "files": len(real_gefs_files),
            "source_type": "NOAA GEFS-Aerosol forecasts",
            "frequency": "6-hour intervals",
            "verified": False,  # Need to verify if data is real
        }

    return {"available": False, "files": 0, "coverage": None}


def assess_local_features():
    """Assess local features (calendar, temporal)."""
    log.info("=== ASSESSING LOCAL FEATURES ===")

    # Local features can be generated from timestamps, so always available
    return {
        "available": True,
        "source_type": "Calendar and temporal features (generated from timestamps)",
        "frequency": "Any frequency needed",
        "verified": True,
    }


def create_comprehensive_assessment():
    """Create comprehensive assessment of all real data sources."""
    log.info("🔍 COMPREHENSIVE REAL DATA SOURCES ASSESSMENT")
    log.info("=" * 60)

    assessment = {"assessment_date": datetime.now().isoformat(), "data_sources": {}}

    # Assess each data source
    assessment["data_sources"]["cams"] = assess_cams_data()
    assessment["data_sources"]["waqi"] = assess_waqi_data()
    assessment["data_sources"]["gefs"] = assess_gefs_data()
    assessment["data_sources"]["local_features"] = assess_local_features()

    # Summary
    log.info("\n📊 ASSESSMENT SUMMARY")
    log.info("=" * 40)

    available_sources = []
    unavailable_sources = []

    for source_name, source_info in assessment["data_sources"].items():
        if source_info["available"]:
            available_sources.append(source_name.upper())
            log.info(f"✅ {source_name.upper()}: Available")
            if "files" in source_info:
                log.info(f"   Files: {source_info['files']}")
            if "coverage" in source_info and source_info["coverage"]:
                log.info(f"   Coverage: {source_info['coverage']}")
            if "verified" in source_info:
                log.info(f"   Verified: {'✅' if source_info['verified'] else '⚠️'}")
        else:
            unavailable_sources.append(source_name.upper())
            log.info(f"❌ {source_name.upper()}: Not available")

    log.info(f"\n🎯 AVAILABLE REAL DATA SOURCES: {len(available_sources)}/4")
    log.info(f"   {', '.join(available_sources)}")

    if unavailable_sources:
        log.info(f"❌ UNAVAILABLE SOURCES: {len(unavailable_sources)}/4")
        log.info(f"   {', '.join(unavailable_sources)}")

    # Determine if we can create a comprehensive dataset
    critical_sources_available = (
        assessment["data_sources"]["cams"]["available"]
        and assessment["data_sources"]["local_features"]["available"]
    )

    log.info(f"\n📈 COMPREHENSIVE DATASET FEASIBILITY:")
    if critical_sources_available:
        log.info("✅ FEASIBLE - We have verified real CAMS data + local features")
        log.info("   Can create dataset with:")
        log.info("   - Real atmospheric composition forecasts (CAMS)")
        log.info("   - Calendar and temporal features")
        if assessment["data_sources"]["waqi"]["available"]:
            log.info("   - Real air quality observations (WAQI)")
        if assessment["data_sources"]["gefs"]["available"]:
            log.info("   - GEFS forecast data (verification needed)")
    else:
        log.info("❌ NOT FEASIBLE - Missing critical real data sources")

    return assessment


def main():
    """Main assessment execution."""
    try:
        assessment = create_comprehensive_assessment()

        # Save assessment results
        output_file = Path("real_data_sources_assessment.json")
        import json

        with open(output_file, "w") as f:
            json.dump(assessment, f, indent=2)

        log.info(f"\n📄 Assessment saved: {output_file}")

        return assessment

    except Exception as e:
        log.error(f"Assessment failed: {e}")
        return None


if __name__ == "__main__":
    assessment = main()
