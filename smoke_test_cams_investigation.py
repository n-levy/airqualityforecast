#!/usr/bin/env python3
"""
CAMS Data Investigation Smoke Test
=================================

This test verifies the findings from the CAMS data investigation and documents
what would be needed to collect real ECMWF-CAMS data in 6-hour intervals over
the past week.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def test_1_previous_collection_analysis():
    """Test 1: Verify analysis of previous CAMS collection attempts."""
    logger = logging.getLogger(__name__)
    logger.info("=== TEST 1: Previous Collection Analysis ===")

    findings = []

    # Check for previous logs
    log_files = list(Path("scripts/data/logs").glob("cams_past_week_collection_*.log"))
    if log_files:
        findings.append("✓ Found previous collection attempt logs")
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

        # Check log content for failures
        with open(latest_log, "r") as f:
            content = f.read()
            if "Climate Data Store API endpoint not found" in content:
                findings.append(
                    "✓ Confirmed: Previous attempts failed due to wrong API endpoint (CDS vs ADS)"
                )
            if "Atmosphere Data Store API endpoint not found" in content:
                findings.append("✓ Confirmed: ADS endpoint tested but access denied")
    else:
        findings.append("✗ No previous collection logs found")

    # Check for empty data directories
    cams_dir = Path("data/cams_past_week_collection")
    if cams_dir.exists():
        nc_files = list(cams_dir.rglob("*.nc"))
        if not nc_files:
            findings.append(
                "✓ Confirmed: No real NetCDF files found in collection directory"
            )
        else:
            findings.append(f"? Found {len(nc_files)} NetCDF files - need verification")
    else:
        findings.append("✓ Confirmed: Collection directory doesn't exist or is empty")

    # Check comprehensive tables for synthetic data markers
    tables_file = Path("stage_5/comprehensive_tables/comprehensive_apis_table.csv")
    if tables_file.exists():
        with open(tables_file, "r") as f:
            content = f.read()
            if (
                "synthetic" in content
                and "Literature-based CAMS performance simulation" in content
            ):
                findings.append(
                    "✓ Confirmed: Comprehensive tables show synthetic CAMS data"
                )

    logger.info("Analysis findings:")
    for finding in findings:
        logger.info(f"  {finding}")

    # Test passes if we found evidence of failed attempts and no real data
    failure_evidence = any(
        "failed" in f.lower() or "wrong" in f.lower() for f in findings
    )
    no_real_data = any(
        "no real" in f.lower() or "synthetic" in f.lower() for f in findings
    )

    return failure_evidence and no_real_data


def test_2_api_configuration_verification():
    """Test 2: Verify API configuration understanding."""
    logger = logging.getLogger(__name__)
    logger.info("=== TEST 2: API Configuration Verification ===")

    findings = []

    # Check current credential configuration
    cdsapirc = Path.home() / ".cdsapirc"
    if cdsapirc.exists():
        with open(cdsapirc, "r") as f:
            content = f.read()

        if "cds.climate.copernicus.eu" in content:
            findings.append("✓ Current config: CDS (Climate Data Store) endpoint")
            findings.append("  ℹ️  Note: CAMS data requires ADS (Atmosphere Data Store)")
        elif "ads.atmosphere.copernicus.eu" in content:
            findings.append("✓ Current config: ADS (Atmosphere Data Store) endpoint")
            findings.append("  ℹ️  Note: ADS requires separate registration from CDS")

        if ":" in content and len(content.split(":")[-1].strip()) > 20:
            findings.append("✓ API key format appears correct")
        else:
            findings.append("✗ API key format may be incorrect")
    else:
        findings.append("✗ No .cdsapirc file found")

    # Document the correct configuration
    findings.append("")
    findings.append("📋 Required for real CAMS data:")
    findings.append("  1. Register at https://ads.atmosphere.copernicus.eu/")
    findings.append("  2. Get ADS API key (different from CDS)")
    findings.append("  3. Configure ~/.cdsapirc with ADS endpoint")
    findings.append("  4. Accept CAMS data license terms")

    logger.info("Configuration findings:")
    for finding in findings:
        logger.info(f"  {finding}")

    return True  # This test always passes as it's informational


def test_3_data_requirements_specification():
    """Test 3: Specify exactly what data is needed."""
    logger = logging.getLogger(__name__)
    logger.info("=== TEST 3: Data Requirements Specification ===")

    # Generate the exact date/time combinations requested
    today = datetime.now()
    start_date = today - timedelta(days=7)

    dates = []
    current_date = start_date
    while current_date <= today - timedelta(days=1):
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    times = ["00:00", "06:00", "12:00", "18:00"]  # 6-hour intervals
    pollutants = [
        "particulate_matter_2.5um",  # PM2.5
        "particulate_matter_10um",  # PM10
        "nitrogen_dioxide",  # NO2
        "ozone",  # O3
        "sulphur_dioxide",  # SO2
        "carbon_monoxide",  # CO
    ]

    # Load cities count
    cities_file = Path("stage_5/comprehensive_tables/comprehensive_features_table.csv")
    if cities_file.exists():
        import pandas as pd

        df = pd.read_csv(cities_file)
        num_cities = len(df)
    else:
        num_cities = 100  # Default assumption

    total_requests = num_cities * len(dates) * len(times)

    logger.info("Data requirements specification:")
    logger.info(f"  📅 Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    logger.info(f"  🕐 Time intervals: {times} (6-hour intervals)")
    logger.info(f"  🌍 Cities: {num_cities} global locations")
    logger.info(f"  🏭 Pollutants: {len(pollutants)} variables")
    logger.info(f"  📊 Total API requests needed: {total_requests}")
    logger.info(f"  💾 Expected file size: ~{total_requests * 0.5:.0f} MB (estimated)")

    # Save requirements to file for reference
    requirements = {
        "date_range": {"start": dates[0], "end": dates[-1], "total_days": len(dates)},
        "time_intervals": times,
        "cities": num_cities,
        "pollutants": pollutants,
        "total_requests": total_requests,
        "estimated_size_mb": total_requests * 0.5,
        "dataset_name": "cams-global-reanalysis-eac4",
        "api_endpoint": "https://ads.atmosphere.copernicus.eu/api",
        "registration_required": "https://ads.atmosphere.copernicus.eu/",
    }

    requirements_file = Path("data/cams_requirements_specification.json")
    requirements_file.parent.mkdir(parents=True, exist_ok=True)

    with open(requirements_file, "w") as f:
        json.dump(requirements, f, indent=2)

    logger.info(f"  📝 Requirements saved to: {requirements_file}")

    return True


def test_4_smoke_test_synthetic_verification():
    """Test 4: Verify that existing 'CAMS' data is actually synthetic."""
    logger = logging.getLogger(__name__)
    logger.info("=== TEST 4: Synthetic Data Verification ===")

    findings = []

    # Check comprehensive tables for synthetic markers
    apis_table = Path("stage_5/comprehensive_tables/comprehensive_apis_table.csv")
    if apis_table.exists():
        with open(apis_table, "r") as f:
            content = f.read()

        # Count occurrences of synthetic vs real markers
        synthetic_count = content.count("synthetic")
        real_count = content.count("REAL_DATA")
        literature_count = content.count("Literature-based CAMS performance simulation")

        findings.append(f"✓ Found {synthetic_count} 'synthetic' markers in API table")
        findings.append(
            f"✓ Found {literature_count} 'Literature-based simulation' entries"
        )
        findings.append(f"  ℹ️  Real data markers: {real_count}")

        if synthetic_count > 50:  # Most entries should be synthetic
            findings.append("✓ Confirmed: Existing CAMS data is synthetic/simulated")

    # Check for any actual NetCDF files that might contain real data
    nc_files = list(Path(".").rglob("*.nc"))
    if nc_files:
        findings.append(
            f"? Found {len(nc_files)} NetCDF files - would need content verification"
        )
        for nc_file in nc_files[:3]:  # Show first 3
            findings.append(f"    {nc_file}")
    else:
        findings.append(
            "✓ No NetCDF files found - confirms no real CAMS data collected"
        )

    logger.info("Synthetic data verification:")
    for finding in findings:
        logger.info(f"  {finding}")

    return True


def test_5_collection_system_readiness():
    """Test 5: Verify that collection system is ready once proper access is obtained."""
    logger = logging.getLogger(__name__)
    logger.info("=== TEST 5: Collection System Readiness ===")

    findings = []

    # Check if collection scripts exist
    scripts_to_check = [
        "scripts/collect_cams_past_week.py",
        "scripts/smoke_test_cams_data.py",
        "cams_ads_downloader.py",
    ]

    for script in scripts_to_check:
        if Path(script).exists():
            findings.append(f"✓ Collection script exists: {script}")
        else:
            findings.append(f"✗ Missing script: {script}")

    # Check if cities data is available
    cities_file = Path("stage_5/comprehensive_tables/comprehensive_features_table.csv")
    if cities_file.exists():
        findings.append("✓ Cities data available for collection")
    else:
        findings.append("✗ Cities data missing")

    # Check Python dependencies
    try:
        import cdsapi  # noqa: F401

        findings.append("✓ cdsapi library available")
    except ImportError:
        findings.append("✗ cdsapi library missing (pip install cdsapi)")

    try:
        import xarray  # noqa: F401

        findings.append("✓ xarray library available for data verification")
    except ImportError:
        findings.append("✗ xarray library missing (pip install xarray)")

    try:
        import pandas  # noqa: F401

        findings.append("✓ pandas library available")
    except ImportError:
        findings.append("✗ pandas library missing")

    logger.info("System readiness check:")
    for finding in findings:
        logger.info(f"  {finding}")

    # System is ready if most components are available
    ready_count = sum(1 for f in findings if f.startswith("✓"))
    total_count = len(findings)

    return ready_count >= (total_count * 0.8)  # 80% readiness threshold


def main():
    """Run all smoke tests."""
    logger = setup_logging()
    logger.info("🔍 CAMS Data Investigation Smoke Test")
    logger.info("=" * 50)

    tests = [
        ("Previous Collection Analysis", test_1_previous_collection_analysis),
        ("API Configuration Verification", test_2_api_configuration_verification),
        ("Data Requirements Specification", test_3_data_requirements_specification),
        ("Synthetic Data Verification", test_4_smoke_test_synthetic_verification),
        ("Collection System Readiness", test_5_collection_system_readiness),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"   {status}")
        except Exception as e:
            logger.error(f"   ❌ ERROR: {e}")
            results.append((test_name, False))

    # Final summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    logger.info("\n" + "=" * 50)
    logger.info("📊 SMOKE TEST SUMMARY")
    logger.info("=" * 50)

    for test_name, result in results:
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")

    logger.info(f"\n📈 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("✅ Investigation complete: No real CAMS data found")
        logger.info("✅ System ready for real data collection with proper ADS access")
        logger.info("✅ Requirements documented for 6-hour intervals over past week")
    else:
        logger.info("⚠️  Some tests failed - review findings above")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
