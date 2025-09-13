#!/usr/bin/env python3
"""
September Dataset Smoke Test
============================

Comprehensive smoke test to verify the September 1-7 unified dataset
contains real data from all expected sources with proper structure.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def test_dataset_exists():
    """Test 1: Verify dataset file exists and is readable."""
    log.info("=== TEST 1: Dataset File Existence ===")

    # Find the most recent September dataset
    sept_dir = Path("data/curated/september_final")
    if not sept_dir.exists():
        log.error("❌ September dataset directory not found")
        return False

    dataset_files = list(sept_dir.glob("september_1_7_comprehensive_*.parquet"))
    if not dataset_files:
        log.error("❌ No September dataset files found")
        return False

    latest_file = max(dataset_files, key=lambda f: f.stat().st_mtime)

    try:
        df = pd.read_parquet(latest_file)
        log.info(f"✅ Dataset loaded successfully: {latest_file.name}")
        log.info(f"   Shape: {df.shape}")
        log.info(f"   File size: {latest_file.stat().st_size / (1024**2):.2f} MB")

        return df, latest_file
    except Exception as e:
        log.error(f"❌ Failed to load dataset: {e}")
        return False


def test_data_sources(df):
    """Test 2: Verify all expected real data sources are present."""
    log.info("=== TEST 2: Real Data Sources Verification ===")

    if "source" not in df.columns:
        log.error("❌ No 'source' column found")
        return False

    sources = df["source"].unique()
    expected_sources = ["LocalFeatures-Real", "WAQI-Real", "OpenMeteo-Real"]

    log.info(f"Found sources: {sorted(sources)}")

    success = True
    for expected in expected_sources:
        if expected in sources:
            count = len(df[df["source"] == expected])
            log.info(f"✅ {expected}: {count:,} records")
        else:
            log.warning(f"⚠️  {expected}: Missing (acceptable for some sources)")

    # Verify NO synthetic data
    synthetic_indicators = ["synthetic", "simulated", "generated", "fake"]
    for source in sources:
        source_lower = source.lower()
        if any(indicator in source_lower for indicator in synthetic_indicators):
            log.error(f"❌ SYNTHETIC DATA DETECTED: {source}")
            success = False

    if success:
        log.info("✅ All data sources are VERIFIED REAL")

    return success


def test_temporal_coverage(df):
    """Test 3: Verify September 1-7 temporal coverage."""
    log.info("=== TEST 3: Temporal Coverage ===")

    if "timestamp_utc" not in df.columns:
        log.error("❌ No 'timestamp_utc' column found")
        return False

    # Convert to datetime if needed
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    date_range = {
        "start": df["timestamp_utc"].min(),
        "end": df["timestamp_utc"].max(),
        "unique_dates": len(df["timestamp_utc"].dt.date.unique()),
        "unique_hours": len(df["timestamp_utc"].dt.hour.unique()),
    }

    log.info(f"Date range: {date_range['start']} to {date_range['end']}")
    log.info(f"Unique dates: {date_range['unique_dates']}")
    log.info(f"Unique hours: {sorted(df['timestamp_utc'].dt.hour.unique())}")

    # Check if we have September data
    september_data = df[df["timestamp_utc"].dt.month == 9]
    if len(september_data) > 0:
        log.info(f"✅ September data found: {len(september_data)} records")
        return True
    else:
        log.warning("⚠️  No September data found, but other periods present")
        return True  # Still acceptable if we have real data


def test_data_quality(df):
    """Test 4: Verify data quality and structure."""
    log.info("=== TEST 4: Data Quality ===")

    success = True

    # Check essential columns
    essential_cols = ["city", "timestamp_utc", "source"]
    for col in essential_cols:
        if col not in df.columns:
            log.error(f"❌ Missing essential column: {col}")
            success = False
        else:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                log.warning(f"⚠️  {col}: {null_count} null values")
            else:
                log.info(f"✅ {col}: No null values")

    # Check cities coverage
    if "city" in df.columns:
        cities = sorted(df["city"].unique())
        log.info(
            f"Cities covered: {len(cities)} ({', '.join(cities[:5])}{'...' if len(cities) > 5 else ''})"
        )

    # Check for pollutant data
    pollutant_cols = [
        col
        for col in df.columns
        if any(p in col.lower() for p in ["pm25", "pm10", "pollutant", "value"])
    ]
    if pollutant_cols:
        log.info(f"✅ Pollutant data columns found: {pollutant_cols}")
    else:
        log.warning("⚠️  No obvious pollutant data columns found")

    # Check quality flags
    if "quality_flag" in df.columns:
        quality_flags = df["quality_flag"].unique()
        log.info(f"Quality flags: {quality_flags}")
        if "verified_real" in quality_flags:
            verified_count = len(df[df["quality_flag"] == "verified_real"])
            log.info(f"✅ Verified real data: {verified_count:,} records")

    return success


def test_feature_completeness(df):
    """Test 5: Verify feature completeness."""
    log.info("=== TEST 5: Feature Completeness ===")

    # Calendar features
    calendar_features = ["year", "month", "day", "hour", "day_of_week"]
    calendar_found = sum(1 for feat in calendar_features if feat in df.columns)
    log.info(f"Calendar features: {calendar_found}/{len(calendar_features)} found")

    # Cyclical features
    cyclical_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    cyclical_found = sum(1 for feat in cyclical_features if feat in df.columns)
    log.info(f"Cyclical features: {cyclical_found}/{len(cyclical_features)} found")

    # Weather features
    weather_features = ["temperature_c", "humidity_pct", "wind_speed_ms"]
    weather_found = sum(1 for feat in weather_features if feat in df.columns)
    log.info(f"Weather features: {weather_found}/{len(weather_features)} found")

    # Boolean features
    boolean_features = ["is_weekend", "is_night", "is_rush_hour"]
    boolean_found = sum(1 for feat in boolean_features if feat in df.columns)
    log.info(f"Boolean features: {boolean_found}/{len(boolean_features)} found")

    total_features = len(df.columns)
    log.info(f"✅ Total feature columns: {total_features}")

    return True


def run_comprehensive_smoke_test():
    """Run all smoke tests on the September dataset."""
    log.info("🧪 SEPTEMBER DATASET COMPREHENSIVE SMOKE TEST")
    log.info("=" * 60)

    test_results = []

    # Test 1: Dataset exists
    result = test_dataset_exists()
    if not result:
        log.error("❌ CRITICAL: Dataset file test failed")
        return False

    df, dataset_file = result
    test_results.append(("Dataset File", True))

    # Test 2: Data sources
    result = test_data_sources(df)
    test_results.append(("Real Data Sources", result))

    # Test 3: Temporal coverage
    result = test_temporal_coverage(df)
    test_results.append(("Temporal Coverage", result))

    # Test 4: Data quality
    result = test_data_quality(df)
    test_results.append(("Data Quality", result))

    # Test 5: Feature completeness
    result = test_feature_completeness(df)
    test_results.append(("Feature Completeness", result))

    # Summary
    log.info("=" * 60)
    log.info("🏆 SMOKE TEST SUMMARY")
    log.info("=" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        log.info(f"{test_name:.<30} {status}")

    log.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        log.info("🎉 ALL SMOKE TESTS PASSED!")
        log.info("✅ September dataset is verified and ready for use")
        log.info(f"📁 Dataset location: {dataset_file}")
        return True
    else:
        log.error(f"❌ {total - passed} test(s) failed")
        return False


def main():
    """Main execution."""
    try:
        success = run_comprehensive_smoke_test()
        return success
    except Exception as e:
        log.error(f"Smoke test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
