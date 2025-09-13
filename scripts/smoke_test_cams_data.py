#!/usr/bin/env python3
"""
Smoke test for CAMS past week data collection.

This script verifies that the collected CAMS data meets quality expectations:
- Files exist and are valid NetCDF format
- Contains expected variables (PM2.5, PM10, NO2, O3, SO2, CO)
- Has reasonable data values (not all NaN, within expected ranges)
- Covers expected time range and intervals
- Provenance files are present and complete
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging():
    """Setup logging configuration."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def find_collection_data():
    """Find the most recent CAMS past week collection data."""
    data_dir = Path("data/cams_past_week_collection")

    if not data_dir.exists():
        raise FileNotFoundError(f"Collection data directory not found: {data_dir}")

    # Find summary files
    summary_files = list(data_dir.glob("collection_summary_*.json"))
    if not summary_files:
        raise FileNotFoundError(f"No collection summary files found in {data_dir}")

    # Get the most recent summary
    latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)

    with open(latest_summary, "r") as f:
        summary = json.load(f)

    logging.info(f"Found collection summary: {latest_summary}")
    logging.info(f"Collection timestamp: {summary['collection_timestamp']}")
    logging.info(
        f"Cities: {summary['total_cities']}, Files: {summary['total_files_collected']}"
    )

    return data_dir, summary


def test_file_existence_and_format(data_dir, summary):
    """Test that files exist and are valid NetCDF format."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing file existence and format ===")

    total_files = 0
    valid_files = 0
    missing_files = 0
    invalid_files = 0

    for city_result in summary["city_results"]:
        city_name = city_result["city"]
        files = city_result["files_collected"]

        logger.info(f"Testing {len(files)} files for {city_name}")

        for file_path in files:
            total_files += 1
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                missing_files += 1
                logger.warning(f"Missing file: {file_path}")
                continue

            # Check if file is valid NetCDF
            try:
                with xr.open_dataset(file_path) as ds:
                    # Basic validation - should have data variables
                    if len(ds.data_vars) == 0:
                        invalid_files += 1
                        logger.warning(f"No data variables in: {file_path.name}")
                    else:
                        valid_files += 1

            except Exception as e:
                invalid_files += 1
                logger.error(f"Invalid NetCDF file {file_path.name}: {e}")

    # Summary
    logger.info("File format test results:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Valid files: {valid_files}")
    logger.info(f"  Missing files: {missing_files}")
    logger.info(f"  Invalid files: {invalid_files}")

    success_rate = valid_files / total_files if total_files > 0 else 0
    logger.info(f"  Success rate: {success_rate:.1%}")

    return success_rate >= 0.9  # 90% success rate threshold


def test_data_variables_and_values(data_dir, summary):
    """Test that files contain expected variables with reasonable values."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing data variables and values ===")

    expected_pollutants = [
        "particulate_matter_2.5um",
        "particulate_matter_10um",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "carbon_monoxide",
    ]

    # Define reasonable value ranges for pollutants (μg/m³)
    value_ranges = {
        "particulate_matter_2.5um": (0, 500),  # PM2.5
        "particulate_matter_10um": (0, 1000),  # PM10
        "nitrogen_dioxide": (0, 200),  # NO2
        "ozone": (0, 300),  # O3
        "sulphur_dioxide": (0, 100),  # SO2
        "carbon_monoxide": (0, 50000),  # CO (different units)
    }

    files_tested = 0
    files_with_good_data = 0
    variable_stats = {
        var: {"found": 0, "valid_values": 0} for var in expected_pollutants
    }

    # Test a sample of files from different cities
    for city_result in summary["city_results"][
        :5
    ]:  # Test first 5 cities for efficiency
        city_name = city_result["city"]
        files = city_result["files_collected"][:2]  # Test 2 files per city

        logger.info(f"Testing data quality for {city_name}")

        for file_path in files:
            files_tested += 1
            file_path = Path(file_path)

            if not file_path.exists():
                continue

            try:
                with xr.open_dataset(file_path) as ds:
                    file_has_good_data = True

                    # Check each expected variable
                    for var_name in expected_pollutants:
                        # Variable might have different names in the dataset
                        found_vars = [
                            v
                            for v in ds.data_vars
                            if var_name.replace("_", "").replace(".", "")
                            in v.replace("_", "").replace(".", "")
                        ]

                        if found_vars:
                            variable_stats[var_name]["found"] += 1

                            # Check data values
                            var_data = ds[found_vars[0]]

                            # Check for all NaN
                            if not var_data.isnull().all():
                                # Check value ranges
                                min_val, max_val = value_ranges[var_name]
                                actual_values = var_data.values.flatten()
                                valid_values = actual_values[~np.isnan(actual_values)]

                                if len(valid_values) > 0:
                                    within_range = np.all(
                                        (valid_values >= min_val)
                                        & (valid_values <= max_val)
                                    )
                                    if within_range:
                                        variable_stats[var_name]["valid_values"] += 1
                                    else:
                                        logger.warning(
                                            f"Values out of range for {var_name} in "
                                            f"{file_path.name}: min={valid_values.min():.2f}, "
                                            f"max={valid_values.max():.2f}"
                                        )
                                        file_has_good_data = False
                                else:
                                    logger.warning(
                                        f"No valid values for {var_name} in {file_path.name}"
                                    )
                                    file_has_good_data = False
                            else:
                                logger.warning(
                                    f"All NaN values for {var_name} in {file_path.name}"
                                )
                                file_has_good_data = False
                        else:
                            logger.warning(
                                f"Variable {var_name} not found in {file_path.name}"
                            )
                            file_has_good_data = False

                    if file_has_good_data:
                        files_with_good_data += 1

            except Exception as e:
                logger.error(f"Error testing {file_path.name}: {e}")

    # Summary
    logger.info("Data quality test results:")
    logger.info(f"  Files tested: {files_tested}")
    logger.info(f"  Files with good data: {files_with_good_data}")

    for var_name, stats in variable_stats.items():
        logger.info(
            f"  {var_name}: found in {stats['found']}/{files_tested} files, "
            f"valid values in {stats['valid_values']}/{stats['found']} occurrences"
        )

    data_quality_rate = files_with_good_data / files_tested if files_tested > 0 else 0
    logger.info(f"  Data quality rate: {data_quality_rate:.1%}")

    return data_quality_rate >= 0.8  # 80% data quality threshold


def test_provenance_files(data_dir, summary):
    """Test that provenance files exist and contain expected metadata."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing provenance files ===")

    total_provenance_files = 0
    valid_provenance_files = 0

    for city_result in summary["city_results"][:3]:  # Test first 3 cities
        files = city_result["files_collected"][:2]  # Test 2 files per city

        for file_path in files:
            file_path = Path(file_path)
            provenance_file = Path(str(file_path) + ".provenance.json")

            total_provenance_files += 1

            if not provenance_file.exists():
                logger.warning(f"Missing provenance file: {provenance_file.name}")
                continue

            try:
                with open(provenance_file, "r") as f:
                    provenance = json.load(f)

                # Check required fields
                required_fields = [
                    "dataset",
                    "request",
                    "api_url",
                    "sha256",
                    "size_bytes",
                ]
                has_all_fields = all(field in provenance for field in required_fields)

                if has_all_fields:
                    valid_provenance_files += 1

                    # Check dataset name
                    if provenance["dataset"] != "cams-global-reanalysis-eac4":
                        logger.warning(
                            f"Unexpected dataset in {provenance_file.name}: {provenance['dataset']}"
                        )
                else:
                    missing_fields = [f for f in required_fields if f not in provenance]
                    logger.warning(
                        f"Missing fields in {provenance_file.name}: {missing_fields}"
                    )

            except Exception as e:
                logger.error(
                    f"Error reading provenance file {provenance_file.name}: {e}"
                )

    provenance_rate = (
        valid_provenance_files / total_provenance_files
        if total_provenance_files > 0
        else 0
    )
    logger.info("Provenance test results:")
    logger.info(f"  Provenance files tested: {total_provenance_files}")
    logger.info(f"  Valid provenance files: {valid_provenance_files}")
    logger.info(f"  Provenance success rate: {provenance_rate:.1%}")

    return provenance_rate >= 0.9


def test_collection_completeness(summary):
    """Test that collection covers expected time range and cities."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing collection completeness ===")

    # Expected parameters
    expected_cities = 100
    expected_days = 8  # Past week (7 days ago to today)
    expected_times_per_day = 4  # 6-hour intervals
    expected_files_per_city = expected_days * expected_times_per_day

    actual_cities = summary["total_cities"]
    actual_success_rate = summary["overall_success_rate"]

    logger.info("Collection completeness:")
    logger.info(f"  Expected cities: {expected_cities}")
    logger.info(f"  Actual cities: {actual_cities}")
    logger.info(f"  Expected files per city: {expected_files_per_city}")
    logger.info(f"  Overall success rate: {actual_success_rate:.1%}")

    # Check individual city completeness
    cities_with_good_coverage = 0
    for city_result in summary["city_results"]:
        city_rate = city_result["success_rate"]
        if city_rate >= 0.8:  # 80% threshold per city
            cities_with_good_coverage += 1

    city_coverage_rate = (
        cities_with_good_coverage / actual_cities if actual_cities > 0 else 0
    )
    logger.info(
        f"  Cities with good coverage (≥80%): "
        f"{cities_with_good_coverage}/{actual_cities} ({city_coverage_rate:.1%})"
    )

    # Overall assessment
    completeness_ok = (
        actual_cities >= expected_cities * 0.9  # At least 90% of cities
        and actual_success_rate >= 0.7  # At least 70% overall success
        and city_coverage_rate >= 0.8  # At least 80% of cities have good coverage
    )

    return completeness_ok


def run_smoke_test():
    """Run all smoke tests."""
    logger = setup_logging()
    logger.info("Starting CAMS past week data smoke test")

    try:
        # Find collection data
        data_dir, summary = find_collection_data()

        # Run tests
        tests = [
            (
                "File existence and format",
                test_file_existence_and_format,
                data_dir,
                summary,
            ),
            (
                "Data variables and values",
                test_data_variables_and_values,
                data_dir,
                summary,
            ),
            ("Provenance files", test_provenance_files, data_dir, summary),
            ("Collection completeness", test_collection_completeness, summary),
        ]

        results = []
        for test_name, test_func, *args in tests:
            logger.info(f"\n--- Running {test_name} test ---")
            try:
                result = test_func(*args)
                results.append((test_name, result))
                status = "PASS" if result else "FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name} test failed with error: {e}")
                results.append((test_name, False))

        # Final summary
        passed_tests = sum(1 for _, result in results if result)
        total_tests = len(results)

        logger.info("\n=== SMOKE TEST SUMMARY ===")
        for test_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status} {test_name}")

        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            logger.info("🎉 All smoke tests PASSED! CAMS data collection is verified.")
            return True
        else:
            logger.warning(
                f"⚠️  {total_tests - passed_tests} smoke tests FAILED. Review the issues above."
            )
            return False

    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
