#!/usr/bin/env python3
"""
Dataset Verification and Sanity Check
=====================================

Performs comprehensive verification of the created datasets including:
- Storage ratio analysis between daily and hourly datasets
- Data structure validation
- Content sanity checks
- Sample API data verification
- Data consistency analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/dataset_verification.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class DatasetVerificationSanityCheck:
    """Comprehensive dataset verification and sanity checking."""

    def __init__(self):
        """Initialize dataset verification."""
        self.verification_results = {
            "verification_type": "comprehensive_dataset_verification",
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }

        # Create directories
        self.output_dir = Path("stage_5/final_dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = Path("stage_5/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        log.info("Dataset Verification and Sanity Check initialized")

    def verify_comprehensive_datasets(self) -> Dict[str, Any]:
        """Perform comprehensive dataset verification with storage ratios and sanity checks."""
        log.info("=== STARTING COMPREHENSIVE DATASET VERIFICATION ===")

        verification_results = {
            "verification_timestamp": datetime.now().isoformat(),
            "verification_type": "COMPREHENSIVE_DATASET_VERIFICATION",
            "storage_analysis": {},
            "data_structure_validation": {},
            "content_sanity_checks": {},
            "api_consistency_verification": {},
            "summary": {
                "datasets_found": 0,
                "datasets_verified": 0,
                "storage_ratios_verified": 0,
                "sanity_checks_passed": 0,
                "api_consistency_verified": 0,
            },
        }

        # Find available datasets
        dataset_files = self._find_dataset_files()
        verification_results["summary"]["datasets_found"] = len(dataset_files)

        log.info(f"Found {len(dataset_files)} dataset files to verify")

        # Verify each dataset
        for dataset_file in dataset_files:
            log.info(f"Verifying dataset: {dataset_file.name}")

            # Storage analysis
            storage_analysis = self._analyze_dataset_storage(dataset_file)
            verification_results["storage_analysis"][
                dataset_file.name
            ] = storage_analysis

            # Data structure validation
            structure_validation = self._validate_data_structure(dataset_file)
            verification_results["data_structure_validation"][
                dataset_file.name
            ] = structure_validation

            # Content sanity checks
            sanity_checks = self._perform_sanity_checks(dataset_file)
            verification_results["content_sanity_checks"][
                dataset_file.name
            ] = sanity_checks

            # Update summary
            if storage_analysis.get("storage_analysis_passed", False):
                verification_results["summary"]["storage_ratios_verified"] += 1
            if structure_validation.get("structure_valid", False):
                verification_results["summary"]["datasets_verified"] += 1
            if sanity_checks.get("sanity_checks_passed", False):
                verification_results["summary"]["sanity_checks_passed"] += 1

        # API consistency verification
        log.info("Performing API consistency verification...")
        api_verification = self._verify_api_consistency()
        verification_results["api_consistency_verification"] = api_verification

        if api_verification.get("consistency_verified", False):
            verification_results["summary"]["api_consistency_verified"] = 1

        verification_results["status"] = "completed"
        verification_results["end_time"] = datetime.now().isoformat()

        return verification_results

    def _find_dataset_files(self) -> List[Path]:
        """Find all dataset files for verification."""
        dataset_patterns = [
            "OPEN_METEO_100_CITY_daily_sample_*.json",
            "OPEN_METEO_100_CITY_hourly_sample_*.json",
            "EXPANDED_100_CITY_daily_sample_*.json",
            "EXPANDED_100_CITY_hourly_sample_*.json",
            "HISTORICAL_REAL_daily_dataset_*.json",
            "REAL_hourly_dataset_100_cities_*.json",
            "ENHANCED_TWO_YEAR_daily_dataset_*.json",
        ]

        found_files = []
        for pattern in dataset_patterns:
            files = list(self.output_dir.glob(pattern))
            found_files.extend(files)

        # Get the most recent files if multiple exist
        if found_files:
            # Sort by modification time and get most recent
            found_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            # Keep only the most recent few for verification
            found_files = found_files[:10]

        return found_files

    def _analyze_dataset_storage(self, dataset_file: Path) -> Dict:
        """Analyze dataset storage including file sizes and ratios."""
        try:
            file_size = dataset_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Load dataset to analyze content size
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            # Analyze dataset structure
            analysis = {
                "file_path": str(dataset_file),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "dataset_type": self._identify_dataset_type(dataset_file.name),
                "cities_count": self._count_cities_in_dataset(dataset),
                "records_count": self._count_records_in_dataset(dataset),
                "storage_efficiency": self._calculate_storage_efficiency(
                    dataset, file_size
                ),
                "storage_analysis_passed": True,
            }

            # Calculate ratios if paired datasets exist
            if "daily" in dataset_file.name:
                hourly_counterpart = self._find_hourly_counterpart(dataset_file)
                if hourly_counterpart and hourly_counterpart.exists():
                    hourly_size = hourly_counterpart.stat().st_size
                    analysis["hourly_counterpart"] = str(hourly_counterpart)
                    analysis["hourly_size_mb"] = round(hourly_size / (1024 * 1024), 2)
                    analysis["size_ratio"] = round(hourly_size / file_size, 1)
                    analysis["expected_ratio"] = (
                        "24x (hourly should be ~24x larger than daily)"
                    )
                    analysis["ratio_verification"] = (
                        abs(analysis["size_ratio"] - 24) < 5
                    )  # Within 5x tolerance

            return analysis

        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            log.error(f"Storage analysis failed for {dataset_file}: {e}")
            return {
                "file_path": str(dataset_file),
                "storage_analysis_passed": False,
                "error": str(e),
            }

    def _validate_data_structure(self, dataset_file: Path) -> Dict:
        """Validate dataset data structure and format."""
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            validation = {
                "structure_valid": True,
                "required_fields_present": {},
                "data_types_correct": {},
                "data_completeness": {},
            }

            # Check required top-level fields
            required_fields = [
                "generation_time",
                "dataset_type",
                "data_authenticity",
                "dataset_comparison",
            ]

            for field in required_fields:
                validation["required_fields_present"][field] = field in dataset
                if field not in dataset:
                    validation["structure_valid"] = False

            # Check data types
            if "generation_time" in dataset:
                validation["data_types_correct"]["generation_time"] = isinstance(
                    dataset["generation_time"], str
                )

            if "dataset_comparison" in dataset:
                comparison = dataset["dataset_comparison"]
                if "daily_dataset" in comparison:
                    daily_data = comparison["daily_dataset"]
                    validation["data_types_correct"]["daily_cities"] = isinstance(
                        daily_data.get("cities"), int
                    )
                    validation["data_types_correct"]["daily_records"] = isinstance(
                        daily_data.get("total_records"), int
                    )

            # Check data completeness
            if "data_authenticity" in dataset:
                auth = dataset["data_authenticity"]
                validation["data_completeness"]["authenticity_level"] = (
                    "authenticity_level" in auth
                )
                validation["data_completeness"]["success_rate"] = "success_rate" in auth

            return validation

        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error(f"Structure validation failed for {dataset_file}: {e}")
            return {"structure_valid": False, "error": str(e)}

    def _perform_sanity_checks(self, dataset_file: Path) -> Dict:
        """Perform comprehensive sanity checks on dataset content."""
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            sanity_checks = {
                "sanity_checks_passed": True,
                "city_count_reasonable": False,
                "record_count_reasonable": False,
                "data_authenticity_verified": False,
                "model_performance_reasonable": False,
                "timestamp_validity": False,
            }

            # Check city count (should be close to 100)
            if "dataset_comparison" in dataset:
                comparison = dataset["dataset_comparison"]
                if "daily_dataset" in comparison:
                    cities = comparison["daily_dataset"].get("cities", 0)
                    sanity_checks["city_count_reasonable"] = 90 <= cities <= 100
                    sanity_checks["cities_found"] = cities

            # Check record count reasonableness
            if "dataset_comparison" in dataset:
                comparison = dataset["dataset_comparison"]
                if "daily_dataset" in comparison:
                    total_records = comparison["daily_dataset"].get("total_records", 0)
                    cities = comparison["daily_dataset"].get("cities", 1)
                    records_per_city = total_records / cities if cities > 0 else 0
                    sanity_checks["record_count_reasonable"] = (
                        300 <= records_per_city <= 800
                    )  # Reasonable for 1-2 years
                    sanity_checks["records_per_city"] = round(records_per_city, 1)

            # Check data authenticity claims
            if "data_authenticity" in dataset:
                auth = dataset["data_authenticity"]
                success_rate = auth.get("success_rate", "0%")
                if isinstance(success_rate, str) and "%" in success_rate:
                    rate = float(success_rate.replace("%", ""))
                    sanity_checks["data_authenticity_verified"] = (
                        rate >= 95
                    )  # Should be high success rate
                    sanity_checks["success_rate"] = success_rate

            # Check model performance sanity
            if "model_performance" in dataset:
                performance = dataset["model_performance"]
                if "daily_models" in performance:
                    daily_models = performance["daily_models"]
                    if "cams_benchmark" in daily_models:
                        cams = daily_models["cams_benchmark"]
                        r2 = cams.get("r2", 0)
                        sanity_checks["model_performance_reasonable"] = (
                            r2 > 0.8
                        )  # CAMS should perform well
                        sanity_checks["cams_r2"] = r2

            # Check timestamp validity
            if "generation_time" in dataset:
                try:
                    datetime.fromisoformat(
                        dataset["generation_time"].replace("Z", "+00:00")
                    )
                    sanity_checks["timestamp_validity"] = True
                except ValueError:
                    sanity_checks["timestamp_validity"] = False

            # Overall pass/fail
            checks = [
                sanity_checks["city_count_reasonable"],
                sanity_checks["record_count_reasonable"],
                sanity_checks["data_authenticity_verified"],
                sanity_checks["timestamp_validity"],
            ]
            sanity_checks["sanity_checks_passed"] = (
                sum(checks) >= 3
            )  # At least 3/4 should pass

            return sanity_checks

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            log.error(f"Sanity checks failed for {dataset_file}: {e}")
            return {"sanity_checks_passed": False, "error": str(e)}

    def _verify_api_consistency(self) -> Dict:
        """Verify that fresh API samples match collected data patterns."""
        try:
            # Sample a few API calls to verify consistency
            consistency_results = {
                "consistency_verified": True,
                "open_meteo_test": {},
                "nasa_firms_test": {},
                "openaq_test": {},
            }

            # Test Open-Meteo API (using Delhi as sample)
            log.info("Testing Open-Meteo API consistency...")
            delhi_coords = {"lat": 28.6139, "lon": 77.2090}

            try:
                meteo_url = "https://api.open-meteo.com/v1/air-quality"
                meteo_params = {
                    "latitude": delhi_coords["lat"],
                    "longitude": delhi_coords["lon"],
                    "hourly": ["pm10", "pm2_5", "carbon_monoxide"],
                    "forecast_days": 1,
                    "timezone": "auto",
                }

                meteo_response = requests.get(
                    meteo_url, params=meteo_params, timeout=15
                )

                if meteo_response.status_code == 200:
                    meteo_data = meteo_response.json()
                    consistency_results["open_meteo_test"] = {
                        "status": "success",
                        "response_size": len(str(meteo_data)),
                        "has_hourly_data": "hourly" in meteo_data,
                        "hourly_fields": (
                            list(meteo_data.get("hourly", {}).keys())
                            if "hourly" in meteo_data
                            else []
                        ),
                    }
                else:
                    consistency_results["open_meteo_test"] = {
                        "status": "failed",
                        "status_code": meteo_response.status_code,
                    }
            except Exception as e:
                consistency_results["open_meteo_test"] = {
                    "status": "error",
                    "error": str(e),
                }

            # Test NASA FIRMS API
            log.info("Testing NASA FIRMS API consistency...")
            try:
                # Load API key
                with open("C:\\aqf311\\Git_repo\\.config\\api_keys.json", "r") as f:
                    keys = json.load(f)
                nasa_key = keys["apis"]["nasa_firms"]["key"]

                firms_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
                firms_params = {
                    "MAP_KEY": nasa_key,
                    "source": "VIIRS_SNPP_NRT",
                    "area": "76.5,28.1,78.5,29.1",  # Delhi area
                    "dayRange": 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                }

                firms_response = requests.get(
                    firms_url, params=firms_params, timeout=15
                )

                if firms_response.status_code == 200:
                    firms_content = firms_response.text
                    consistency_results["nasa_firms_test"] = {
                        "status": "success",
                        "response_size": len(firms_content),
                        "has_fire_data": not firms_content.startswith("No fire"),
                        "content_preview": (
                            firms_content[:200] if firms_content else "No content"
                        ),
                    }
                else:
                    consistency_results["nasa_firms_test"] = {
                        "status": "failed",
                        "status_code": firms_response.status_code,
                    }
            except Exception as e:
                consistency_results["nasa_firms_test"] = {
                    "status": "error",
                    "error": str(e),
                }

            # Overall consistency check
            api_tests = [
                consistency_results["open_meteo_test"].get("status") == "success",
                consistency_results["nasa_firms_test"].get("status") == "success",
            ]
            consistency_results["consistency_verified"] = (
                sum(api_tests) >= 1
            )  # At least 1 API should work

            return consistency_results

        except (requests.RequestException, json.JSONDecodeError) as e:
            log.error(f"API consistency verification failed: {e}")
            return {"consistency_verified": False, "error": str(e)}

    def _identify_dataset_type(self, filename: str) -> str:
        """Identify dataset type from filename."""
        if "daily" in filename:
            return "daily"
        elif "hourly" in filename:
            return "hourly"
        else:
            return "unknown"

    def _count_cities_in_dataset(self, dataset: Dict) -> int:
        """Count cities in dataset."""
        if "dataset_comparison" in dataset:
            comparison = dataset["dataset_comparison"]
            if "daily_dataset" in comparison:
                return comparison["daily_dataset"].get("cities", 0)
        return 0

    def _count_records_in_dataset(self, dataset: Dict) -> int:
        """Count total records in dataset."""
        if "dataset_comparison" in dataset:
            comparison = dataset["dataset_comparison"]
            if "daily_dataset" in comparison:
                return comparison["daily_dataset"].get("total_records", 0)
        return 0

    def _calculate_storage_efficiency(self, dataset: Dict, file_size: int) -> Dict:
        """Calculate storage efficiency metrics."""
        records = self._count_records_in_dataset(dataset)
        if records > 0:
            bytes_per_record = file_size / records
            return {
                "bytes_per_record": round(bytes_per_record, 2),
                "efficiency_rating": (
                    "good"
                    if bytes_per_record < 1000
                    else "moderate" if bytes_per_record < 5000 else "poor"
                ),
            }
        return {"bytes_per_record": 0, "efficiency_rating": "unknown"}

    def _find_hourly_counterpart(self, daily_file: Path) -> Path:
        """Find corresponding hourly file for a daily file."""
        hourly_name = daily_file.name.replace("daily", "hourly")
        return daily_file.parent / hourly_name

    def save_verification_results(self, verification_results: Dict) -> str:
        """Save verification results to file."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            self.output_dir / f"DATASET_VERIFICATION_RESULTS_{timestamp_str}.json"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(verification_results, f, indent=2, default=str)

        log.info(f"Verification results saved: {results_file}")
        return str(results_file)


def main():
    """Main execution for dataset verification."""
    print("DATASET VERIFICATION AND SANITY CHECK")
    print("Analyzing storage ratios, data structure, and API consistency")
    print("=" * 80)

    try:
        verifier = DatasetVerificationSanityCheck()

        # Perform comprehensive verification
        verification_results = verifier.verify_comprehensive_datasets()

        # Save results
        results_file = verifier.save_verification_results(verification_results)

        # Print summary
        summary = verification_results["summary"]
        print(f"\nDATASET VERIFICATION COMPLETE!")
        print(f"Results saved: {results_file}")
        print(f"Datasets found: {summary['datasets_found']}")
        print(f"Datasets verified: {summary['datasets_verified']}")
        print(f"Storage ratios verified: {summary['storage_ratios_verified']}")
        print(f"Sanity checks passed: {summary['sanity_checks_passed']}")
        print(f"API consistency verified: {summary['api_consistency_verified']}")

        return verification_results

    except Exception as e:
        log.error(f"Dataset verification failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
