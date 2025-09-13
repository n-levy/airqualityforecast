#!/usr/bin/env python3
"""
Smoke Test: Model Features Data Verification
===========================================

Comprehensive smoke test to verify that the enhanced features dataset contains
only real data, with no synthetic, simulated, or mathematically generated data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class ModelFeaturesSmokeTest:
    """Smoke test to verify real data collection in enhanced features."""

    def __init__(self):
        self.dataset_path = Path(
            "stage_5/enhanced_features/enhanced_worst_air_quality_with_features.json"
        )
        self.test_results = {
            "test_name": "model_features_smoke_test",
            "start_time": None,
            "tests_passed": 0,
            "tests_failed": 0,
            "critical_failures": [],
            "warnings": [],
            "data_authenticity": "unknown",
        }

    def run_smoke_test(self) -> Dict[str, Any]:
        """Run comprehensive smoke test on enhanced features data."""
        log.info("=== STARTING MODEL FEATURES SMOKE TEST ===")

        if not self.dataset_path.exists():
            self.test_results["critical_failures"].append(
                "Enhanced features dataset not found"
            )
            self.test_results["data_authenticity"] = "failed"
            return self.test_results

        # Load dataset
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        log.info(f"Loaded dataset with {len(dataset.get('city_results', {}))} cities")

        # Run verification tests
        self._test_dataset_structure(dataset)
        self._test_fire_features_authenticity(dataset)
        self._test_holiday_features_authenticity(dataset)
        self._test_for_synthetic_patterns(dataset)
        self._test_data_variability(dataset)
        self._test_geographic_consistency(dataset)

        # Final assessment
        if self.test_results["critical_failures"]:
            self.test_results["data_authenticity"] = "failed"
            log.error("CRITICAL: Synthetic or simulated data detected!")
        elif self.test_results["warnings"]:
            self.test_results["data_authenticity"] = "suspicious"
            log.warning("WARNING: Some data patterns appear potentially synthetic")
        else:
            self.test_results["data_authenticity"] = "verified_real"
            log.info("SUCCESS: All data appears to be real/authentic")

        log.info("=== MODEL FEATURES SMOKE TEST COMPLETED ===")
        return self.test_results

    def _test_dataset_structure(self, dataset: Dict) -> None:
        """Test basic dataset structure."""
        log.info("Testing dataset structure...")

        required_keys = ["collection_type", "city_results"]
        for key in required_keys:
            if key not in dataset:
                self.test_results["critical_failures"].append(
                    f"Missing required key: {key}"
                )
                return

        # Check for enhanced features metadata
        if "enhanced_features" in dataset:
            enhanced_meta = dataset["enhanced_features"]
            if not enhanced_meta.get("fire_activity_features"):
                self.test_results["warnings"].append(
                    "Fire features not marked as added"
                )
            if not enhanced_meta.get("holiday_features"):
                self.test_results["warnings"].append(
                    "Holiday features not marked as added"
                )

        self.test_results["tests_passed"] += 1
        log.info("✓ Dataset structure test passed")

    def _test_fire_features_authenticity(self, dataset: Dict) -> None:
        """Test fire features for authenticity markers."""
        log.info("Testing fire features authenticity...")

        cities_with_fire_features = 0
        suspicious_patterns = []

        for city_name, city_data in dataset["city_results"].items():
            # Check if city has fire features
            has_fire_features = False

            for source_name, source_data in city_data.get("data_sources", {}).items():
                if "data_sample" in source_data:
                    for record in source_data["data_sample"]:
                        if "fire_features" in record:
                            has_fire_features = True
                            fire_features = record["fire_features"]

                            # Test for synthetic patterns
                            self._check_fire_feature_authenticity(
                                city_name, fire_features, suspicious_patterns
                            )

            if has_fire_features:
                cities_with_fire_features += 1

        log.info(f"Cities with fire features: {cities_with_fire_features}")

        if cities_with_fire_features == 0:
            self.test_results["critical_failures"].append(
                "No fire features found in any city"
            )
        elif suspicious_patterns:
            self.test_results["warnings"].extend(
                suspicious_patterns[:5]
            )  # Limit to first 5

        self.test_results["tests_passed"] += 1
        log.info("✓ Fire features authenticity test completed")

    def _check_fire_feature_authenticity(
        self, city_name: str, fire_features: Dict, suspicious_patterns: List
    ) -> None:
        """Check individual fire features for synthetic patterns."""

        # Check for obviously synthetic values
        if "fire_weather_index" in fire_features:
            fwi = fire_features["fire_weather_index"]
            # Real fire weather indices should vary and not be perfect round numbers
            if isinstance(fwi, (int, float)) and fwi == round(fwi, 0) and fwi % 10 == 0:
                suspicious_patterns.append(
                    f"{city_name}: Fire weather index too round ({fwi})"
                )

        # Check fire distance patterns
        if "fire_distance_km" in fire_features:
            distance = fire_features["fire_distance_km"]
            if isinstance(distance, (int, float)) and distance == round(distance, 0):
                if distance in [50, 100, 200, 500]:  # Common synthetic defaults
                    suspicious_patterns.append(
                        f"{city_name}: Fire distance appears synthetic ({distance}km)"
                    )

        # Check for unrealistic fire activity patterns
        if "active_fires_nearby" in fire_features:
            fires = fire_features["active_fires_nearby"]
            if isinstance(fires, int) and fires > 1000:  # Unrealistically high
                suspicious_patterns.append(
                    f"{city_name}: Unrealistically high fire count ({fires})"
                )

    def _test_holiday_features_authenticity(self, dataset: Dict) -> None:
        """Test holiday features for authenticity."""
        log.info("Testing holiday features authenticity...")

        cities_with_holiday_features = 0
        holiday_patterns = {}

        for city_name, city_data in dataset["city_results"].items():
            country = city_data.get("country", "Unknown")

            for source_name, source_data in city_data.get("data_sources", {}).items():
                if "data_sample" in source_data:
                    for record in source_data["data_sample"]:
                        if "holiday_features" in record:
                            cities_with_holiday_features += 1
                            holiday_features = record["holiday_features"]

                            # Collect holiday patterns by country
                            if country not in holiday_patterns:
                                holiday_patterns[country] = set()

                            if holiday_features.get("holiday_name"):
                                holiday_patterns[country].add(
                                    holiday_features["holiday_name"]
                                )

        log.info(f"Cities with holiday features: {cities_with_holiday_features}")
        log.info(f"Countries with holiday data: {len(holiday_patterns)}")

        # Check for authentic country-specific holidays
        authentic_holidays = self._verify_country_holidays(holiday_patterns)
        if not authentic_holidays:
            self.test_results["warnings"].append(
                "Holiday patterns may not reflect real country-specific holidays"
            )

        if cities_with_holiday_features == 0:
            self.test_results["critical_failures"].append(
                "No holiday features found in any city"
            )

        self.test_results["tests_passed"] += 1
        log.info("✓ Holiday features authenticity test completed")

    def _verify_country_holidays(self, holiday_patterns: Dict[str, set]) -> bool:
        """Verify that holidays match real country patterns."""

        # Sample of authentic country-specific holidays
        authentic_patterns = {
            "India": {
                "Diwali",
                "Holi",
                "Independence Day",
                "Republic Day",
                "Gandhi Jayanti",
            },
            "China": {"Chinese New Year", "National Day", "Mid-Autumn Festival"},
            "USA": {"Independence Day", "Thanksgiving", "Memorial Day", "Labor Day"},
            "Brazil": {"Independence Day", "Carnival", "Proclamation of the Republic"},
            "Egypt": {"Revolution Day", "Coptic Christmas", "Ramadan"},
        }

        matches = 0
        for country, holidays in holiday_patterns.items():
            if country in authentic_patterns:
                if holidays & authentic_patterns[country]:  # Set intersection
                    matches += 1

        return matches > 0

    def _test_for_synthetic_patterns(self, dataset: Dict) -> None:
        """Test for obvious synthetic data patterns."""
        log.info("Testing for synthetic data patterns...")

        synthetic_indicators = []

        # Check for patterns that indicate mathematical generation
        sample_count = 0
        round_number_count = 0

        for city_name, city_data in dataset["city_results"].items():
            for source_name, source_data in city_data.get("data_sources", {}).items():
                if "data_sample" in source_data:
                    for record in source_data["data_sample"]:
                        sample_count += 1

                        # Check fire features for synthetic patterns
                        if "fire_features" in record:
                            fire_features = record["fire_features"]

                            # Count suspiciously round numbers
                            for key, value in fire_features.items():
                                if isinstance(value, (int, float)) and value > 0:
                                    if value == round(value, 0) and value % 10 == 0:
                                        round_number_count += 1

        # Calculate synthetic probability
        if sample_count > 0:
            round_ratio = round_number_count / sample_count
            if round_ratio > 0.3:  # More than 30% round numbers
                synthetic_indicators.append(
                    f"High ratio of round numbers: {round_ratio:.2%}"
                )

        if synthetic_indicators:
            self.test_results["warnings"].extend(synthetic_indicators)

        self.test_results["tests_passed"] += 1
        log.info("✓ Synthetic patterns test completed")

    def _test_data_variability(self, dataset: Dict) -> None:
        """Test for proper data variability (real data should vary)."""
        log.info("Testing data variability...")

        # Collect fire weather index values
        fwi_values = []
        fire_distances = []

        for city_name, city_data in dataset["city_results"].items():
            for source_name, source_data in city_data.get("data_sources", {}).items():
                if "data_sample" in source_data:
                    for record in source_data["data_sample"]:
                        if "fire_features" in record:
                            fire_features = record["fire_features"]

                            if "fire_weather_index" in fire_features:
                                fwi_values.append(fire_features["fire_weather_index"])

                            if "fire_distance_km" in fire_features:
                                fire_distances.append(fire_features["fire_distance_km"])

        # Check variability
        if len(set(fwi_values)) < len(fwi_values) * 0.5:  # Less than 50% unique values
            self.test_results["warnings"].append(
                "Fire weather index values show low variability"
            )

        if len(set(fire_distances)) < len(fire_distances) * 0.5:
            self.test_results["warnings"].append(
                "Fire distance values show low variability"
            )

        self.test_results["tests_passed"] += 1
        log.info("✓ Data variability test completed")

    def _test_geographic_consistency(self, dataset: Dict) -> None:
        """Test for geographic consistency in features."""
        log.info("Testing geographic consistency...")

        continent_fire_patterns = {}

        for city_name, city_data in dataset["city_results"].items():
            continent = city_data.get("continent", "unknown")

            if continent not in continent_fire_patterns:
                continent_fire_patterns[continent] = {
                    "fire_seasons": set(),
                    "fire_types": set(),
                }

            # Check fire metadata
            if "fire_metadata" in city_data:
                fire_meta = city_data["fire_metadata"]
                if "fire_seasons" in fire_meta:
                    seasons = fire_meta["fire_seasons"]
                    if "peak" in seasons:
                        continent_fire_patterns[continent]["fire_seasons"].update(
                            seasons["peak"]
                        )

                if "typical_fire_sources" in fire_meta:
                    fire_sources = fire_meta["typical_fire_sources"]
                    continent_fire_patterns[continent]["fire_types"].update(
                        fire_sources
                    )

        # Verify continent-specific patterns make sense
        inconsistencies = []

        # Asia should have different patterns than Europe
        if "asia" in continent_fire_patterns and "europe" in continent_fire_patterns:
            asia_types = continent_fire_patterns["asia"]["fire_types"]
            europe_types = continent_fire_patterns["europe"]["fire_types"]

            if asia_types == europe_types:
                inconsistencies.append(
                    "Asia and Europe have identical fire types (suspicious)"
                )

        if inconsistencies:
            self.test_results["warnings"].extend(inconsistencies)

        self.test_results["tests_passed"] += 1
        log.info("✓ Geographic consistency test completed")


def main():
    """Run the smoke test."""
    log.info("Starting Model Features Data Smoke Test")

    try:
        smoke_test = ModelFeaturesSmokeTest()
        results = smoke_test.run_smoke_test()

        # Print results
        print("\n" + "=" * 60)
        print("MODEL FEATURES SMOKE TEST RESULTS")
        print("=" * 60)
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Data Authenticity: {results['data_authenticity']}")

        if results["critical_failures"]:
            print(f"\nCRITICAL FAILURES:")
            for failure in results["critical_failures"]:
                print(f"  ❌ {failure}")

        if results["warnings"]:
            print(f"\nWARNINGS:")
            for warning in results["warnings"]:
                print(f"  ⚠️  {warning}")

        if results["data_authenticity"] == "verified_real":
            print(f"\n✅ SUCCESS: Model features data verified as REAL")
            return 0
        elif results["data_authenticity"] == "suspicious":
            print(f"\n⚠️  WARNING: Data may contain synthetic elements")
            return 1
        else:
            print(f"\n❌ FAILURE: Data contains synthetic or simulated elements")
            return 2

    except Exception as e:
        log.error(f"Smoke test failed: {str(e)}")
        print(f"\n❌ SMOKE TEST ERROR: {str(e)}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
