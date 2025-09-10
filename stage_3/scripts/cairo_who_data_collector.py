#!/usr/bin/env python3
"""
Cairo WHO Data Collection - Week 1, Day 4
==========================================

Fourth step of the Global 100-City Data Collection Strategy.
Test WHO Global Health Observatory data access for Cairo, Egypt.

Objective: Validate WHO data source availability and document access patterns for African cities.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import json

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class CairoWHOCollector:
    """Test WHO data collection for Cairo as representative African city."""

    def __init__(self, output_dir: str = "data/analysis/cairo_who_test"):
        """Initialize Cairo WHO data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cairo city configuration
        self.city_config = {
            "name": "Cairo",
            "country": "Egypt",
            "lat": 30.0444,
            "lon": 31.2357,
            "aqi_standard": "WHO",
            "continent": "africa",
        }

        self.session = self._create_session()

        log.info("Cairo WHO Data Collector initialized")
        log.info(f"Output directory: {self.output_dir}")
        log.info(
            f"Target city: {self.city_config['name']}, {self.city_config['country']}"
        )

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set user agent for respectful scraping
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        return session

    def test_who_official_sources(self) -> Dict:
        """Test various WHO and health organization data source endpoints for air quality data."""

        log.info(
            "Testing WHO and international health organization data source availability..."
        )

        # Known WHO and health organization endpoints to test
        test_sources = {
            "who_main": "https://www.who.int/",
            "who_gho": "https://www.who.int/data/gho",
            "who_air_quality": "https://www.who.int/health-topics/air-pollution",
            "who_ambient_air": "https://www.who.int/data/gho/data/themes/air-pollution",
            "who_database": "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/concentrations-of-fine-particulate-matter-(pm2-5)",
            "unep_air": "https://www.unep.org/explore-topics/air",
        }

        results = {}

        for source_name, url in test_sources.items():
            log.info(f"Testing {source_name}: {url}")

            try:
                response = self.session.get(url, timeout=30)
                results[source_name] = {
                    "url": url,
                    "status_code": response.status_code,
                    "accessible": response.status_code == 200,
                    "content_length": (
                        len(response.content) if response.status_code == 200 else 0
                    ),
                    "content_type": response.headers.get("content-type", "unknown"),
                    "tested_at": datetime.now().isoformat(),
                }

                if response.status_code == 200:
                    log.info(f"✅ {source_name} accessible")
                else:
                    log.warning(
                        f"⚠️ {source_name} returned status {response.status_code}"
                    )

            except Exception as e:
                log.error(f"❌ {source_name} failed: {str(e)}")
                results[source_name] = {
                    "url": url,
                    "status_code": None,
                    "accessible": False,
                    "error": str(e),
                    "tested_at": datetime.now().isoformat(),
                }

            # Respectful delay between requests
            time.sleep(1)

        return results

    def search_alternative_cairo_sources(self) -> Dict:
        """Search for alternative Cairo and satellite air quality data sources."""

        log.info("Searching for alternative Cairo air quality data sources...")

        # Alternative Cairo/Egypt/satellite air quality sources
        alternative_sources = {
            "nasa_modis": "https://modis.gsfc.nasa.gov/",
            "nasa_earthdata": "https://earthdata.nasa.gov/",
            "iqair_cairo": "https://www.iqair.com/egypt/cairo",
            "waqi_cairo": "https://waqi.info/",
            "egypt_eeaa": "https://www.eeaa.gov.eg/",
        }

        results = {}

        for source_name, url in alternative_sources.items():
            log.info(f"Testing {source_name}: {url}")

            try:
                response = self.session.get(url, timeout=30)
                results[source_name] = {
                    "url": url,
                    "status_code": response.status_code,
                    "accessible": response.status_code == 200,
                    "content_type": response.headers.get("content-type", "unknown"),
                    "tested_at": datetime.now().isoformat(),
                }

                if response.status_code == 200:
                    log.info(f"✅ {source_name} accessible")
                    # Check if content mentions air quality data
                    content_lower = response.text.lower()
                    has_air_quality = any(
                        term in content_lower
                        for term in [
                            "air quality",
                            "pm2.5",
                            "pm10",
                            "no2",
                            "ozone",
                            "pollution",
                            "cairo",
                            "egypt",
                            "modis",
                            "satellite",
                        ]
                    )
                    results[source_name]["has_air_quality_content"] = has_air_quality
                else:
                    log.warning(
                        f"⚠️ {source_name} returned status {response.status_code}"
                    )

            except Exception as e:
                log.error(f"❌ {source_name} failed: {str(e)}")
                results[source_name] = {
                    "url": url,
                    "status_code": None,
                    "accessible": False,
                    "error": str(e),
                    "tested_at": datetime.now().isoformat(),
                }

            time.sleep(1)

        return results

    def document_who_guidelines_calculation(self) -> Dict:
        """Document WHO Air Quality Guidelines adaptation for Cairo."""

        log.info("Documenting WHO Air Quality Guidelines calculation method...")

        # WHO Air Quality Guidelines adaptation bands and thresholds
        who_aqi_bands = {
            "1": {
                "range": "0-25",
                "category": "Low Risk",
                "color": "#00e400",
                "health_implications": "Air quality is satisfactory and poses little or no risk",
                "pm25_max": 15,
                "pm10_max": 45,
                "no2_max": 25,
                "o3_max": 100,
                "so2_max": 40,
            },
            "2": {
                "range": "26-50",
                "category": "Moderate Risk",
                "color": "#ffff00",
                "health_implications": "Air quality is acceptable for most people, sensitive individuals may experience minor issues",
                "pm25_max": 25,
                "pm10_max": 75,
                "no2_max": 50,
                "o3_max": 160,
                "so2_max": 80,
            },
            "3": {
                "range": "51-75",
                "category": "High Risk",
                "color": "#ff7e00",
                "health_implications": "Members of sensitive groups may experience health effects",
                "pm25_max": 37.5,
                "pm10_max": 100,
                "no2_max": 100,
                "o3_max": 200,
                "so2_max": 200,
            },
            "4": {
                "range": "76-100",
                "category": "Very High Risk",
                "color": "#ff0000",
                "health_implications": "Everyone may begin to experience health effects, sensitive groups more seriously",
                "pm25_max": 75,
                "pm10_max": 150,
                "no2_max": 200,
                "o3_max": 300,
                "so2_max": 500,
            },
            "5": {
                "range": "100+",
                "category": "Extremely High Risk",
                "color": "#8f3f97",
                "health_implications": "Health alert: everyone may experience serious health effects",
                "pm25_max": float("inf"),
                "pm10_max": float("inf"),
                "no2_max": float("inf"),
                "o3_max": float("inf"),
                "so2_max": float("inf"),
            },
        }

        return {
            "standard_name": "WHO Air Quality Guidelines Adaptation",
            "scale": "0-100+",
            "categories": who_aqi_bands,
            "calculation_method": "worst_pollutant",
            "pollutants": ["PM2.5", "PM10", "NO2", "O3", "SO2"],
            "units": {
                "PM2.5": "μg/m³",
                "PM10": "μg/m³",
                "NO2": "μg/m³",
                "O3": "μg/m³",
                "SO2": "μg/m³",
            },
            "reference": "https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines",
            "notes": "Adapted WHO guidelines for African cities with limited monitoring infrastructure",
        }

    def calculate_who_aqi(self, pollutants: Dict[str, float]) -> Dict:
        """Calculate WHO AQI adaptation from pollutant concentrations."""

        who_doc = self.document_who_guidelines_calculation()
        bands = who_doc["categories"]

        individual_aqis = {}

        for pollutant, concentration in pollutants.items():
            if concentration is None or np.isnan(concentration):
                individual_aqis[pollutant] = None
                continue

            pollutant_key = f"{pollutant.lower()}_max"

            for band_num, band_info in bands.items():
                if pollutant_key in band_info:
                    if concentration <= band_info[pollutant_key]:
                        # Simple linear interpolation within band
                        if band_num == "1":
                            aqi_low, aqi_high = 0, 25
                            conc_low = 0
                        elif band_num == "2":
                            aqi_low, aqi_high = 26, 50
                            conc_low = bands["1"][pollutant_key]
                        elif band_num == "3":
                            aqi_low, aqi_high = 51, 75
                            conc_low = bands["2"][pollutant_key]
                        elif band_num == "4":
                            aqi_low, aqi_high = 76, 100
                            conc_low = bands["3"][pollutant_key]
                        else:  # band_num == "5"
                            aqi_low, aqi_high = 100, 150
                            conc_low = bands["4"][pollutant_key]

                        conc_high = band_info[pollutant_key]

                        if conc_high == float("inf"):
                            aqi_value = 150  # Maximum WHO AQI
                        else:
                            # Linear interpolation
                            aqi_value = (
                                (aqi_high - aqi_low) / (conc_high - conc_low)
                            ) * (concentration - conc_low) + aqi_low

                        individual_aqis[pollutant] = {
                            "aqi_value": round(aqi_value),
                            "category": band_info["category"],
                            "concentration": concentration,
                            "band": int(band_num),
                        }
                        break

        # Overall WHO AQI is the worst individual pollutant AQI
        valid_aqis = [v["aqi_value"] for v in individual_aqis.values() if v is not None]

        if valid_aqis:
            overall_aqi = max(valid_aqis)

            # Determine overall category
            if overall_aqi <= 25:
                overall_category = "Low Risk"
                overall_band = 1
            elif overall_aqi <= 50:
                overall_category = "Moderate Risk"
                overall_band = 2
            elif overall_aqi <= 75:
                overall_category = "High Risk"
                overall_band = 3
            elif overall_aqi <= 100:
                overall_category = "Very High Risk"
                overall_band = 4
            else:
                overall_category = "Extremely High Risk"
                overall_band = 5

            # Find dominant pollutant
            dominant_pollutant = None
            for pollutant, aqi_info in individual_aqis.items():
                if aqi_info and aqi_info["aqi_value"] == overall_aqi:
                    dominant_pollutant = pollutant
                    break
        else:
            overall_aqi = None
            overall_category = "No Data"
            overall_band = None
            dominant_pollutant = None

        return {
            "overall_who_aqi": overall_aqi,
            "overall_category": overall_category,
            "overall_band": overall_band,
            "dominant_pollutant": dominant_pollutant,
            "individual_aqis": individual_aqis,
            "calculated_at": datetime.now().isoformat(),
        }

    def create_test_summary(self, who_results: Dict, alternative_results: Dict) -> Dict:
        """Create summary of Cairo WHO data collection test."""

        summary = {
            "test_info": {
                "city": self.city_config,
                "test_date": datetime.now().isoformat(),
                "test_objective": "Week 1, Day 4: Validate WHO data source for Cairo",
                "phase": "Phase 1 - Proof of Concept",
            },
            "data_source_tests": {
                "who_official_sources": who_results,
                "alternative_sources": alternative_results,
            },
            "who_guidelines_documentation": self.document_who_guidelines_calculation(),
            "findings": {
                "accessible_sources": 0,
                "total_sources_tested": len(who_results) + len(alternative_results),
                "recommended_approach": "",
                "challenges_identified": [],
                "next_steps": [],
            },
        }

        # Analyze results
        accessible_who = sum(
            1 for r in who_results.values() if r.get("accessible", False)
        )
        accessible_alt = sum(
            1 for r in alternative_results.values() if r.get("accessible", False)
        )

        summary["findings"]["accessible_sources"] = accessible_who + accessible_alt
        summary["findings"]["who_sources_accessible"] = accessible_who
        summary["findings"]["alternative_sources_accessible"] = accessible_alt

        # Recommendations based on results
        if accessible_who > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Proceed with WHO official data sources"
            summary["findings"]["next_steps"].append(
                "Implement WHO data extraction methods"
            )
            summary["findings"]["next_steps"].append(
                "Integrate satellite data for validation"
            )
        elif accessible_alt > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Use satellite and alternative sources (NASA MODIS, IQAir, WAQI)"
            summary["findings"]["next_steps"].append(
                "Implement satellite data processing methods"
            )
            summary["findings"]["next_steps"].append(
                "Validate satellite estimates against limited ground truth"
            )
        else:
            summary["findings"][
                "recommended_approach"
            ] = "Rely primarily on NASA satellite data for African cities"
            summary["findings"]["challenges_identified"].append(
                "Limited ground truth data availability in Africa"
            )
            summary["findings"]["next_steps"].append(
                "Develop satellite-only methodology"
            )
            summary["findings"]["next_steps"].append(
                "Contact African environmental agencies for partnership"
            )

        # Add Africa-specific considerations
        summary["findings"]["challenges_identified"].append(
            "Limited monitoring infrastructure in African cities (as expected)"
        )
        summary["findings"]["next_steps"].append(
            "Prepare satellite data as primary source for African cities"
        )

        return summary

    def save_test_results(self, summary: Dict) -> None:
        """Save test results to output directory."""

        # Save main summary
        summary_path = self.output_dir / "cairo_who_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Test summary saved to {summary_path}")

        # Save simplified CSV for easy viewing
        csv_data = []

        for source_type in ["who_official_sources", "alternative_sources"]:
            for source_name, result in summary["data_source_tests"][
                source_type
            ].items():
                csv_data.append(
                    {
                        "source_type": source_type,
                        "source_name": source_name,
                        "url": result["url"],
                        "accessible": result.get("accessible", False),
                        "status_code": result.get("status_code"),
                        "has_air_quality": result.get(
                            "has_air_quality_content", "unknown"
                        ),
                    }
                )

        csv_path = self.output_dir / "cairo_sources_test_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        log.info(f"CSV results saved to {csv_path}")


def main():
    """Execute Cairo WHO data collection test - Week 1, Day 4."""

    log.info("Starting Cairo WHO Data Collection Test - Week 1, Day 4")
    log.info("=" * 60)

    # Initialize collector
    collector = CairoWHOCollector()

    # Test WHO official sources
    log.info(
        "Phase 1: Testing WHO and international health organization data sources..."
    )
    who_results = collector.test_who_official_sources()

    # Test alternative sources
    log.info("Phase 2: Testing alternative Cairo and satellite air quality sources...")
    alternative_results = collector.search_alternative_cairo_sources()

    # Create test summary
    log.info("Phase 3: Creating test summary and recommendations...")
    summary = collector.create_test_summary(who_results, alternative_results)

    # Save results
    collector.save_test_results(summary)

    # Print summary report
    print("\n" + "=" * 60)
    print("CAIRO WHO DATA COLLECTION TEST - WEEK 1, DAY 4")
    print("=" * 60)

    print(f"\nTEST OBJECTIVE:")
    print(f"Validate WHO data source availability for Cairo, Egypt")
    print(f"Document access patterns and WHO Guidelines calculation compatibility")

    print(f"\nRESULTS SUMMARY:")
    print(f"• Total sources tested: {summary['findings']['total_sources_tested']}")
    print(f"• Accessible sources: {summary['findings']['accessible_sources']}")
    print(
        f"• WHO official sources accessible: {summary['findings']['who_sources_accessible']}"
    )
    print(
        f"• Alternative sources accessible: {summary['findings']['alternative_sources_accessible']}"
    )

    print(f"\nRECOMMENDATION:")
    print(f"• {summary['findings']['recommended_approach']}")

    if summary["findings"]["challenges_identified"]:
        print(f"\nCHALLENGES IDENTIFIED:")
        for challenge in summary["findings"]["challenges_identified"]:
            print(f"• {challenge}")

    print(f"\nNEXT STEPS:")
    for step in summary["findings"]["next_steps"]:
        print(f"• {step}")

    print(f"\nWHO AIR QUALITY GUIDELINES CALCULATION:")
    who_doc = summary["who_guidelines_documentation"]
    print(f"• Standard: {who_doc['standard_name']}")
    print(f"• Scale: {who_doc['scale']} ({len(who_doc['categories'])} categories)")
    print(f"• Pollutants: {', '.join(who_doc['pollutants'])}")
    print(
        f"• Method: {who_doc['calculation_method']} (worst pollutant determines overall AQI)"
    )

    # Test WHO AQI calculation with sample data
    print(f"\nSAMPLE WHO AQI CALCULATION:")
    sample_pollutants = {"PM2.5": 35.0, "PM10": 85.0, "NO2": 45.0, "O3": 120.0}
    sample_aqi = collector.calculate_who_aqi(sample_pollutants)
    if sample_aqi.get("overall_who_aqi"):
        print(
            f"• Sample pollutants: PM2.5=35ug/m3, PM10=85ug/m3, NO2=45ug/m3, O3=120ug/m3"
        )
        print(
            f"• Calculated WHO AQI: {sample_aqi['overall_who_aqi']} ({sample_aqi['overall_category']})"
        )
        print(f"• Dominant pollutant: {sample_aqi['dominant_pollutant']}")
        if sample_aqi["overall_band"]:
            health_info = who_doc["categories"][str(sample_aqi["overall_band"])][
                "health_implications"
            ]
            print(f"• Health implications: {health_info}")

    print("\n" + "=" * 60)
    print("WEEK 1, DAY 4 TEST COMPLETE")
    print("Cairo WHO data source assessment finished")
    print("Ready to proceed based on findings and recommendations")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
