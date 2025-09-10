#!/usr/bin/env python3
"""
Toronto Environment Canada Data Collection - Week 1, Day 2
==========================================================

Second step of the Global 100-City Data Collection Strategy.
Test Environment Canada air quality data access for Toronto.

Objective: Validate Environment Canada data source availability and document access patterns.
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


class TorontoEnvironmentCanadaCollector:
    """Test Environment Canada data collection for Toronto as representative North American city."""

    def __init__(self, output_dir: str = "data/analysis/toronto_envcan_test"):
        """Initialize Toronto Environment Canada data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Toronto city configuration
        self.city_config = {
            "name": "Toronto",
            "country": "Canada",
            "lat": 43.6532,
            "lon": -79.3832,
            "aqi_standard": "Canadian",
            "continent": "north_america",
        }

        self.session = self._create_session()

        log.info("Toronto Environment Canada Data Collector initialized")
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

    def test_environment_canada_sources(self) -> Dict:
        """Test various Environment Canada data source endpoints for Toronto air quality data."""

        log.info("Testing Environment Canada data source availability...")

        # Known Environment Canada endpoints to test
        test_sources = {
            "envcan_aqhi": "https://weather.gc.ca/airquality/pages/index_e.html",
            "envcan_national_aps": "https://www.canada.ca/en/environment-climate-change/services/air-pollution/monitoring-networks-data/national-air-pollution-surveillance.html",
            "envcan_opendata": "https://open.canada.ca/data/en",
            "envcan_weather_air": "https://weather.gc.ca/airquality/",
            "envcan_datamart": "https://dd.weather.gc.ca/",
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

    def search_alternative_toronto_sources(self) -> Dict:
        """Search for alternative Toronto air quality data sources."""

        log.info("Searching for alternative Toronto air quality data sources...")

        # Alternative Canadian/Toronto air quality sources
        alternative_sources = {
            "ontario_ministry": "https://www.ontario.ca/page/air-quality-ontario",
            "toronto_opendata": "https://open.toronto.ca/",
            "airnow_canada": "https://www.airnow.gov/",
            "ontario_aqhi": "http://www.airqualityontario.com/",
            "canada_gc_data": "https://data.gc.ca/",
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
                            "aqhi",
                            "pm2.5",
                            "pm10",
                            "no2",
                            "ozone",
                            "pollution",
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

    def document_canadian_aqhi_calculation(self) -> Dict:
        """Document Canadian AQHI calculation method for Toronto."""

        log.info("Documenting Canadian AQHI calculation method...")

        # Canadian Air Quality Health Index bands and thresholds
        aqhi_bands = {
            "1": {
                "range": "1-3",
                "category": "Low Risk",
                "color": "#00CCFF",
                "health_message": "Ideal air quality for outdoor activities",
            },
            "2": {
                "range": "4-6",
                "category": "Moderate Risk",
                "color": "#FFFF00",
                "health_message": "No need to modify your usual outdoor activities unless you experience symptoms such as coughing and throat irritation",
            },
            "3": {
                "range": "7-10",
                "category": "High Risk",
                "color": "#FF6600",
                "health_message": "Consider reducing or rescheduling strenuous activities outdoors if you are experiencing symptoms",
            },
            "4": {
                "range": "10+",
                "category": "Very High Risk",
                "color": "#FF0000",
                "health_message": "Reduce or reschedule strenuous activities outdoors, especially if you experience symptoms such as coughing and throat irritation",
            },
        }

        return {
            "standard_name": "Canadian Air Quality Health Index (AQHI)",
            "scale": "1-10+",
            "categories": aqhi_bands,
            "calculation_method": "formula_based",
            "formula": "AQHI = (10/10.4) * [100 * ((exp(0.000537 * O3) - 1) + (exp(0.000871 * NO2) - 1) + (exp(0.000487 * PM2.5) - 1))]",
            "pollutants": ["O3", "NO2", "PM2.5"],
            "units": {
                "O3": "ppb",
                "NO2": "ppb",
                "PM2.5": "μg/m³",
            },
            "reference": "https://weather.gc.ca/airquality/pages/aqhi-desc_e.html",
        }

    def calculate_aqhi(self, pollutants: Dict[str, float]) -> Dict:
        """Calculate AQHI from pollutant concentrations."""

        aqhi_doc = self.document_canadian_aqhi_calculation()
        bands = aqhi_doc["categories"]

        # Extract pollutant concentrations
        o3 = pollutants.get("O3", None)
        no2 = pollutants.get("NO2", None)
        pm25 = pollutants.get("PM2.5", None)

        if all(v is not None and not np.isnan(v) for v in [o3, no2, pm25]):
            # Calculate AQHI using the official formula
            # AQHI = (10/10.4) * [100 * ((exp(0.000537 * O3) - 1) + (exp(0.000871 * NO2) - 1) + (exp(0.000487 * PM2.5) - 1))]

            try:
                term1 = np.exp(0.000537 * o3) - 1
                term2 = np.exp(0.000871 * no2) - 1
                term3 = np.exp(0.000487 * pm25) - 1

                aqhi_value = (10 / 10.4) * 100 * (term1 + term2 + term3)
                aqhi_rounded = round(aqhi_value)

                # Determine category
                if aqhi_rounded <= 3:
                    category_info = bands["1"]
                elif aqhi_rounded <= 6:
                    category_info = bands["2"]
                elif aqhi_rounded <= 10:
                    category_info = bands["3"]
                else:
                    category_info = bands["4"]

                return {
                    "aqhi_value": aqhi_rounded,
                    "aqhi_exact": aqhi_value,
                    "category": category_info["category"],
                    "health_message": category_info["health_message"],
                    "color": category_info["color"],
                    "pollutant_contributions": {
                        "O3_contribution": term1,
                        "NO2_contribution": term2,
                        "PM25_contribution": term3,
                    },
                    "calculated_at": datetime.now().isoformat(),
                }

            except Exception as e:
                return {
                    "aqhi_value": None,
                    "error": f"Calculation failed: {str(e)}",
                    "calculated_at": datetime.now().isoformat(),
                }
        else:
            return {
                "aqhi_value": None,
                "error": "Missing required pollutant data (O3, NO2, PM2.5)",
                "available_pollutants": {
                    k: v for k, v in pollutants.items() if v is not None
                },
                "calculated_at": datetime.now().isoformat(),
            }

    def create_test_summary(
        self, envcan_results: Dict, alternative_results: Dict
    ) -> Dict:
        """Create summary of Toronto Environment Canada data collection test."""

        summary = {
            "test_info": {
                "city": self.city_config,
                "test_date": datetime.now().isoformat(),
                "test_objective": "Week 1, Day 2: Validate Environment Canada data source for Toronto",
                "phase": "Phase 1 - Proof of Concept",
            },
            "data_source_tests": {
                "environment_canada_sources": envcan_results,
                "alternative_sources": alternative_results,
            },
            "aqhi_documentation": self.document_canadian_aqhi_calculation(),
            "findings": {
                "accessible_sources": 0,
                "total_sources_tested": len(envcan_results) + len(alternative_results),
                "recommended_approach": "",
                "challenges_identified": [],
                "next_steps": [],
            },
        }

        # Analyze results
        accessible_envcan = sum(
            1 for r in envcan_results.values() if r.get("accessible", False)
        )
        accessible_alt = sum(
            1 for r in alternative_results.values() if r.get("accessible", False)
        )

        summary["findings"]["accessible_sources"] = accessible_envcan + accessible_alt
        summary["findings"]["environment_canada_sources_accessible"] = accessible_envcan
        summary["findings"]["alternative_sources_accessible"] = accessible_alt

        # Recommendations based on results
        if accessible_envcan > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Proceed with Environment Canada official sources"
            summary["findings"]["next_steps"].append(
                "Implement Environment Canada data extraction methods"
            )
        elif accessible_alt > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Use alternative Canadian air quality sources"
            summary["findings"]["next_steps"].append(
                "Implement alternative source extraction methods"
            )
        else:
            summary["findings"][
                "recommended_approach"
            ] = "Investigate API-based access or request manual data"
            summary["findings"]["challenges_identified"].append(
                "No direct web access to data sources"
            )
            summary["findings"]["next_steps"].append(
                "Research Environment Canada API documentation"
            )
            summary["findings"]["next_steps"].append(
                "Contact Environment Canada support for data access guidance"
            )

        return summary

    def save_test_results(self, summary: Dict) -> None:
        """Save test results to output directory."""

        # Save main summary
        summary_path = self.output_dir / "toronto_envcan_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Test summary saved to {summary_path}")

        # Save simplified CSV for easy viewing
        csv_data = []

        for source_type in ["environment_canada_sources", "alternative_sources"]:
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

        csv_path = self.output_dir / "toronto_sources_test_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        log.info(f"CSV results saved to {csv_path}")


def main():
    """Execute Toronto Environment Canada data collection test - Week 1, Day 2."""

    log.info("Starting Toronto Environment Canada Data Collection Test - Week 1, Day 2")
    log.info("=" * 70)

    # Initialize collector
    collector = TorontoEnvironmentCanadaCollector()

    # Test Environment Canada official sources
    log.info("Phase 1: Testing Environment Canada official data sources...")
    envcan_results = collector.test_environment_canada_sources()

    # Test alternative sources
    log.info("Phase 2: Testing alternative Toronto air quality sources...")
    alternative_results = collector.search_alternative_toronto_sources()

    # Create test summary
    log.info("Phase 3: Creating test summary and recommendations...")
    summary = collector.create_test_summary(envcan_results, alternative_results)

    # Save results
    collector.save_test_results(summary)

    # Print summary report
    print("\n" + "=" * 70)
    print("TORONTO ENVIRONMENT CANADA DATA COLLECTION TEST - WEEK 1, DAY 2")
    print("=" * 70)

    print(f"\nTEST OBJECTIVE:")
    print(f"Validate Environment Canada data source availability for Toronto")
    print(f"Document access patterns and Canadian AQHI calculation compatibility")

    print(f"\nRESULTS SUMMARY:")
    print(f"• Total sources tested: {summary['findings']['total_sources_tested']}")
    print(f"• Accessible sources: {summary['findings']['accessible_sources']}")
    print(
        f"• Environment Canada sources accessible: {summary['findings']['environment_canada_sources_accessible']}"
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

    print(f"\nCanadian AQHI CALCULATION:")
    aqhi_doc = summary["aqhi_documentation"]
    print(f"• Standard: {aqhi_doc['standard_name']}")
    print(f"• Scale: {aqhi_doc['scale']} ({len(aqhi_doc['categories'])} categories)")
    print(f"• Pollutants: {', '.join(aqhi_doc['pollutants'])}")
    print(f"• Method: {aqhi_doc['calculation_method']} (formula-based calculation)")

    # Test AQHI calculation with sample data
    print(f"\nSAMPLE AQHI CALCULATION:")
    sample_pollutants = {"O3": 30.0, "NO2": 25.0, "PM2.5": 15.0}
    sample_aqhi = collector.calculate_aqhi(sample_pollutants)
    if sample_aqhi.get("aqhi_value"):
        print(f"• Sample pollutants: O3=30ppb, NO2=25ppb, PM2.5=15ug/m3")
        print(
            f"• Calculated AQHI: {sample_aqhi['aqhi_value']} ({sample_aqhi['category']})"
        )
        print(f"• Health message: {sample_aqhi['health_message']}")

    print("\n" + "=" * 70)
    print("WEEK 1, DAY 2 TEST COMPLETE")
    print("Toronto Environment Canada data source assessment finished")
    print("Ready to proceed based on findings and recommendations")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
