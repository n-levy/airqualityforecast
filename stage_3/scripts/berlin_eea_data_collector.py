#!/usr/bin/env python3
"""
Berlin EEA Data Collection - Week 1, Day 1
==========================================

First step of the Global 100-City Data Collection Strategy.
Test EEA air quality data access for Berlin, Germany.

Objective: Validate EEA data source availability and document access patterns.
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


class BerlinEEACollector:
    """Test EEA data collection for Berlin as representative European city."""

    def __init__(self, output_dir: str = "data/analysis/berlin_eea_test"):
        """Initialize Berlin EEA data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Berlin city configuration
        self.city_config = {
            "name": "Berlin",
            "country": "Germany",
            "lat": 52.5200,
            "lon": 13.4050,
            "aqi_standard": "EAQI",
            "continent": "europe",
        }

        self.session = self._create_session()

        log.info("Berlin EEA Data Collector initialized")
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

    def test_eea_data_sources(self) -> Dict:
        """Test various EEA data source endpoints for Berlin air quality data."""

        log.info("Testing EEA data source availability...")

        # Known EEA endpoints to test
        test_sources = {
            "eea_datahub": "https://www.eea.europa.eu/en/datahub",
            "eea_discomap": "https://discomap.eea.europa.eu/",
            "eea_aqereporting": "https://www.eea.europa.eu/data-and-maps/data/aqereporting-9",
            "eea_air_quality": "https://www.eea.europa.eu/themes/air/air-quality",
            "eionet_cdr": "https://cdr.eionet.europa.eu/",
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

    def search_alternative_berlin_sources(self) -> Dict:
        """Search for alternative Berlin air quality data sources."""

        log.info("Searching for alternative Berlin air quality data sources...")

        # Alternative German/Berlin air quality sources
        alternative_sources = {
            "berlin_luftdaten": "https://luftdaten.berlin.de/",
            "umweltbundesamt": "https://www.umweltbundesamt.de/daten/luft/luftdaten",
            "berlin_senate": "https://www.berlin.de/sen/uvk/umwelt/luft/luftguete/",
            "uba_current": "https://www.umweltbundesamt.de/daten/luft/luftdaten/luftqualitaet/eJWa",
            "berlin_opendata": "https://daten.berlin.de/",
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
                            "luftqualität",
                            "air quality",
                            "pm2.5",
                            "pm10",
                            "no2",
                            "ozone",
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

    def document_eaqi_calculation(self) -> Dict:
        """Document EAQI calculation method for Berlin."""

        log.info("Documenting EAQI calculation method...")

        # European Air Quality Index bands and thresholds
        eaqi_bands = {
            "1": {
                "category": "Very Good",
                "color": "#50f0e6",
                "pm25_max": 10,
                "pm10_max": 20,
                "no2_max": 40,
                "o3_max": 60,
                "so2_max": 100,
            },
            "2": {
                "category": "Good",
                "color": "#50ccaa",
                "pm25_max": 20,
                "pm10_max": 40,
                "no2_max": 90,
                "o3_max": 120,
                "so2_max": 200,
            },
            "3": {
                "category": "Fair",
                "color": "#f0e641",
                "pm25_max": 25,
                "pm10_max": 50,
                "no2_max": 120,
                "o3_max": 180,
                "so2_max": 350,
            },
            "4": {
                "category": "Poor",
                "color": "#ff5050",
                "pm25_max": 50,
                "pm10_max": 100,
                "no2_max": 230,
                "o3_max": 240,
                "so2_max": 500,
            },
            "5": {
                "category": "Very Poor",
                "color": "#960032",
                "pm25_max": 75,
                "pm10_max": 150,
                "no2_max": 340,
                "o3_max": 380,
                "so2_max": 750,
            },
            "6": {
                "category": "Extremely Poor",
                "color": "#7d2181",
                "pm25_max": float("inf"),
                "pm10_max": float("inf"),
                "no2_max": float("inf"),
                "o3_max": float("inf"),
                "so2_max": float("inf"),
            },
        }

        return {
            "standard_name": "European Air Quality Index (EAQI)",
            "scale": "1-6",
            "categories": eaqi_bands,
            "calculation_method": "worst_pollutant",
            "pollutants": ["PM2.5", "PM10", "NO2", "O3", "SO2"],
            "units": {
                "PM2.5": "μg/m³",
                "PM10": "μg/m³",
                "NO2": "μg/m³",
                "O3": "μg/m³",
                "SO2": "μg/m³",
            },
            "reference": "https://www.eea.europa.eu/themes/air/air-quality-index",
        }

    def calculate_eaqi(self, pollutants: Dict[str, float]) -> Dict:
        """Calculate EAQI from pollutant concentrations."""

        eaqi_doc = self.document_eaqi_calculation()
        bands = eaqi_doc["categories"]

        individual_aqis = {}

        for pollutant, concentration in pollutants.items():
            if concentration is None or np.isnan(concentration):
                individual_aqis[pollutant] = None
                continue

            pollutant_key = f"{pollutant.lower()}_max"

            for band_num, band_info in bands.items():
                if pollutant_key in band_info:
                    if concentration <= band_info[pollutant_key]:
                        individual_aqis[pollutant] = {
                            "band": int(band_num),
                            "category": band_info["category"],
                            "concentration": concentration,
                        }
                        break

        # Overall EAQI is the worst individual pollutant AQI
        valid_aqis = [v["band"] for v in individual_aqis.values() if v is not None]

        if valid_aqis:
            overall_band = max(valid_aqis)
            overall_category = bands[str(overall_band)]["category"]

            # Find dominant pollutant
            dominant_pollutant = None
            for pollutant, aqi_info in individual_aqis.items():
                if aqi_info and aqi_info["band"] == overall_band:
                    dominant_pollutant = pollutant
                    break
        else:
            overall_band = None
            overall_category = "No Data"
            dominant_pollutant = None

        return {
            "overall_eaqi": overall_band,
            "overall_category": overall_category,
            "dominant_pollutant": dominant_pollutant,
            "individual_aqis": individual_aqis,
            "calculated_at": datetime.now().isoformat(),
        }

    def create_test_summary(self, eea_results: Dict, alternative_results: Dict) -> Dict:
        """Create summary of Berlin EEA data collection test."""

        summary = {
            "test_info": {
                "city": self.city_config,
                "test_date": datetime.now().isoformat(),
                "test_objective": "Week 1, Day 1: Validate EEA data source for Berlin",
                "phase": "Phase 1 - Proof of Concept",
            },
            "data_source_tests": {
                "eea_official_sources": eea_results,
                "alternative_sources": alternative_results,
            },
            "eaqi_documentation": self.document_eaqi_calculation(),
            "findings": {
                "accessible_sources": 0,
                "total_sources_tested": len(eea_results) + len(alternative_results),
                "recommended_approach": "",
                "challenges_identified": [],
                "next_steps": [],
            },
        }

        # Analyze results
        accessible_eea = sum(
            1 for r in eea_results.values() if r.get("accessible", False)
        )
        accessible_alt = sum(
            1 for r in alternative_results.values() if r.get("accessible", False)
        )

        summary["findings"]["accessible_sources"] = accessible_eea + accessible_alt
        summary["findings"]["eea_sources_accessible"] = accessible_eea
        summary["findings"]["alternative_sources_accessible"] = accessible_alt

        # Recommendations based on results
        if accessible_eea > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Proceed with EEA official sources"
            summary["findings"]["next_steps"].append(
                "Implement EEA data extraction methods"
            )
        elif accessible_alt > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Use alternative German air quality sources"
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
            summary["findings"]["next_steps"].append("Research EEA API documentation")
            summary["findings"]["next_steps"].append(
                "Contact EEA support for data access guidance"
            )

        return summary

    def save_test_results(self, summary: Dict) -> None:
        """Save test results to output directory."""

        # Save main summary
        summary_path = self.output_dir / "berlin_eea_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Test summary saved to {summary_path}")

        # Save simplified CSV for easy viewing
        csv_data = []

        for source_type in ["eea_official_sources", "alternative_sources"]:
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

        csv_path = self.output_dir / "berlin_sources_test_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        log.info(f"CSV results saved to {csv_path}")


def main():
    """Execute Berlin EEA data collection test - Week 1, Day 1."""

    log.info("Starting Berlin EEA Data Collection Test - Week 1, Day 1")
    log.info("=" * 60)

    # Initialize collector
    collector = BerlinEEACollector()

    # Test EEA official sources
    log.info("Phase 1: Testing EEA official data sources...")
    eea_results = collector.test_eea_data_sources()

    # Test alternative sources
    log.info("Phase 2: Testing alternative Berlin air quality sources...")
    alternative_results = collector.search_alternative_berlin_sources()

    # Create test summary
    log.info("Phase 3: Creating test summary and recommendations...")
    summary = collector.create_test_summary(eea_results, alternative_results)

    # Save results
    collector.save_test_results(summary)

    # Print summary report
    print("\n" + "=" * 60)
    print("BERLIN EEA DATA COLLECTION TEST - WEEK 1, DAY 1")
    print("=" * 60)

    print(f"\nTEST OBJECTIVE:")
    print(f"Validate EEA data source availability for Berlin, Germany")
    print(f"Document access patterns and EAQI calculation compatibility")

    print(f"\nRESULTS SUMMARY:")
    print(f"• Total sources tested: {summary['findings']['total_sources_tested']}")
    print(f"• Accessible sources: {summary['findings']['accessible_sources']}")
    print(
        f"• EEA official sources accessible: {summary['findings']['eea_sources_accessible']}"
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

    print(f"\nEAQI CALCULATION:")
    eaqi_doc = summary["eaqi_documentation"]
    print(f"• Standard: {eaqi_doc['standard_name']}")
    print(f"• Scale: {eaqi_doc['scale']} ({len(eaqi_doc['categories'])} categories)")
    print(f"• Pollutants: {', '.join(eaqi_doc['pollutants'])}")
    print(
        f"• Method: {eaqi_doc['calculation_method']} (worst pollutant determines overall EAQI)"
    )

    print("\n" + "=" * 60)
    print("WEEK 1, DAY 1 TEST COMPLETE")
    print("Berlin EEA data source assessment finished")
    print("Ready to proceed based on findings and recommendations")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
