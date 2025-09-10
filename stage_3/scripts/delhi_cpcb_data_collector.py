#!/usr/bin/env python3
"""
Delhi CPCB Portal Data Collection - Week 1, Day 3
=================================================

Third step of the Global 100-City Data Collection Strategy.
Test Indian Central Pollution Control Board (CPCB) portal data access for Delhi.

Objective: Validate CPCB portal data source availability and document access patterns.
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


class DelhiCPCBCollector:
    """Test CPCB portal data collection for Delhi as representative Asian city."""

    def __init__(self, output_dir: str = "data/analysis/delhi_cpcb_test"):
        """Initialize Delhi CPCB data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Delhi city configuration
        self.city_config = {
            "name": "Delhi",
            "country": "India",
            "lat": 28.6139,
            "lon": 77.2090,
            "aqi_standard": "Indian",
            "continent": "asia",
        }

        self.session = self._create_session()

        log.info("Delhi CPCB Data Collector initialized")
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

    def test_cpcb_official_sources(self) -> Dict:
        """Test various CPCB and Indian government data source endpoints for Delhi air quality data."""

        log.info("Testing CPCB and Indian government data source availability...")

        # Known CPCB and Indian government endpoints to test
        test_sources = {
            "cpcb_main": "https://cpcb.nic.in/",
            "cpcb_aqms": "https://app.cpcbccr.com/",
            "cpcb_realtime": "https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing",
            "moefcc_portal": "https://moef.gov.in/",
            "india_gov_env": "https://www.india.gov.in/topics/environment",
            "delhi_pollution": "https://dpcc.delhigovt.nic.in/",
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

    def search_alternative_delhi_sources(self) -> Dict:
        """Search for alternative Delhi air quality data sources."""

        log.info("Searching for alternative Delhi air quality data sources...")

        # Alternative Indian/Delhi air quality sources
        alternative_sources = {
            "waqi_delhi": "https://waqi.info/",
            "iqair_delhi": "https://www.iqair.com/india/delhi",
            "safar_india": "https://safar.tropmet.res.in/",
            "aqi_in": "https://aqi.in/dashboard/india/delhi/delhi",
            "delhi_opendata": "https://data.gov.in/",
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
                            "aqi",
                            "pm2.5",
                            "pm10",
                            "no2",
                            "ozone",
                            "pollution",
                            "delhi",
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

    def document_indian_aqi_calculation(self) -> Dict:
        """Document Indian National AQI calculation method for Delhi."""

        log.info("Documenting Indian National AQI calculation method...")

        # Indian National Air Quality Index bands and thresholds
        indian_aqi_bands = {
            "1": {
                "range": "0-50",
                "category": "Good",
                "color": "#009933",
                "health_implications": "Minimal impact",
                "pm25_max": 30,
                "pm10_max": 50,
                "no2_max": 40,
                "o3_max": 50,
                "so2_max": 40,
                "co_max": 1.0,
            },
            "2": {
                "range": "51-100",
                "category": "Satisfactory",
                "color": "#58FF09",
                "health_implications": "Minor breathing discomfort to sensitive people",
                "pm25_max": 60,
                "pm10_max": 100,
                "no2_max": 80,
                "o3_max": 100,
                "so2_max": 80,
                "co_max": 2.0,
            },
            "3": {
                "range": "101-200",
                "category": "Moderately Polluted",
                "color": "#FFFF00",
                "health_implications": "Breathing discomfort to people with lung, asthma and heart diseases",
                "pm25_max": 90,
                "pm10_max": 250,
                "no2_max": 180,
                "o3_max": 168,
                "so2_max": 380,
                "co_max": 10.0,
            },
            "4": {
                "range": "201-300",
                "category": "Poor",
                "color": "#FF9933",
                "health_implications": "Breathing discomfort to most people on prolonged exposure",
                "pm25_max": 120,
                "pm10_max": 350,
                "no2_max": 280,
                "o3_max": 208,
                "so2_max": 800,
                "co_max": 17.0,
            },
            "5": {
                "range": "301-400",
                "category": "Very Poor",
                "color": "#FF3300",
                "health_implications": "Respiratory illness on prolonged exposure",
                "pm25_max": 250,
                "pm10_max": 430,
                "no2_max": 400,
                "o3_max": 748,
                "so2_max": 1600,
                "co_max": 34.0,
            },
            "6": {
                "range": "401-500",
                "category": "Severe",
                "color": "#990000",
                "health_implications": "Affects healthy people and seriously impacts those with existing diseases",
                "pm25_max": float("inf"),
                "pm10_max": float("inf"),
                "no2_max": float("inf"),
                "o3_max": float("inf"),
                "so2_max": float("inf"),
                "co_max": float("inf"),
            },
        }

        return {
            "standard_name": "Indian National Air Quality Index",
            "scale": "0-500",
            "categories": indian_aqi_bands,
            "calculation_method": "worst_pollutant",
            "pollutants": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
            "units": {
                "PM2.5": "μg/m³",
                "PM10": "μg/m³",
                "NO2": "μg/m³",
                "O3": "μg/m³",
                "SO2": "μg/m³",
                "CO": "mg/m³",
            },
            "reference": "https://cpcb.nic.in/air-quality-standard/",
        }

    def calculate_indian_aqi(self, pollutants: Dict[str, float]) -> Dict:
        """Calculate Indian National AQI from pollutant concentrations."""

        aqi_doc = self.document_indian_aqi_calculation()
        bands = aqi_doc["categories"]

        individual_aqis = {}

        for pollutant, concentration in pollutants.items():
            if concentration is None or np.isnan(concentration):
                individual_aqis[pollutant] = None
                continue

            pollutant_key = f"{pollutant.lower()}_max"

            for band_num, band_info in bands.items():
                if pollutant_key in band_info:
                    if concentration <= band_info[pollutant_key]:
                        # Calculate sub-index using linear interpolation
                        if band_num == "1":
                            aqi_low, aqi_high = 0, 50
                            conc_low = 0
                        elif band_num == "2":
                            aqi_low, aqi_high = 51, 100
                            conc_low = bands["1"][pollutant_key]
                        elif band_num == "3":
                            aqi_low, aqi_high = 101, 200
                            conc_low = bands["2"][pollutant_key]
                        elif band_num == "4":
                            aqi_low, aqi_high = 201, 300
                            conc_low = bands["3"][pollutant_key]
                        elif band_num == "5":
                            aqi_low, aqi_high = 301, 400
                            conc_low = bands["4"][pollutant_key]
                        else:  # band_num == "6"
                            aqi_low, aqi_high = 401, 500
                            conc_low = bands["5"][pollutant_key]

                        conc_high = band_info[pollutant_key]

                        if conc_high == float("inf"):
                            sub_index = 500  # Maximum AQI for severe category
                        else:
                            # Linear interpolation formula
                            sub_index = (
                                (aqi_high - aqi_low) / (conc_high - conc_low)
                            ) * (concentration - conc_low) + aqi_low

                        individual_aqis[pollutant] = {
                            "sub_index": round(sub_index),
                            "category": band_info["category"],
                            "concentration": concentration,
                            "band": int(band_num),
                        }
                        break

        # Overall AQI is the worst individual pollutant sub-index
        valid_aqis = [v["sub_index"] for v in individual_aqis.values() if v is not None]

        if valid_aqis:
            overall_aqi = max(valid_aqis)

            # Determine overall category
            if overall_aqi <= 50:
                overall_category = "Good"
                overall_band = 1
            elif overall_aqi <= 100:
                overall_category = "Satisfactory"
                overall_band = 2
            elif overall_aqi <= 200:
                overall_category = "Moderately Polluted"
                overall_band = 3
            elif overall_aqi <= 300:
                overall_category = "Poor"
                overall_band = 4
            elif overall_aqi <= 400:
                overall_category = "Very Poor"
                overall_band = 5
            else:
                overall_category = "Severe"
                overall_band = 6

            # Find dominant pollutant
            dominant_pollutant = None
            for pollutant, aqi_info in individual_aqis.items():
                if aqi_info and aqi_info["sub_index"] == overall_aqi:
                    dominant_pollutant = pollutant
                    break
        else:
            overall_aqi = None
            overall_category = "No Data"
            overall_band = None
            dominant_pollutant = None

        return {
            "overall_aqi": overall_aqi,
            "overall_category": overall_category,
            "overall_band": overall_band,
            "dominant_pollutant": dominant_pollutant,
            "individual_aqis": individual_aqis,
            "calculated_at": datetime.now().isoformat(),
        }

    def create_test_summary(
        self, cpcb_results: Dict, alternative_results: Dict
    ) -> Dict:
        """Create summary of Delhi CPCB data collection test."""

        summary = {
            "test_info": {
                "city": self.city_config,
                "test_date": datetime.now().isoformat(),
                "test_objective": "Week 1, Day 3: Validate CPCB portal data source for Delhi",
                "phase": "Phase 1 - Proof of Concept",
            },
            "data_source_tests": {
                "cpcb_official_sources": cpcb_results,
                "alternative_sources": alternative_results,
            },
            "indian_aqi_documentation": self.document_indian_aqi_calculation(),
            "findings": {
                "accessible_sources": 0,
                "total_sources_tested": len(cpcb_results) + len(alternative_results),
                "recommended_approach": "",
                "challenges_identified": [],
                "next_steps": [],
            },
        }

        # Analyze results
        accessible_cpcb = sum(
            1 for r in cpcb_results.values() if r.get("accessible", False)
        )
        accessible_alt = sum(
            1 for r in alternative_results.values() if r.get("accessible", False)
        )

        summary["findings"]["accessible_sources"] = accessible_cpcb + accessible_alt
        summary["findings"]["cpcb_sources_accessible"] = accessible_cpcb
        summary["findings"]["alternative_sources_accessible"] = accessible_alt

        # Recommendations based on results
        if accessible_cpcb > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Proceed with CPCB official sources"
            summary["findings"]["next_steps"].append(
                "Implement CPCB portal data extraction methods"
            )
            summary["findings"]["next_steps"].append(
                "Navigate government portal authentication if required"
            )
        elif accessible_alt > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Use alternative Indian air quality sources (WAQI, IQAir)"
            summary["findings"]["next_steps"].append(
                "Implement alternative source extraction methods"
            )
            summary["findings"]["next_steps"].append(
                "Validate data quality against CPCB when possible"
            )
        else:
            summary["findings"][
                "recommended_approach"
            ] = "Investigate API-based access or satellite data fallback"
            summary["findings"]["challenges_identified"].append(
                "No direct web access to CPCB or alternative data sources"
            )
            summary["findings"]["next_steps"].append("Research CPCB API documentation")
            summary["findings"]["next_steps"].append(
                "Contact CPCB support for data access guidance"
            )
            summary["findings"]["next_steps"].append(
                "Prepare NASA satellite data as fallback for Asian cities"
            )

        # Add complexity assessment
        if accessible_cpcb < len(cpcb_results) // 2:
            summary["findings"]["challenges_identified"].append(
                "Government portal access complexity confirmed (as expected)"
            )

        return summary

    def save_test_results(self, summary: Dict) -> None:
        """Save test results to output directory."""

        # Save main summary
        summary_path = self.output_dir / "delhi_cpcb_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Test summary saved to {summary_path}")

        # Save simplified CSV for easy viewing
        csv_data = []

        for source_type in ["cpcb_official_sources", "alternative_sources"]:
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

        csv_path = self.output_dir / "delhi_sources_test_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        log.info(f"CSV results saved to {csv_path}")


def main():
    """Execute Delhi CPCB data collection test - Week 1, Day 3."""

    log.info("Starting Delhi CPCB Data Collection Test - Week 1, Day 3")
    log.info("=" * 65)

    # Initialize collector
    collector = DelhiCPCBCollector()

    # Test CPCB official sources
    log.info("Phase 1: Testing CPCB and Indian government official data sources...")
    cpcb_results = collector.test_cpcb_official_sources()

    # Test alternative sources
    log.info("Phase 2: Testing alternative Delhi air quality sources...")
    alternative_results = collector.search_alternative_delhi_sources()

    # Create test summary
    log.info("Phase 3: Creating test summary and recommendations...")
    summary = collector.create_test_summary(cpcb_results, alternative_results)

    # Save results
    collector.save_test_results(summary)

    # Print summary report
    print("\n" + "=" * 65)
    print("DELHI CPCB DATA COLLECTION TEST - WEEK 1, DAY 3")
    print("=" * 65)

    print(f"\nTEST OBJECTIVE:")
    print(f"Validate CPCB portal data source availability for Delhi")
    print(f"Document access patterns and Indian National AQI calculation compatibility")

    print(f"\nRESULTS SUMMARY:")
    print(f"• Total sources tested: {summary['findings']['total_sources_tested']}")
    print(f"• Accessible sources: {summary['findings']['accessible_sources']}")
    print(
        f"• CPCB official sources accessible: {summary['findings']['cpcb_sources_accessible']}"
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

    print(f"\nINDIAN NATIONAL AQI CALCULATION:")
    aqi_doc = summary["indian_aqi_documentation"]
    print(f"• Standard: {aqi_doc['standard_name']}")
    print(f"• Scale: {aqi_doc['scale']} ({len(aqi_doc['categories'])} categories)")
    print(f"• Pollutants: {', '.join(aqi_doc['pollutants'])}")
    print(
        f"• Method: {aqi_doc['calculation_method']} (worst pollutant determines overall AQI)"
    )

    # Test Indian AQI calculation with sample data
    print(f"\nSAMPLE INDIAN AQI CALCULATION:")
    sample_pollutants = {"PM2.5": 85.0, "PM10": 150.0, "NO2": 65.0, "O3": 80.0}
    sample_aqi = collector.calculate_indian_aqi(sample_pollutants)
    if sample_aqi.get("overall_aqi"):
        print(
            f"• Sample pollutants: PM2.5=85ug/m3, PM10=150ug/m3, NO2=65ug/m3, O3=80ug/m3"
        )
        print(
            f"• Calculated AQI: {sample_aqi['overall_aqi']} ({sample_aqi['overall_category']})"
        )
        print(f"• Dominant pollutant: {sample_aqi['dominant_pollutant']}")
        if sample_aqi["overall_band"]:
            health_info = aqi_doc["categories"][str(sample_aqi["overall_band"])][
                "health_implications"
            ]
            print(f"• Health implications: {health_info}")

    print("\n" + "=" * 65)
    print("WEEK 1, DAY 3 TEST COMPLETE")
    print("Delhi CPCB portal data source assessment finished")
    print("Ready to proceed based on findings and recommendations")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    exit(main())
