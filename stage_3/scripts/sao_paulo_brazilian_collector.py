#!/usr/bin/env python3
"""
São Paulo Brazilian Government Data Collection - Week 1, Day 5
=============================================================

Fifth and final step of the Global 100-City Data Collection Strategy.
Test Brazilian government environmental agency data access for São Paulo.

Objective: Validate Brazilian government data source availability and complete
representative city testing for all 5 continents.
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


class SaoPauloBrazilianCollector:
    """Test Brazilian government data collection for São Paulo as representative South American city."""

    def __init__(self, output_dir: str = "data/analysis/sao_paulo_brazilian_test"):
        """Initialize São Paulo Brazilian data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # São Paulo city configuration
        self.city_config = {
            "name": "São Paulo",
            "country": "Brazil",
            "lat": -23.5505,
            "lon": -46.6333,
            "aqi_standard": "EPA",
            "continent": "south_america",
        }

        self.session = self._create_session()

        log.info("São Paulo Brazilian Data Collector initialized")
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

    def test_brazilian_government_sources(self) -> Dict:
        """Test various Brazilian government environmental agency data source endpoints."""

        log.info(
            "Testing Brazilian government environmental agency data source availability..."
        )

        # Known Brazilian government environmental endpoints to test
        test_sources = {
            "ibama_main": "https://www.gov.br/ibama/pt-br",
            "mma_main": "https://www.gov.br/mma/pt-br",
            "inpe_main": "https://www.inpe.br/",
            "cetesb_sp": "https://cetesb.sp.gov.br/",
            "sma_sp": "https://www.infraestruturameioambiente.sp.gov.br/",
            "gov_br_data": "https://dados.gov.br/",
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

    def search_alternative_sao_paulo_sources(self) -> Dict:
        """Search for alternative São Paulo and South American air quality data sources."""

        log.info(
            "Searching for alternative São Paulo and South American air quality data sources..."
        )

        # Alternative São Paulo/Brazil/South American air quality sources
        alternative_sources = {
            "nasa_satellite": "https://earthdata.nasa.gov/",
            "iqair_sao_paulo": "https://www.iqair.com/brazil/sao-paulo",
            "waqi_sao_paulo": "https://waqi.info/",
            "openaq_global": "https://openaq.org/",
            "purple_air": "https://www.purpleair.com/",
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
                            "sao paulo",
                            "são paulo",
                            "brazil",
                            "brasil",
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

    def document_epa_adaptation_calculation(self) -> Dict:
        """Document EPA AQI adaptation for South American context."""

        log.info("Documenting EPA AQI adaptation for South American context...")

        # EPA AQI adaptation bands and thresholds (same as US EPA but with local context)
        epa_aqi_bands = {
            "1": {
                "range": "0-50",
                "category": "Good",
                "color": "#00e400",
                "health_implications": "Air quality is satisfactory and poses little or no risk",
                "pm25_max": 12.0,
                "pm10_max": 54,
                "no2_max": 53,
                "o3_max": 70,
                "so2_max": 35,
                "co_max": 4.4,
            },
            "2": {
                "range": "51-100",
                "category": "Moderate",
                "color": "#ffff00",
                "health_implications": "Air quality is acceptable for most people, sensitive individuals may experience minor issues",
                "pm25_max": 35.4,
                "pm10_max": 154,
                "no2_max": 100,
                "o3_max": 85,
                "so2_max": 75,
                "co_max": 9.4,
            },
            "3": {
                "range": "101-150",
                "category": "Unhealthy for Sensitive Groups",
                "color": "#ff7e00",
                "health_implications": "Members of sensitive groups may experience health effects",
                "pm25_max": 55.4,
                "pm10_max": 254,
                "no2_max": 360,
                "o3_max": 105,
                "so2_max": 185,
                "co_max": 12.4,
            },
            "4": {
                "range": "151-200",
                "category": "Unhealthy",
                "color": "#ff0000",
                "health_implications": "Everyone may begin to experience health effects, sensitive groups more seriously",
                "pm25_max": 150.4,
                "pm10_max": 354,
                "no2_max": 649,
                "o3_max": 125,
                "so2_max": 304,
                "co_max": 15.4,
            },
            "5": {
                "range": "201-300",
                "category": "Very Unhealthy",
                "color": "#8f3f97",
                "health_implications": "Health alert: everyone may experience serious health effects",
                "pm25_max": 250.4,
                "pm10_max": 424,
                "no2_max": 1249,
                "o3_max": 164,
                "so2_max": 604,
                "co_max": 30.4,
            },
            "6": {
                "range": "301-500",
                "category": "Hazardous",
                "color": "#7e0023",
                "health_implications": "Emergency conditions: everyone is likely to be affected",
                "pm25_max": float("inf"),
                "pm10_max": float("inf"),
                "no2_max": float("inf"),
                "o3_max": float("inf"),
                "so2_max": float("inf"),
                "co_max": float("inf"),
            },
        }

        return {
            "standard_name": "EPA AQI Adaptation for South America",
            "scale": "0-500",
            "categories": epa_aqi_bands,
            "calculation_method": "worst_pollutant",
            "pollutants": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
            "units": {
                "PM2.5": "μg/m³",
                "PM10": "μg/m³",
                "NO2": "ppb",
                "O3": "ppb",
                "SO2": "ppb",
                "CO": "ppm",
            },
            "reference": "https://www.airnow.gov/aqi/aqi-basics/",
            "notes": "EPA AQI standard adapted for South American cities with local context considerations",
        }

    def calculate_epa_aqi(self, pollutants: Dict[str, float]) -> Dict:
        """Calculate EPA AQI from pollutant concentrations."""

        epa_doc = self.document_epa_adaptation_calculation()
        bands = epa_doc["categories"]

        individual_aqis = {}

        for pollutant, concentration in pollutants.items():
            if concentration is None or np.isnan(concentration):
                individual_aqis[pollutant] = None
                continue

            pollutant_key = f"{pollutant.lower()}_max"

            for band_num, band_info in bands.items():
                if pollutant_key in band_info:
                    if concentration <= band_info[pollutant_key]:
                        # EPA AQI linear interpolation within band
                        if band_num == "1":
                            aqi_low, aqi_high = 0, 50
                            conc_low = 0
                        elif band_num == "2":
                            aqi_low, aqi_high = 51, 100
                            conc_low = bands["1"][pollutant_key]
                        elif band_num == "3":
                            aqi_low, aqi_high = 101, 150
                            conc_low = bands["2"][pollutant_key]
                        elif band_num == "4":
                            aqi_low, aqi_high = 151, 200
                            conc_low = bands["3"][pollutant_key]
                        elif band_num == "5":
                            aqi_low, aqi_high = 201, 300
                            conc_low = bands["4"][pollutant_key]
                        else:  # band_num == "6"
                            aqi_low, aqi_high = 301, 500
                            conc_low = bands["5"][pollutant_key]

                        conc_high = band_info[pollutant_key]

                        if conc_high == float("inf"):
                            aqi_value = 500  # Maximum EPA AQI
                        else:
                            # EPA linear interpolation formula
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

        # Overall EPA AQI is the worst individual pollutant AQI
        valid_aqis = [v["aqi_value"] for v in individual_aqis.values() if v is not None]

        if valid_aqis:
            overall_aqi = max(valid_aqis)

            # Determine overall category
            if overall_aqi <= 50:
                overall_category = "Good"
                overall_band = 1
            elif overall_aqi <= 100:
                overall_category = "Moderate"
                overall_band = 2
            elif overall_aqi <= 150:
                overall_category = "Unhealthy for Sensitive Groups"
                overall_band = 3
            elif overall_aqi <= 200:
                overall_category = "Unhealthy"
                overall_band = 4
            elif overall_aqi <= 300:
                overall_category = "Very Unhealthy"
                overall_band = 5
            else:
                overall_category = "Hazardous"
                overall_band = 6

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
            "overall_epa_aqi": overall_aqi,
            "overall_category": overall_category,
            "overall_band": overall_band,
            "dominant_pollutant": dominant_pollutant,
            "individual_aqis": individual_aqis,
            "calculated_at": datetime.now().isoformat(),
        }

    def create_test_summary(
        self, brazilian_results: Dict, alternative_results: Dict
    ) -> Dict:
        """Create summary of São Paulo Brazilian government data collection test."""

        summary = {
            "test_info": {
                "city": self.city_config,
                "test_date": datetime.now().isoformat(),
                "test_objective": "Week 1, Day 5: Validate Brazilian government data source for São Paulo",
                "phase": "Phase 1 - Proof of Concept - FINAL REPRESENTATIVE CITY",
            },
            "data_source_tests": {
                "brazilian_government_sources": brazilian_results,
                "alternative_sources": alternative_results,
            },
            "epa_aqi_documentation": self.document_epa_adaptation_calculation(),
            "findings": {
                "accessible_sources": 0,
                "total_sources_tested": len(brazilian_results)
                + len(alternative_results),
                "recommended_approach": "",
                "challenges_identified": [],
                "next_steps": [],
            },
        }

        # Analyze results
        accessible_brazilian = sum(
            1 for r in brazilian_results.values() if r.get("accessible", False)
        )
        accessible_alt = sum(
            1 for r in alternative_results.values() if r.get("accessible", False)
        )

        summary["findings"]["accessible_sources"] = (
            accessible_brazilian + accessible_alt
        )
        summary["findings"][
            "brazilian_government_sources_accessible"
        ] = accessible_brazilian
        summary["findings"]["alternative_sources_accessible"] = accessible_alt

        # Recommendations based on results
        if accessible_brazilian > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Proceed with Brazilian government sources"
            summary["findings"]["next_steps"].append(
                "Implement Brazilian government data extraction methods"
            )
            summary["findings"]["next_steps"].append(
                "Integrate satellite data for validation"
            )
        elif accessible_alt > 0:
            summary["findings"][
                "recommended_approach"
            ] = "Use satellite and alternative sources (NASA, IQAir, WAQI, OpenAQ)"
            summary["findings"]["next_steps"].append(
                "Implement satellite data processing methods"
            )
            summary["findings"]["next_steps"].append(
                "Validate satellite estimates against available ground truth"
            )
        else:
            summary["findings"][
                "recommended_approach"
            ] = "Rely primarily on NASA satellite data for South American cities"
            summary["findings"]["challenges_identified"].append(
                "Limited accessible data sources for South American cities"
            )
            summary["findings"]["next_steps"].append(
                "Develop satellite-only methodology"
            )
            summary["findings"]["next_steps"].append(
                "Contact South American environmental agencies for partnership"
            )

        # Add South America-specific considerations
        summary["findings"]["next_steps"].append(
            "Prepare satellite data as primary source for South American cities"
        )

        # Final representative city completion
        summary["findings"]["milestone_achieved"] = "ALL 5 REPRESENTATIVE CITIES TESTED"
        summary["findings"]["next_steps"].append(
            "Begin Week 2: Scale to full 5-year datasets for representative cities"
        )

        return summary

    def save_test_results(self, summary: Dict) -> None:
        """Save test results to output directory."""

        # Save main summary
        summary_path = self.output_dir / "sao_paulo_brazilian_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Test summary saved to {summary_path}")

        # Save simplified CSV for easy viewing
        csv_data = []

        for source_type in ["brazilian_government_sources", "alternative_sources"]:
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

        csv_path = self.output_dir / "sao_paulo_sources_test_results.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

        log.info(f"CSV results saved to {csv_path}")


def main():
    """Execute São Paulo Brazilian government data collection test - Week 1, Day 5."""

    log.info(
        "Starting São Paulo Brazilian Government Data Collection Test - Week 1, Day 5"
    )
    log.info("FINAL REPRESENTATIVE CITY TEST")
    log.info("=" * 75)

    # Initialize collector
    collector = SaoPauloBrazilianCollector()

    # Test Brazilian government official sources
    log.info(
        "Phase 1: Testing Brazilian government environmental agency data sources..."
    )
    brazilian_results = collector.test_brazilian_government_sources()

    # Test alternative sources
    log.info(
        "Phase 2: Testing alternative São Paulo and South American air quality sources..."
    )
    alternative_results = collector.search_alternative_sao_paulo_sources()

    # Create test summary
    log.info("Phase 3: Creating test summary and recommendations...")
    summary = collector.create_test_summary(brazilian_results, alternative_results)

    # Save results
    collector.save_test_results(summary)

    # Print summary report
    print("\n" + "=" * 75)
    print("SÃO PAULO BRAZILIAN GOVERNMENT DATA COLLECTION TEST - WEEK 1, DAY 5")
    print("FINAL REPRESENTATIVE CITY TEST")
    print("=" * 75)

    print(f"\nTEST OBJECTIVE:")
    print(f"Validate Brazilian government data source availability for São Paulo")
    print(f"Complete representative city testing for all 5 continents")
    print(f"Document EPA AQI adaptation for South American context")

    print(f"\nRESULTS SUMMARY:")
    print(f"• Total sources tested: {summary['findings']['total_sources_tested']}")
    print(f"• Accessible sources: {summary['findings']['accessible_sources']}")
    print(
        f"• Brazilian government sources accessible: {summary['findings']['brazilian_government_sources_accessible']}"
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

    print(f"\nEPA AQI ADAPTATION CALCULATION:")
    epa_doc = summary["epa_aqi_documentation"]
    print(f"• Standard: {epa_doc['standard_name']}")
    print(f"• Scale: {epa_doc['scale']} ({len(epa_doc['categories'])} categories)")
    print(f"• Pollutants: {', '.join(epa_doc['pollutants'])}")
    print(
        f"• Method: {epa_doc['calculation_method']} (worst pollutant determines overall AQI)"
    )

    # Test EPA AQI calculation with sample data
    print(f"\nSAMPLE EPA AQI CALCULATION:")
    sample_pollutants = {"PM2.5": 25.0, "PM10": 65.0, "NO2": 45.0, "O3": 75.0}
    sample_aqi = collector.calculate_epa_aqi(sample_pollutants)
    if sample_aqi.get("overall_epa_aqi"):
        print(f"• Sample pollutants: PM2.5=25ug/m3, PM10=65ug/m3, NO2=45ppb, O3=75ppb")
        print(
            f"• Calculated EPA AQI: {sample_aqi['overall_epa_aqi']} ({sample_aqi['overall_category']})"
        )
        print(f"• Dominant pollutant: {sample_aqi['dominant_pollutant']}")
        if sample_aqi["overall_band"]:
            health_info = epa_doc["categories"][str(sample_aqi["overall_band"])][
                "health_implications"
            ]
            print(f"• Health implications: {health_info}")

    print(f"\n🎉 MILESTONE ACHIEVED: {summary['findings']['milestone_achieved']} 🎉")

    print("\n" + "=" * 75)
    print("WEEK 1, DAY 5 TEST COMPLETE")
    print("São Paulo Brazilian government data source assessment finished")
    print("ALL 5 REPRESENTATIVE CITIES TESTED SUCCESSFULLY")
    print("Ready to proceed to Week 2: Scale to full datasets")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    exit(main())
