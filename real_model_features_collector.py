#!/usr/bin/env python3
"""
Real Model Features Collector
============================

Collects REAL fire activity and holiday data from authentic sources:
- NASA FIRMS (Fire Information for Resource Management System)
- date.nager.at (Public Holiday API)

No synthetic data, no simulation, no mathematical generation.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("real_model_features_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class RateLimitedSession:
    """Session with rate limiting and retry logic."""

    def __init__(self, rate_limit_delay: float = 1.0):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def get(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited GET request."""
        # Ensure rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        try:
            response = self.session.get(url, **kwargs)
            self.last_request_time = time.time()
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            log.warning(f"Request failed for {url}: {e}")
            self.last_request_time = time.time()
            raise


class RealFireDataCollector:
    """Collect REAL fire detection data from NASA FIRMS API."""

    def __init__(self):
        self.session = RateLimitedSession(rate_limit_delay=2.0)
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"

        # Country code mapping for FIRMS API
        self.country_codes = {
            "India": "IND",
            "Pakistan": "PAK",
            "Bangladesh": "BGD",
            "China": "CHN",
            "Afghanistan": "AFG",
            "Mongolia": "MNG",
            "Kazakhstan": "KAZ",
            "Tajikistan": "TJK",
            "Uzbekistan": "UZB",
            "Egypt": "EGY",
            "Sudan": "SDN",
            "Chad": "TCD",
            "Mali": "MLI",
            "Burkina Faso": "BFA",
            "Nigeria": "NGA",
            "Ghana": "GHA",
            "Uganda": "UGA",
            "Senegal": "SEN",
            "Ivory Coast": "CIV",
            "Morocco": "MAR",
            "Libya": "LBY",
            "Cameroon": "CMR",
            "Republic of the Congo": "COG",
            "Democratic Republic of the Congo": "COD",
            "North Macedonia": "MKD",
            "Bosnia and Herzegovina": "BIH",
            "Bulgaria": "BGR",
            "Poland": "POL",
            "Czech Republic": "CZE",
            "Romania": "ROU",
            "Serbia": "SRB",
            "Italy": "ITA",
            "Hungary": "HUN",
            "Slovakia": "SVK",
            "Mexico": "MEX",
            "USA": "USA",
            "Canada": "CAN",
            "Peru": "PER",
            "Bolivia": "BOL",
            "Chile": "CHL",
            "Brazil": "BRA",
            "Colombia": "COL",
            "Ecuador": "ECU",
            "Argentina": "ARG",
            "Uruguay": "URY",
        }

    def get_country_fires(self, country: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get real fire detections for a country from NASA FIRMS."""
        country_code = self.country_codes.get(country)
        if not country_code:
            log.warning(f"No country code mapping for {country}")
            return []

        # FIRMS provides data for last 24h, 48h, 7d
        if days <= 1:
            period = "24h"
        elif days <= 2:
            period = "48h"
        else:
            period = "7d"

        # Use SUOMI-NPP VIIRS C2 data (most recent and accurate)
        csv_url = f"https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_{country_code}_{period}.csv"

        try:
            log.info(
                f"Fetching real fire data for {country} ({country_code}) from NASA FIRMS"
            )
            response = self.session.get(csv_url)

            # Parse CSV data
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                log.info(f"No fire data found for {country}")
                return []

            headers = lines[0].split(",")
            fires = []

            for line in lines[1:]:
                values = line.split(",")
                if len(values) >= len(headers):
                    fire_data = dict(zip(headers, values))
                    try:
                        fires.append(
                            {
                                "latitude": float(fire_data.get("latitude", 0)),
                                "longitude": float(fire_data.get("longitude", 0)),
                                "brightness": float(fire_data.get("brightness", 0)),
                                "confidence": fire_data.get("confidence", "unknown"),
                                "frp": float(
                                    fire_data.get("frp", 0)
                                ),  # Fire Radiative Power
                                "scan": float(fire_data.get("scan", 0)),
                                "track": float(fire_data.get("track", 0)),
                                "acq_date": fire_data.get("acq_date", ""),
                                "acq_time": fire_data.get("acq_time", ""),
                                "satellite": fire_data.get("satellite", "SUOMI_NPP"),
                                "instrument": fire_data.get("instrument", "VIIRS"),
                                "data_source": "NASA_FIRMS_REAL",
                            }
                        )
                    except ValueError as e:
                        log.warning(f"Invalid fire data in {country}: {e}")
                        continue

            log.info(f"Collected {len(fires)} real fire detections for {country}")
            return fires

        except Exception as e:
            log.error(f"Failed to get real fire data for {country}: {e}")
            return []

    def calculate_fire_impact(
        self,
        fires: List[Dict],
        city_lat: float,
        city_lon: float,
        radius_km: float = 100,
    ) -> Dict[str, Any]:
        """Calculate real fire impact metrics for a city."""
        if not fires:
            return {
                "fire_count": 0,
                "total_frp": 0.0,
                "avg_distance_km": 0.0,
                "nearest_fire_km": 0.0,
                "high_confidence_fires": 0,
                "total_brightness": 0.0,
                "data_source": "NASA_FIRMS_REAL",
            }

        nearby_fires = []
        distances = []
        high_conf_count = 0
        total_frp = 0.0
        total_brightness = 0.0

        for fire in fires:
            # Calculate distance using Haversine formula
            fire_lat = fire.get("latitude", 0)
            fire_lon = fire.get("longitude", 0)

            distance = self._haversine_distance(city_lat, city_lon, fire_lat, fire_lon)

            if distance <= radius_km:
                nearby_fires.append(fire)
                distances.append(distance)

                if fire.get("confidence") in ["high", "h"]:
                    high_conf_count += 1

                total_frp += fire.get("frp", 0)
                total_brightness += fire.get("brightness", 0)

        return {
            "fire_count": len(nearby_fires),
            "total_frp": round(total_frp, 2),
            "avg_distance_km": (
                round(sum(distances) / len(distances), 2) if distances else 0.0
            ),
            "nearest_fire_km": round(min(distances), 2) if distances else 0.0,
            "high_confidence_fires": high_conf_count,
            "total_brightness": round(total_brightness, 2),
            "data_source": "NASA_FIRMS_REAL",
        }

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return R * c


class RealHolidayDataCollector:
    """Collect REAL holiday data from public holiday APIs."""

    def __init__(self):
        self.session = RateLimitedSession(rate_limit_delay=1.0)

        # Country code mapping for holiday API
        self.country_codes = {
            "India": "IN",
            "Pakistan": "PK",
            "Bangladesh": "BD",
            "China": "CN",
            "Afghanistan": "AF",
            "Mongolia": "MN",
            "Kazakhstan": "KZ",
            "Tajikistan": "TJ",
            "Uzbekistan": "UZ",
            "Egypt": "EG",
            "Sudan": "SD",
            "Chad": "TD",
            "Mali": "ML",
            "Burkina Faso": "BF",
            "Nigeria": "NG",
            "Ghana": "GH",
            "Uganda": "UG",
            "Senegal": "SN",
            "Ivory Coast": "CI",
            "Morocco": "MA",
            "Libya": "LY",
            "Cameroon": "CM",
            "Republic of the Congo": "CG",
            "Democratic Republic of the Congo": "CD",
            "North Macedonia": "MK",
            "Bosnia and Herzegovina": "BA",
            "Bulgaria": "BG",
            "Poland": "PL",
            "Czech Republic": "CZ",
            "Romania": "RO",
            "Serbia": "RS",
            "Italy": "IT",
            "Hungary": "HU",
            "Slovakia": "SK",
            "Mexico": "MX",
            "USA": "US",
            "Canada": "CA",
            "Peru": "PE",
            "Bolivia": "BO",
            "Chile": "CL",
            "Brazil": "BR",
            "Colombia": "CO",
            "Ecuador": "EC",
            "Argentina": "AR",
            "Uruguay": "UY",
        }

    def get_country_holidays(
        self, country: str, year: int = None
    ) -> List[Dict[str, Any]]:
        """Get real public holidays for a country from date.nager.at API."""
        if year is None:
            year = datetime.now().year

        country_code = self.country_codes.get(country)
        if not country_code:
            log.warning(f"No country code mapping for {country}")
            return []

        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

        try:
            log.info(
                f"Fetching real holiday data for {country} ({country_code}) from date.nager.at"
            )
            response = self.session.get(url)
            holidays = response.json()

            real_holidays = []
            for holiday in holidays:
                real_holidays.append(
                    {
                        "date": holiday["date"],
                        "name": holiday["name"],
                        "local_name": holiday["localName"],
                        "country_code": holiday["countryCode"],
                        "fixed": holiday["fixed"],
                        "global": holiday["global"],
                        "counties": holiday.get("counties", []),
                        "launch_year": holiday.get("launchYear"),
                        "data_source": "date.nager.at_REAL",
                    }
                )

            log.info(f"Collected {len(real_holidays)} real holidays for {country}")
            return real_holidays

        except Exception as e:
            log.error(f"Failed to get real holiday data for {country}: {e}")
            return []

    def calculate_holiday_impact(
        self, holidays: List[Dict], target_date: str
    ) -> Dict[str, Any]:
        """Calculate real holiday impact for a specific date."""
        if not holidays:
            return {
                "is_holiday": False,
                "holiday_name": None,
                "days_to_next_holiday": 365,
                "days_from_last_holiday": 365,
                "holiday_type": "none",
                "data_source": "date.nager.at_REAL",
            }

        # Parse target date
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            log.warning(f"Invalid date format: {target_date}")
            return {"error": "invalid_date"}

        # Check if target date is a holiday
        current_holiday = None
        for holiday in holidays:
            if holiday["date"] == target_date:
                current_holiday = holiday
                break

        # Calculate distances to holidays
        distances = []
        for holiday in holidays:
            try:
                holiday_dt = datetime.strptime(holiday["date"], "%Y-%m-%d")
                days_diff = (holiday_dt - target_dt).days
                distances.append((days_diff, holiday))
            except ValueError:
                continue

        # Find next and previous holidays
        future_holidays = [(days, h) for days, h in distances if days > 0]
        past_holidays = [(abs(days), h) for days, h in distances if days < 0]

        days_to_next = min(future_holidays)[0] if future_holidays else 365
        days_from_last = min(past_holidays)[0] if past_holidays else 365

        return {
            "is_holiday": current_holiday is not None,
            "holiday_name": current_holiday["name"] if current_holiday else None,
            "local_name": current_holiday["local_name"] if current_holiday else None,
            "days_to_next_holiday": days_to_next,
            "days_from_last_holiday": days_from_last,
            "holiday_type": (
                self._classify_holiday_type(current_holiday["name"])
                if current_holiday
                else "none"
            ),
            "is_fixed_holiday": current_holiday["fixed"] if current_holiday else False,
            "is_global_holiday": (
                current_holiday["global"] if current_holiday else False
            ),
            "data_source": "date.nager.at_REAL",
        }

    def _classify_holiday_type(self, holiday_name: str) -> str:
        """Classify holiday type based on name."""
        if not holiday_name:
            return "unknown"

        name_lower = holiday_name.lower()

        religious_keywords = [
            "christmas",
            "easter",
            "good friday",
            "epiphany",
            "assumption",
            "all saints",
        ]
        national_keywords = [
            "independence",
            "national",
            "republic",
            "constitution",
            "liberation",
        ]
        cultural_keywords = ["new year", "labour", "workers", "international"]

        if any(keyword in name_lower for keyword in religious_keywords):
            return "religious"
        elif any(keyword in name_lower for keyword in national_keywords):
            return "national"
        elif any(keyword in name_lower for keyword in cultural_keywords):
            return "cultural"
        else:
            return "other"


class RealModelFeaturesCollector:
    """Main collector that combines real fire and holiday data."""

    def __init__(self):
        self.fire_collector = RealFireDataCollector()
        self.holiday_collector = RealHolidayDataCollector()
        self.output_dir = Path("stage_5/real_model_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_real_features(self, dataset_path: str) -> Dict[str, Any]:
        """Collect real model features for all cities in the dataset."""
        log.info("=== STARTING REAL MODEL FEATURES COLLECTION ===")

        # Load existing dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results = {
            "collection_type": "real_model_features",
            "start_time": datetime.now().isoformat(),
            "data_sources": ["NASA_FIRMS", "date.nager.at"],
            "cities_processed": 0,
            "total_fire_detections": 0,
            "total_holidays_collected": 0,
            "city_results": {},
        }

        for city_name, city_data in dataset["city_results"].items():
            log.info(f"Collecting real features for {city_name}")

            country = city_data.get("country", "Unknown")
            coordinates = city_data.get("coordinates", {})
            lat = coordinates.get("lat", 0)
            lon = coordinates.get("lon", 0)

            # Collect real fire data
            fires = self.fire_collector.get_country_fires(country, days=7)
            fire_impact = self.fire_collector.calculate_fire_impact(fires, lat, lon)

            # Collect real holiday data
            holidays = self.holiday_collector.get_country_holidays(country)

            # Add real features to city data
            city_data["real_fire_data"] = {
                "raw_fires": fires[:10],  # Store first 10 fires as sample
                "fire_impact_metrics": fire_impact,
                "collection_timestamp": datetime.now().isoformat(),
            }

            city_data["real_holiday_data"] = {
                "country_holidays": holidays,
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Add real features to existing data samples
            self._add_real_features_to_samples(city_data, fires, holidays, lat, lon)

            results["city_results"][city_name] = city_data
            results["cities_processed"] += 1
            results["total_fire_detections"] += len(fires)
            results["total_holidays_collected"] += len(holidays)

            # Rate limiting between cities
            time.sleep(1)

        # Save results
        results["end_time"] = datetime.now().isoformat()
        output_file = (
            self.output_dir
            / f"real_model_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log.info(f"Real model features saved to: {output_file}")
        log.info("=== REAL MODEL FEATURES COLLECTION COMPLETED ===")

        return results

    def _add_real_features_to_samples(
        self, city_data: Dict, fires: List, holidays: List, lat: float, lon: float
    ):
        """Add real features to existing data samples."""
        for source_name, source_data in city_data.get("data_sources", {}).items():
            if "data_sample" in source_data:
                for record in source_data["data_sample"]:
                    # Add real fire features
                    record["real_fire_features"] = (
                        self.fire_collector.calculate_fire_impact(fires, lat, lon)
                    )

                    # Add real holiday features
                    record_date = record.get(
                        "date", datetime.now().strftime("%Y-%m-%d")
                    )
                    record["real_holiday_features"] = (
                        self.holiday_collector.calculate_holiday_impact(
                            holidays, record_date
                        )
                    )

            if "historical_data_sample" in source_data:
                for record in source_data["historical_data_sample"]:
                    # Add real fire features
                    record["real_fire_features"] = (
                        self.fire_collector.calculate_fire_impact(fires, lat, lon)
                    )

                    # Add real holiday features
                    record_date = record.get(
                        "date", datetime.now().strftime("%Y-%m-%d")
                    )
                    record["real_holiday_features"] = (
                        self.holiday_collector.calculate_holiday_impact(
                            holidays, record_date
                        )
                    )


def main():
    """Main execution for real model features collection."""
    log.info("Starting Real Model Features Collection")

    try:
        collector = RealModelFeaturesCollector()

        # Use the expanded dataset as input
        dataset_path = (
            "stage_5/expanded_worst_air_quality/expanded_worst_air_quality_results.json"
        )

        if not Path(dataset_path).exists():
            log.error(f"Input dataset not found: {dataset_path}")
            return 1

        results = collector.collect_real_features(dataset_path)

        log.info("Real Model Features Collection completed successfully")
        log.info(f"Processed {results['cities_processed']} cities")
        log.info(f"Collected {results['total_fire_detections']} real fire detections")
        log.info(f"Collected {results['total_holidays_collected']} real holidays")

        return 0

    except KeyboardInterrupt:
        log.info("Collection interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Real model features collection failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
