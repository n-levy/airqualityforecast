#!/usr/bin/env python3
"""
Real Model Features Collector - FIXED VERSION
============================================

Collects REAL fire activity and holiday data from authentic sources:
- NASA MODIS/VIIRS active fire data via web services
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
    """Collect REAL fire detection data from NASA MODIS/VIIRS."""

    def __init__(self):
        self.session = RateLimitedSession(rate_limit_delay=3.0)
        # Using MODIS Collection 6.1 data from NASA
        self.base_url = (
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/txt"
        )

    def get_global_fires(self, days: int = 1) -> List[Dict[str, Any]]:
        """Get real fire detections globally from NASA MODIS."""
        if days <= 1:
            period = "24h"
        elif days <= 2:
            period = "48h"
        else:
            period = "7d"

        # Use global MODIS active fire data
        txt_url = f"https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/txt/MODIS_C6_1_Global_{period}.txt"

        try:
            log.info(f"Fetching real global fire data from NASA MODIS")
            response = self.session.get(txt_url)

            # Parse TXT data
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                log.info("No global fire data found")
                return []

            # Skip header line
            fires = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 13:  # MODIS format has 13+ columns
                    try:
                        fires.append(
                            {
                                "latitude": float(parts[0]),
                                "longitude": float(parts[1]),
                                "brightness": float(parts[2]),
                                "scan": float(parts[3]),
                                "track": float(parts[4]),
                                "acq_date": parts[5],
                                "acq_time": parts[6],
                                "satellite": parts[7],
                                "confidence": float(parts[8]),
                                "version": parts[9],
                                "bright_t31": (
                                    float(parts[10]) if parts[10] != "" else 0
                                ),
                                "frp": float(parts[11]) if parts[11] != "" else 0,
                                "daynight": parts[12] if len(parts) > 12 else "D",
                                "data_source": "NASA_MODIS_REAL",
                            }
                        )
                    except (ValueError, IndexError) as e:
                        log.warning(f"Invalid fire data: {e}")
                        continue

            log.info(f"Collected {len(fires)} real fire detections globally")
            return fires

        except Exception as e:
            log.error(f"Failed to get real global fire data: {e}")
            return []

    def filter_fires_by_location(
        self, fires: List[Dict], lat: float, lon: float, radius_km: float = 200
    ) -> List[Dict]:
        """Filter fires within radius of a location."""
        nearby_fires = []

        for fire in fires:
            fire_lat = fire.get("latitude", 0)
            fire_lon = fire.get("longitude", 0)

            distance = self._haversine_distance(lat, lon, fire_lat, fire_lon)

            if distance <= radius_km:
                fire["distance_km"] = round(distance, 2)
                nearby_fires.append(fire)

        return nearby_fires

    def calculate_fire_impact(
        self,
        fires: List[Dict],
        city_lat: float,
        city_lon: float,
        radius_km: float = 200,
    ) -> Dict[str, Any]:
        """Calculate real fire impact metrics for a city."""
        nearby_fires = self.filter_fires_by_location(
            fires, city_lat, city_lon, radius_km
        )

        if not nearby_fires:
            return {
                "fire_count": 0,
                "total_frp": 0.0,
                "avg_distance_km": 0.0,
                "nearest_fire_km": 0.0,
                "high_confidence_fires": 0,
                "total_brightness": 0.0,
                "day_fires": 0,
                "night_fires": 0,
                "data_source": "NASA_MODIS_REAL",
                "search_radius_km": radius_km,
            }

        distances = []
        high_conf_count = 0
        total_frp = 0.0
        total_brightness = 0.0
        day_fires = 0
        night_fires = 0

        for fire in nearby_fires:
            distances.append(fire.get("distance_km", 0))

            if fire.get("confidence", 0) >= 80:  # High confidence threshold
                high_conf_count += 1

            total_frp += fire.get("frp", 0)
            total_brightness += fire.get("brightness", 0)

            if fire.get("daynight", "D") == "D":
                day_fires += 1
            else:
                night_fires += 1

        return {
            "fire_count": len(nearby_fires),
            "total_frp": round(total_frp, 2),
            "avg_distance_km": (
                round(sum(distances) / len(distances), 2) if distances else 0.0
            ),
            "nearest_fire_km": round(min(distances), 2) if distances else 0.0,
            "high_confidence_fires": high_conf_count,
            "total_brightness": round(total_brightness, 2),
            "day_fires": day_fires,
            "night_fires": night_fires,
            "data_source": "NASA_MODIS_REAL",
            "search_radius_km": radius_km,
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
            log.info(f"Fetching real holiday data for {country} ({country_code})")
            response = self.session.get(url)
            holidays = response.json()

            if not isinstance(holidays, list):
                log.warning(f"Unexpected holiday data format for {country}")
                return []

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
            return {"error": "invalid_date", "data_source": "date.nager.at_REAL"}

        # Check if target date is a holiday
        current_holiday = None
        for holiday in holidays:
            if holiday["date"] == target_date:
                current_holiday = holiday
                break

        # Calculate distances to holidays
        future_distances = []
        past_distances = []

        for holiday in holidays:
            try:
                holiday_dt = datetime.strptime(holiday["date"], "%Y-%m-%d")
                days_diff = (holiday_dt - target_dt).days

                if days_diff > 0:
                    future_distances.append(days_diff)
                elif days_diff < 0:
                    past_distances.append(abs(days_diff))

            except ValueError:
                continue

        days_to_next = min(future_distances) if future_distances else 365
        days_from_last = min(past_distances) if past_distances else 365

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
        self.global_fires = None  # Cache global fires to avoid repeated downloads

    def collect_real_features(self, dataset_path: str) -> Dict[str, Any]:
        """Collect real model features for all cities in the dataset."""
        log.info("=== STARTING REAL MODEL FEATURES COLLECTION ===")

        # Load existing dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results = {
            "collection_type": "real_model_features",
            "start_time": datetime.now().isoformat(),
            "data_sources": ["NASA_MODIS", "date.nager.at"],
            "cities_processed": 0,
            "total_fire_detections": 0,
            "total_holidays_collected": 0,
            "city_results": {},
        }

        # Get global fires once to avoid repeated API calls
        log.info("Fetching global fire data from NASA MODIS...")
        self.global_fires = self.fire_collector.get_global_fires(days=7)
        results["global_fires_downloaded"] = len(self.global_fires)

        for city_name, city_data in dataset["city_results"].items():
            log.info(f"Processing real features for {city_name}")

            country = city_data.get("country", "Unknown")
            coordinates = city_data.get("coordinates", {})
            lat = coordinates.get("lat", 0)
            lon = coordinates.get("lon", 0)

            # Calculate fire impact using global fires
            fire_impact = self.fire_collector.calculate_fire_impact(
                self.global_fires, lat, lon
            )

            # Collect real holiday data
            holidays = self.holiday_collector.get_country_holidays(country)

            # Add real features to city data
            city_data["real_fire_data"] = {
                "fire_impact_metrics": fire_impact,
                "collection_timestamp": datetime.now().isoformat(),
                "search_radius_km": 200,
            }

            city_data["real_holiday_data"] = {
                "country_holidays": holidays,
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Add real features to existing data samples
            self._add_real_features_to_samples(city_data, holidays, lat, lon)

            results["city_results"][city_name] = city_data
            results["cities_processed"] += 1
            results["total_fire_detections"] += fire_impact.get("fire_count", 0)
            results["total_holidays_collected"] += len(holidays)

            # Rate limiting between cities
            time.sleep(0.5)

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
        self, city_data: Dict, holidays: List, lat: float, lon: float
    ):
        """Add real features to existing data samples."""
        for source_name, source_data in city_data.get("data_sources", {}).items():
            if "data_sample" in source_data:
                for record in source_data["data_sample"]:
                    # Add real fire features (use global fire impact for consistency)
                    fire_impact = self.fire_collector.calculate_fire_impact(
                        self.global_fires, lat, lon
                    )
                    record["real_fire_features"] = fire_impact

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
                    fire_impact = self.fire_collector.calculate_fire_impact(
                        self.global_fires, lat, lon
                    )
                    record["real_fire_features"] = fire_impact

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
        log.info(
            f"Global fires downloaded: {results.get('global_fires_downloaded', 0)}"
        )
        log.info(
            f"Total fire detections near cities: {results['total_fire_detections']}"
        )
        log.info(f"Total holidays collected: {results['total_holidays_collected']}")

        return 0

    except KeyboardInterrupt:
        log.info("Collection interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Real model features collection failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
