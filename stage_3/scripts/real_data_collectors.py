#!/usr/bin/env python3
"""
Real Data Collection System using Free APIs

This module implements data collectors for various external data sources
using only free APIs to enhance air quality forecasting features.

Free APIs Used:
- OpenWeatherMap (weather data)
- NASA FIRMS (fire detection)
- OpenStreetMap/Overpass API (construction, traffic infrastructure)
- Public Holiday APIs
- NASA Earth Data (satellite observations)
- USGS Earthquake API (geological events)
"""

from __future__ import annotations

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
import math

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# City coordinates for data collection
CITY_COORDS = {
    "Berlin": (52.5200, 13.4050),
    "Hamburg": (53.5511, 9.9937),
    "Munich": (48.1351, 11.5820),
}


class RateLimitedSession:
    """Session with rate limiting and retry logic."""

    def __init__(self, rate_limit_delay: float = 1.0):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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


class WeatherDataCollector:
    """Collect weather data from OpenWeatherMap free API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather collector.

        To get free API key:
        1. Sign up at https://openweathermap.org/api
        2. Free tier: 1,000 calls/day, 60 calls/minute
        """
        self.api_key = api_key or "demo_key"  # Use demo for testing
        self.session = RateLimitedSession(
            rate_limit_delay=1.1
        )  # 60 calls/minute = 1 call/second
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather conditions."""
        url = f"{self.base_url}/weather"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}

        try:
            response = self.session.get(url, params=params)
            data = response.json()

            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "wind_direction": data.get("wind", {}).get("deg", 0),
                "cloud_cover": data.get("clouds", {}).get("all", 0),
                "visibility": data.get("visibility", 10000) / 1000,  # Convert m to km
                "weather_description": data["weather"][0]["description"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            log.error(f"Failed to get weather data for {lat}, {lon}: {e}")
            return {}

    def get_forecast(
        self, lat: float, lon: float, days: int = 5
    ) -> List[Dict[str, Any]]:
        """Get weather forecast (5-day free tier)."""
        url = f"{self.base_url}/forecast"
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}

        try:
            response = self.session.get(url, params=params)
            data = response.json()

            forecasts = []
            for item in data["list"][
                : days * 8
            ]:  # 8 forecasts per day (3-hour intervals)
                forecasts.append(
                    {
                        "datetime": item["dt_txt"],
                        "temperature": item["main"]["temp"],
                        "humidity": item["main"]["humidity"],
                        "pressure": item["main"]["pressure"],
                        "wind_speed": item.get("wind", {}).get("speed", 0),
                        "wind_direction": item.get("wind", {}).get("deg", 0),
                        "cloud_cover": item.get("clouds", {}).get("all", 0),
                        "precipitation": item.get("rain", {}).get("3h", 0)
                        + item.get("snow", {}).get("3h", 0),
                        "weather_description": item["weather"][0]["description"],
                    }
                )

            return forecasts
        except Exception as e:
            log.error(f"Failed to get forecast for {lat}, {lon}: {e}")
            return []


class FireDataCollector:
    """Collect fire detection data from NASA FIRMS."""

    def __init__(self):
        """
        NASA FIRMS (Fire Information for Resource Management System) - Free
        No API key required for basic access
        """
        self.session = RateLimitedSession(rate_limit_delay=2.0)  # Be conservative
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"

    def get_fires_by_country(
        self, country_code: str = "DEU", days: int = 1
    ) -> List[Dict[str, Any]]:
        """Get fire detections for a country (Germany = DEU)."""
        url = f"{self.base_url}/csv"

        # FIRMS provides data for last 24h, 48h, 7d
        if days <= 1:
            period = "24h"
        elif days <= 2:
            period = "48h"
        else:
            period = "7d"

        # Use direct CSV download (no API key required)
        csv_url = f"https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_{country_code}_{period}.csv"

        try:
            response = self.session.get(csv_url)

            # Parse CSV data
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                return []

            headers = lines[0].split(",")
            fires = []

            for line in lines[1:]:
                values = line.split(",")
                if len(values) >= len(headers):
                    fire_data = dict(zip(headers, values))
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
                        }
                    )

            return fires
        except Exception as e:
            log.error(f"Failed to get fire data for {country_code}: {e}")
            return []

    def calculate_fire_impact(
        self,
        fires: List[Dict],
        city_lat: float,
        city_lon: float,
        radius_km: float = 100,
    ) -> Dict[str, float]:
        """Calculate fire impact metrics for a city."""
        if not fires:
            return {
                "fire_count": 0,
                "total_frp": 0,
                "avg_distance": 0,
                "max_brightness": 0,
            }

        nearby_fires = []

        for fire in fires:
            distance = self._haversine_distance(
                city_lat, city_lon, fire["latitude"], fire["longitude"]
            )

            if distance <= radius_km:
                fire["distance"] = distance
                nearby_fires.append(fire)

        if not nearby_fires:
            return {
                "fire_count": 0,
                "total_frp": 0,
                "avg_distance": 0,
                "max_brightness": 0,
            }

        return {
            "fire_count": len(nearby_fires),
            "total_frp": sum(f["frp"] for f in nearby_fires),
            "avg_distance": np.mean([f["distance"] for f in nearby_fires]),
            "max_brightness": max(f["brightness"] for f in nearby_fires),
            "high_confidence_fires": sum(
                1 for f in nearby_fires if f["confidence"] in ["high", "h"]
            ),
        }

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points on Earth."""
        R = 6371  # Earth radius in kilometers

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return distance


class OSMDataCollector:
    """Collect infrastructure data from OpenStreetMap using Overpass API."""

    def __init__(self):
        """
        OpenStreetMap Overpass API - Free
        Rate limit: reasonable use (a few queries per minute)
        """
        self.session = RateLimitedSession(rate_limit_delay=3.0)  # Be very conservative
        self.overpass_url = "http://overpass-api.de/api/interpreter"

    def get_construction_sites(
        self, lat: float, lon: float, radius_km: float = 20
    ) -> List[Dict[str, Any]]:
        """Get construction sites around a city."""
        # Create bounding box
        lat_offset = radius_km / 111.0  # Rough conversion km to degrees
        lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

        bbox = f"{lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset}"

        query = f"""
        [out:json][timeout:25];
        (
          way["construction"]["construction"!="no"]({bbox});
          way["building"="construction"]({bbox});
          way["landuse"="construction"]({bbox});
          relation["construction"]["construction"!="no"]({bbox});
        );
        out geom;
        """

        try:
            response = self.session.get(self.overpass_url, params={"data": query})
            data = response.json()

            construction_sites = []
            for element in data.get("elements", []):
                if element["type"] in ["way", "relation"]:
                    tags = element.get("tags", {})
                    construction_sites.append(
                        {
                            "type": element["type"],
                            "construction_type": tags.get("construction", "unknown"),
                            "building_type": tags.get("building", ""),
                            "landuse": tags.get("landuse", ""),
                            "name": tags.get("name", ""),
                            "start_date": tags.get("start_date", ""),
                            "geometry": element.get("geometry", []),
                        }
                    )

            return construction_sites
        except Exception as e:
            log.error(f"Failed to get construction data for {lat}, {lon}: {e}")
            return []

    def get_traffic_infrastructure(
        self, lat: float, lon: float, radius_km: float = 20
    ) -> Dict[str, int]:
        """Get traffic infrastructure density."""
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * math.cos(math.radians(lat)))
        bbox = f"{lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset}"

        query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"motorway|trunk|primary|secondary"]({bbox});
          way["railway"="rail"]({bbox});
          node["amenity"="fuel"]({bbox});
          way["landuse"="industrial"]({bbox});
        );
        out count;
        """

        try:
            response = self.session.get(self.overpass_url, params={"data": query})
            data = response.json()

            # Count different types of infrastructure
            infrastructure = {
                "major_roads": 0,
                "railways": 0,
                "fuel_stations": 0,
                "industrial_areas": 0,
            }

            for element in data.get("elements", []):
                tags = element.get("tags", {})
                highway = tags.get("highway", "")
                railway = tags.get("railway", "")
                amenity = tags.get("amenity", "")
                landuse = tags.get("landuse", "")

                if highway in ["motorway", "trunk", "primary", "secondary"]:
                    infrastructure["major_roads"] += 1
                elif railway == "rail":
                    infrastructure["railways"] += 1
                elif amenity == "fuel":
                    infrastructure["fuel_stations"] += 1
                elif landuse == "industrial":
                    infrastructure["industrial_areas"] += 1

            return infrastructure
        except Exception as e:
            log.error(f"Failed to get traffic infrastructure for {lat}, {lon}: {e}")
            return {
                "major_roads": 0,
                "railways": 0,
                "fuel_stations": 0,
                "industrial_areas": 0,
            }


class HolidayDataCollector:
    """Collect holiday data from free APIs."""

    def __init__(self):
        """
        Public Holiday APIs - Free
        - date.nager.at (free, no key required)
        - abstractapi.com (free tier: 1000 requests/month)
        """
        self.session = RateLimitedSession(rate_limit_delay=1.0)

    def get_public_holidays(
        self, country_code: str = "DE", year: int = None
    ) -> List[Dict[str, Any]]:
        """Get public holidays for a country."""
        if year is None:
            year = datetime.now().year

        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

        try:
            response = self.session.get(url)
            holidays = response.json()

            return [
                {
                    "date": holiday["date"],
                    "name": holiday["name"],
                    "local_name": holiday["localName"],
                    "country_code": holiday["countryCode"],
                    "fixed": holiday["fixed"],
                    "global": holiday["global"],
                    "counties": holiday.get("counties", []),
                    "launch_year": holiday.get("launchYear"),
                }
                for holiday in holidays
            ]
        except Exception as e:
            log.error(f"Failed to get holidays for {country_code} {year}: {e}")
            return []

    def get_school_holidays(self, state: str = "BE") -> List[Dict[str, Any]]:
        """
        Get school holidays (simplified for German states).
        Using predefined data since comprehensive school holiday APIs are limited.
        """
        # Simplified school holiday data for German states
        # In production, you'd integrate with education ministry APIs
        school_holidays_2025 = {
            "BE": [  # Berlin
                {"name": "Summer Holidays", "start": "2025-07-17", "end": "2025-08-29"},
                {"name": "Autumn Holidays", "start": "2025-10-20", "end": "2025-11-01"},
                {
                    "name": "Christmas Holidays",
                    "start": "2025-12-22",
                    "end": "2026-01-02",
                },
            ],
            "HH": [  # Hamburg
                {"name": "Summer Holidays", "start": "2025-07-17", "end": "2025-08-27"},
                {"name": "Autumn Holidays", "start": "2025-10-18", "end": "2025-10-31"},
                {
                    "name": "Christmas Holidays",
                    "start": "2025-12-22",
                    "end": "2026-01-02",
                },
            ],
            "BY": [  # Bavaria (Munich)
                {"name": "Summer Holidays", "start": "2025-07-28", "end": "2025-09-15"},
                {"name": "Autumn Holidays", "start": "2025-10-27", "end": "2025-10-31"},
                {
                    "name": "Christmas Holidays",
                    "start": "2025-12-24",
                    "end": "2026-01-07",
                },
            ],
        }

        return school_holidays_2025.get(state, [])


class EarthquakeDataCollector:
    """Collect earthquake data from USGS (free API)."""

    def __init__(self):
        """
        USGS Earthquake API - Free
        No API key required
        """
        self.session = RateLimitedSession(rate_limit_delay=1.0)
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def get_recent_earthquakes(
        self, lat: float, lon: float, radius_km: float = 200, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent earthquakes around a location."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        params = {
            "format": "geojson",
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": radius_km,
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": 2.0,  # Only significant earthquakes
        }

        try:
            response = self.session.get(self.base_url, params=params)
            data = response.json()

            earthquakes = []
            for feature in data.get("features", []):
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]

                earthquakes.append(
                    {
                        "magnitude": props.get("mag", 0),
                        "place": props.get("place", ""),
                        "time": props.get("time", 0),
                        "depth": coords[2] if len(coords) > 2 else 0,
                        "latitude": coords[1],
                        "longitude": coords[0],
                        "significance": props.get("sig", 0),
                        "type": props.get("type", "earthquake"),
                    }
                )

            return earthquakes
        except Exception as e:
            log.error(f"Failed to get earthquake data for {lat}, {lon}: {e}")
            return []


class RealDataIntegrator:
    """Main class to integrate all real data sources."""

    def __init__(self, weather_api_key: Optional[str] = None):
        """Initialize all data collectors."""
        self.weather = WeatherDataCollector(weather_api_key)
        self.fire = FireDataCollector()
        self.osm = OSMDataCollector()
        self.holidays = HolidayDataCollector()
        self.earthquakes = EarthquakeDataCollector()

    def collect_all_data(
        self, cities: Dict[str, Tuple[float, float]], date: datetime = None
    ) -> pd.DataFrame:
        """Collect all available real data for specified cities."""
        if date is None:
            date = datetime.now()

        all_data = []

        log.info(f"Collecting real data for {len(cities)} cities...")

        # Get country-wide data once
        log.info("Collecting fire data...")
        fires = self.fire.get_fires_by_country("DEU", days=7)

        log.info("Collecting holiday data...")
        holidays = self.holidays.get_public_holidays("DE", date.year)
        school_holidays_be = self.holidays.get_school_holidays("BE")
        school_holidays_hh = self.holidays.get_school_holidays("HH")
        school_holidays_by = self.holidays.get_school_holidays("BY")

        for city_name, (lat, lon) in cities.items():
            log.info(f"Collecting data for {city_name}...")

            city_data = {
                "city": city_name,
                "date": date.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,
                "data_collection_time": datetime.utcnow().isoformat(),
            }

            # Weather data
            try:
                weather = self.weather.get_current_weather(lat, lon)
                city_data.update({f"weather_{k}": v for k, v in weather.items()})
            except Exception as e:
                log.warning(f"Failed to get weather for {city_name}: {e}")

            # Fire impact
            try:
                fire_impact = self.fire.calculate_fire_impact(
                    fires, lat, lon, radius_km=100
                )
                city_data.update({f"fire_{k}": v for k, v in fire_impact.items()})
            except Exception as e:
                log.warning(f"Failed to calculate fire impact for {city_name}: {e}")

            # Construction sites
            try:
                construction = self.osm.get_construction_sites(lat, lon, radius_km=20)
                city_data["construction_site_count"] = len(construction)

                # Construction types
                construction_types = {}
                for site in construction:
                    const_type = site.get("construction_type", "unknown")
                    construction_types[const_type] = (
                        construction_types.get(const_type, 0) + 1
                    )

                city_data.update(
                    {f"construction_{k}": v for k, v in construction_types.items()}
                )
            except Exception as e:
                log.warning(f"Failed to get construction data for {city_name}: {e}")

            # Traffic infrastructure
            try:
                infrastructure = self.osm.get_traffic_infrastructure(
                    lat, lon, radius_km=20
                )
                city_data.update(
                    {f"infrastructure_{k}": v for k, v in infrastructure.items()}
                )
            except Exception as e:
                log.warning(f"Failed to get infrastructure data for {city_name}: {e}")

            # Earthquake activity
            try:
                earthquakes = self.earthquakes.get_recent_earthquakes(
                    lat, lon, radius_km=200, days=30
                )
                city_data["earthquake_count"] = len(earthquakes)
                if earthquakes:
                    city_data["max_earthquake_magnitude"] = max(
                        eq["magnitude"] for eq in earthquakes
                    )
                    city_data["avg_earthquake_depth"] = np.mean(
                        [eq["depth"] for eq in earthquakes]
                    )
                else:
                    city_data["max_earthquake_magnitude"] = 0
                    city_data["avg_earthquake_depth"] = 0
            except Exception as e:
                log.warning(f"Failed to get earthquake data for {city_name}: {e}")

            # Holiday flags
            date_str = date.strftime("%Y-%m-%d")
            city_data["is_public_holiday"] = any(
                h["date"] == date_str for h in holidays
            )

            # School holidays (city-specific)
            school_map = {
                "Berlin": school_holidays_be,
                "Hamburg": school_holidays_hh,
                "Munich": school_holidays_by,
            }
            city_school_holidays = school_map.get(city_name, [])

            is_school_holiday = False
            for holiday in city_school_holidays:
                start_date = datetime.strptime(holiday["start"], "%Y-%m-%d").date()
                end_date = datetime.strptime(holiday["end"], "%Y-%m-%d").date()
                if start_date <= date.date() <= end_date:
                    is_school_holiday = True
                    city_data["school_holiday_name"] = holiday["name"]
                    break

            city_data["is_school_holiday"] = is_school_holiday
            if not is_school_holiday:
                city_data["school_holiday_name"] = "none"

            all_data.append(city_data)

            # Rate limiting between cities
            time.sleep(2)

        df = pd.DataFrame(all_data)
        log.info(f"Collected real data: {df.shape[0]} rows, {df.shape[1]} columns")

        return df


def main():
    """Test the real data collection system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect real external data for air quality forecasting"
    )
    parser.add_argument("--weather-api-key", help="OpenWeatherMap API key (optional)")
    parser.add_argument(
        "--output", default="data/real_external_data.csv", help="Output file path"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Berlin", "Hamburg", "Munich"],
        help="Cities to collect data for",
    )

    args = parser.parse_args()

    # Setup cities
    cities = {city: CITY_COORDS[city] for city in args.cities if city in CITY_COORDS}

    if not cities:
        log.error("No valid cities specified")
        return 1

    # Initialize data integrator
    integrator = RealDataIntegrator(args.weather_api_key)

    # Collect data
    try:
        df = integrator.collect_all_data(cities)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        log.info(f"Real data collection completed: {output_path}")
        log.info(f"Data shape: {df.shape}")

        # Print sample
        print("\nSample real data collected:")
        print("=" * 80)
        for col in df.columns[:10]:  # Show first 10 columns
            print(f"{col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")

        return 0

    except Exception as e:
        log.error(f"Data collection failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
