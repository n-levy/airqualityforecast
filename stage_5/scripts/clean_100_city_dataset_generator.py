#!/usr/bin/env python3
"""
Clean 100-City Dataset Generator
===============================

Creates clean datasets with only:
- Ground truth: OpenAQ real measured data (using API key)
- Forecasts: Open-Meteo weather-based air quality predictions
- Internal features: Holiday, temporal, geographic (no pattern-based synthetic)

Removes all pattern-based synthetic data (fire features, WAQI fallbacks, etc.)
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/clean_dataset_generation.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class Clean100CityDatasetGenerator:
    """Generate clean datasets with only external APIs and internal system features."""

    def __init__(self):
        """Initialize clean dataset generator."""
        self.generation_results = {
            "generation_type": "clean_100_city_dataset",
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }

        # Create directories
        self.output_dir = Path("stage_5/final_dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = Path("stage_5/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load API keys
        try:
            with open(".config/api_keys.json", "r") as f:
                keys = json.load(f)
            self.openaq_key = keys["apis"]["openaq"]["key"]
            self.nasa_firms_key = keys["apis"].get("nasa_firms", {}).get("key")
            log.info("OpenAQ API key loaded successfully")
            if self.nasa_firms_key:
                log.info("NASA FIRMS MAP_KEY loaded successfully")
        except Exception as e:
            log.error(f"Failed to load API keys: {e}")
            self.openaq_key = None
            self.nasa_firms_key = None

        # 100 cities (same as original project)
        self.cities = self._load_city_list()

        log.info("Clean 100-City Dataset Generator initialized")

    def _load_city_list(self) -> List[Dict]:
        """Load the standard 100 cities list."""
        return [
            # Asia (20 cities)
            {
                "name": "Delhi",
                "country": "IN",
                "lat": 28.6139,
                "lon": 77.2090,
                "continent": "Asia",
            },
            {
                "name": "Lahore",
                "country": "PK",
                "lat": 31.5497,
                "lon": 74.3436,
                "continent": "Asia",
            },
            {
                "name": "Beijing",
                "country": "CN",
                "lat": 39.9042,
                "lon": 116.4074,
                "continent": "Asia",
            },
            {
                "name": "Dhaka",
                "country": "BD",
                "lat": 23.8103,
                "lon": 90.4125,
                "continent": "Asia",
            },
            {
                "name": "Mumbai",
                "country": "IN",
                "lat": 19.0760,
                "lon": 72.8777,
                "continent": "Asia",
            },
            {
                "name": "Karachi",
                "country": "PK",
                "lat": 24.8607,
                "lon": 67.0011,
                "continent": "Asia",
            },
            {
                "name": "Shanghai",
                "country": "CN",
                "lat": 31.2304,
                "lon": 121.4737,
                "continent": "Asia",
            },
            {
                "name": "Kolkata",
                "country": "IN",
                "lat": 22.5726,
                "lon": 88.3639,
                "continent": "Asia",
            },
            {
                "name": "Bangkok",
                "country": "TH",
                "lat": 13.7563,
                "lon": 100.5018,
                "continent": "Asia",
            },
            {
                "name": "Jakarta",
                "country": "ID",
                "lat": -6.2088,
                "lon": 106.8456,
                "continent": "Asia",
            },
            {
                "name": "Manila",
                "country": "PH",
                "lat": 14.5995,
                "lon": 120.9842,
                "continent": "Asia",
            },
            {
                "name": "Ho Chi Minh City",
                "country": "VN",
                "lat": 10.8231,
                "lon": 106.6297,
                "continent": "Asia",
            },
            {
                "name": "Hanoi",
                "country": "VN",
                "lat": 21.0285,
                "lon": 105.8542,
                "continent": "Asia",
            },
            {
                "name": "Seoul",
                "country": "KR",
                "lat": 37.5665,
                "lon": 126.9780,
                "continent": "Asia",
            },
            {
                "name": "Taipei",
                "country": "TW",
                "lat": 25.0330,
                "lon": 121.5654,
                "continent": "Asia",
            },
            {
                "name": "Ulaanbaatar",
                "country": "MN",
                "lat": 47.8864,
                "lon": 106.9057,
                "continent": "Asia",
            },
            {
                "name": "Almaty",
                "country": "KZ",
                "lat": 43.2220,
                "lon": 76.8512,
                "continent": "Asia",
            },
            {
                "name": "Tashkent",
                "country": "UZ",
                "lat": 41.2995,
                "lon": 69.2401,
                "continent": "Asia",
            },
            {
                "name": "Tehran",
                "country": "IR",
                "lat": 35.6892,
                "lon": 51.3890,
                "continent": "Asia",
            },
            {
                "name": "Kabul",
                "country": "AF",
                "lat": 34.5553,
                "lon": 69.2075,
                "continent": "Asia",
            },
            # Africa (20 cities)
            {
                "name": "N'Djamena",
                "country": "TD",
                "lat": 12.1348,
                "lon": 15.0557,
                "continent": "Africa",
            },
            {
                "name": "Cairo",
                "country": "EG",
                "lat": 30.0444,
                "lon": 31.2357,
                "continent": "Africa",
            },
            {
                "name": "Lagos",
                "country": "NG",
                "lat": 6.5244,
                "lon": 3.3792,
                "continent": "Africa",
            },
            {
                "name": "Accra",
                "country": "GH",
                "lat": 5.6037,
                "lon": -0.1870,
                "continent": "Africa",
            },
            {
                "name": "Khartoum",
                "country": "SD",
                "lat": 15.5007,
                "lon": 32.5599,
                "continent": "Africa",
            },
            {
                "name": "Kampala",
                "country": "UG",
                "lat": 0.3476,
                "lon": 32.5825,
                "continent": "Africa",
            },
            {
                "name": "Nairobi",
                "country": "KE",
                "lat": -1.2921,
                "lon": 36.8219,
                "continent": "Africa",
            },
            {
                "name": "Abidjan",
                "country": "CI",
                "lat": 5.3600,
                "lon": -4.0083,
                "continent": "Africa",
            },
            {
                "name": "Bamako",
                "country": "ML",
                "lat": 12.6392,
                "lon": -8.0029,
                "continent": "Africa",
            },
            {
                "name": "Ouagadougou",
                "country": "BF",
                "lat": 12.3714,
                "lon": -1.5197,
                "continent": "Africa",
            },
            {
                "name": "Dakar",
                "country": "SN",
                "lat": 14.7167,
                "lon": -17.4677,
                "continent": "Africa",
            },
            {
                "name": "Kinshasa",
                "country": "CD",
                "lat": -4.4419,
                "lon": 15.2663,
                "continent": "Africa",
            },
            {
                "name": "Casablanca",
                "country": "MA",
                "lat": 33.5731,
                "lon": -7.5898,
                "continent": "Africa",
            },
            {
                "name": "Johannesburg",
                "country": "ZA",
                "lat": -26.2041,
                "lon": 28.0473,
                "continent": "Africa",
            },
            {
                "name": "Addis Ababa",
                "country": "ET",
                "lat": 9.1450,
                "lon": 38.7451,
                "continent": "Africa",
            },
            {
                "name": "Dar es Salaam",
                "country": "TZ",
                "lat": -6.7924,
                "lon": 39.2083,
                "continent": "Africa",
            },
            {
                "name": "Algiers",
                "country": "DZ",
                "lat": 36.7538,
                "lon": 3.0588,
                "continent": "Africa",
            },
            {
                "name": "Tunis",
                "country": "TN",
                "lat": 36.8065,
                "lon": 10.1815,
                "continent": "Africa",
            },
            {
                "name": "Maputo",
                "country": "MZ",
                "lat": -25.9692,
                "lon": 32.5732,
                "continent": "Africa",
            },
            {
                "name": "Cape Town",
                "country": "ZA",
                "lat": -33.9249,
                "lon": 18.4241,
                "continent": "Africa",
            },
            # Europe (20 cities)
            {
                "name": "Skopje",
                "country": "MK",
                "lat": 41.9973,
                "lon": 21.4280,
                "continent": "Europe",
            },
            {
                "name": "Sarajevo",
                "country": "BA",
                "lat": 43.8563,
                "lon": 18.4131,
                "continent": "Europe",
            },
            {
                "name": "Sofia",
                "country": "BG",
                "lat": 42.6977,
                "lon": 23.3219,
                "continent": "Europe",
            },
            {
                "name": "Plovdiv",
                "country": "BG",
                "lat": 42.1354,
                "lon": 24.7453,
                "continent": "Europe",
            },
            {
                "name": "Bucharest",
                "country": "RO",
                "lat": 44.4268,
                "lon": 26.1025,
                "continent": "Europe",
            },
            {
                "name": "Belgrade",
                "country": "RS",
                "lat": 44.7866,
                "lon": 20.4489,
                "continent": "Europe",
            },
            {
                "name": "Warsaw",
                "country": "PL",
                "lat": 52.2297,
                "lon": 21.0122,
                "continent": "Europe",
            },
            {
                "name": "Krakow",
                "country": "PL",
                "lat": 50.0647,
                "lon": 19.9450,
                "continent": "Europe",
            },
            {
                "name": "Prague",
                "country": "CZ",
                "lat": 50.0755,
                "lon": 14.4378,
                "continent": "Europe",
            },
            {
                "name": "Budapest",
                "country": "HU",
                "lat": 47.4979,
                "lon": 19.0402,
                "continent": "Europe",
            },
            {
                "name": "Milan",
                "country": "IT",
                "lat": 45.4642,
                "lon": 9.1900,
                "continent": "Europe",
            },
            {
                "name": "Turin",
                "country": "IT",
                "lat": 45.0703,
                "lon": 7.6869,
                "continent": "Europe",
            },
            {
                "name": "Naples",
                "country": "IT",
                "lat": 40.8518,
                "lon": 14.2681,
                "continent": "Europe",
            },
            {
                "name": "Athens",
                "country": "GR",
                "lat": 37.9838,
                "lon": 23.7275,
                "continent": "Europe",
            },
            {
                "name": "Madrid",
                "country": "ES",
                "lat": 40.4168,
                "lon": -3.7038,
                "continent": "Europe",
            },
            {
                "name": "Barcelona",
                "country": "ES",
                "lat": 41.3851,
                "lon": 2.1734,
                "continent": "Europe",
            },
            {
                "name": "Paris",
                "country": "FR",
                "lat": 48.8566,
                "lon": 2.3522,
                "continent": "Europe",
            },
            {
                "name": "London",
                "country": "GB",
                "lat": 51.5074,
                "lon": -0.1278,
                "continent": "Europe",
            },
            {
                "name": "Berlin",
                "country": "DE",
                "lat": 52.5200,
                "lon": 13.4050,
                "continent": "Europe",
            },
            {
                "name": "Amsterdam",
                "country": "NL",
                "lat": 52.3676,
                "lon": 4.9041,
                "continent": "Europe",
            },
            # North America (20 cities) - includes Sacramento replacement
            {
                "name": "Mexicali",
                "country": "MX",
                "lat": 32.6245,
                "lon": -115.4523,
                "continent": "North America",
            },
            {
                "name": "Mexico City",
                "country": "MX",
                "lat": 19.4326,
                "lon": -99.1332,
                "continent": "North America",
            },
            {
                "name": "Guadalajara",
                "country": "MX",
                "lat": 20.6597,
                "lon": -103.3496,
                "continent": "North America",
            },
            {
                "name": "Tijuana",
                "country": "MX",
                "lat": 32.5149,
                "lon": -117.0382,
                "continent": "North America",
            },
            {
                "name": "Monterrey",
                "country": "MX",
                "lat": 25.6866,
                "lon": -100.3161,
                "continent": "North America",
            },
            {
                "name": "Los Angeles",
                "country": "US",
                "lat": 34.0522,
                "lon": -118.2437,
                "continent": "North America",
            },
            {
                "name": "Sacramento",
                "country": "US",
                "lat": 38.5816,
                "lon": -121.4944,
                "continent": "North America",
            },  # Replaced Fresno
            {
                "name": "Phoenix",
                "country": "US",
                "lat": 33.4484,
                "lon": -112.0740,
                "continent": "North America",
            },
            {
                "name": "Houston",
                "country": "US",
                "lat": 29.7604,
                "lon": -95.3698,
                "continent": "North America",
            },
            {
                "name": "New York",
                "country": "US",
                "lat": 40.7128,
                "lon": -74.0060,
                "continent": "North America",
            },
            {
                "name": "Chicago",
                "country": "US",
                "lat": 41.8781,
                "lon": -87.6298,
                "continent": "North America",
            },
            {
                "name": "Denver",
                "country": "US",
                "lat": 39.7392,
                "lon": -104.9903,
                "continent": "North America",
            },
            {
                "name": "Detroit",
                "country": "US",
                "lat": 42.3314,
                "lon": -83.0458,
                "continent": "North America",
            },
            {
                "name": "Atlanta",
                "country": "US",
                "lat": 33.7490,
                "lon": -84.3880,
                "continent": "North America",
            },
            {
                "name": "Philadelphia",
                "country": "US",
                "lat": 39.9526,
                "lon": -75.1652,
                "continent": "North America",
            },
            {
                "name": "Toronto",
                "country": "CA",
                "lat": 43.6532,
                "lon": -79.3832,
                "continent": "North America",
            },
            {
                "name": "Montreal",
                "country": "CA",
                "lat": 45.5017,
                "lon": -73.5673,
                "continent": "North America",
            },
            {
                "name": "Vancouver",
                "country": "CA",
                "lat": 49.2827,
                "lon": -123.1207,
                "continent": "North America",
            },
            {
                "name": "Calgary",
                "country": "CA",
                "lat": 51.0447,
                "lon": -114.0719,
                "continent": "North America",
            },
            {
                "name": "Ottawa",
                "country": "CA",
                "lat": 45.4215,
                "lon": -75.6972,
                "continent": "North America",
            },
            # South America (20 cities)
            {
                "name": "Lima",
                "country": "PE",
                "lat": -12.0464,
                "lon": -77.0428,
                "continent": "South America",
            },
            {
                "name": "Santiago",
                "country": "CL",
                "lat": -33.4489,
                "lon": -70.6693,
                "continent": "South America",
            },
            {
                "name": "São Paulo",
                "country": "BR",
                "lat": -23.5505,
                "lon": -46.6333,
                "continent": "South America",
            },
            {
                "name": "Rio de Janeiro",
                "country": "BR",
                "lat": -22.9068,
                "lon": -43.1729,
                "continent": "South America",
            },
            {
                "name": "Bogotá",
                "country": "CO",
                "lat": 4.7110,
                "lon": -74.0721,
                "continent": "South America",
            },
            {
                "name": "La Paz",
                "country": "BO",
                "lat": -16.5000,
                "lon": -68.1193,
                "continent": "South America",
            },
            {
                "name": "Medellín",
                "country": "CO",
                "lat": 6.2442,
                "lon": -75.5812,
                "continent": "South America",
            },
            {
                "name": "Buenos Aires",
                "country": "AR",
                "lat": -34.6118,
                "lon": -58.3960,
                "continent": "South America",
            },
            {
                "name": "Quito",
                "country": "EC",
                "lat": -0.1807,
                "lon": -78.4678,
                "continent": "South America",
            },
            {
                "name": "Caracas",
                "country": "VE",
                "lat": 10.4806,
                "lon": -66.9036,
                "continent": "South America",
            },
            {
                "name": "Belo Horizonte",
                "country": "BR",
                "lat": -19.8157,
                "lon": -43.9542,
                "continent": "South America",
            },
            {
                "name": "Brasília",
                "country": "BR",
                "lat": -15.8267,
                "lon": -47.9218,
                "continent": "South America",
            },
            {
                "name": "Porto Alegre",
                "country": "BR",
                "lat": -30.0346,
                "lon": -51.2177,
                "continent": "South America",
            },
            {
                "name": "Montevideo",
                "country": "UY",
                "lat": -34.9011,
                "lon": -56.1645,
                "continent": "South America",
            },
            {
                "name": "Asunción",
                "country": "PY",
                "lat": -25.2637,
                "lon": -57.5759,
                "continent": "South America",
            },
            {
                "name": "Córdoba",
                "country": "AR",
                "lat": -31.4201,
                "lon": -64.1888,
                "continent": "South America",
            },
            {
                "name": "Valparaíso",
                "country": "CL",
                "lat": -33.0458,
                "lon": -71.6197,
                "continent": "South America",
            },
            {
                "name": "Cali",
                "country": "CO",
                "lat": 3.4516,
                "lon": -76.5320,
                "continent": "South America",
            },
            {
                "name": "Curitiba",
                "country": "BR",
                "lat": -25.4284,
                "lon": -49.2733,
                "continent": "South America",
            },
            {
                "name": "Fortaleza",
                "country": "BR",
                "lat": -3.7319,
                "lon": -38.5267,
                "continent": "South America",
            },
        ]

    def collect_openaq_ground_truth(self, city: Dict) -> Dict:
        """Collect ground truth data from OpenAQ API."""
        if not self.openaq_key:
            log.warning(f"No OpenAQ API key available for {city['name']}")
            return {"status": "no_api_key", "data": None}

        try:
            # Find nearest OpenAQ station
            station_url = "https://api.openaq.org/v3/locations"
            station_params = {
                "coordinates": f"{city['lat']},{city['lon']}",
                "radius": 100000,  # 100km radius
                "limit": 5,
                "sort": "lastUpdated",
                "order": "desc",
            }
            station_headers = {
                "User-Agent": "AQF311-Research/1.0",
                "Accept": "application/json",
                "X-API-Key": self.openaq_key,
            }

            station_response = requests.get(
                station_url, headers=station_headers, params=station_params, timeout=15
            )

            if station_response.status_code == 200:
                stations = station_response.json().get("results", [])
                if stations:
                    station_id = stations[0].get("id")

                    # Get recent measurements
                    measurements_url = "https://api.openaq.org/v3/measurements"
                    measurements_params = {
                        "location_id": station_id,
                        "date_from": (datetime.now() - timedelta(days=7)).strftime(
                            "%Y-%m-%d"
                        ),
                        "date_to": datetime.now().strftime("%Y-%m-%d"),
                        "limit": 100,
                        "sort": "datetime",
                        "order": "desc",
                    }

                    measurements_response = requests.get(
                        measurements_url,
                        headers=station_headers,
                        params=measurements_params,
                        timeout=20,
                    )

                    if measurements_response.status_code == 200:
                        measurements = measurements_response.json().get("results", [])

                        # Process measurements
                        processed_data = self._process_openaq_measurements(measurements)

                        return {
                            "status": "success",
                            "station_id": station_id,
                            "station_name": stations[0].get("name", "Unknown"),
                            "measurements_count": len(measurements),
                            "data": processed_data,
                        }

            return {"status": "no_data", "data": None}

        except Exception as e:
            log.error(f"OpenAQ collection failed for {city['name']}: {e}")
            return {"status": "error", "error": str(e), "data": None}

    def _process_openaq_measurements(self, measurements: List[Dict]) -> Dict:
        """Process OpenAQ measurements into structured format."""
        pollutant_data = {}

        for measurement in measurements:
            param = measurement.get("parameter")
            value = measurement.get("value")
            unit = measurement.get("unit")
            date_utc = measurement.get("date", {}).get("utc")

            if param not in pollutant_data:
                pollutant_data[param] = {
                    "parameter": param,
                    "unit": unit,
                    "values": [],
                    "timestamps": [],
                }

            pollutant_data[param]["values"].append(value)
            pollutant_data[param]["timestamps"].append(date_utc)

        # Calculate statistics
        for param, data in pollutant_data.items():
            values = [v for v in data["values"] if v is not None]
            if values:
                data["count"] = len(values)
                data["mean"] = sum(values) / len(values)
                data["min"] = min(values)
                data["max"] = max(values)
                data["latest"] = data["values"][0] if data["values"] else None
                data["latest_timestamp"] = (
                    data["timestamps"][0] if data["timestamps"] else None
                )

        return pollutant_data

    def collect_openmeteo_forecasts(self, city: Dict) -> Dict:
        """Collect forecast data from Open-Meteo API."""
        try:
            # Current and forecast air quality
            forecast_url = "https://api.open-meteo.com/v1/air-quality"
            forecast_params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "hourly": [
                    "pm10",
                    "pm2_5",
                    "carbon_monoxide",
                    "nitrogen_dioxide",
                    "sulphur_dioxide",
                    "ozone",
                    "european_aqi",
                ],
                "past_days": 7,
                "forecast_days": 7,
                "timezone": "auto",
            }

            forecast_response = requests.get(
                forecast_url, params=forecast_params, timeout=30
            )

            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()

                # Process forecast data
                processed_forecasts = self._process_openmeteo_forecasts(forecast_data)

                return {"status": "success", "data": processed_forecasts}

            return {"status": "no_data", "data": None}

        except Exception as e:
            log.error(f"Open-Meteo collection failed for {city['name']}: {e}")
            return {"status": "error", "error": str(e), "data": None}

    def _process_openmeteo_forecasts(self, forecast_data: Dict) -> Dict:
        """Process Open-Meteo forecast data."""
        hourly = forecast_data.get("hourly", {})

        # Create forecast records
        forecast_records = []
        times = hourly.get("time", [])

        for i, time_str in enumerate(times):
            record = {
                "datetime": time_str,
                "pm10": (
                    hourly.get("pm10", [])[i]
                    if i < len(hourly.get("pm10", []))
                    else None
                ),
                "pm2_5": (
                    hourly.get("pm2_5", [])[i]
                    if i < len(hourly.get("pm2_5", []))
                    else None
                ),
                "co": (
                    hourly.get("carbon_monoxide", [])[i]
                    if i < len(hourly.get("carbon_monoxide", []))
                    else None
                ),
                "no2": (
                    hourly.get("nitrogen_dioxide", [])[i]
                    if i < len(hourly.get("nitrogen_dioxide", []))
                    else None
                ),
                "so2": (
                    hourly.get("sulphur_dioxide", [])[i]
                    if i < len(hourly.get("sulphur_dioxide", []))
                    else None
                ),
                "o3": (
                    hourly.get("ozone", [])[i]
                    if i < len(hourly.get("ozone", []))
                    else None
                ),
                "european_aqi": (
                    hourly.get("european_aqi", [])[i]
                    if i < len(hourly.get("european_aqi", []))
                    else None
                ),
            }
            forecast_records.append(record)

        return {
            "forecast_records": forecast_records,
            "total_records": len(forecast_records),
            "date_range": {
                "start": times[0] if times else None,
                "end": times[-1] if times else None,
            },
        }

    def collect_nasa_firms_fire_data(self, city: Dict) -> Dict:
        """Collect real fire data from NASA FIRMS API."""
        if not self.nasa_firms_key:
            log.warning(f"No NASA FIRMS MAP_KEY available for {city['name']}")
            return {"status": "no_api_key", "data": None}

        try:
            # Define area around city (100km radius approx 1 degree)
            buffer = 1.0  # degrees
            west = city["lon"] - buffer
            south = city["lat"] - buffer
            east = city["lon"] + buffer
            north = city["lat"] + buffer

            # NASA FIRMS API endpoint
            firms_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            firms_params = {
                "MAP_KEY": self.nasa_firms_key,
                "source": "VIIRS_SNPP_NRT",  # Near Real-Time VIIRS data
                "area": f"{west},{south},{east},{north}",
                "dayRange": 7,  # Last 7 days
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            firms_response = requests.get(firms_url, params=firms_params, timeout=30)

            if firms_response.status_code == 200:
                # Parse CSV response
                csv_content = firms_response.text

                if csv_content.strip() and not csv_content.startswith("No fire"):
                    # Process CSV data
                    fire_data = self._process_nasa_firms_csv(csv_content, city)

                    return {"status": "success", "data": fire_data}
                else:
                    return {
                        "status": "no_fires",
                        "data": {
                            "fire_count": 0,
                            "active_fires": [],
                            "fire_summary": "No active fires detected",
                        },
                    }
            else:
                log.warning(
                    f"FIRMS API error for {city['name']}: {firms_response.status_code}"
                )
                return {"status": "api_error", "data": None}

        except Exception as e:
            log.error(f"NASA FIRMS collection failed for {city['name']}: {e}")
            return {"status": "error", "error": str(e), "data": None}

    def _process_nasa_firms_csv(self, csv_content: str, city: Dict) -> Dict:
        """Process NASA FIRMS CSV response."""
        import csv
        import io
        from math import sqrt

        lines = csv_content.strip().split("\n")
        if len(lines) < 2:  # Header + at least one data row
            return {
                "fire_count": 0,
                "active_fires": [],
                "fire_summary": "No fire data available",
            }

        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        fires = []

        city_lat, city_lon = city["lat"], city["lon"]

        for row in csv_reader:
            try:
                fire_lat = float(row.get("latitude", 0))
                fire_lon = float(row.get("longitude", 0))
                confidence = float(row.get("confidence", 0))
                brightness = float(row.get("brightness", 0))

                # Calculate distance from city center
                distance = (
                    sqrt((fire_lat - city_lat) ** 2 + (fire_lon - city_lon) ** 2) * 111
                )  # Approx km

                fire_record = {
                    "latitude": fire_lat,
                    "longitude": fire_lon,
                    "brightness": brightness,
                    "confidence": confidence,
                    "distance_km": round(distance, 2),
                    "acq_date": row.get("acq_date"),
                    "acq_time": row.get("acq_time"),
                    "satellite": row.get("satellite", "VIIRS"),
                    "instrument": row.get("instrument", "VIIRS"),
                }
                fires.append(fire_record)

            except (ValueError, TypeError):
                continue

        # Calculate fire statistics
        if fires:
            total_fires = len(fires)
            avg_distance = sum(f["distance_km"] for f in fires) / total_fires
            closest_fire = min(fires, key=lambda f: f["distance_km"])
            avg_confidence = sum(f["confidence"] for f in fires) / total_fires
            high_confidence_fires = len([f for f in fires if f["confidence"] >= 75])

            fire_summary = {
                "fire_count": total_fires,
                "active_fires": fires[:10],  # Limit to first 10 for storage
                "fire_statistics": {
                    "total_detected": total_fires,
                    "high_confidence_fires": high_confidence_fires,
                    "average_distance_km": round(avg_distance, 2),
                    "closest_fire_distance_km": closest_fire["distance_km"],
                    "average_confidence": round(avg_confidence, 1),
                    "max_brightness": max(f["brightness"] for f in fires),
                    "detection_period": "Last 7 days",
                },
                "fire_impact_assessment": {
                    "fire_risk_level": self._assess_fire_risk(
                        total_fires, avg_distance, avg_confidence
                    ),
                    "potential_pm25_impact": self._estimate_pm25_impact(fires, city),
                    "air_quality_alert": high_confidence_fires > 0
                    and avg_distance < 50,
                },
            }
        else:
            fire_summary = {
                "fire_count": 0,
                "active_fires": [],
                "fire_statistics": {"total_detected": 0},
                "fire_impact_assessment": {
                    "fire_risk_level": "low",
                    "potential_pm25_impact": "minimal",
                },
            }

        return fire_summary

    def _assess_fire_risk(
        self, fire_count: int, avg_distance: float, avg_confidence: float
    ) -> str:
        """Assess fire risk level based on fire data."""
        if fire_count == 0:
            return "low"
        elif fire_count < 5 and avg_distance > 50:
            return "low"
        elif fire_count < 10 and avg_distance > 25:
            return "moderate"
        elif avg_confidence > 75 and avg_distance < 25:
            return "high"
        else:
            return "moderate"

    def _estimate_pm25_impact(self, fires: List[Dict], city: Dict) -> str:
        """Estimate PM2.5 impact from fires."""
        if not fires:
            return "minimal"

        # Consider fires within 100km
        nearby_fires = [f for f in fires if f["distance_km"] < 100]
        high_confidence_nearby = [f for f in nearby_fires if f["confidence"] >= 70]

        if len(high_confidence_nearby) > 5:
            return "significant"
        elif len(high_confidence_nearby) > 0:
            return "moderate"
        elif len(nearby_fires) > 10:
            return "moderate"
        else:
            return "minimal"

    def generate_internal_features(self, city: Dict, record_date: str) -> Dict:
        """Generate only internal system features (no pattern-based synthetic)."""
        # Parse date
        try:
            date_obj = datetime.strptime(record_date, "%Y-%m-%d")
        except:
            try:
                date_obj = datetime.fromisoformat(record_date.replace("Z", "+00:00"))
            except:
                date_obj = datetime.now()

        features = {}

        # Temporal features (internal system)
        features["temporal"] = {
            "month": date_obj.month,
            "day_of_year": date_obj.timetuple().tm_yday,
            "weekday": date_obj.weekday(),
            "is_weekend": date_obj.weekday() >= 5,
            "season": self._get_season(date_obj.month),
            "quarter": (date_obj.month - 1) // 3 + 1,
        }

        # Geographic features (internal system)
        features["geographic"] = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "continent": city["continent"],
            "hemisphere": "Northern" if city["lat"] >= 0 else "Southern",
            "climate_zone": self._get_climate_zone(city["lat"]),
            "coastal_proximity": self._estimate_coastal_proximity(
                city["lat"], city["lon"]
            ),
        }

        # Holiday features (internal system - basic major holidays only)
        features["holidays"] = {
            "is_new_year": date_obj.month == 1 and date_obj.day == 1,
            "is_christmas": date_obj.month == 12 and date_obj.day == 25,
            "holiday_season": date_obj.month in [12, 1]
            or (date_obj.month == 12 and date_obj.day >= 20),
        }

        return features

    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _get_climate_zone(self, lat: float) -> str:
        """Get climate zone from latitude."""
        abs_lat = abs(lat)
        if abs_lat <= 23.5:
            return "tropical"
        elif abs_lat <= 35:
            return "subtropical"
        elif abs_lat <= 60:
            return "temperate"
        else:
            return "polar"

    def _estimate_coastal_proximity(self, lat: float, lon: float) -> str:
        """Simple coastal proximity estimation."""
        # This is a very basic estimation - in reality would need geographic data
        # For now, just classify based on major known coastal cities
        coastal_cities = [
            (-33.9249, 18.4241),  # Cape Town
            (40.7128, -74.0060),  # New York
            (51.5074, -0.1278),  # London
            (35.6892, 51.3890),  # Tehran (inland)
        ]

        # Simple distance check to nearest coast (very approximate)
        min_distance = float("inf")
        for coast_lat, coast_lon in coastal_cities:
            distance = ((lat - coast_lat) ** 2 + (lon - coast_lon) ** 2) ** 0.5
            min_distance = min(min_distance, distance)

        if min_distance < 5:  # Very close to known coastal city
            return "coastal"
        elif min_distance < 15:
            return "near_coastal"
        else:
            return "inland"

    def generate_clean_dataset(self) -> Dict[str, Any]:
        """Generate clean dataset with only external APIs and internal features."""
        log.info("=== STARTING CLEAN 100-CITY DATASET GENERATION ===")

        dataset_results = {
            "generation_timestamp": datetime.now().isoformat(),
            "dataset_type": "CLEAN_100_CITY_DATASET",
            "data_sources": {
                "ground_truth": "OpenAQ Real Measured Data (API authenticated)",
                "forecasts": "Open-Meteo Weather-Based Air Quality Predictions",
                "fire_data": "NASA FIRMS Real Fire Detection Data (API authenticated)",
                "internal_features": "Holiday, Temporal, Geographic (system-generated)",
            },
            "excluded_sources": [
                "Pattern-based fire activity modeling",
                "Synthetic WAQI fallback data",
                "CAMS synthetic benchmarks",
                "Extreme pollution scenarios",
                "All pattern-based synthetic data",
            ],
            "cities_data": [],
            "summary": {
                "total_cities": len(self.cities),
                "successful_openaq": 0,
                "successful_openmeteo": 0,
                "successful_nasa_firms": 0,
                "cities_with_complete_data": 0,
            },
        }

        for i, city in enumerate(self.cities, 1):
            log.info(
                f"[{i:3}/{len(self.cities)}] Processing {city['name']}, {city['country']}"
            )

            city_result = {
                "city_metadata": {
                    "name": city["name"],
                    "country": city["country"],
                    "continent": city["continent"],
                    "coordinates": {"lat": city["lat"], "lon": city["lon"]},
                },
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Collect OpenAQ ground truth
            log.info(f"  Collecting OpenAQ ground truth...")
            openaq_result = self.collect_openaq_ground_truth(city)
            city_result["ground_truth"] = openaq_result

            if openaq_result["status"] == "success":
                dataset_results["summary"]["successful_openaq"] += 1

            # Collect Open-Meteo forecasts
            log.info(f"  Collecting Open-Meteo forecasts...")
            openmeteo_result = self.collect_openmeteo_forecasts(city)
            city_result["forecasts"] = openmeteo_result

            if openmeteo_result["status"] == "success":
                dataset_results["summary"]["successful_openmeteo"] += 1

            # Collect NASA FIRMS fire data
            log.info(f"  Collecting NASA FIRMS fire data...")
            nasa_firms_result = self.collect_nasa_firms_fire_data(city)
            city_result["fire_data"] = nasa_firms_result

            if nasa_firms_result["status"] in ["success", "no_fires"]:
                dataset_results["summary"]["successful_nasa_firms"] += 1

            # Generate internal features for sample dates
            log.info(f"  Generating internal system features...")
            sample_dates = [
                datetime.now().strftime("%Y-%m-%d"),
                (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            ]

            city_result["internal_features"] = {}
            for date in sample_dates:
                city_result["internal_features"][date] = (
                    self.generate_internal_features(city, date)
                )

            # Check completeness
            has_ground_truth = openaq_result["status"] == "success"
            has_forecasts = openmeteo_result["status"] == "success"
            has_fire_data = nasa_firms_result["status"] in ["success", "no_fires"]

            # Count cities with at least ground truth OR forecasts as having useful data
            if has_ground_truth and has_forecasts and has_fire_data:
                dataset_results["summary"]["cities_with_complete_data"] += 1
                city_result["data_completeness"] = "complete"
            elif (has_ground_truth or has_forecasts) and has_fire_data:
                city_result["data_completeness"] = "partial_with_fire"
            elif has_ground_truth or has_forecasts:
                city_result["data_completeness"] = "partial"
            else:
                city_result["data_completeness"] = "incomplete"

            dataset_results["cities_data"].append(city_result)

            # Progress update
            if i % 10 == 0:
                openaq_success = dataset_results["summary"]["successful_openaq"]
                openmeteo_success = dataset_results["summary"]["successful_openmeteo"]
                nasa_firms_success = dataset_results["summary"]["successful_nasa_firms"]
                complete_data = dataset_results["summary"]["cities_with_complete_data"]

                log.info(f"Progress: {i}/100 cities processed")
                log.info(
                    f"  OpenAQ success: {openaq_success}/{i} ({openaq_success/i*100:.1f}%)"
                )
                log.info(
                    f"  Open-Meteo success: {openmeteo_success}/{i} ({openmeteo_success/i*100:.1f}%)"
                )
                log.info(
                    f"  NASA FIRMS success: {nasa_firms_success}/{i} ({nasa_firms_success/i*100:.1f}%)"
                )
                log.info(
                    f"  Complete data: {complete_data}/{i} ({complete_data/i*100:.1f}%)"
                )

        return dataset_results

    def save_clean_dataset(self, dataset_results: Dict) -> str:
        """Save the clean dataset."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full clean dataset
        clean_file = self.output_dir / f"CLEAN_100_CITY_dataset_{timestamp_str}.json"
        with open(clean_file, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=2, default=str)

        log.info(f"Clean dataset saved: {clean_file}")

        # Generate summary report
        summary = dataset_results["summary"]
        summary_report = {
            "generation_timestamp": dataset_results["generation_timestamp"],
            "dataset_type": "CLEAN_100_CITY_DATASET",
            "data_authenticity": "External APIs + Internal System Features Only",
            "excluded_synthetic": "All pattern-based modeling removed",
            "summary_statistics": summary,
            "data_sources": dataset_results["data_sources"],
            "excluded_sources": dataset_results["excluded_sources"],
        }

        summary_file = self.output_dir / f"CLEAN_100_CITY_summary_{timestamp_str}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2, default=str)

        log.info(f"Summary report saved: {summary_file}")
        log.info("=== CLEAN DATASET GENERATION COMPLETED ===")

        return str(clean_file)


def main():
    """Main execution."""
    print("CLEAN 100-CITY DATASET GENERATOR")
    print("Generating datasets with only external APIs and internal system features")
    print("Removing all pattern-based synthetic data")
    print("=" * 80)

    generator = Clean100CityDatasetGenerator()

    # Generate clean dataset
    dataset_results = generator.generate_clean_dataset()

    # Save results
    clean_file = generator.save_clean_dataset(dataset_results)

    summary = dataset_results["summary"]
    print(f"\nCLEAN DATASET GENERATION COMPLETE!")
    print(f"Dataset saved: {clean_file}")
    print(f"Cities processed: {summary['total_cities']}")
    print(
        f"OpenAQ success: {summary['successful_openaq']}/{summary['total_cities']} ({summary['successful_openaq']/summary['total_cities']*100:.1f}%)"
    )
    print(
        f"Open-Meteo success: {summary['successful_openmeteo']}/{summary['total_cities']} ({summary['successful_openmeteo']/summary['total_cities']*100:.1f}%)"
    )
    print(
        f"NASA FIRMS success: {summary['successful_nasa_firms']}/{summary['total_cities']} ({summary['successful_nasa_firms']/summary['total_cities']*100:.1f}%)"
    )
    print(
        f"Complete data: {summary['cities_with_complete_data']}/{summary['total_cities']} ({summary['cities_with_complete_data']/summary['total_cities']*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
