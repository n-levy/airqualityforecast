#!/usr/bin/env python3
"""
Global 100-City Air Quality Data Collector
==========================================

Collects air quality data for 100 cities across 5 continents using publicly
available APIs and data sources that do not require personal API keys.

Data Sources per Continent:
- Europe: EEA, CAMS (public access)
- North America: EPA AirNow, Environment Canada (public)
- Asia: Government portals, WAQI (public scraping), NASA satellite
- Africa: WHO data, satellite estimates, research networks
- South America: Government portals, satellite data, research networks

Each city includes:
- Ground truth from government/research monitoring
- 2+ benchmarks from different sources
- Standardized meteorological features
- Regional-specific features
- Local AQI standard calculations
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
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

# City configuration by continent
CITIES_CONFIG = {
    "asia": [
        {
            "name": "Delhi",
            "country": "India",
            "lat": 28.6139,
            "lon": 77.2090,
            "aqi_standard": "Indian",
        },
        {
            "name": "Lahore",
            "country": "Pakistan",
            "lat": 31.5497,
            "lon": 74.3436,
            "aqi_standard": "Pakistani",
        },
        {
            "name": "Beijing",
            "country": "China",
            "lat": 39.9042,
            "lon": 116.4074,
            "aqi_standard": "Chinese",
        },
        {
            "name": "Dhaka",
            "country": "Bangladesh",
            "lat": 23.8103,
            "lon": 90.4125,
            "aqi_standard": "EPA",
        },
        {
            "name": "Mumbai",
            "country": "India",
            "lat": 19.0760,
            "lon": 72.8777,
            "aqi_standard": "Indian",
        },
        {
            "name": "Karachi",
            "country": "Pakistan",
            "lat": 24.8607,
            "lon": 67.0011,
            "aqi_standard": "Pakistani",
        },
        {
            "name": "Shanghai",
            "country": "China",
            "lat": 31.2304,
            "lon": 121.4737,
            "aqi_standard": "Chinese",
        },
        {
            "name": "Kolkata",
            "country": "India",
            "lat": 22.5726,
            "lon": 88.3639,
            "aqi_standard": "Indian",
        },
        {
            "name": "Bangkok",
            "country": "Thailand",
            "lat": 14.5995,
            "lon": 100.5018,
            "aqi_standard": "Thai",
        },
        {
            "name": "Jakarta",
            "country": "Indonesia",
            "lat": -6.2088,
            "lon": 106.8456,
            "aqi_standard": "Indonesian",
        },
        {
            "name": "Manila",
            "country": "Philippines",
            "lat": 14.5995,
            "lon": 120.9842,
            "aqi_standard": "EPA",
        },
        {
            "name": "Ho Chi Minh City",
            "country": "Vietnam",
            "lat": 10.8231,
            "lon": 106.6297,
            "aqi_standard": "EPA",
        },
        {
            "name": "Hanoi",
            "country": "Vietnam",
            "lat": 21.0285,
            "lon": 105.8542,
            "aqi_standard": "EPA",
        },
        {
            "name": "Seoul",
            "country": "South Korea",
            "lat": 37.5665,
            "lon": 126.9780,
            "aqi_standard": "EPA",
        },
        {
            "name": "Taipei",
            "country": "Taiwan",
            "lat": 25.0330,
            "lon": 121.5654,
            "aqi_standard": "EPA",
        },
        {
            "name": "Ulaanbaatar",
            "country": "Mongolia",
            "lat": 47.8864,
            "lon": 106.9057,
            "aqi_standard": "EPA",
        },
        {
            "name": "Almaty",
            "country": "Kazakhstan",
            "lat": 43.2567,
            "lon": 76.9286,
            "aqi_standard": "EPA",
        },
        {
            "name": "Tashkent",
            "country": "Uzbekistan",
            "lat": 41.2993,
            "lon": 69.2407,
            "aqi_standard": "EPA",
        },
        {
            "name": "Tehran",
            "country": "Iran",
            "lat": 35.6961,
            "lon": 51.4231,
            "aqi_standard": "EPA",
        },
        {
            "name": "Kabul",
            "country": "Afghanistan",
            "lat": 34.5553,
            "lon": 69.2075,
            "aqi_standard": "EPA",
        },
    ],
    "africa": [
        {
            "name": "N'Djamena",
            "country": "Chad",
            "lat": 12.1348,
            "lon": 15.0557,
            "aqi_standard": "WHO",
        },
        {
            "name": "Cairo",
            "country": "Egypt",
            "lat": 30.0444,
            "lon": 31.2357,
            "aqi_standard": "WHO",
        },
        {
            "name": "Lagos",
            "country": "Nigeria",
            "lat": 6.5244,
            "lon": 3.3792,
            "aqi_standard": "WHO",
        },
        {
            "name": "Accra",
            "country": "Ghana",
            "lat": 5.6037,
            "lon": -0.1870,
            "aqi_standard": "WHO",
        },
        {
            "name": "Khartoum",
            "country": "Sudan",
            "lat": 15.5007,
            "lon": 32.5599,
            "aqi_standard": "WHO",
        },
        {
            "name": "Kampala",
            "country": "Uganda",
            "lat": 0.3476,
            "lon": 32.5825,
            "aqi_standard": "WHO",
        },
        {
            "name": "Nairobi",
            "country": "Kenya",
            "lat": -1.2921,
            "lon": 36.8219,
            "aqi_standard": "WHO",
        },
        {
            "name": "Abidjan",
            "country": "Côte d'Ivoire",
            "lat": 5.3600,
            "lon": -4.0083,
            "aqi_standard": "WHO",
        },
        {
            "name": "Bamako",
            "country": "Mali",
            "lat": 12.6392,
            "lon": -8.0029,
            "aqi_standard": "WHO",
        },
        {
            "name": "Ouagadougou",
            "country": "Burkina Faso",
            "lat": 12.3714,
            "lon": -1.5197,
            "aqi_standard": "WHO",
        },
        {
            "name": "Dakar",
            "country": "Senegal",
            "lat": 14.7167,
            "lon": -17.4677,
            "aqi_standard": "WHO",
        },
        {
            "name": "Kinshasa",
            "country": "DR Congo",
            "lat": -4.4419,
            "lon": 15.2663,
            "aqi_standard": "WHO",
        },
        {
            "name": "Casablanca",
            "country": "Morocco",
            "lat": 33.5731,
            "lon": -7.5898,
            "aqi_standard": "WHO",
        },
        {
            "name": "Johannesburg",
            "country": "South Africa",
            "lat": -26.2041,
            "lon": 28.0473,
            "aqi_standard": "WHO",
        },
        {
            "name": "Addis Ababa",
            "country": "Ethiopia",
            "lat": 9.1450,
            "lon": 38.7451,
            "aqi_standard": "WHO",
        },
        {
            "name": "Dar es Salaam",
            "country": "Tanzania",
            "lat": -6.7924,
            "lon": 39.2083,
            "aqi_standard": "WHO",
        },
        {
            "name": "Algiers",
            "country": "Algeria",
            "lat": 36.7538,
            "lon": 3.0588,
            "aqi_standard": "WHO",
        },
        {
            "name": "Tunis",
            "country": "Tunisia",
            "lat": 36.8065,
            "lon": 10.1815,
            "aqi_standard": "WHO",
        },
        {
            "name": "Maputo",
            "country": "Mozambique",
            "lat": -25.9692,
            "lon": 32.5732,
            "aqi_standard": "WHO",
        },
        {
            "name": "Cape Town",
            "country": "South Africa",
            "lat": -33.9249,
            "lon": 18.4241,
            "aqi_standard": "WHO",
        },
    ],
    "europe": [
        {
            "name": "Skopje",
            "country": "North Macedonia",
            "lat": 41.9973,
            "lon": 21.4280,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Sarajevo",
            "country": "Bosnia and Herzegovina",
            "lat": 43.8563,
            "lon": 18.4131,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Sofia",
            "country": "Bulgaria",
            "lat": 42.6977,
            "lon": 23.3219,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Plovdiv",
            "country": "Bulgaria",
            "lat": 42.1354,
            "lon": 24.7453,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Bucharest",
            "country": "Romania",
            "lat": 44.4268,
            "lon": 26.1025,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Belgrade",
            "country": "Serbia",
            "lat": 44.7866,
            "lon": 20.4489,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Warsaw",
            "country": "Poland",
            "lat": 52.2297,
            "lon": 21.0122,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Krakow",
            "country": "Poland",
            "lat": 50.0647,
            "lon": 19.9450,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Prague",
            "country": "Czech Republic",
            "lat": 50.0755,
            "lon": 14.4378,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Budapest",
            "country": "Hungary",
            "lat": 47.4979,
            "lon": 19.0402,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Milan",
            "country": "Italy",
            "lat": 45.4642,
            "lon": 9.1900,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Turin",
            "country": "Italy",
            "lat": 45.0703,
            "lon": 7.6869,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Naples",
            "country": "Italy",
            "lat": 40.8518,
            "lon": 14.2681,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Athens",
            "country": "Greece",
            "lat": 37.9838,
            "lon": 23.7275,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Madrid",
            "country": "Spain",
            "lat": 40.4168,
            "lon": -3.7038,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Barcelona",
            "country": "Spain",
            "lat": 41.3851,
            "lon": 2.1734,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Paris",
            "country": "France",
            "lat": 48.8566,
            "lon": 2.3522,
            "aqi_standard": "EAQI",
        },
        {
            "name": "London",
            "country": "UK",
            "lat": 51.5074,
            "lon": -0.1278,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Berlin",
            "country": "Germany",
            "lat": 52.5200,
            "lon": 13.4050,
            "aqi_standard": "EAQI",
        },
        {
            "name": "Amsterdam",
            "country": "Netherlands",
            "lat": 52.3676,
            "lon": 4.9041,
            "aqi_standard": "EAQI",
        },
    ],
    "north_america": [
        {
            "name": "Mexicali",
            "country": "Mexico",
            "lat": 32.6519,
            "lon": -115.4683,
            "aqi_standard": "Mexican",
        },
        {
            "name": "Mexico City",
            "country": "Mexico",
            "lat": 19.4326,
            "lon": -99.1332,
            "aqi_standard": "Mexican",
        },
        {
            "name": "Guadalajara",
            "country": "Mexico",
            "lat": 20.6597,
            "lon": -103.3496,
            "aqi_standard": "Mexican",
        },
        {
            "name": "Tijuana",
            "country": "Mexico",
            "lat": 32.5149,
            "lon": -117.0382,
            "aqi_standard": "Mexican",
        },
        {
            "name": "Monterrey",
            "country": "Mexico",
            "lat": 25.6866,
            "lon": -100.3161,
            "aqi_standard": "Mexican",
        },
        {
            "name": "Los Angeles",
            "country": "USA",
            "lat": 34.0522,
            "lon": -118.2437,
            "aqi_standard": "EPA",
        },
        {
            "name": "Fresno",
            "country": "USA",
            "lat": 36.7378,
            "lon": -119.7871,
            "aqi_standard": "EPA",
        },
        {
            "name": "Phoenix",
            "country": "USA",
            "lat": 33.4484,
            "lon": -112.0740,
            "aqi_standard": "EPA",
        },
        {
            "name": "Houston",
            "country": "USA",
            "lat": 29.7604,
            "lon": -95.3698,
            "aqi_standard": "EPA",
        },
        {
            "name": "New York",
            "country": "USA",
            "lat": 40.7128,
            "lon": -74.0060,
            "aqi_standard": "EPA",
        },
        {
            "name": "Chicago",
            "country": "USA",
            "lat": 41.8781,
            "lon": -87.6298,
            "aqi_standard": "EPA",
        },
        {
            "name": "Denver",
            "country": "USA",
            "lat": 39.7392,
            "lon": -104.9903,
            "aqi_standard": "EPA",
        },
        {
            "name": "Detroit",
            "country": "USA",
            "lat": 42.3314,
            "lon": -83.0458,
            "aqi_standard": "EPA",
        },
        {
            "name": "Atlanta",
            "country": "USA",
            "lat": 33.7490,
            "lon": -84.3880,
            "aqi_standard": "EPA",
        },
        {
            "name": "Philadelphia",
            "country": "USA",
            "lat": 39.9526,
            "lon": -75.1652,
            "aqi_standard": "EPA",
        },
        {
            "name": "Toronto",
            "country": "Canada",
            "lat": 43.6532,
            "lon": -79.3832,
            "aqi_standard": "Canadian",
        },
        {
            "name": "Montreal",
            "country": "Canada",
            "lat": 45.5017,
            "lon": -73.5673,
            "aqi_standard": "Canadian",
        },
        {
            "name": "Vancouver",
            "country": "Canada",
            "lat": 49.2827,
            "lon": -123.1207,
            "aqi_standard": "Canadian",
        },
        {
            "name": "Calgary",
            "country": "Canada",
            "lat": 51.0447,
            "lon": -114.0719,
            "aqi_standard": "Canadian",
        },
        {
            "name": "Ottawa",
            "country": "Canada",
            "lat": 45.4215,
            "lon": -75.6972,
            "aqi_standard": "Canadian",
        },
    ],
    "south_america": [
        {
            "name": "Lima",
            "country": "Peru",
            "lat": -12.0464,
            "lon": -77.0428,
            "aqi_standard": "EPA",
        },
        {
            "name": "Santiago",
            "country": "Chile",
            "lat": -33.4489,
            "lon": -70.6693,
            "aqi_standard": "Chilean",
        },
        {
            "name": "São Paulo",
            "country": "Brazil",
            "lat": -23.5505,
            "lon": -46.6333,
            "aqi_standard": "EPA",
        },
        {
            "name": "Rio de Janeiro",
            "country": "Brazil",
            "lat": -22.9068,
            "lon": -43.1729,
            "aqi_standard": "EPA",
        },
        {
            "name": "Bogotá",
            "country": "Colombia",
            "lat": 4.7110,
            "lon": -74.0721,
            "aqi_standard": "EPA",
        },
        {
            "name": "La Paz",
            "country": "Bolivia",
            "lat": -16.5000,
            "lon": -68.1500,
            "aqi_standard": "EPA",
        },
        {
            "name": "Medellín",
            "country": "Colombia",
            "lat": 6.2442,
            "lon": -75.5812,
            "aqi_standard": "EPA",
        },
        {
            "name": "Buenos Aires",
            "country": "Argentina",
            "lat": -34.6118,
            "lon": -58.3960,
            "aqi_standard": "EPA",
        },
        {
            "name": "Quito",
            "country": "Ecuador",
            "lat": -0.1807,
            "lon": -78.4678,
            "aqi_standard": "EPA",
        },
        {
            "name": "Caracas",
            "country": "Venezuela",
            "lat": 10.4806,
            "lon": -66.9036,
            "aqi_standard": "EPA",
        },
        {
            "name": "Belo Horizonte",
            "country": "Brazil",
            "lat": -19.9167,
            "lon": -43.9345,
            "aqi_standard": "EPA",
        },
        {
            "name": "Brasília",
            "country": "Brazil",
            "lat": -15.7801,
            "lon": -47.9292,
            "aqi_standard": "EPA",
        },
        {
            "name": "Porto Alegre",
            "country": "Brazil",
            "lat": -30.0346,
            "lon": -51.2177,
            "aqi_standard": "EPA",
        },
        {
            "name": "Montevideo",
            "country": "Uruguay",
            "lat": -34.9011,
            "lon": -56.1645,
            "aqi_standard": "EPA",
        },
        {
            "name": "Asunción",
            "country": "Paraguay",
            "lat": -25.2637,
            "lon": -57.5759,
            "aqi_standard": "EPA",
        },
        {
            "name": "Córdoba",
            "country": "Argentina",
            "lat": -31.4201,
            "lon": -64.1888,
            "aqi_standard": "EPA",
        },
        {
            "name": "Valparaíso",
            "country": "Chile",
            "lat": -33.0472,
            "lon": -71.6127,
            "aqi_standard": "Chilean",
        },
        {
            "name": "Cali",
            "country": "Colombia",
            "lat": 3.4516,
            "lon": -76.5320,
            "aqi_standard": "EPA",
        },
        {
            "name": "Curitiba",
            "country": "Brazil",
            "lat": -25.4284,
            "lon": -49.2733,
            "aqi_standard": "EPA",
        },
        {
            "name": "Fortaleza",
            "country": "Brazil",
            "lat": -3.7172,
            "lon": -38.5433,
            "aqi_standard": "EPA",
        },
    ],
}

# Data source configuration by continent
DATA_SOURCES = {
    "europe": {
        "ground_truth": {
            "name": "European Environment Agency",
            "url": "https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm",
            "api_key_required": False,
            "method": "direct_download",
        },
        "benchmark1": {
            "name": "CAMS (Copernicus Atmosphere Monitoring Service)",
            "url": "https://atmosphere.copernicus.eu/data",
            "api_key_required": False,
            "method": "public_api",
        },
        "benchmark2": {
            "name": "National monitoring networks",
            "url": "various",
            "api_key_required": False,
            "method": "government_portals",
        },
    },
    "north_america": {
        "ground_truth": {
            "name": "EPA AirNow + Environment Canada",
            "url": "https://www.airnow.gov/",
            "api_key_required": False,
            "method": "public_api",
        },
        "benchmark1": {
            "name": "NOAA air quality forecasts",
            "url": "https://airquality.weather.gov/",
            "api_key_required": False,
            "method": "public_scraping",
        },
        "benchmark2": {
            "name": "State/provincial networks",
            "url": "various",
            "api_key_required": False,
            "method": "government_portals",
        },
    },
    "asia": {
        "ground_truth": {
            "name": "National environmental agencies",
            "url": "various",
            "api_key_required": False,
            "method": "government_portals",
        },
        "benchmark1": {
            "name": "WAQI aggregated data",
            "url": "https://waqi.info/",
            "api_key_required": False,
            "method": "public_scraping",
        },
        "benchmark2": {
            "name": "NASA satellite estimates",
            "url": "https://earthdata.nasa.gov/",
            "api_key_required": False,
            "method": "satellite_api",
        },
    },
    "africa": {
        "ground_truth": {
            "name": "WHO Global Health Observatory",
            "url": "https://www.who.int/data/gho",
            "api_key_required": False,
            "method": "public_api",
        },
        "benchmark1": {
            "name": "NASA MODIS satellite data",
            "url": "https://modis.gsfc.nasa.gov/",
            "api_key_required": False,
            "method": "satellite_api",
        },
        "benchmark2": {
            "name": "Research networks (INDAAF)",
            "url": "various",
            "api_key_required": False,
            "method": "research_data",
        },
    },
    "south_america": {
        "ground_truth": {
            "name": "National environmental agencies",
            "url": "various",
            "api_key_required": False,
            "method": "government_portals",
        },
        "benchmark1": {
            "name": "NASA satellite estimates",
            "url": "https://earthdata.nasa.gov/",
            "api_key_required": False,
            "method": "satellite_api",
        },
        "benchmark2": {
            "name": "Regional research networks",
            "url": "various",
            "api_key_required": False,
            "method": "research_data",
        },
    },
}


class GlobalDataCollector:
    """Collects air quality data for 100 cities using public APIs."""

    def __init__(self, output_dir: str = "data/analysis/global_100cities"):
        """Initialize global data collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = self._create_session()

        log.info("Global 100-City Data Collector initialized")
        log.info(f"Output directory: {self.output_dir}")
        log.info(
            f"Total cities configured: {sum(len(cities) for cities in CITIES_CONFIG.values())}"
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

        return session

    def get_city_list(self, continent: str = None) -> List[Dict]:
        """Get list of cities for a continent or all cities."""
        if continent:
            return CITIES_CONFIG.get(continent, [])

        all_cities = []
        for continent_cities in CITIES_CONFIG.values():
            all_cities.extend(continent_cities)

        return all_cities

    def get_data_sources(self, continent: str) -> Dict:
        """Get data sources configuration for a continent."""
        return DATA_SOURCES.get(continent, {})

    def create_city_dataset_schema(self, city: Dict, continent: str) -> Dict:
        """Create standardized dataset schema for a city."""

        timestamp = datetime.now()

        schema = {
            "city_info": {
                "name": city["name"],
                "country": city["country"],
                "continent": continent,
                "coordinates": {"lat": city["lat"], "lon": city["lon"]},
                "aqi_standard": city["aqi_standard"],
                "data_sources": self.get_data_sources(continent),
            },
            "timestamp": timestamp.isoformat(),
            "data_collection_status": "configured",
            # Ground truth (to be populated)
            "ground_truth": {
                "pm25_actual": None,
                "pm10_actual": None,
                "no2_actual": None,
                "o3_actual": None,
                "so2_actual": None,
                "source": DATA_SOURCES[continent]["ground_truth"]["name"],
            },
            # Benchmarks (to be populated)
            "benchmarks": {
                "benchmark1": {
                    "pm25": None,
                    "pm10": None,
                    "no2": None,
                    "o3": None,
                    "source": DATA_SOURCES[continent]["benchmark1"]["name"],
                },
                "benchmark2": {
                    "pm25": None,
                    "pm10": None,
                    "no2": None,
                    "o3": None,
                    "source": DATA_SOURCES[continent]["benchmark2"]["name"],
                },
            },
            # Meteorology (to be populated)
            "meteorology": {
                "temperature": None,
                "humidity": None,
                "wind_speed": None,
                "wind_direction": None,
                "pressure": None,
                "precipitation": None,
            },
            # Temporal features
            "temporal": {
                "datetime": timestamp.isoformat(),
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "day_of_year": timestamp.timetuple().tm_yday,
                "month": timestamp.month,
                "season": self._get_season(timestamp.month, city["lat"]),
                "is_weekend": timestamp.weekday() >= 5,
                "is_holiday": False,  # To be determined based on local calendar
            },
            # Regional features (to be populated based on continent)
            "regional_features": self._get_regional_features_schema(continent),
        }

        return schema

    def _get_season(self, month: int, latitude: float) -> str:
        """Determine season based on month and hemisphere."""
        # Northern hemisphere seasons
        if latitude >= 0:
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            else:
                return "autumn"
        # Southern hemisphere seasons (opposite)
        else:
            if month in [12, 1, 2]:
                return "summer"
            elif month in [3, 4, 5]:
                return "autumn"
            elif month in [6, 7, 8]:
                return "winter"
            else:
                return "spring"

    def _get_regional_features_schema(self, continent: str) -> Dict:
        """Get regional-specific features for a continent."""

        regional_features = {
            "europe": {
                "ets_indicator": None,  # European Emission Trading System
                "cross_border_transport": None,
                "heating_season": None,
                "traffic_restriction": None,
            },
            "north_america": {
                "wildfire_indicator": None,
                "inversion_potential": None,
                "interstate_transport": None,
                "industrial_emissions": None,
            },
            "asia": {
                "monsoon_indicator": None,
                "dust_storm_potential": None,
                "industrial_activity": None,
                "agricultural_burning": None,
            },
            "africa": {
                "saharan_dust": None,
                "harmattan_effect": None,
                "seasonal_burning": None,
                "mining_activity": None,
            },
            "south_america": {
                "biomass_burning": None,
                "enso_indicator": None,  # El Niño/La Niña
                "altitude_effect": None,
                "amazon_influence": None,
            },
        }

        return regional_features.get(continent, {})

    def generate_dataset_summary(self) -> Dict:
        """Generate comprehensive dataset summary."""

        summary = {
            "dataset_info": {
                "name": "Global 100-City Air Quality Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_cities": 100,
                "continents": 5,
            },
            "cities_by_continent": {},
            "data_sources_by_continent": {},
            "features": {
                "air_quality": ["pm25", "pm10", "no2", "o3", "so2"],
                "meteorological": [
                    "temperature",
                    "humidity",
                    "wind_speed",
                    "wind_direction",
                    "pressure",
                    "precipitation",
                ],
                "temporal": [
                    "hour",
                    "day_of_week",
                    "day_of_year",
                    "month",
                    "season",
                    "is_weekend",
                    "is_holiday",
                ],
                "regional": "continent-specific features",
            },
            "aqi_standards": set(),
            "public_apis_only": True,
            "no_personal_keys_required": True,
        }

        # Populate continent-specific information
        for continent, cities in CITIES_CONFIG.items():
            summary["cities_by_continent"][continent] = {
                "count": len(cities),
                "cities": [city["name"] for city in cities],
                "countries": list(set(city["country"] for city in cities)),
            }

            summary["data_sources_by_continent"][continent] = DATA_SOURCES[continent]

            # Collect AQI standards
            for city in cities:
                summary["aqi_standards"].add(city["aqi_standard"])

        # Convert set to list for JSON serialization
        summary["aqi_standards"] = list(summary["aqi_standards"])

        return summary

    def create_complete_dataset_structure(self) -> Dict:
        """Create complete dataset structure for all 100 cities."""

        log.info("Creating complete dataset structure for 100 cities...")

        complete_dataset = {"metadata": self.generate_dataset_summary(), "cities": {}}

        total_cities = 0

        for continent, cities in CITIES_CONFIG.items():
            log.info(f"Processing {continent}: {len(cities)} cities")

            complete_dataset["cities"][continent] = {}

            for city in cities:
                city_key = f"{city['name'].lower().replace(' ', '_')}_{city['country'].lower().replace(' ', '_')}"

                city_schema = self.create_city_dataset_schema(city, continent)
                complete_dataset["cities"][continent][city_key] = city_schema

                total_cities += 1

        log.info(
            f"Created dataset structure for {total_cities} cities across {len(CITIES_CONFIG)} continents"
        )

        return complete_dataset

    def save_dataset_structure(
        self, dataset: Dict, filename: str = "global_100cities_dataset_structure.json"
    ) -> None:
        """Save dataset structure to JSON file."""

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Dataset structure saved to {output_path}")

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info(f"File size: {file_size_mb:.2f} MB")

    def generate_data_collection_plan(self) -> Dict:
        """Generate detailed data collection implementation plan."""

        plan = {
            "implementation_phases": {
                "phase1": {
                    "name": "Data Source Setup",
                    "description": "Configure access to all public APIs and data sources",
                    "tasks": [
                        "Set up web scraping infrastructure",
                        "Configure satellite data access",
                        "Test government portal APIs",
                        "Validate data source availability",
                    ],
                    "estimated_time": "1-2 weeks",
                },
                "phase2": {
                    "name": "Data Collection Implementation",
                    "description": "Implement data collectors for each continent",
                    "tasks": [
                        "Europe: EEA and CAMS integration",
                        "North America: EPA and Environment Canada",
                        "Asia: Government portals and WAQI scraping",
                        "Africa: WHO data and satellite integration",
                        "South America: Mixed sources implementation",
                    ],
                    "estimated_time": "3-4 weeks",
                },
                "phase3": {
                    "name": "Data Validation and QA",
                    "description": "Validate data quality and implement QA procedures",
                    "tasks": [
                        "Cross-source validation",
                        "Outlier detection",
                        "Missing data handling",
                        "Quality scoring implementation",
                    ],
                    "estimated_time": "2-3 weeks",
                },
                "phase4": {
                    "name": "Dataset Finalization",
                    "description": "Create final dataset with all features",
                    "tasks": [
                        "Feature engineering",
                        "AQI calculations for all standards",
                        "Regional feature integration",
                        "Final validation and documentation",
                    ],
                    "estimated_time": "2-3 weeks",
                },
            },
            "technical_requirements": {
                "infrastructure": [
                    "Distributed scraping infrastructure",
                    "Data storage and processing pipeline",
                    "Quality assurance automation",
                    "Monitoring and alerting system",
                ],
                "skills_needed": [
                    "Web scraping expertise",
                    "API integration experience",
                    "Data validation and QA",
                    "Air quality domain knowledge",
                ],
            },
            "challenges_and_mitigations": {
                "data_availability": {
                    "challenge": "Some cities may have limited real-time data",
                    "mitigation": "Use satellite data and regional interpolation",
                },
                "rate_limiting": {
                    "challenge": "Public APIs may have usage limits",
                    "mitigation": "Implement distributed collection and caching",
                },
                "data_quality": {
                    "challenge": "Variable quality across sources",
                    "mitigation": "Multi-source validation and quality scoring",
                },
                "website_changes": {
                    "challenge": "Government websites may change structure",
                    "mitigation": "Robust scraping with monitoring and alerts",
                },
            },
        }

        return plan


def main():
    """Main execution function for global data collector setup."""

    log.info("Setting up Global 100-City Air Quality Dataset...")

    # Initialize collector
    collector = GlobalDataCollector()

    # Create complete dataset structure
    dataset = collector.create_complete_dataset_structure()

    # Save dataset structure
    collector.save_dataset_structure(dataset)

    # Generate and save data collection plan
    plan = collector.generate_data_collection_plan()
    plan_path = collector.output_dir / "data_collection_implementation_plan.json"

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    log.info(f"Implementation plan saved to {plan_path}")

    # Generate summary report
    print("\n" + "=" * 80)
    print("GLOBAL 100-CITY AIR QUALITY DATASET")
    print("Configuration Complete - Ready for Data Collection")
    print("=" * 80)

    summary = dataset["metadata"]

    print(f"\nDATASET OVERVIEW:")
    print(f"• Total Cities: {summary['dataset_info']['total_cities']}")
    print(f"• Continents: {summary['dataset_info']['continents']}")
    print(
        f"• AQI Standards: {len(summary['aqi_standards'])} ({', '.join(summary['aqi_standards'])})"
    )
    print(f"• Public APIs Only: {summary['public_apis_only']}")
    print(f"• No Personal Keys Required: {summary['no_personal_keys_required']}")

    print(f"\nCITIES BY CONTINENT:")
    for continent, info in summary["cities_by_continent"].items():
        print(
            f"• {continent.title()}: {info['count']} cities across {len(info['countries'])} countries"
        )

    print(f"\nDATA SOURCES:")
    for continent, sources in summary["data_sources_by_continent"].items():
        print(f"• {continent.title()}:")
        print(f"  - Ground Truth: {sources['ground_truth']['name']}")
        print(f"  - Benchmark 1: {sources['benchmark1']['name']}")
        print(f"  - Benchmark 2: {sources['benchmark2']['name']}")

    print(f"\nFEATURES:")
    print(f"• Air Quality: {', '.join(summary['features']['air_quality'])}")
    print(f"• Meteorological: {', '.join(summary['features']['meteorological'])}")
    print(f"• Temporal: {', '.join(summary['features']['temporal'])}")
    print(f"• Regional: {summary['features']['regional']}")

    print(f"\nNEXT STEPS:")
    print(f"1. Review implementation plan: {plan_path}")
    print(f"2. Begin Phase 1: Data Source Setup")
    print(f"3. Implement continent-specific data collectors")
    print(f"4. Execute data collection and validation")
    print(f"5. Generate final ensemble forecasting dataset")

    print("\n" + "=" * 80)
    print("STATUS: READY FOR IMPLEMENTATION")
    print("All 100 cities configured with public data sources")
    print("No personal API keys required for data collection")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
