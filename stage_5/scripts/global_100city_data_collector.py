#!/usr/bin/env python3
"""
Global 100-City Air Quality Dataset Collector
===========================================

Phase 1: Infrastructure Setup
Comprehensive data collection framework for generating the ultra-minimal 100-city
air quality dataset with 5 years of daily data (2020-09-11 to 2025-09-11).

Features:
- Continental pattern-based collection (Berlin, São Paulo, Toronto, Delhi, Cairo)
- Multi-source validation (ground truth + 2 benchmarks per city)
- Local AQI calculations (11 regional standards)
- Public APIs only (no personal keys required)
- Ultra-minimal storage approach
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# Configure logging
Path("stage_5/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/data_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class Global100CityCollector:
    """
    Main collector class for the Global 100-City Air Quality Dataset.
    Implements continental patterns with multi-source validation.
    """

    def __init__(
        self, output_dir: str = "stage_5/data", config_dir: str = "stage_5/config"
    ):
        """Initialize the global data collector."""
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        self.logs_dir = Path("stage_5/logs")

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Date range: 5 years of daily data
        self.end_date = datetime.now().date()
        self.start_date = self.end_date - timedelta(days=5 * 365)  # 5 years
        self.total_days = (self.end_date - self.start_date).days

        log.info(
            f"Collection period: {self.start_date} to {self.end_date} ({self.total_days} days)"
        )

        # Load configuration
        self._load_configurations()

        # Initialize HTTP session with retries
        self.session = self._create_session()

        # Collection statistics
        self.stats = {
            "cities_processed": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records": 0,
            "data_quality_score": 0.0,
            "start_time": datetime.now(),
            "current_phase": "initialization",
        }

        log.info("Global 100-City Collector initialized successfully")

    def _load_configurations(self):
        """Load city configurations and data source specifications."""

        # Continental patterns from previous analysis
        self.continental_patterns = {
            "europe": {
                "pattern_name": "Berlin Pattern",
                "success_rate": 0.85,
                "data_sources": {
                    "ground_truth": "EEA (European Environment Agency)",
                    "benchmark1": "CAMS (Copernicus Atmosphere Monitoring)",
                    "benchmark2": "National monitoring networks",
                },
                "aqi_standard": "European EAQI",
            },
            "south_america": {
                "pattern_name": "São Paulo Pattern",
                "success_rate": 0.85,
                "data_sources": {
                    "ground_truth": "Government agencies",
                    "benchmark1": "NASA satellite estimates",
                    "benchmark2": "Regional research networks",
                },
                "aqi_standard": "EPA adaptations, Chilean ICA",
            },
            "north_america": {
                "pattern_name": "Toronto Pattern",
                "success_rate": 0.70,
                "data_sources": {
                    "ground_truth": "EPA AirNow + Environment Canada",
                    "benchmark1": "NOAA air quality forecasts",
                    "benchmark2": "State/provincial networks",
                },
                "aqi_standard": "US EPA, Canadian AQHI, Mexican IMECA",
            },
            "asia": {
                "pattern_name": "Delhi Pattern",
                "success_rate": 0.50,
                "data_sources": {
                    "ground_truth": "National environmental agencies",
                    "benchmark1": "WAQI aggregated data",
                    "benchmark2": "NASA satellite estimates",
                },
                "aqi_standard": "Indian, Chinese, Thai, Local standards",
            },
            "africa": {
                "pattern_name": "Cairo Pattern",
                "success_rate": 0.55,
                "data_sources": {
                    "ground_truth": "WHO Global Health Observatory",
                    "benchmark1": "NASA MODIS satellite data",
                    "benchmark2": "Research networks (INDAAF)",
                },
                "aqi_standard": "WHO Guidelines",
            },
        }

        # 100 cities configuration (from existing specification)
        self.cities_config = {
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
                    "lat": 13.7563,
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
                    "lat": 43.2220,
                    "lon": 76.8512,
                    "aqi_standard": "EPA",
                },
                {
                    "name": "Tashkent",
                    "country": "Uzbekistan",
                    "lat": 41.2995,
                    "lon": 69.2401,
                    "aqi_standard": "EPA",
                },
                {
                    "name": "Tehran",
                    "country": "Iran",
                    "lat": 35.6892,
                    "lon": 51.3890,
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
            "europe": [
                {
                    "name": "Skopje",
                    "country": "North Macedonia",
                    "lat": 41.9973,
                    "lon": 21.4280,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Sarajevo",
                    "country": "Bosnia and Herzegovina",
                    "lat": 43.8563,
                    "lon": 18.4131,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Sofia",
                    "country": "Bulgaria",
                    "lat": 42.6977,
                    "lon": 23.3219,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Plovdiv",
                    "country": "Bulgaria",
                    "lat": 42.1354,
                    "lon": 24.7453,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Bucharest",
                    "country": "Romania",
                    "lat": 44.4268,
                    "lon": 26.1025,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Belgrade",
                    "country": "Serbia",
                    "lat": 44.7866,
                    "lon": 20.4489,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Warsaw",
                    "country": "Poland",
                    "lat": 52.2297,
                    "lon": 21.0122,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Krakow",
                    "country": "Poland",
                    "lat": 50.0647,
                    "lon": 19.9450,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Prague",
                    "country": "Czech Republic",
                    "lat": 50.0755,
                    "lon": 14.4378,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Budapest",
                    "country": "Hungary",
                    "lat": 47.4979,
                    "lon": 19.0402,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Milan",
                    "country": "Italy",
                    "lat": 45.4642,
                    "lon": 9.1900,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Turin",
                    "country": "Italy",
                    "lat": 45.0703,
                    "lon": 7.6869,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Naples",
                    "country": "Italy",
                    "lat": 40.8518,
                    "lon": 14.2681,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Athens",
                    "country": "Greece",
                    "lat": 37.9755,
                    "lon": 23.7348,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Madrid",
                    "country": "Spain",
                    "lat": 40.4168,
                    "lon": -3.7038,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Barcelona",
                    "country": "Spain",
                    "lat": 41.3851,
                    "lon": 2.1734,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Paris",
                    "country": "France",
                    "lat": 48.8566,
                    "lon": 2.3522,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "London",
                    "country": "UK",
                    "lat": 51.5074,
                    "lon": -0.1278,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Berlin",
                    "country": "Germany",
                    "lat": 52.5200,
                    "lon": 13.4050,
                    "aqi_standard": "European EAQI",
                },
                {
                    "name": "Amsterdam",
                    "country": "Netherlands",
                    "lat": 52.3676,
                    "lon": 4.9041,
                    "aqi_standard": "European EAQI",
                },
            ],
            "north_america": [
                {
                    "name": "Mexicali",
                    "country": "Mexico",
                    "lat": 32.6519,
                    "lon": -115.4683,
                    "aqi_standard": "Mexican IMECA",
                },
                {
                    "name": "Mexico City",
                    "country": "Mexico",
                    "lat": 19.4326,
                    "lon": -99.1332,
                    "aqi_standard": "Mexican IMECA",
                },
                {
                    "name": "Guadalajara",
                    "country": "Mexico",
                    "lat": 20.6597,
                    "lon": -103.3496,
                    "aqi_standard": "Mexican IMECA",
                },
                {
                    "name": "Tijuana",
                    "country": "Mexico",
                    "lat": 32.5149,
                    "lon": -117.0382,
                    "aqi_standard": "Mexican IMECA",
                },
                {
                    "name": "Monterrey",
                    "country": "Mexico",
                    "lat": 25.6866,
                    "lon": -100.3161,
                    "aqi_standard": "Mexican IMECA",
                },
                {
                    "name": "Los Angeles",
                    "country": "USA",
                    "lat": 34.0522,
                    "lon": -118.2437,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Fresno",
                    "country": "USA",
                    "lat": 36.7378,
                    "lon": -119.7871,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Phoenix",
                    "country": "USA",
                    "lat": 33.4484,
                    "lon": -112.0740,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Houston",
                    "country": "USA",
                    "lat": 29.7604,
                    "lon": -95.3698,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "New York",
                    "country": "USA",
                    "lat": 40.7128,
                    "lon": -74.0060,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Chicago",
                    "country": "USA",
                    "lat": 41.8781,
                    "lon": -87.6298,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Denver",
                    "country": "USA",
                    "lat": 39.7392,
                    "lon": -104.9903,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Detroit",
                    "country": "USA",
                    "lat": 42.3314,
                    "lon": -83.0458,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Atlanta",
                    "country": "USA",
                    "lat": 33.7490,
                    "lon": -84.3880,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Philadelphia",
                    "country": "USA",
                    "lat": 39.9526,
                    "lon": -75.1652,
                    "aqi_standard": "US EPA",
                },
                {
                    "name": "Toronto",
                    "country": "Canada",
                    "lat": 43.6532,
                    "lon": -79.3832,
                    "aqi_standard": "Canadian AQHI",
                },
                {
                    "name": "Montreal",
                    "country": "Canada",
                    "lat": 45.5017,
                    "lon": -73.5673,
                    "aqi_standard": "Canadian AQHI",
                },
                {
                    "name": "Vancouver",
                    "country": "Canada",
                    "lat": 49.2827,
                    "lon": -123.1207,
                    "aqi_standard": "Canadian AQHI",
                },
                {
                    "name": "Calgary",
                    "country": "Canada",
                    "lat": 51.0447,
                    "lon": -114.0719,
                    "aqi_standard": "Canadian AQHI",
                },
                {
                    "name": "Ottawa",
                    "country": "Canada",
                    "lat": 45.4215,
                    "lon": -75.6972,
                    "aqi_standard": "Canadian AQHI",
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
                    "lon": 40.4897,
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
                    "aqi_standard": "Chilean ICA",
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
                    "lon": -68.1193,
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
                    "lat": -19.8157,
                    "lon": -43.9542,
                    "aqi_standard": "EPA",
                },
                {
                    "name": "Brasília",
                    "country": "Brazil",
                    "lat": -15.8267,
                    "lon": -47.9218,
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
                    "aqi_standard": "Chilean ICA",
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
                    "lon": -38.5434,
                    "aqi_standard": "EPA",
                },
            ],
        }

        # Data source endpoints (public APIs only)
        self.data_sources = {
            "europe": {
                "ground_truth": {
                    "name": "EEA",
                    "url": "https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm",
                    "method": "direct_download",
                },
                "benchmark1": {
                    "name": "CAMS",
                    "url": "https://atmosphere.copernicus.eu/data",
                    "method": "public_api",
                },
                "benchmark2": {
                    "name": "National Networks",
                    "url": "various",
                    "method": "government_portals",
                },
            },
            "north_america": {
                "ground_truth": {
                    "name": "EPA AirNow + Environment Canada",
                    "url": "https://www.airnow.gov/",
                    "method": "public_api",
                },
                "benchmark1": {
                    "name": "NOAA",
                    "url": "https://airquality.weather.gov/",
                    "method": "public_scraping",
                },
                "benchmark2": {
                    "name": "State/Provincial",
                    "url": "various",
                    "method": "government_portals",
                },
            },
            "asia": {
                "ground_truth": {
                    "name": "Government Portals",
                    "url": "various",
                    "method": "government_portals",
                },
                "benchmark1": {
                    "name": "WAQI",
                    "url": "https://waqi.info/",
                    "method": "public_scraping",
                },
                "benchmark2": {
                    "name": "NASA Satellite",
                    "url": "https://earthdata.nasa.gov/",
                    "method": "satellite_api",
                },
            },
            "africa": {
                "ground_truth": {
                    "name": "WHO",
                    "url": "https://www.who.int/data/gho",
                    "method": "public_api",
                },
                "benchmark1": {
                    "name": "NASA MODIS",
                    "url": "https://modis.gsfc.nasa.gov/",
                    "method": "satellite_api",
                },
                "benchmark2": {
                    "name": "Research Networks",
                    "url": "various",
                    "method": "research_data",
                },
            },
            "south_america": {
                "ground_truth": {
                    "name": "Government Agencies",
                    "url": "various",
                    "method": "government_portals",
                },
                "benchmark1": {
                    "name": "NASA Satellite",
                    "url": "https://earthdata.nasa.gov/",
                    "method": "satellite_api",
                },
                "benchmark2": {
                    "name": "Research Networks",
                    "url": "various",
                    "method": "research_data",
                },
            },
        }

        log.info("Configurations loaded successfully")
        log.info(
            f"Total cities configured: {sum(len(cities) for cities in self.cities_config.values())}"
        )

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Headers
        session.headers.update(
            {
                "User-Agent": "Global-100City-AirQuality-Collector/1.0 (Research)",
                "Accept": "application/json, text/csv, */*",
            }
        )

        return session

    def initialize_infrastructure(self) -> Dict[str, Any]:
        """
        Step 1: Initialize Collection Framework
        Set up data collection infrastructure, database structure,
        logging and test basic connectivity.
        """
        log.info("=== STEP 1: INITIALIZING COLLECTION FRAMEWORK ===")

        results = {
            "step": 1,
            "name": "Initialize Collection Framework",
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "details": {},
        }

        try:
            # 1. Create directory structure
            directories = [
                "stage_5/data/raw",
                "stage_5/data/processed",
                "stage_5/data/final",
                "stage_5/logs",
                "stage_5/config",
                "stage_5/metadata",
                "stage_5/quality_reports",
            ]

            for dir_path in directories:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            log.info("Directory structure created")
            results["details"]["directories"] = "created"

            # 2. Save configurations to files
            config_files = {
                "cities_config.json": self.cities_config,
                "continental_patterns.json": self.continental_patterns,
                "data_sources.json": self.data_sources,
            }

            for filename, config in config_files.items():
                config_path = self.config_dir / filename
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                log.info(f"Configuration saved: {filename}")

            results["details"]["configurations"] = "saved"

            # 3. Initialize collection metadata
            metadata = {
                "project": "Global 100-City Air Quality Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "collection_period": {
                    "start_date": self.start_date.isoformat(),
                    "end_date": self.end_date.isoformat(),
                    "total_days": self.total_days,
                },
                "total_cities": 100,
                "continents": 5,
                "expected_records": 100 * self.total_days,
                "data_resolution": "daily",
                "storage_approach": "ultra-minimal",
                "api_key_required": False,
                "continental_patterns": {
                    continent: pattern["pattern_name"]
                    for continent, pattern in self.continental_patterns.items()
                },
            }

            metadata_path = Path("stage_5/metadata/collection_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            log.info("Collection metadata initialized")
            results["details"]["metadata"] = "initialized"

            # 4. Test basic connectivity
            test_urls = [
                "https://httpbin.org/get",  # Basic connectivity test
                "https://waqi.info/",  # WAQI accessibility
                "https://www.airnow.gov/",  # EPA AirNow accessibility
            ]

            connectivity_results = {}
            for i, url in enumerate(test_urls):
                try:
                    response = self.session.get(url, timeout=10)
                    connectivity_results[f"test_{i+1}"] = {
                        "url": url,
                        "status_code": response.status_code,
                        "accessible": response.status_code == 200,
                    }
                    log.info(
                        f"Connectivity test {i+1}: {url} (Status: {response.status_code})"
                    )
                except Exception as e:
                    connectivity_results[f"test_{i+1}"] = {
                        "url": url,
                        "error": str(e),
                        "accessible": False,
                    }
                    log.warning(f"Connectivity test {i+1} failed: {url} - {str(e)}")

                time.sleep(1)  # Rate limiting

            results["details"]["connectivity_tests"] = connectivity_results

            # 5. Initialize statistics tracking
            self.stats.update(
                {
                    "infrastructure_initialized": True,
                    "total_expected_records": 100 * self.total_days,
                    "continental_patterns": len(self.continental_patterns),
                    "data_sources_configured": sum(
                        len(sources) for sources in self.data_sources.values()
                    ),
                }
            )

            # 6. Save initial progress (convert datetime to string)
            progress_path = Path("stage_5/logs/collection_progress.json")
            stats_serializable = {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.stats.items()
            }
            with open(progress_path, "w") as f:
                json.dump(
                    {
                        "phase": "Phase 1: Infrastructure Setup",
                        "current_step": 1,
                        "completed_steps": ["Step 1: Initialize Collection Framework"],
                        "stats": stats_serializable,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            log.info("=== STEP 1 COMPLETED SUCCESSFULLY ===")
            log.info(
                f"Expected dataset size: ~{(100 * self.total_days * 80 * 4) / (1024**3):.1f} GB"
            )
            log.info(f"Collection period: {self.total_days} days per city")
            log.info("Infrastructure ready for data collection")

        except Exception as e:
            log.error(f"Step 1 failed: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise

        return results


def main():
    """Main execution function."""
    log.info("Starting Global 100-City Data Collection - Phase 1")

    try:
        # Initialize collector
        collector = Global100CityCollector()

        # Execute Step 1: Initialize Collection Framework
        step1_results = collector.initialize_infrastructure()

        # Save results
        results_path = Path("stage_5/logs/step1_results.json")
        with open(results_path, "w") as f:
            json.dump(step1_results, f, indent=2)

        log.info("Phase 1 - Step 1 completed successfully")
        log.info(f"Results saved to: {results_path}")

        return step1_results

    except Exception as e:
        log.error(f"Phase 1 - Step 1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
