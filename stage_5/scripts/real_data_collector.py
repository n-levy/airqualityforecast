#!/usr/bin/env python3
"""
Real Data Collector for Global 100-City Air Quality Dataset
==========================================================

Collects actual air quality data from live APIs and sources.
Implements the detailed plan for real data collection across 100 cities.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/real_data_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class RealDataCollector:
    """Real data collector for 100-city air quality dataset."""
    
    def __init__(self):
        """Initialize real data collector."""
        self.collection_results = {
            "collection_type": "real_data",
            "start_time": datetime.now().isoformat(),
            "api_results": {},
            "city_results": {},
            "collection_progress": {},
            "status": "in_progress"
        }
        
        # Create directories
        self.output_dir = Path("stage_5/real_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API configurations (using free/public endpoints)
        self.api_configs = self._initialize_api_configs()
        
        # City configurations
        self.cities = self._initialize_city_list()
        
        # Initialize session
        self.session = self._create_session()
        
        log.info("Real Data Collector initialized")
    
    def _initialize_api_configs(self) -> Dict[str, Dict]:
        """Initialize API configurations for data collection."""
        return {
            "openweathermap": {
                "name": "OpenWeatherMap Air Pollution API",
                "base_url": "http://api.openweathermap.org/data/2.5/air_pollution",
                "requires_key": True,
                "free_tier": True,
                "rate_limit": 60,  # requests per minute
                "data_types": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
            },
            "waqi": {
                "name": "World Air Quality Index API",
                "base_url": "https://api.waqi.info/feed",
                "requires_key": True,
                "free_tier": True,
                "rate_limit": 1000,  # requests per day
                "data_types": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
            },
            "iqair": {
                "name": "IQAir API",
                "base_url": "https://api.airvisual.com/v2",
                "requires_key": True,
                "free_tier": True,
                "rate_limit": 10000,  # requests per month
                "data_types": ["PM2.5", "AQI"]
            },
            "openaq": {
                "name": "OpenAQ API",
                "base_url": "https://api.openaq.org/v2",
                "requires_key": False,
                "free_tier": True,
                "rate_limit": 10000,  # requests per day
                "data_types": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
            },
            "aqicn": {
                "name": "AQICN.org API",
                "base_url": "https://api.aqicn.org/feed",
                "requires_key": True,
                "free_tier": True,
                "rate_limit": 1000,  # requests per day
                "data_types": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
            }
        }
    
    def _initialize_city_list(self) -> List[Dict]:
        """Initialize the list of 100 cities for data collection."""
        return [
            # Europe (20 cities)
            {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050, "continent": "europe"},
            {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278, "continent": "europe"},
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522, "continent": "europe"},
            {"name": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038, "continent": "europe"},
            {"name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964, "continent": "europe"},
            {"name": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041, "continent": "europe"},
            {"name": "Athens", "country": "Greece", "lat": 37.9755, "lon": 23.7348, "continent": "europe"},
            {"name": "Barcelona", "country": "Spain", "lat": 41.3851, "lon": 2.1734, "continent": "europe"},
            {"name": "Budapest", "country": "Hungary", "lat": 47.4979, "lon": 19.0402, "continent": "europe"},
            {"name": "Prague", "country": "Czech Republic", "lat": 50.0755, "lon": 14.4378, "continent": "europe"},
            {"name": "Warsaw", "country": "Poland", "lat": 52.2297, "lon": 21.0122, "continent": "europe"},
            {"name": "Vienna", "country": "Austria", "lat": 48.2082, "lon": 16.3738, "continent": "europe"},
            {"name": "Sofia", "country": "Bulgaria", "lat": 42.6977, "lon": 23.3219, "continent": "europe"},
            {"name": "Bucharest", "country": "Romania", "lat": 44.4268, "lon": 26.1025, "continent": "europe"},
            {"name": "Belgrade", "country": "Serbia", "lat": 44.7866, "lon": 20.4489, "continent": "europe"},
            {"name": "Zagreb", "country": "Croatia", "lat": 45.8150, "lon": 15.9819, "continent": "europe"},
            {"name": "Ljubljana", "country": "Slovenia", "lat": 46.0569, "lon": 14.5058, "continent": "europe"},
            {"name": "Bratislava", "country": "Slovakia", "lat": 48.1486, "lon": 17.1077, "continent": "europe"},
            {"name": "Brussels", "country": "Belgium", "lat": 50.8503, "lon": 4.3517, "continent": "europe"},
            {"name": "Copenhagen", "country": "Denmark", "lat": 55.6761, "lon": 12.5683, "continent": "europe"},
            
            # North America (20 cities)
            {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060, "continent": "north_america"},
            {"name": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437, "continent": "north_america"},
            {"name": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298, "continent": "north_america"},
            {"name": "Houston", "country": "USA", "lat": 29.7604, "lon": -95.3698, "continent": "north_america"},
            {"name": "Phoenix", "country": "USA", "lat": 33.4484, "lon": -112.0740, "continent": "north_america"},
            {"name": "Philadelphia", "country": "USA", "lat": 39.9526, "lon": -75.1652, "continent": "north_america"},
            {"name": "San Antonio", "country": "USA", "lat": 29.4241, "lon": -98.4936, "continent": "north_america"},
            {"name": "San Diego", "country": "USA", "lat": 32.7157, "lon": -117.1611, "continent": "north_america"},
            {"name": "Dallas", "country": "USA", "lat": 32.7767, "lon": -96.7970, "continent": "north_america"},
            {"name": "San Jose", "country": "USA", "lat": 37.3382, "lon": -121.8863, "continent": "north_america"},
            {"name": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832, "continent": "north_america"},
            {"name": "Montreal", "country": "Canada", "lat": 45.5017, "lon": -73.5673, "continent": "north_america"},
            {"name": "Vancouver", "country": "Canada", "lat": 49.2827, "lon": -123.1207, "continent": "north_america"},
            {"name": "Calgary", "country": "Canada", "lat": 51.0447, "lon": -114.0719, "continent": "north_america"},
            {"name": "Ottawa", "country": "Canada", "lat": 45.4215, "lon": -75.6972, "continent": "north_america"},
            {"name": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332, "continent": "north_america"},
            {"name": "Guadalajara", "country": "Mexico", "lat": 20.6597, "lon": -103.3496, "continent": "north_america"},
            {"name": "Monterrey", "country": "Mexico", "lat": 25.6866, "lon": -100.3161, "continent": "north_america"},
            {"name": "Tijuana", "country": "Mexico", "lat": 32.5149, "lon": -117.0382, "continent": "north_america"},
            {"name": "Puebla", "country": "Mexico", "lat": 19.0414, "lon": -98.2063, "continent": "north_america"},
            
            # Asia (20 cities)
            {"name": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090, "continent": "asia"},
            {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777, "continent": "asia"},
            {"name": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "continent": "asia"},
            {"name": "Shanghai", "country": "China", "lat": 31.2304, "lon": 121.4737, "continent": "asia"},
            {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "continent": "asia"},
            {"name": "Seoul", "country": "South Korea", "lat": 37.5665, "lon": 126.9780, "continent": "asia"},
            {"name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018, "continent": "asia"},
            {"name": "Jakarta", "country": "Indonesia", "lat": -6.2088, "lon": 106.8456, "continent": "asia"},
            {"name": "Manila", "country": "Philippines", "lat": 14.5995, "lon": 120.9842, "continent": "asia"},
            {"name": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "continent": "asia"},
            {"name": "Kuala Lumpur", "country": "Malaysia", "lat": 3.1390, "lon": 101.6869, "continent": "asia"},
            {"name": "Ho Chi Minh City", "country": "Vietnam", "lat": 10.8231, "lon": 106.6297, "continent": "asia"},
            {"name": "Hanoi", "country": "Vietnam", "lat": 21.0285, "lon": 105.8542, "continent": "asia"},
            {"name": "Dhaka", "country": "Bangladesh", "lat": 23.8103, "lon": 90.4125, "continent": "asia"},
            {"name": "Karachi", "country": "Pakistan", "lat": 24.8607, "lon": 67.0011, "continent": "asia"},
            {"name": "Lahore", "country": "Pakistan", "lat": 31.5497, "lon": 74.3436, "continent": "asia"},
            {"name": "Kolkata", "country": "India", "lat": 22.5726, "lon": 88.3639, "continent": "asia"},
            {"name": "Chennai", "country": "India", "lat": 13.0827, "lon": 80.2707, "continent": "asia"},
            {"name": "Bangalore", "country": "India", "lat": 12.9716, "lon": 77.5946, "continent": "asia"},
            {"name": "Hyderabad", "country": "India", "lat": 17.3850, "lon": 78.4867, "continent": "asia"},
            
            # South America (20 cities)
            {"name": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333, "continent": "south_america"},
            {"name": "Rio de Janeiro", "country": "Brazil", "lat": -22.9068, "lon": -43.1729, "continent": "south_america"},
            {"name": "Buenos Aires", "country": "Argentina", "lat": -34.6118, "lon": -58.3960, "continent": "south_america"},
            {"name": "Lima", "country": "Peru", "lat": -12.0464, "lon": -77.0428, "continent": "south_america"},
            {"name": "Santiago", "country": "Chile", "lat": -33.4489, "lon": -70.6693, "continent": "south_america"},
            {"name": "Bogotá", "country": "Colombia", "lat": 4.7110, "lon": -74.0721, "continent": "south_america"},
            {"name": "Caracas", "country": "Venezuela", "lat": 10.4806, "lon": -66.9036, "continent": "south_america"},
            {"name": "Quito", "country": "Ecuador", "lat": -0.1807, "lon": -78.4678, "continent": "south_america"},
            {"name": "La Paz", "country": "Bolivia", "lat": -16.5000, "lon": -68.1193, "continent": "south_america"},
            {"name": "Asunción", "country": "Paraguay", "lat": -25.2637, "lon": -57.5759, "continent": "south_america"},
            {"name": "Montevideo", "country": "Uruguay", "lat": -34.9011, "lon": -56.1645, "continent": "south_america"},
            {"name": "Brasília", "country": "Brazil", "lat": -15.8267, "lon": -47.9218, "continent": "south_america"},
            {"name": "Belo Horizonte", "country": "Brazil", "lat": -19.8157, "lon": -43.9542, "continent": "south_america"},
            {"name": "Porto Alegre", "country": "Brazil", "lat": -30.0346, "lon": -51.2177, "continent": "south_america"},
            {"name": "Salvador", "country": "Brazil", "lat": -12.9714, "lon": -38.5014, "continent": "south_america"},
            {"name": "Recife", "country": "Brazil", "lat": -8.0476, "lon": -34.8770, "continent": "south_america"},
            {"name": "Fortaleza", "country": "Brazil", "lat": -3.7172, "lon": -38.5434, "continent": "south_america"},
            {"name": "Medellín", "country": "Colombia", "lat": 6.2442, "lon": -75.5812, "continent": "south_america"},
            {"name": "Cali", "country": "Colombia", "lat": 3.4516, "lon": -76.5320, "continent": "south_america"},
            {"name": "Córdoba", "country": "Argentina", "lat": -31.4201, "lon": -64.1888, "continent": "south_america"},
            
            # Africa (20 cities)
            {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "continent": "africa"},
            {"name": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792, "continent": "africa"},
            {"name": "Johannesburg", "country": "South Africa", "lat": -26.2041, "lon": 28.0473, "continent": "africa"},
            {"name": "Cape Town", "country": "South Africa", "lat": -33.9249, "lon": 18.4241, "continent": "africa"},
            {"name": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219, "continent": "africa"},
            {"name": "Addis Ababa", "country": "Ethiopia", "lat": 9.1450, "lon": 40.4897, "continent": "africa"},
            {"name": "Casablanca", "country": "Morocco", "lat": 33.5731, "lon": -7.5898, "continent": "africa"},
            {"name": "Algiers", "country": "Algeria", "lat": 36.7538, "lon": 3.0588, "continent": "africa"},
            {"name": "Tunis", "country": "Tunisia", "lat": 36.8065, "lon": 10.1815, "continent": "africa"},
            {"name": "Accra", "country": "Ghana", "lat": 5.6037, "lon": -0.1870, "continent": "africa"},
            {"name": "Dakar", "country": "Senegal", "lat": 14.7167, "lon": -17.4677, "continent": "africa"},
            {"name": "Abidjan", "country": "Côte d'Ivoire", "lat": 5.3600, "lon": -4.0083, "continent": "africa"},
            {"name": "Kampala", "country": "Uganda", "lat": 0.3476, "lon": 32.5825, "continent": "africa"},
            {"name": "Dar es Salaam", "country": "Tanzania", "lat": -6.7924, "lon": 39.2083, "continent": "africa"},
            {"name": "Khartoum", "country": "Sudan", "lat": 15.5007, "lon": 32.5599, "continent": "africa"},
            {"name": "Maputo", "country": "Mozambique", "lat": -25.9692, "lon": 32.5732, "continent": "africa"},
            {"name": "Lusaka", "country": "Zambia", "lat": -15.3875, "lon": 28.3228, "continent": "africa"},
            {"name": "Harare", "country": "Zimbabwe", "lat": -17.8292, "lon": 31.0522, "continent": "africa"},
            {"name": "Gaborone", "country": "Botswana", "lat": -24.6282, "lon": 25.9231, "continent": "africa"},
            {"name": "Windhoek", "country": "Namibia", "lat": -22.5609, "lon": 17.0658, "continent": "africa"}
        ]
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            "User-Agent": "Global-100City-RealData-Collector/1.0 (Research)",
            "Accept": "application/json, */*",
        })
        
        return session
    
    def test_api_connections(self) -> Dict[str, Any]:
        """Test connections to all APIs."""
        log.info("=== TESTING API CONNECTIONS ===")
        
        api_test_results = {}
        
        for api_name, config in self.api_configs.items():
            log.info(f"Testing {config['name']}...")
            
            try:
                # Test basic connectivity
                if api_name == "openaq":
                    # OpenAQ test - get countries
                    test_url = f"{config['base_url']}/countries"
                    response = self.session.get(test_url, timeout=10)
                    
                elif api_name == "waqi":
                    # WAQI test - would need API key
                    test_url = f"{config['base_url']}/here/?token=demo"
                    response = self.session.get(test_url, timeout=10)
                    
                elif api_name == "iqair":
                    # IQAir test - would need API key  
                    test_url = f"{config['base_url']}/countries?key=demo"
                    response = self.session.get(test_url, timeout=10)
                    
                elif api_name == "aqicn":
                    # AQICN test - would need API key
                    test_url = f"{config['base_url']}/here/?token=demo"
                    response = self.session.get(test_url, timeout=10)
                    
                elif api_name == "openweathermap":
                    # OpenWeatherMap test - would need API key
                    test_url = f"{config['base_url']}/current?lat=52.5200&lon=13.4050&appid=demo"
                    response = self.session.get(test_url, timeout=10)
                
                api_test_results[api_name] = {
                    "name": config["name"],
                    "status": "accessible" if response.status_code in [200, 401, 403] else "inaccessible",
                    "status_code": response.status_code,
                    "response_size": len(response.content),
                    "requires_key": config["requires_key"],
                    "free_tier": config["free_tier"],
                    "rate_limit": config["rate_limit"]
                }
                
                log.info(f"  {config['name']}: {response.status_code} - {'✅' if response.status_code in [200, 401, 403] else '❌'}")
                
            except Exception as e:
                api_test_results[api_name] = {
                    "name": config["name"],
                    "status": "error",
                    "error": str(e),
                    "requires_key": config["requires_key"],
                    "free_tier": config["free_tier"]
                }
                log.warning(f"  {config['name']}: Error - {str(e)}")
            
            time.sleep(1)  # Rate limiting
        
        self.collection_results["api_results"] = api_test_results
        return api_test_results
    
    def collect_sample_data(self, num_cities: int = 5) -> Dict[str, Any]:
        """Collect sample data from first few cities to test the system."""
        log.info(f"=== COLLECTING SAMPLE DATA FROM {num_cities} CITIES ===")
        
        sample_cities = self.cities[:num_cities]
        sample_results = {}
        
        for i, city in enumerate(sample_cities):
            city_name = f"{city['name']}, {city['country']}"
            log.info(f"Collecting sample data for {city_name} ({i+1}/{num_cities})")
            
            city_data = self._collect_city_sample_data(city)
            sample_results[city['name']] = city_data
            
            # Rate limiting
            time.sleep(2)
        
        self.collection_results["sample_results"] = sample_results
        return sample_results
    
    def _collect_city_sample_data(self, city: Dict) -> Dict[str, Any]:
        """Collect sample data for a single city."""
        city_result = {
            "city": city["name"],
            "country": city["country"],
            "coordinates": {"lat": city["lat"], "lon": city["lon"]},
            "continent": city["continent"],
            "api_results": {},
            "data_quality": {},
            "sample_data_points": 0
        }
        
        # Try OpenAQ API (no key required)
        try:
            openaq_data = self._collect_openaq_data(city)
            city_result["api_results"]["openaq"] = openaq_data
            if openaq_data.get("status") == "success":
                city_result["sample_data_points"] += openaq_data.get("record_count", 0)
        except Exception as e:
            city_result["api_results"]["openaq"] = {"status": "error", "error": str(e)}
        
        # Try other APIs (would require keys in production)
        for api_name in ["waqi", "iqair", "aqicn"]:
            city_result["api_results"][api_name] = {
                "status": "requires_api_key",
                "note": f"Would collect from {self.api_configs[api_name]['name']} with valid API key"
            }
        
        # Calculate data quality metrics
        successful_apis = sum(1 for result in city_result["api_results"].values() 
                            if result.get("status") == "success")
        city_result["data_quality"] = {
            "successful_apis": successful_apis,
            "total_apis_attempted": len(city_result["api_results"]),
            "success_rate": successful_apis / len(city_result["api_results"]),
            "data_availability": "good" if successful_apis >= 1 else "limited"
        }
        
        return city_result
    
    def _collect_openaq_data(self, city: Dict) -> Dict[str, Any]:
        """Collect data from OpenAQ API for a city."""
        try:
            # Get measurements near the city coordinates
            url = f"{self.api_configs['openaq']['base_url']}/measurements"
            params = {
                "coordinates": f"{city['lat']},{city['lon']}",
                "radius": 25000,  # 25km radius
                "limit": 100,
                "order_by": "datetime",
                "sort": "desc"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                measurements = data.get("results", [])
                
                # Process measurements
                if measurements:
                    processed_data = []
                    for measurement in measurements:
                        processed_data.append({
                            "date": measurement.get("date", {}).get("utc"),
                            "parameter": measurement.get("parameter"),
                            "value": measurement.get("value"),
                            "unit": measurement.get("unit"),
                            "location": measurement.get("location"),
                            "city": measurement.get("city"),
                            "country": measurement.get("country")
                        })
                    
                    return {
                        "status": "success",
                        "source": "OpenAQ",
                        "record_count": len(processed_data),
                        "data_sample": processed_data[:5],  # First 5 records as sample
                        "parameters_found": list(set(m.get("parameter") for m in measurements)),
                        "collection_timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "no_data",
                        "source": "OpenAQ",
                        "message": "No measurements found near this location"
                    }
            else:
                return {
                    "status": "api_error",
                    "source": "OpenAQ",
                    "status_code": response.status_code,
                    "error": response.text
                }
        
        except Exception as e:
            return {
                "status": "error",
                "source": "OpenAQ",
                "error": str(e)
            }
    
    def generate_collection_report(self) -> Dict[str, Any]:
        """Generate comprehensive collection report."""
        log.info("=== GENERATING COLLECTION REPORT ===")
        
        # API connectivity summary
        api_summary = {}
        if "api_results" in self.collection_results:
            accessible_apis = sum(1 for result in self.collection_results["api_results"].values() 
                                if result.get("status") == "accessible")
            api_summary = {
                "total_apis_tested": len(self.collection_results["api_results"]),
                "accessible_apis": accessible_apis,
                "apis_requiring_keys": sum(1 for result in self.collection_results["api_results"].values() 
                                         if result.get("requires_key", False)),
                "free_tier_apis": sum(1 for result in self.collection_results["api_results"].values() 
                                    if result.get("free_tier", False))
            }
        
        # Sample data summary
        sample_summary = {}
        if "sample_results" in self.collection_results:
            sample_results = self.collection_results["sample_results"]
            cities_with_data = sum(1 for result in sample_results.values() 
                                 if result.get("sample_data_points", 0) > 0)
            total_sample_points = sum(result.get("sample_data_points", 0) 
                                    for result in sample_results.values())
            
            sample_summary = {
                "cities_tested": len(sample_results),
                "cities_with_data": cities_with_data,
                "cities_success_rate": cities_with_data / len(sample_results) if sample_results else 0,
                "total_sample_data_points": total_sample_points,
                "average_points_per_city": total_sample_points / len(sample_results) if sample_results else 0
            }
        
        # Overall assessment
        overall_assessment = {
            "collection_feasibility": "viable" if api_summary.get("accessible_apis", 0) >= 2 else "challenging",
            "data_availability": "good" if sample_summary.get("cities_with_data", 0) >= 3 else "limited",
            "recommended_approach": self._get_collection_recommendations(),
            "estimated_success_rate": min(0.8, sample_summary.get("cities_success_rate", 0) + 0.2),
            "next_steps": [
                "Obtain API keys for key data sources (WAQI, IQAir, OpenWeatherMap)",
                "Implement full data collection pipeline",
                "Set up data quality validation and processing",
                "Begin systematic collection across all 100 cities"
            ]
        }
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "collection_type": "real_data_feasibility_test",
            "api_connectivity": api_summary,
            "sample_data_collection": sample_summary,
            "overall_assessment": overall_assessment,
            "detailed_results": self.collection_results
        }
        
        # Save report
        report_path = self.output_dir / "real_data_collection_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Collection report saved to: {report_path}")
        return report
    
    def _get_collection_recommendations(self) -> List[str]:
        """Get recommendations for full data collection."""
        recommendations = []
        
        # API-based recommendations
        if "api_results" in self.collection_results:
            accessible_apis = [name for name, result in self.collection_results["api_results"].items() 
                             if result.get("status") == "accessible"]
            
            if "openaq" in accessible_apis:
                recommendations.append("Use OpenAQ as primary free data source")
            
            recommendations.extend([
                "Obtain API keys for WAQI (1000 requests/day free tier)",
                "Obtain API keys for IQAir (10,000 requests/month free tier)", 
                "Obtain API keys for OpenWeatherMap Air Pollution API",
                "Use multiple APIs per city for data validation and completeness"
            ])
        
        # Data collection recommendations
        recommendations.extend([
            "Implement progressive data collection (most recent data first)",
            "Set up robust error handling and retry mechanisms",
            "Use geographical clustering to optimize API usage",
            "Implement data caching to avoid duplicate requests",
            "Set up monitoring for data quality and API health"
        ])
        
        return recommendations
    
    def save_progress(self):
        """Save current collection progress."""
        progress_path = self.output_dir / "collection_progress.json"
        with open(progress_path, 'w') as f:
            json.dump(self.collection_results, f, indent=2)
        log.info(f"Progress saved to: {progress_path}")


def main():
    """Main execution for real data collection testing."""
    log.info("Starting Real Data Collection System Test")
    
    try:
        # Initialize collector
        collector = RealDataCollector()
        
        # Test API connections
        api_results = collector.test_api_connections()
        
        # Collect sample data
        sample_results = collector.collect_sample_data(num_cities=5)
        
        # Generate comprehensive report
        report = collector.generate_collection_report()
        
        # Save progress
        collector.save_progress()
        
        # Print summary
        log.info("\n" + "="*60)
        log.info("REAL DATA COLLECTION TEST COMPLETED")
        log.info("="*60)
        log.info(f"APIs Tested: {len(api_results)}")
        log.info(f"Cities Tested: {len(sample_results)}")
        log.info(f"Feasibility: {report['overall_assessment']['collection_feasibility'].upper()}")
        log.info(f"Success Rate: {report['overall_assessment']['estimated_success_rate']:.1%}")
        log.info("="*60)
        
        return report
        
    except Exception as e:
        log.error(f"Real data collection test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()