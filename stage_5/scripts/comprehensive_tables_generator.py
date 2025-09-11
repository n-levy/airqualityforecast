#!/usr/bin/env python3
"""
Comprehensive Tables Generator
=============================

Generates comprehensive tables for:
1. All features for every city
2. All APIs used for every city
3. Local AQI standard calculations applied to every city

Saves tables as CSV files and JSON summaries in the project files.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/comprehensive_tables_generation.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class ComprehensiveTablesGenerator:
    """Generator for comprehensive tables of city features, APIs, and AQI standards."""

    def __init__(self):
        """Initialize comprehensive tables generator."""
        self.generation_results = {
            "generation_type": "comprehensive_tables",
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }

        # Create directories
        self.output_dir = Path("stage_5/comprehensive_tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load enhanced dataset
        self.input_file = Path(
            "stage_5/enhanced_features/enhanced_worst_air_quality_with_features.json"
        )

        log.info("Comprehensive Tables Generator initialized")

    def generate_all_tables(self) -> Dict[str, Any]:
        """Generate all comprehensive tables."""
        log.info("=== STARTING COMPREHENSIVE TABLES GENERATION ===")

        # Load enhanced dataset
        with open(self.input_file, "r") as f:
            dataset = json.load(f)

        log.info(f"Loaded enhanced dataset with {len(dataset['city_results'])} cities")

        # Generate tables
        features_table = self._generate_features_table(dataset)
        apis_table = self._generate_apis_table(dataset)
        aqi_standards_table = self._generate_aqi_standards_table(dataset)

        # Save tables
        self._save_tables(features_table, apis_table, aqi_standards_table)

        log.info("=== COMPREHENSIVE TABLES GENERATION COMPLETED ===")

        return {
            "status": "completed",
            "tables_generated": 3,
            "total_cities": len(dataset["city_results"]),
            "output_directory": str(self.output_dir),
        }

    def _generate_features_table(self, dataset: Dict) -> pd.DataFrame:
        """Generate comprehensive features table for all cities."""
        log.info("Generating comprehensive features table...")

        features_data = []

        for city_name, city_data in dataset["city_results"].items():
            # Basic city information
            city_info = {
                "City": city_name,
                "Country": city_data.get("country", "Unknown"),
                "Continent": city_data.get("continent", "unknown").title(),
                "Latitude": city_data.get("coordinates", {}).get("lat", 0),
                "Longitude": city_data.get("coordinates", {}).get("lon", 0),
                "Average_AQI": city_data.get("avg_aqi", 0),
                "Average_PM25": city_data.get("avg_pm25", 0),
                "Total_Records": city_data.get("total_records", 0),
                "Successful_Sources": city_data.get("successful_sources", 0),
            }

            # Pollutant features (from sample data)
            pollutant_features = self._extract_pollutant_features(city_data)
            city_info.update(pollutant_features)

            # Meteorological features
            meteorological_features = self._extract_meteorological_features(city_data)
            city_info.update(meteorological_features)

            # Fire activity features
            fire_features = self._extract_fire_features(city_data)
            city_info.update(fire_features)

            # Holiday features
            holiday_features = self._extract_holiday_features(city_data)
            city_info.update(holiday_features)

            # Temporal features
            temporal_features = self._extract_temporal_features(city_data)
            city_info.update(temporal_features)

            # Data quality features
            quality_features = self._extract_quality_features(city_data)
            city_info.update(quality_features)

            features_data.append(city_info)

        features_df = pd.DataFrame(features_data)
        log.info(
            f"Generated features table with {len(features_df)} cities and {len(features_df.columns)} features"
        )

        return features_df

    def _generate_apis_table(self, dataset: Dict) -> pd.DataFrame:
        """Generate comprehensive APIs table for all cities."""
        log.info("Generating comprehensive APIs table...")

        apis_data = []

        for city_name, city_data in dataset["city_results"].items():
            city_info = {
                "City": city_name,
                "Country": city_data.get("country", "Unknown"),
                "Continent": city_data.get("continent", "unknown").title(),
                "Latitude": city_data.get("coordinates", {}).get("lat", 0),
                "Longitude": city_data.get("coordinates", {}).get("lon", 0),
            }

            # Extract API information from data sources
            data_sources = city_data.get("data_sources", {})

            # API availability and status
            for api_name in [
                "waqi",
                "openweathermap",
                "realistic_high_pollution",
                "enhanced_pollution_scenarios",
            ]:
                api_data = data_sources.get(api_name, {})
                api_status = api_data.get("status", "not_attempted")
                api_source = api_data.get("source", "Unknown")
                api_records = api_data.get("record_count", 0)

                city_info[f"{api_name.upper()}_Status"] = api_status
                city_info[f"{api_name.upper()}_Source"] = api_source
                city_info[f"{api_name.upper()}_Records"] = api_records
                city_info[f"{api_name.upper()}_Available"] = api_status in [
                    "success",
                    "partial_success",
                ]

            # API summary statistics
            successful_apis = sum(
                1
                for source in data_sources.values()
                if source.get("status") in ["success", "partial_success"]
            )
            total_apis = len(data_sources)

            city_info["Total_APIs_Attempted"] = total_apis
            city_info["Successful_APIs"] = successful_apis
            city_info["API_Success_Rate"] = round(
                successful_apis / total_apis if total_apis > 0 else 0, 3
            )

            # Data quality indicators
            city_info["Primary_Data_Source"] = self._get_primary_data_source(
                data_sources
            )
            city_info["Data_Quality_Level"] = self._assess_data_quality(data_sources)
            city_info["Real_Data_Available"] = any(
                source.get("source", "").startswith("WAQI")
                or source.get("source", "").startswith("OpenWeatherMap")
                for source in data_sources.values()
            )

            # API-specific features
            city_info["Fire_Features_Added"] = any(
                source.get("fire_features_added", False)
                for source in data_sources.values()
            )
            city_info["Holiday_Features_Added"] = any(
                source.get("holiday_features_added", False)
                for source in data_sources.values()
            )

            apis_data.append(city_info)

        apis_df = pd.DataFrame(apis_data)
        log.info(
            f"Generated APIs table with {len(apis_df)} cities and {len(apis_df.columns)} API-related features"
        )

        return apis_df

    def _generate_aqi_standards_table(self, dataset: Dict) -> pd.DataFrame:
        """Generate comprehensive AQI standards table for all cities."""
        log.info("Generating comprehensive AQI standards table...")

        aqi_data = []

        # Define AQI standards by country/region
        aqi_standards = self._get_aqi_standards_mapping()

        for city_name, city_data in dataset["city_results"].items():
            country = city_data.get("country", "Unknown")
            continent = city_data.get("continent", "unknown")

            # Determine applicable AQI standard
            aqi_standard = self._determine_aqi_standard(country, continent)
            standard_details = aqi_standards.get(aqi_standard, aqi_standards["US EPA"])

            city_info = {
                "City": city_name,
                "Country": country,
                "Continent": continent.title(),
                "AQI_Standard": aqi_standard,
                "Standard_Name": standard_details["name"],
                "Standard_Authority": standard_details["authority"],
                "Implementation_Year": standard_details.get(
                    "implementation_year", "Unknown"
                ),
            }

            # Pollutant-specific AQI breakpoints and calculations
            for pollutant in ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]:
                if pollutant.lower().replace(".", "") in standard_details.get(
                    "breakpoints", {}
                ):
                    breakpoints = standard_details["breakpoints"][
                        pollutant.lower().replace(".", "")
                    ]
                    city_info[f"{pollutant}_Breakpoints"] = str(
                        breakpoints[:4]
                    )  # First 4 breakpoints
                    city_info[f"{pollutant}_Units"] = standard_details.get(
                        "units", {}
                    ).get(pollutant.lower().replace(".", ""), "µg/m³")
                    city_info[f"{pollutant}_Good_Threshold"] = (
                        breakpoints[1] if len(breakpoints) > 1 else 0
                    )
                    city_info[f"{pollutant}_Moderate_Threshold"] = (
                        breakpoints[2] if len(breakpoints) > 2 else 0
                    )
                    city_info[f"{pollutant}_Unhealthy_Threshold"] = (
                        breakpoints[3] if len(breakpoints) > 3 else 0
                    )
                else:
                    city_info[f"{pollutant}_Breakpoints"] = "Not defined"
                    city_info[f"{pollutant}_Units"] = "µg/m³"
                    city_info[f"{pollutant}_Good_Threshold"] = 0
                    city_info[f"{pollutant}_Moderate_Threshold"] = 0
                    city_info[f"{pollutant}_Unhealthy_Threshold"] = 0

            # AQI calculation characteristics
            city_info["AQI_Scale"] = standard_details.get("scale", "0-500")
            city_info["AQI_Categories"] = len(standard_details.get("categories", []))
            city_info["Predominant_Pollutant_Method"] = standard_details.get(
                "predominant_method", "Maximum AQI"
            )
            city_info["Real_Time_Calculation"] = standard_details.get("real_time", True)
            city_info["Health_Advisory_Levels"] = len(
                standard_details.get("health_advisories", [])
            )

            # Regional characteristics
            city_info["Regional_Adjustments"] = standard_details.get(
                "regional_adjustments", False
            )
            city_info["Climate_Considerations"] = standard_details.get(
                "climate_considerations", False
            )
            city_info["Local_Pollutant_Priority"] = self._get_local_pollutant_priority(
                country, continent
            )

            aqi_data.append(city_info)

        aqi_df = pd.DataFrame(aqi_data)
        log.info(
            f"Generated AQI standards table with {len(aqi_df)} cities and {len(aqi_df.columns)} AQI-related features"
        )

        return aqi_df

    def _extract_pollutant_features(self, city_data: Dict) -> Dict:
        """Extract pollutant concentration features from city data."""
        features = {}

        # Get sample data from the best available source
        sample_data = self._get_best_sample_data(city_data)

        if sample_data and "pollutants" in sample_data:
            pollutants = sample_data["pollutants"]
            for pollutant, value in pollutants.items():
                features[f"{pollutant}_Concentration"] = value
                features[f"{pollutant}_Available"] = True
        else:
            # Default pollutant availability
            for pollutant in ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]:
                features[f"{pollutant}_Concentration"] = 0
                features[f"{pollutant}_Available"] = False

        return features

    def _extract_meteorological_features(self, city_data: Dict) -> Dict:
        """Extract meteorological features from city data."""
        features = {}

        sample_data = self._get_best_sample_data(city_data)

        if sample_data and "meteorology" in sample_data:
            meteorology = sample_data["meteorology"]
            features["Temperature_C"] = meteorology.get("temperature", 0)
            features["Humidity_Percent"] = meteorology.get("humidity", 0)
            features["Wind_Speed_ms"] = meteorology.get("wind_speed", 0)
            features["Pressure_hPa"] = meteorology.get("pressure", 0)
            features["Visibility_km"] = meteorology.get("visibility_km", 10)
            features["Meteorology_Available"] = True
        else:
            features["Temperature_C"] = 0
            features["Humidity_Percent"] = 0
            features["Wind_Speed_ms"] = 0
            features["Pressure_hPa"] = 0
            features["Visibility_km"] = 0
            features["Meteorology_Available"] = False

        return features

    def _extract_fire_features(self, city_data: Dict) -> Dict:
        """Extract fire activity features from city data."""
        features = {}

        # Fire metadata
        fire_metadata = city_data.get("fire_metadata", {})
        fire_seasons = fire_metadata.get("fire_seasons", {})

        features["Fire_Peak_Months"] = str(fire_seasons.get("peak", []))
        features["Fire_High_Months"] = str(fire_seasons.get("high", []))
        features["Fire_Risk_Level"] = fire_metadata.get("fire_risk_factors", {}).get(
            "climate_fire_risk", "unknown"
        )
        features["Primary_Fire_Source"] = (
            fire_metadata.get("typical_fire_sources", ["unknown"])[0]
            if fire_metadata.get("typical_fire_sources")
            else "unknown"
        )

        # Sample fire features
        sample_data = self._get_best_sample_data(city_data)
        if sample_data and "fire_features" in sample_data:
            fire_features = sample_data["fire_features"]
            features["Fire_Weather_Index"] = fire_features.get("fire_weather_index", 0)
            features["Fire_Danger_Rating"] = fire_features.get(
                "fire_danger_rating", "unknown"
            )
            features["Active_Fires_Nearby"] = fire_features.get(
                "active_fires_nearby", 0
            )
            features["Fire_PM25_Contribution"] = fire_features.get(
                "fire_pm25_contribution", 0
            )
            features["Fire_Features_Available"] = True
        else:
            features["Fire_Weather_Index"] = 0
            features["Fire_Danger_Rating"] = "unknown"
            features["Active_Fires_Nearby"] = 0
            features["Fire_PM25_Contribution"] = 0
            features["Fire_Features_Available"] = False

        return features

    def _extract_holiday_features(self, city_data: Dict) -> Dict:
        """Extract holiday features from city data."""
        features = {}

        # Holiday metadata
        holiday_metadata = city_data.get("holiday_metadata", {})
        major_holidays = holiday_metadata.get("major_holidays", {})

        features["Total_Major_Holidays"] = len(major_holidays)
        features["Has_Religious_Holidays"] = any(
            "christmas" in h.lower()
            or "easter" in h.lower()
            or "diwali" in h.lower()
            or "eid" in h.lower()
            for h in major_holidays.keys()
        )
        features["Has_National_Holidays"] = any(
            "independence" in h.lower() or "national" in h.lower()
            for h in major_holidays.keys()
        )
        features["Holiday_Pollution_Impact"] = holiday_metadata.get(
            "holiday_pollution_impact", {}
        ).get("fireworks_emissions", "moderate")

        # Sample holiday features
        sample_data = self._get_best_sample_data(city_data)
        if sample_data and "holiday_features" in sample_data:
            holiday_features = sample_data["holiday_features"]
            features["Holiday_Pollution_Multiplier"] = holiday_features.get(
                "holiday_pollution_multiplier", 1.0
            )
            features["Fireworks_Likely"] = holiday_features.get(
                "fireworks_likely", False
            )
            features["Holiday_Season"] = holiday_features.get(
                "holiday_season", "regular_period"
            )
            features["Holiday_Features_Available"] = True
        else:
            features["Holiday_Pollution_Multiplier"] = 1.0
            features["Fireworks_Likely"] = False
            features["Holiday_Season"] = "regular_period"
            features["Holiday_Features_Available"] = False

        return features

    def _extract_temporal_features(self, city_data: Dict) -> Dict:
        """Extract temporal features from city data."""
        features = {}

        sample_data = self._get_best_sample_data(city_data)

        if sample_data:
            # Extract temporal information
            if "date" in sample_data:
                try:
                    from datetime import datetime

                    date_obj = datetime.strptime(sample_data["date"], "%Y-%m-%d")
                    features["Sample_Month"] = date_obj.month
                    features["Sample_Day_of_Year"] = date_obj.timetuple().tm_yday
                    features["Sample_Weekday"] = date_obj.weekday()
                    features["Sample_Is_Weekend"] = date_obj.weekday() >= 5
                    features["Temporal_Features_Available"] = True
                except:
                    features["Sample_Month"] = 0
                    features["Sample_Day_of_Year"] = 0
                    features["Sample_Weekday"] = 0
                    features["Sample_Is_Weekend"] = False
                    features["Temporal_Features_Available"] = False
            else:
                features["Sample_Month"] = 0
                features["Sample_Day_of_Year"] = 0
                features["Sample_Weekday"] = 0
                features["Sample_Is_Weekend"] = False
                features["Temporal_Features_Available"] = False

        return features

    def _extract_quality_features(self, city_data: Dict) -> Dict:
        """Extract data quality features from city data."""
        features = {}

        data_sources = city_data.get("data_sources", {})

        # Data completeness
        features["Data_Completeness_Score"] = self._calculate_completeness_score(
            data_sources
        )
        features["Has_Real_Data"] = any(
            "WAQI" in source.get("source", "") for source in data_sources.values()
        )
        features["Has_Synthetic_Data"] = any(
            "Synthetic" in source.get("source", "")
            or "Realistic" in source.get("source", "")
            for source in data_sources.values()
        )
        features["Has_Extreme_Scenarios"] = any(
            "Enhanced Pollution Scenarios" in source.get("source", "")
            for source in data_sources.values()
        )

        # Quality assessment
        features["Overall_Data_Quality"] = self._assess_overall_data_quality(city_data)
        features["Data_Coverage_Days"] = self._estimate_data_coverage(data_sources)
        features["Data_Sources_Count"] = len(data_sources)

        return features

    def _get_best_sample_data(self, city_data: Dict) -> Optional[Dict]:
        """Get the best available sample data from city data sources."""
        data_sources = city_data.get("data_sources", {})

        # Priority order for data sources
        priority_order = [
            "waqi",
            "realistic_high_pollution",
            "enhanced_pollution_scenarios",
            "openweathermap",
        ]

        for source_name in priority_order:
            if source_name in data_sources:
                source_data = data_sources[source_name]
                if source_data.get("status") == "success":
                    # Try data_sample first, then historical_data_sample
                    if "data_sample" in source_data and source_data["data_sample"]:
                        return source_data["data_sample"][0]
                    elif (
                        "historical_data_sample" in source_data
                        and source_data["historical_data_sample"]
                    ):
                        return source_data["historical_data_sample"][0]

        return None

    def _get_primary_data_source(self, data_sources: Dict) -> str:
        """Determine the primary data source for a city."""
        successful_sources = [
            name
            for name, source in data_sources.items()
            if source.get("status") in ["success", "partial_success"]
        ]

        if not successful_sources:
            return "None"

        # Priority order
        priority_order = [
            "waqi",
            "openweathermap",
            "realistic_high_pollution",
            "enhanced_pollution_scenarios",
        ]

        for source_name in priority_order:
            if source_name in successful_sources:
                return source_name.upper()

        return successful_sources[0].upper()

    def _assess_data_quality(self, data_sources: Dict) -> str:
        """Assess overall data quality level."""
        successful_sources = sum(
            1
            for source in data_sources.values()
            if source.get("status") in ["success", "partial_success"]
        )
        total_sources = len(data_sources)

        if successful_sources == 0:
            return "No Data"
        elif successful_sources == total_sources:
            return "Excellent"
        elif successful_sources >= total_sources * 0.8:
            return "Very Good"
        elif successful_sources >= total_sources * 0.6:
            return "Good"
        elif successful_sources >= total_sources * 0.4:
            return "Fair"
        else:
            return "Poor"

    def _calculate_completeness_score(self, data_sources: Dict) -> float:
        """Calculate data completeness score (0-1)."""
        if not data_sources:
            return 0.0

        total_records = sum(
            source.get("record_count", 0) for source in data_sources.values()
        )
        expected_records = (
            760  # Expected records per city (365 daily + 30 scenarios + 365 historical)
        )

        return min(1.0, total_records / expected_records)

    def _assess_overall_data_quality(self, city_data: Dict) -> str:
        """Assess overall data quality for a city."""
        total_records = city_data.get("total_records", 0)
        successful_sources = city_data.get("successful_sources", 0)

        if total_records >= 700 and successful_sources >= 3:
            return "Excellent"
        elif total_records >= 500 and successful_sources >= 2:
            return "Very Good"
        elif total_records >= 300 and successful_sources >= 2:
            return "Good"
        elif total_records >= 100 and successful_sources >= 1:
            return "Fair"
        else:
            return "Poor"

    def _estimate_data_coverage(self, data_sources: Dict) -> int:
        """Estimate data coverage in days."""
        max_records = max(
            (source.get("record_count", 0) for source in data_sources.values()),
            default=0,
        )
        # Assuming daily records, but accounting for multiple types (daily + scenarios + historical)
        return min(365, max_records // 2)  # Conservative estimate

    def _get_aqi_standards_mapping(self) -> Dict:
        """Get comprehensive AQI standards mapping."""
        return {
            "US EPA": {
                "name": "US Environmental Protection Agency AQI",
                "authority": "US EPA",
                "implementation_year": 1999,
                "scale": "0-500",
                "categories": [
                    "Good",
                    "Moderate",
                    "Unhealthy for Sensitive Groups",
                    "Unhealthy",
                    "Very Unhealthy",
                    "Hazardous",
                ],
                "breakpoints": {
                    "pm25": [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],
                    "pm10": [0, 54, 154, 254, 354, 424, 504, 604],
                    "no2": [0, 53, 100, 360, 649, 1249, 1649, 2049],
                    "o3": [0, 54, 70, 85, 105, 200, 300, 500],
                    "so2": [0, 35, 75, 185, 304, 604, 804, 1004],
                    "co": [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4],
                },
                "units": {
                    "pm25": "µg/m³",
                    "pm10": "µg/m³",
                    "no2": "ppb",
                    "o3": "ppb",
                    "so2": "ppb",
                    "co": "ppm",
                },
                "predominant_method": "Maximum AQI",
                "real_time": True,
                "health_advisories": [
                    "Good",
                    "Moderate",
                    "Unhealthy for Sensitive Groups",
                    "Unhealthy",
                    "Very Unhealthy",
                    "Hazardous",
                ],
                "regional_adjustments": False,
                "climate_considerations": False,
            },
            "European EAQI": {
                "name": "European Air Quality Index",
                "authority": "European Environment Agency",
                "implementation_year": 2017,
                "scale": "0-100+",
                "categories": [
                    "Good",
                    "Fair",
                    "Moderate",
                    "Poor",
                    "Very Poor",
                    "Extremely Poor",
                ],
                "breakpoints": {
                    "pm25": [0, 10, 20, 25, 50, 75, 800],
                    "pm10": [0, 20, 40, 50, 100, 150, 1200],
                    "no2": [0, 40, 90, 120, 230, 340, 1000],
                    "o3": [0, 50, 100, 130, 240, 380, 800],
                    "so2": [0, 100, 200, 350, 500, 750, 1250],
                },
                "units": {
                    "pm25": "µg/m³",
                    "pm10": "µg/m³",
                    "no2": "µg/m³",
                    "o3": "µg/m³",
                    "so2": "µg/m³",
                },
                "predominant_method": "Common Air Quality Index",
                "real_time": True,
                "health_advisories": [
                    "Good",
                    "Fair",
                    "Moderate",
                    "Poor",
                    "Very Poor",
                    "Extremely Poor",
                ],
                "regional_adjustments": True,
                "climate_considerations": True,
            },
            "Chinese AQI": {
                "name": "Chinese Air Quality Index",
                "authority": "Ministry of Environmental Protection of China",
                "implementation_year": 2012,
                "scale": "0-500",
                "categories": [
                    "Excellent",
                    "Good",
                    "Lightly Polluted",
                    "Moderately Polluted",
                    "Heavily Polluted",
                    "Severely Polluted",
                ],
                "breakpoints": {
                    "pm25": [0, 35, 75, 115, 150, 250, 350, 500],
                    "pm10": [0, 50, 150, 250, 350, 420, 500, 600],
                    "no2": [0, 40, 80, 180, 280, 565, 750, 940],
                    "o3": [0, 100, 160, 215, 265, 800],
                    "so2": [0, 50, 150, 475, 800, 1600, 2100, 2620],
                    "co": [0, 2, 4, 14, 24, 36, 48, 60],
                },
                "units": {
                    "pm25": "µg/m³",
                    "pm10": "µg/m³",
                    "no2": "µg/m³",
                    "o3": "µg/m³",
                    "so2": "µg/m³",
                    "co": "mg/m³",
                },
                "predominant_method": "Maximum AQI",
                "real_time": True,
                "health_advisories": [
                    "Excellent",
                    "Good",
                    "Lightly Polluted",
                    "Moderately Polluted",
                    "Heavily Polluted",
                    "Severely Polluted",
                ],
                "regional_adjustments": True,
                "climate_considerations": True,
            },
            "Indian AQI": {
                "name": "Indian Air Quality Index",
                "authority": "Central Pollution Control Board",
                "implementation_year": 2014,
                "scale": "0-500",
                "categories": [
                    "Good",
                    "Satisfactory",
                    "Moderately Polluted",
                    "Poor",
                    "Very Poor",
                    "Severe",
                ],
                "breakpoints": {
                    "pm25": [0, 30, 60, 90, 120, 250, 380, 500],
                    "pm10": [0, 50, 100, 250, 350, 430, 500, 600],
                    "no2": [0, 40, 80, 180, 280, 400, 500, 600],
                    "o3": [0, 50, 100, 168, 208, 748, 850, 1000],
                    "so2": [0, 40, 80, 380, 800, 1600, 2000, 2400],
                    "co": [0, 1, 2, 10, 17, 34, 46, 57],
                },
                "units": {
                    "pm25": "µg/m³",
                    "pm10": "µg/m³",
                    "no2": "µg/m³",
                    "o3": "µg/m³",
                    "so2": "µg/m³",
                    "co": "mg/m³",
                },
                "predominant_method": "Sub-Index Maximum",
                "real_time": True,
                "health_advisories": [
                    "Good",
                    "Satisfactory",
                    "Moderately Polluted",
                    "Poor",
                    "Very Poor",
                    "Severe",
                ],
                "regional_adjustments": True,
                "climate_considerations": True,
            },
            "Canadian AQHI": {
                "name": "Canadian Air Quality Health Index",
                "authority": "Environment and Climate Change Canada",
                "implementation_year": 2008,
                "scale": "1-10+",
                "categories": [
                    "Low Health Risk",
                    "Moderate Health Risk",
                    "High Health Risk",
                    "Very High Health Risk",
                ],
                "breakpoints": {
                    "pm25": [0, 28, 60, 90, 120, 250],
                    "no2": [0, 30, 60, 120, 240, 480],
                    "o3": [0, 82, 164, 246, 328, 656],
                },
                "units": {"pm25": "µg/m³", "no2": "ppb", "o3": "ppb"},
                "predominant_method": "Health Risk Formula",
                "real_time": True,
                "health_advisories": [
                    "Low Health Risk",
                    "Moderate Health Risk",
                    "High Health Risk",
                    "Very High Health Risk",
                ],
                "regional_adjustments": True,
                "climate_considerations": True,
            },
        }

    def _determine_aqi_standard(self, country: str, continent: str) -> str:
        """Determine the applicable AQI standard for a country."""
        # Country-specific mappings
        country_standards = {
            "USA": "US EPA",
            "China": "Chinese AQI",
            "India": "Indian AQI",
            "Pakistan": "Indian AQI",
            "Bangladesh": "Indian AQI",
            "Canada": "Canadian AQHI",
            "Germany": "European EAQI",
            "UK": "European EAQI",
            "France": "European EAQI",
            "Spain": "European EAQI",
            "Italy": "European EAQI",
            "Poland": "European EAQI",
            "Bulgaria": "European EAQI",
            "Romania": "European EAQI",
            "Serbia": "European EAQI",
            "Hungary": "European EAQI",
            "Czech Republic": "European EAQI",
            "Slovakia": "European EAQI",
            "Bosnia and Herzegovina": "European EAQI",
            "North Macedonia": "European EAQI",
        }

        if country in country_standards:
            return country_standards[country]

        # Continental defaults
        continental_defaults = {
            "europe": "European EAQI",
            "asia": "US EPA",  # Default for countries without specific standards
            "north_america": "US EPA",
            "south_america": "US EPA",
            "africa": "US EPA",
        }

        return continental_defaults.get(continent, "US EPA")

    def _get_local_pollutant_priority(self, country: str, continent: str) -> str:
        """Get the local pollutant priority for a country/region."""
        # Regional pollutant priorities based on common pollution sources
        pollutant_priorities = {
            "India": "PM2.5",
            "China": "PM2.5",
            "Pakistan": "PM2.5",
            "Bangladesh": "PM2.5",
            "USA": "O3",
            "Canada": "PM2.5",
            "Germany": "NO2",
            "UK": "NO2",
            "France": "NO2",
            "Italy": "PM10",
            "Poland": "PM2.5",
            "Mexico": "O3",
            "Brazil": "PM10",
            "Chile": "PM2.5",
            "Egypt": "PM10",
            "Nigeria": "PM2.5",
            "South Africa": "PM10",
        }

        if country in pollutant_priorities:
            return pollutant_priorities[country]

        # Continental defaults
        continental_defaults = {
            "asia": "PM2.5",
            "africa": "PM10",
            "europe": "NO2",
            "north_america": "O3",
            "south_america": "PM10",
        }

        return continental_defaults.get(continent, "PM2.5")

    def _save_tables(
        self, features_df: pd.DataFrame, apis_df: pd.DataFrame, aqi_df: pd.DataFrame
    ):
        """Save all tables to CSV files and generate summaries."""
        log.info("Saving comprehensive tables...")

        # Save CSV files
        features_csv = self.output_dir / "comprehensive_features_table.csv"
        apis_csv = self.output_dir / "comprehensive_apis_table.csv"
        aqi_csv = self.output_dir / "comprehensive_aqi_standards_table.csv"

        features_df.to_csv(features_csv, index=False)
        apis_df.to_csv(apis_csv, index=False)
        aqi_df.to_csv(aqi_csv, index=False)

        log.info(f"Saved features table: {features_csv}")
        log.info(f"Saved APIs table: {apis_csv}")
        log.info(f"Saved AQI standards table: {aqi_csv}")

        # Generate and save summaries
        summaries = {
            "features_summary": {
                "total_cities": len(features_df),
                "total_features": len(features_df.columns),
                "continents": features_df["Continent"].value_counts().to_dict(),
                "avg_aqi_by_continent": features_df.groupby("Continent")["Average_AQI"]
                .mean()
                .round(1)
                .to_dict(),
                "feature_categories": {
                    "basic_info": 9,
                    "pollutant_features": 12,
                    "meteorological_features": 6,
                    "fire_features": 9,
                    "holiday_features": 8,
                    "temporal_features": 5,
                    "quality_features": 8,
                },
            },
            "apis_summary": {
                "total_cities": len(apis_df),
                "api_success_rates": {
                    "WAQI": (
                        apis_df["WAQI_Available"].sum() / len(apis_df) * 100
                    ).round(1),
                    "Realistic_High_Pollution": (
                        apis_df["REALISTIC_HIGH_POLLUTION_Available"].sum()
                        / len(apis_df)
                        * 100
                    ).round(1),
                    "Enhanced_Scenarios": (
                        apis_df["ENHANCED_POLLUTION_SCENARIOS_Available"].sum()
                        / len(apis_df)
                        * 100
                    ).round(1),
                },
                "data_quality_distribution": apis_df["Data_Quality_Level"]
                .value_counts()
                .to_dict(),
                "cities_with_real_data": apis_df["Real_Data_Available"].sum(),
                "average_api_success_rate": apis_df["API_Success_Rate"].mean().round(3),
            },
            "aqi_standards_summary": {
                "total_cities": len(aqi_df),
                "standards_distribution": aqi_df["AQI_Standard"]
                .value_counts()
                .to_dict(),
                "standards_by_continent": aqi_df.groupby("Continent")["AQI_Standard"]
                .apply(lambda x: x.value_counts().to_dict())
                .to_dict(),
                "pollutant_priorities": aqi_df["Local_Pollutant_Priority"]
                .value_counts()
                .to_dict(),
                "standards_with_regional_adjustments": aqi_df[
                    "Regional_Adjustments"
                ].sum(),
            },
            "generation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tables": 3,
                "total_cities_processed": len(features_df),
                "input_dataset": str(self.input_file),
                "output_directory": str(self.output_dir),
            },
        }

        # Save summary
        summary_file = self.output_dir / "comprehensive_tables_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summaries, f, indent=2)

        log.info(f"Saved comprehensive summary: {summary_file}")

        # Generate README
        self._generate_readme(features_df, apis_df, aqi_df, summaries)

    def _generate_readme(
        self,
        features_df: pd.DataFrame,
        apis_df: pd.DataFrame,
        aqi_df: pd.DataFrame,
        summaries: Dict,
    ):
        """Generate README file for the comprehensive tables."""
        readme_content = f"""# Comprehensive Tables for Global Worst Air Quality Cities Dataset

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This directory contains comprehensive tables summarizing all features, APIs, and AQI standards for the {len(features_df)} worst air quality cities globally (20 cities per continent).

## Files

### 1. Comprehensive Features Table (`comprehensive_features_table.csv`)
**{len(features_df)} cities × {len(features_df.columns)} features**

Contains all available features for every city:
- **Basic Information**: City, Country, Continent, Coordinates, AQI/PM2.5 levels
- **Pollutant Concentrations**: PM2.5, PM10, NO2, O3, SO2, CO levels and availability
- **Meteorological Data**: Temperature, humidity, wind speed, pressure, visibility
- **Fire Activity Features**: Fire weather index, danger rating, nearby fires, PM2.5 contribution
- **Holiday Features**: Holiday impacts, pollution multipliers, celebration patterns
- **Temporal Features**: Sample dates, seasonal patterns, weekday/weekend effects
- **Data Quality Features**: Completeness scores, data coverage, source reliability

### 2. Comprehensive APIs Table (`comprehensive_apis_table.csv`)
**{len(apis_df)} cities × {len(apis_df.columns)} API-related features**

Documents all data sources and APIs used for each city:
- **API Status**: Success/failure status for WAQI, OpenWeatherMap, synthetic sources
- **API Performance**: Record counts, success rates, data availability
- **Data Source Quality**: Primary sources, quality levels, real vs synthetic data
- **Feature Enhancement**: Fire and holiday feature integration status

**API Success Rates:**
{chr(10).join(f"- {api}: {rate}%" for api, rate in summaries['apis_summary']['api_success_rates'].items())}

### 3. Comprehensive AQI Standards Table (`comprehensive_aqi_standards_table.csv`)
**{len(aqi_df)} cities × {len(aqi_df.columns)} AQI-related features**

Details the local AQI calculation standards applied to each city:
- **AQI Standards**: Applied standard (US EPA, European EAQI, Chinese AQI, etc.)
- **Pollutant Breakpoints**: Threshold values for Good/Moderate/Unhealthy levels
- **Standard Characteristics**: Scale, categories, calculation methods
- **Regional Adaptations**: Local adjustments and pollutant priorities

**AQI Standards Distribution:**
{chr(10).join(f"- {standard}: {count} cities" for standard, count in summaries['aqi_standards_summary']['standards_distribution'].items())}

## Continental Coverage

{chr(10).join(f"- **{continent}**: {count} cities (Avg AQI: {summaries['features_summary']['avg_aqi_by_continent'][continent]})" for continent, count in summaries['features_summary']['continents'].items())}

## Data Quality Summary

- **Cities with Real Data**: {summaries['apis_summary']['cities_with_real_data']}/100
- **Average API Success Rate**: {summaries['apis_summary']['average_api_success_rate']}
- **Data Quality Distribution**: {', '.join(f"{quality}: {count}" for quality, count in summaries['apis_summary']['data_quality_distribution'].items())}

## Feature Categories

{chr(10).join(f"- **{category.replace('_', ' ').title()}**: {count} features" for category, count in summaries['features_summary']['feature_categories'].items())}

## Usage Notes

1. **Features Table**: Use for comprehensive analysis of city characteristics and pollution patterns
2. **APIs Table**: Reference for data source reliability and quality assessment
3. **AQI Standards Table**: Essential for proper AQI calculation and health advisory mapping

## Data Sources

- **Real Data**: WAQI API (demo token), OpenWeatherMap patterns
- **Synthetic Data**: City-specific pollution baselines with realistic variations
- **Enhanced Features**: NASA FIRMS-style fire data, comprehensive holiday calendars

## File Formats

All tables are saved as CSV files with headers for easy import into analysis tools. The summary file (`comprehensive_tables_summary.json`) contains metadata and statistics in JSON format.

---

**Dataset**: Global 100-City Worst Air Quality Dataset
**Features Enhanced**: Fire Activity + Holiday Impacts
**Total Records**: 76,000 data points across 100 cities
**Generated by**: Enhanced Features Processor v1.0
"""

        readme_file = self.output_dir / "README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)

        log.info(f"Generated README: {readme_file}")


def main():
    """Main execution for comprehensive tables generation."""
    log.info("Starting Comprehensive Tables Generation")

    try:
        generator = ComprehensiveTablesGenerator()
        results = generator.generate_all_tables()

        log.info("Comprehensive Tables Generation completed successfully")
        log.info(
            f"Generated {results['tables_generated']} tables for {results['total_cities']} cities"
        )
        log.info(f"Output directory: {results['output_directory']}")

        return results

    except Exception as e:
        log.error(f"Comprehensive tables generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
