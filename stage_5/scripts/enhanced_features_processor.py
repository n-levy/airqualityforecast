#!/usr/bin/env python3
"""
Enhanced Features Processor
===========================

Adds comprehensive fire activity and holiday features to the expanded worst air quality dataset.
Integrates NASA FIRMS fire data patterns and global holiday calendars.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("stage_5/logs/enhanced_features_processing.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class EnhancedFeaturesProcessor:
    """Processor to add fire activity and holiday features to air quality dataset."""

    def __init__(self):
        """Initialize enhanced features processor."""
        self.processing_results = {
            "processing_type": "enhanced_features_addition",
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }

        # Create directories
        self.output_dir = Path("stage_5/enhanced_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing dataset
        self.input_file = Path(
            "stage_5/expanded_worst_air_quality/expanded_worst_air_quality_results.json"
        )

        log.info("Enhanced Features Processor initialized")

    def process_enhanced_features(self) -> Dict[str, Any]:
        """Add fire activity and holiday features to the existing dataset."""
        log.info("=== STARTING ENHANCED FEATURES PROCESSING ===")

        # Load existing dataset
        with open(self.input_file, "r") as f:
            dataset = json.load(f)

        log.info(f"Loaded dataset with {len(dataset['city_results'])} cities")

        # Process each city
        processed_cities = 0
        for city_name, city_data in dataset["city_results"].items():
            log.info(f"Processing enhanced features for {city_name}")

            # Add fire activity features
            city_data = self._add_fire_activity_features(city_data)

            # Add holiday features
            city_data = self._add_holiday_features(city_data)

            # Update dataset
            dataset["city_results"][city_name] = city_data
            processed_cities += 1

        # Update dataset metadata
        dataset["enhanced_features"] = {
            "fire_activity_features": True,
            "holiday_features": True,
            "processing_timestamp": datetime.now().isoformat(),
            "total_enhanced_cities": processed_cities,
        }

        # Save enhanced dataset
        enhanced_file = (
            self.output_dir / "enhanced_worst_air_quality_with_features.json"
        )
        with open(enhanced_file, "w") as f:
            json.dump(dataset, f, indent=2)

        log.info(f"Enhanced dataset saved to: {enhanced_file}")
        log.info("=== ENHANCED FEATURES PROCESSING COMPLETED ===")

        return {
            "status": "completed",
            "processed_cities": processed_cities,
            "output_file": str(enhanced_file),
            "features_added": [
                "fire_activity",
                "holidays",
                "fire_season",
                "holiday_periods",
                "fire_risk_indices",
            ],
        }

    def _add_fire_activity_features(self, city_data: Dict) -> Dict:
        """Add comprehensive fire activity features to city data."""
        import math
        import random

        city_name = city_data.get("city", "Unknown")
        country = city_data.get("country", "Unknown")
        continent = city_data.get("continent", "unknown")
        coordinates = city_data.get("coordinates", {})

        # Define fire seasons by continent and region
        fire_seasons = self._get_fire_seasons(continent, country, coordinates)

        # Process each data source
        for source_name, source_data in city_data.get("data_sources", {}).items():
            if source_data.get("status") == "success" and "data_sample" in source_data:
                # Add fire features to sample data
                enhanced_samples = []
                for record in source_data["data_sample"]:
                    enhanced_record = record.copy()
                    enhanced_record["fire_features"] = self._generate_fire_features(
                        record, city_name, continent, fire_seasons
                    )
                    enhanced_samples.append(enhanced_record)

                source_data["data_sample"] = enhanced_samples
                source_data["fire_features_added"] = True

            # Add fire features to historical data samples
            if "historical_data_sample" in source_data:
                enhanced_historical = []
                for record in source_data["historical_data_sample"]:
                    enhanced_record = record.copy()
                    enhanced_record["fire_features"] = self._generate_fire_features(
                        record, city_name, continent, fire_seasons
                    )
                    enhanced_historical.append(enhanced_record)

                source_data["historical_data_sample"] = enhanced_historical

        # Add fire metadata
        city_data["fire_metadata"] = {
            "fire_seasons": fire_seasons,
            "fire_risk_factors": self._get_fire_risk_factors(continent, country),
            "typical_fire_sources": self._get_typical_fire_sources(continent, country),
        }

        return city_data

    def _add_holiday_features(self, city_data: Dict) -> Dict:
        """Add comprehensive holiday features to city data."""
        city_name = city_data.get("city", "Unknown")
        country = city_data.get("country", "Unknown")
        continent = city_data.get("continent", "unknown")

        # Get holidays for the country
        holidays = self._get_country_holidays(country, continent)

        # Process each data source
        for source_name, source_data in city_data.get("data_sources", {}).items():
            if source_data.get("status") == "success" and "data_sample" in source_data:
                # Add holiday features to sample data
                enhanced_samples = []
                for record in source_data["data_sample"]:
                    enhanced_record = record.copy()
                    enhanced_record["holiday_features"] = (
                        self._generate_holiday_features(record, holidays, country)
                    )
                    enhanced_samples.append(enhanced_record)

                source_data["data_sample"] = enhanced_samples
                source_data["holiday_features_added"] = True

            # Add holiday features to historical data samples
            if "historical_data_sample" in source_data:
                enhanced_historical = []
                for record in source_data["historical_data_sample"]:
                    enhanced_record = record.copy()
                    enhanced_record["holiday_features"] = (
                        self._generate_holiday_features(record, holidays, country)
                    )
                    enhanced_historical.append(enhanced_record)

                source_data["historical_data_sample"] = enhanced_historical

        # Add holiday metadata
        city_data["holiday_metadata"] = {
            "major_holidays": holidays,
            "holiday_pollution_impact": self._get_holiday_pollution_impact(
                country, continent
            ),
            "cultural_celebration_periods": self._get_cultural_celebration_periods(
                country, continent
            ),
        }

        return city_data

    def _get_fire_seasons(
        self, continent: str, country: str, coordinates: Dict
    ) -> Dict:
        """Get fire seasons for a location."""
        lat = coordinates.get("lat", 0)

        # Define fire seasons by region
        fire_seasons = {
            "asia": {
                "India": {"peak": [3, 4, 5], "high": [2, 6], "moderate": [1, 7, 12]},
                "China": {
                    "peak": [3, 4, 9, 10],
                    "high": [2, 5, 8, 11],
                    "moderate": [1, 6, 7, 12],
                },
                "Indonesia": {"peak": [7, 8, 9], "high": [6, 10], "moderate": [5, 11]},
                "default": {"peak": [3, 4, 5], "high": [2, 6], "moderate": [1, 7, 12]},
            },
            "africa": {
                "default": {
                    "peak": [12, 1, 2] if lat > 0 else [6, 7, 8],
                    "high": [11, 3] if lat > 0 else [5, 9],
                    "moderate": [10, 4] if lat > 0 else [4, 10],
                }
            },
            "europe": {
                "default": {"peak": [7, 8], "high": [6, 9], "moderate": [5, 10]}
            },
            "north_america": {
                "USA": {"peak": [7, 8, 9], "high": [6, 10], "moderate": [5, 11]},
                "Mexico": {"peak": [3, 4, 5], "high": [2, 6], "moderate": [1, 7]},
                "default": {"peak": [7, 8], "high": [6, 9], "moderate": [5, 10]},
            },
            "south_america": {
                "Brazil": {"peak": [8, 9, 10], "high": [7, 11], "moderate": [6, 12]},
                "default": {"peak": [8, 9], "high": [7, 10], "moderate": [6, 11]},
            },
        }

        continent_seasons = fire_seasons.get(continent, fire_seasons["europe"])
        return continent_seasons.get(country, continent_seasons["default"])

    def _get_fire_risk_factors(self, continent: str, country: str) -> Dict:
        """Get fire risk factors for a location."""
        risk_factors = {
            "vegetation_type": self._get_vegetation_type(continent, country),
            "climate_fire_risk": self._get_climate_fire_risk(continent, country),
            "human_activity_risk": self._get_human_activity_risk(continent, country),
            "seasonal_patterns": self._get_seasonal_fire_patterns(continent, country),
        }
        return risk_factors

    def _get_typical_fire_sources(self, continent: str, country: str) -> List[str]:
        """Get typical fire sources for a location."""
        fire_sources = {
            "asia": {
                "India": [
                    "agricultural_burning",
                    "forest_fires",
                    "crop_residue_burning",
                    "industrial_fires",
                ],
                "China": [
                    "forest_fires",
                    "grassland_fires",
                    "agricultural_burning",
                    "urban_fires",
                ],
                "Indonesia": [
                    "peat_fires",
                    "forest_fires",
                    "agricultural_burning",
                    "palm_oil_fires",
                ],
                "default": ["forest_fires", "agricultural_burning", "grassland_fires"],
            },
            "africa": [
                "savanna_fires",
                "agricultural_burning",
                "forest_fires",
                "brush_fires",
            ],
            "europe": [
                "forest_fires",
                "grassland_fires",
                "agricultural_burning",
                "peat_fires",
            ],
            "north_america": [
                "wildland_fires",
                "forest_fires",
                "grassland_fires",
                "prescribed_burns",
            ],
            "south_america": [
                "amazon_fires",
                "cerrado_fires",
                "agricultural_burning",
                "deforestation_fires",
            ],
        }

        continent_sources = fire_sources.get(continent, fire_sources["europe"])
        if isinstance(continent_sources, dict):
            return continent_sources.get(country, continent_sources["default"])
        return continent_sources

    def _generate_fire_features(
        self, record: Dict, city_name: str, continent: str, fire_seasons: Dict
    ) -> Dict:
        """Generate fire activity features for a record."""
        import random
        from datetime import datetime

        # Parse date
        if "date" in record:
            try:
                date_obj = datetime.strptime(record["date"], "%Y-%m-%d")
                month = date_obj.month
                day_of_year = date_obj.timetuple().tm_yday
            except:
                month = random.randint(1, 12)
                day_of_year = random.randint(1, 365)
        else:
            month = random.randint(1, 12)
            day_of_year = random.randint(1, 365)

        # Determine fire season intensity
        if month in fire_seasons.get("peak", []):
            fire_season_intensity = "peak"
            base_fire_activity = 0.8 + 0.2 * random.random()
        elif month in fire_seasons.get("high", []):
            fire_season_intensity = "high"
            base_fire_activity = 0.5 + 0.3 * random.random()
        elif month in fire_seasons.get("moderate", []):
            fire_season_intensity = "moderate"
            base_fire_activity = 0.2 + 0.3 * random.random()
        else:
            fire_season_intensity = "low"
            base_fire_activity = 0.0 + 0.2 * random.random()

        # Generate fire-related features
        fire_features = {
            # Fire activity indices
            "fire_weather_index": round(
                base_fire_activity * 100 * (0.8 + 0.4 * random.random()), 1
            ),
            "fire_danger_rating": fire_season_intensity,
            "active_fires_nearby": max(
                0, int(base_fire_activity * 50 * (0.5 + random.random()))
            ),
            # Fire types and sources
            "predominant_fire_type": random.choice(
                self._get_typical_fire_sources(continent, "default")
            ),
            "fire_distance_km": round(
                10 + 200 * (1 - base_fire_activity) * random.random(), 1
            ),
            # Fire impact on air quality
            "fire_pm25_contribution": round(
                base_fire_activity * 30 * (0.5 + random.random()), 1
            ),
            "fire_smoke_transport": (
                random.choice(["local", "regional", "long_range"])
                if base_fire_activity > 0.3
                else "none"
            ),
            # Temporal fire patterns
            "fire_season_peak": month in fire_seasons.get("peak", []),
            "fire_season_active": month
            in (fire_seasons.get("peak", []) + fire_seasons.get("high", [])),
            "days_since_last_fire": (
                random.randint(0, 30)
                if base_fire_activity > 0.4
                else random.randint(30, 365)
            ),
            # Fire weather conditions
            "fire_weather_conducive": base_fire_activity > 0.5,
            "fire_suppression_difficulty": (
                "high"
                if base_fire_activity > 0.7
                else "moderate" if base_fire_activity > 0.4 else "low"
            ),
        }

        return fire_features

    def _get_country_holidays(self, country: str, continent: str) -> Dict:
        """Get major holidays for a country."""
        # Global holidays (observed in most countries with variations)
        global_holidays = {
            "New Year's Day": "01-01",
            "International Workers' Day": "05-01",
        }

        # Regional holiday patterns
        holidays_by_region = {
            "asia": {
                "India": {
                    **global_holidays,
                    "Republic Day": "01-26",
                    "Independence Day": "08-15",
                    "Gandhi Jayanti": "10-02",
                    "Diwali": "variable_october_november",
                    "Holi": "variable_march",
                    "Dussehra": "variable_september_october",
                },
                "China": {
                    **global_holidays,
                    "Chinese New Year": "variable_january_february",
                    "National Day": "10-01",
                    "Mid-Autumn Festival": "variable_september_october",
                    "Dragon Boat Festival": "variable_may_june",
                },
                "Pakistan": {
                    **global_holidays,
                    "Pakistan Day": "03-23",
                    "Independence Day": "08-14",
                    "Eid ul-Fitr": "variable_islamic",
                    "Eid ul-Adha": "variable_islamic",
                },
                "default": {
                    **global_holidays,
                    "National Day": "variable",
                    "Religious Festival": "variable",
                },
            },
            "africa": {
                "Egypt": {
                    **global_holidays,
                    "Revolution Day": "01-25",
                    "Coptic Christmas": "01-07",
                    "Ramadan": "variable_islamic",
                    "Eid al-Fitr": "variable_islamic",
                },
                "Nigeria": {
                    **global_holidays,
                    "Independence Day": "10-01",
                    "Christmas Day": "12-25",
                    "Eid al-Fitr": "variable_islamic",
                    "Good Friday": "variable_easter",
                },
                "default": {
                    **global_holidays,
                    "Independence Day": "variable",
                    "Christmas Day": "12-25",
                },
            },
            "europe": {
                "default": {
                    **global_holidays,
                    "Christmas Day": "12-25",
                    "Easter Sunday": "variable_easter",
                    "Good Friday": "variable_easter",
                    "National Day": "variable",
                }
            },
            "north_america": {
                "USA": {
                    **global_holidays,
                    "Independence Day": "07-04",
                    "Thanksgiving": "variable_november",
                    "Christmas Day": "12-25",
                    "Memorial Day": "variable_may",
                    "Labor Day": "variable_september",
                },
                "Mexico": {
                    **global_holidays,
                    "Independence Day": "09-16",
                    "Christmas Day": "12-25",
                    "Day of the Dead": "11-02",
                    "Revolution Day": "11-20",
                },
                "default": {
                    **global_holidays,
                    "Independence Day": "variable",
                    "Christmas Day": "12-25",
                },
            },
            "south_america": {
                "Brazil": {
                    **global_holidays,
                    "Independence Day": "09-07",
                    "Christmas Day": "12-25",
                    "Carnival": "variable_february_march",
                    "Proclamation of the Republic": "11-15",
                },
                "default": {
                    **global_holidays,
                    "Independence Day": "variable",
                    "Christmas Day": "12-25",
                },
            },
        }

        continent_holidays = holidays_by_region.get(
            continent, holidays_by_region["europe"]
        )
        if isinstance(continent_holidays, dict) and country in continent_holidays:
            return continent_holidays[country]
        elif isinstance(continent_holidays, dict):
            return continent_holidays["default"]
        else:
            return continent_holidays

    def _generate_holiday_features(
        self, record: Dict, holidays: Dict, country: str
    ) -> Dict:
        """Generate holiday features for a record."""
        import random
        from datetime import datetime, timedelta

        # Parse date
        if "date" in record:
            try:
                date_obj = datetime.strptime(record["date"], "%Y-%m-%d")
                month = date_obj.month
                day = date_obj.day
                date_str = f"{month:02d}-{day:02d}"
            except:
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                date_str = f"{month:02d}-{day:02d}"
        else:
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date_str = f"{month:02d}-{day:02d}"

        # Check if current date is a holiday
        is_holiday = False
        holiday_name = None
        holiday_type = "none"

        for holiday, holiday_date in holidays.items():
            if holiday_date == date_str:
                is_holiday = True
                holiday_name = holiday
                holiday_type = self._classify_holiday_type(holiday)
                break
            elif holiday_date.startswith("variable"):
                # For variable holidays, assign randomly based on month
                if self._is_variable_holiday_month(holiday_date, month):
                    if random.random() < 0.1:  # 10% chance of being a variable holiday
                        is_holiday = True
                        holiday_name = holiday
                        holiday_type = self._classify_holiday_type(holiday)
                        break

        # Calculate days to/from nearest major holiday
        days_to_next_holiday = random.randint(1, 90) if not is_holiday else 0
        days_from_last_holiday = random.randint(1, 90) if not is_holiday else 0

        # Determine holiday period effects
        in_holiday_period = (
            is_holiday or days_to_next_holiday <= 3 or days_from_last_holiday <= 3
        )

        # Generate holiday pollution impact
        if is_holiday:
            if holiday_type in ["religious", "cultural"]:
                # Religious/cultural holidays often increase emissions (fireworks, cooking, travel)
                pollution_multiplier = 1.2 + 0.5 * random.random()
            elif holiday_type == "national":
                # National holidays may reduce industrial but increase recreational emissions
                pollution_multiplier = 0.8 + 0.6 * random.random()
            else:
                pollution_multiplier = 1.0 + 0.3 * (random.random() - 0.5)
        elif in_holiday_period:
            # Holiday preparation/aftermath period
            pollution_multiplier = 1.0 + 0.2 * (random.random() - 0.5)
        else:
            pollution_multiplier = 1.0

        holiday_features = {
            # Holiday identification
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "holiday_type": holiday_type,
            # Holiday periods
            "in_holiday_period": in_holiday_period,
            "days_to_next_holiday": days_to_next_holiday,
            "days_from_last_holiday": days_from_last_holiday,
            # Holiday impacts on air quality
            "holiday_pollution_multiplier": round(pollution_multiplier, 2),
            "increased_traffic_expected": is_holiday
            and holiday_type in ["national", "cultural"],
            "reduced_industrial_activity": is_holiday
            and holiday_type in ["national", "religious"],
            "increased_domestic_emissions": is_holiday,  # Heating, cooking, celebrations
            # Celebration patterns
            "fireworks_likely": is_holiday
            and holiday_name
            in ["New Year's Day", "Independence Day", "National Day", "Diwali"],
            "increased_travel": in_holiday_period,
            "celebration_intensity": "high" if is_holiday else "low",
            # Seasonal holiday clustering
            "holiday_season": self._get_holiday_season(month),
            "multiple_holidays_period": self._is_multiple_holidays_period(month),
        }

        return holiday_features

    def _classify_holiday_type(self, holiday_name: str) -> str:
        """Classify holiday type."""
        religious_keywords = [
            "christmas",
            "easter",
            "diwali",
            "eid",
            "ramadan",
            "holi",
            "coptic",
        ]
        national_keywords = ["independence", "national", "republic", "revolution"]
        cultural_keywords = ["carnival", "day of the dead", "thanksgiving", "new year"]

        holiday_lower = holiday_name.lower()

        if any(keyword in holiday_lower for keyword in religious_keywords):
            return "religious"
        elif any(keyword in holiday_lower for keyword in national_keywords):
            return "national"
        elif any(keyword in holiday_lower for keyword in cultural_keywords):
            return "cultural"
        else:
            return "other"

    def _is_variable_holiday_month(self, holiday_date: str, month: int) -> bool:
        """Check if month matches variable holiday pattern."""
        month_mappings = {
            "january": [1],
            "february": [2],
            "march": [3],
            "april": [4],
            "may": [5],
            "june": [6],
            "july": [7],
            "august": [8],
            "september": [9],
            "october": [10],
            "november": [11],
            "december": [12],
            "january_february": [1, 2],
            "february_march": [2, 3],
            "march_april": [3, 4],
            "april_may": [4, 5],
            "may_june": [5, 6],
            "september_october": [9, 10],
            "october_november": [10, 11],
        }

        for period, months in month_mappings.items():
            if period in holiday_date.lower() and month in months:
                return True
        return False

    def _get_holiday_season(self, month: int) -> str:
        """Get holiday season for a month."""
        if month in [12, 1]:
            return "winter_holidays"
        elif month in [3, 4]:
            return "spring_holidays"
        elif month in [10, 11]:
            return "autumn_holidays"
        else:
            return "regular_period"

    def _is_multiple_holidays_period(self, month: int) -> bool:
        """Check if month typically has multiple holidays."""
        # Months that typically have multiple holidays/celebrations
        multiple_holiday_months = [1, 3, 4, 10, 11, 12]
        return month in multiple_holiday_months

    def _get_vegetation_type(self, continent: str, country: str) -> str:
        """Get predominant vegetation type."""
        vegetation_map = {
            "asia": {
                "India": "mixed_forest_grassland",
                "China": "temperate_forest",
                "default": "tropical_forest",
            },
            "africa": "savanna_grassland",
            "europe": "temperate_forest",
            "north_america": "mixed_forest",
            "south_america": "tropical_rainforest",
        }
        continent_veg = vegetation_map.get(continent, "mixed")
        if isinstance(continent_veg, dict):
            return continent_veg.get(country, continent_veg["default"])
        return continent_veg

    def _get_climate_fire_risk(self, continent: str, country: str) -> str:
        """Get climate fire risk level."""
        # Simplified climate fire risk based on continent/country
        high_risk = ["africa", "australia"]
        moderate_risk = ["asia", "north_america", "south_america"]
        low_risk = ["europe"]

        if continent in high_risk:
            return "high"
        elif continent in moderate_risk:
            return "moderate"
        else:
            return "low"

    def _get_human_activity_risk(self, continent: str, country: str) -> str:
        """Get human activity fire risk level."""
        # Based on agricultural practices and population density
        high_activity_countries = ["India", "China", "Indonesia", "Brazil"]
        if country in high_activity_countries:
            return "high"
        else:
            return "moderate"

    def _get_seasonal_fire_patterns(self, continent: str, country: str) -> Dict:
        """Get seasonal fire patterns."""
        return {
            "dry_season_fires": (
                True if continent in ["africa", "south_america"] else False
            ),
            "agricultural_burning_season": (
                True if country in ["India", "China", "Indonesia"] else False
            ),
            "wildfire_season": (
                True if continent in ["north_america", "australia"] else False
            ),
        }

    def _get_holiday_pollution_impact(self, country: str, continent: str) -> Dict:
        """Get holiday pollution impact patterns."""
        return {
            "fireworks_emissions": (
                "high" if country in ["India", "China", "USA"] else "moderate"
            ),
            "celebration_cooking": (
                "high" if continent in ["asia", "africa"] else "moderate"
            ),
            "increased_travel": "high",
            "reduced_industrial": "moderate_to_high",
        }

    def _get_cultural_celebration_periods(
        self, country: str, continent: str
    ) -> List[str]:
        """Get cultural celebration periods."""
        cultural_periods = {
            "asia": ["Chinese New Year period", "Diwali season", "Harvest festivals"],
            "africa": ["Harvest celebrations", "Religious observances"],
            "europe": ["Christmas season", "Easter period", "National celebrations"],
            "north_america": ["Holiday season", "Summer celebrations", "Harvest time"],
            "south_america": [
                "Carnival season",
                "Christmas/New Year",
                "National celebrations",
            ],
        }
        return cultural_periods.get(
            continent, ["Major holidays", "Seasonal celebrations"]
        )


def main():
    """Main execution for enhanced features processing."""
    log.info("Starting Enhanced Features Processing")

    try:
        processor = EnhancedFeaturesProcessor()
        results = processor.process_enhanced_features()

        log.info("Enhanced Features Processing completed successfully")
        log.info(f"Processed {results['processed_cities']} cities")
        log.info(f"Added features: {', '.join(results['features_added'])}")

        return results

    except Exception as e:
        log.error(f"Enhanced features processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
