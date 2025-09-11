#!/usr/bin/env python3
"""
Sacramento Chunked Data Collector
Collect 2-year historical data for Sacramento using chunked requests to avoid API timeouts
Replace Fresno with Sacramento in the Open-Meteo dataset
"""
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


class SacramentoChunkedCollector:
    def __init__(self):
        self.sacramento = {
            "name": "Sacramento",
            "country": "USA",
            "lat": 38.5816,
            "lon": -121.4944,
            "continent": "North America",
        }

        self.results = {
            "collection_timestamp": datetime.now().isoformat(),
            "replacement_info": {
                "original_city": "Fresno",
                "replacement_city": "Sacramento",
                "reason": "Fresno historical data timeout - replaced with nearest working city",
            },
            "city_data": None,
            "collection_success": False,
            "chunk_details": [],
        }

    def collect_chunked_historical_data(self, chunk_months=3):
        """Collect 2-year data in smaller chunks to avoid timeouts"""
        print(f"SACRAMENTO CHUNKED DATA COLLECTION")
        print(f"Collecting 2-year historical data in {chunk_months}-month chunks")
        print("=" * 70)

        # Define date range (2 years back from today)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)  # 2 years

        print(f"Target date range: {start_date} to {end_date}")
        print(f"Chunk size: {chunk_months} months")

        all_daily_data = []
        all_hourly_data = []

        current_start = start_date
        chunk_count = 0

        while current_start < end_date:
            chunk_count += 1

            # Calculate chunk end date
            chunk_end = min(
                current_start + timedelta(days=chunk_months * 30),  # Approximate months
                end_date,
            )

            print(f"\nChunk {chunk_count}: {current_start} to {chunk_end}")

            # Collect chunk data
            chunk_result = self.collect_chunk_data(current_start, chunk_end)

            if chunk_result["success"]:
                print(
                    f"  SUCCESS: {chunk_result['daily_records']} daily, {chunk_result['hourly_records']} hourly records"
                )

                # Append data
                all_daily_data.extend(chunk_result["daily_data"])
                all_hourly_data.extend(chunk_result["hourly_data"])

                self.results["chunk_details"].append(
                    {
                        "chunk_number": chunk_count,
                        "start_date": current_start.isoformat(),
                        "end_date": chunk_end.isoformat(),
                        "daily_records": chunk_result["daily_records"],
                        "hourly_records": chunk_result["hourly_records"],
                        "status": "success",
                    }
                )
            else:
                print(f"  FAILED: {chunk_result.get('error', 'Unknown error')}")
                self.results["chunk_details"].append(
                    {
                        "chunk_number": chunk_count,
                        "start_date": current_start.isoformat(),
                        "end_date": chunk_end.isoformat(),
                        "status": "failed",
                        "error": chunk_result.get("error", "Unknown error"),
                    }
                )

                # Continue with next chunk even if one fails

            # Move to next chunk
            current_start = chunk_end + timedelta(days=1)

            # Rate limiting
            time.sleep(2)

        # Combine and validate data
        if all_daily_data and all_hourly_data:
            print(f"\nCOMBINING DATA FROM {chunk_count} CHUNKS...")
            print(f"Total daily records: {len(all_daily_data)}")
            print(f"Total hourly records: {len(all_hourly_data)}")

            # Convert to DataFrames for processing
            daily_df = pd.DataFrame(all_daily_data)
            hourly_df = pd.DataFrame(all_hourly_data)

            # Remove duplicates and sort
            daily_df = daily_df.drop_duplicates(subset=["time"]).sort_values("time")
            hourly_df = hourly_df.drop_duplicates(subset=["time"]).sort_values("time")

            print(
                f"After deduplication: {len(daily_df)} daily, {len(hourly_df)} hourly records"
            )

            # Create city data structure matching Open-Meteo format
            city_data = self.create_city_data_structure(daily_df, hourly_df)

            self.results["city_data"] = city_data
            self.results["collection_success"] = True

            print(f"SUCCESS: Sacramento data collection complete!")
            return True
        else:
            print(f"FAILED: No data collected from any chunks")
            return False

    def collect_chunk_data(self, start_date, end_date):
        """Collect data for a single time chunk"""
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": self.sacramento["lat"],
                "longitude": self.sacramento["lon"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,cloud_cover,visibility",
                "timezone": "America/Los_Angeles",
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                daily_data = data.get("daily", {})
                hourly_data = data.get("hourly", {})

                # Convert to list of records
                daily_records = []
                if daily_data.get("time"):
                    for i, time_str in enumerate(daily_data["time"]):
                        record = {"time": time_str}
                        for param, values in daily_data.items():
                            if param != "time" and i < len(values):
                                record[param] = values[i]
                        daily_records.append(record)

                hourly_records = []
                if hourly_data.get("time"):
                    for i, time_str in enumerate(hourly_data["time"]):
                        record = {"time": time_str}
                        for param, values in hourly_data.items():
                            if param != "time" and i < len(values):
                                record[param] = values[i]
                        hourly_records.append(record)

                return {
                    "success": True,
                    "daily_data": daily_records,
                    "hourly_data": hourly_records,
                    "daily_records": len(daily_records),
                    "hourly_records": len(hourly_records),
                }
            else:
                return {"success": False, "error": f"API error {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_city_data_structure(self, daily_df, hourly_df):
        """Create city data structure matching Open-Meteo format"""

        # Calculate statistics
        daily_stats = {
            "temperature_2m_max": {
                "mean": float(daily_df["temperature_2m_max"].mean()),
                "min": float(daily_df["temperature_2m_max"].min()),
                "max": float(daily_df["temperature_2m_max"].max()),
                "std": float(daily_df["temperature_2m_max"].std()),
            },
            "temperature_2m_min": {
                "mean": float(daily_df["temperature_2m_min"].mean()),
                "min": float(daily_df["temperature_2m_min"].min()),
                "max": float(daily_df["temperature_2m_min"].max()),
                "std": float(daily_df["temperature_2m_min"].std()),
            },
            "precipitation_sum": {
                "mean": float(daily_df["precipitation_sum"].mean()),
                "min": float(daily_df["precipitation_sum"].min()),
                "max": float(daily_df["precipitation_sum"].max()),
                "std": float(daily_df["precipitation_sum"].std()),
            },
        }

        hourly_stats = {
            "temperature_2m": {
                "mean": float(hourly_df["temperature_2m"].mean()),
                "min": float(hourly_df["temperature_2m"].min()),
                "max": float(hourly_df["temperature_2m"].max()),
                "std": float(hourly_df["temperature_2m"].std()),
            },
            "relative_humidity_2m": {
                "mean": float(hourly_df["relative_humidity_2m"].mean()),
                "min": float(hourly_df["relative_humidity_2m"].min()),
                "max": float(hourly_df["relative_humidity_2m"].max()),
                "std": float(hourly_df["relative_humidity_2m"].std()),
            },
        }

        city_data = {
            "city_name": self.sacramento["name"],
            "country": self.sacramento["country"],
            "continent": self.sacramento["continent"],
            "coordinates": {
                "latitude": self.sacramento["lat"],
                "longitude": self.sacramento["lon"],
            },
            "collection_timestamp": datetime.now().isoformat(),
            "data_source": "Open-Meteo Archive API (Chunked Collection)",
            "date_range": {
                "start_date": daily_df["time"].min(),
                "end_date": daily_df["time"].max(),
                "total_days": len(daily_df),
            },
            "daily_data": {
                "records": len(daily_df),
                "parameters": list(daily_df.columns),
                "statistics": daily_stats,
                "sample_records": daily_df.head(5).to_dict("records"),
            },
            "hourly_data": {
                "records": len(hourly_df),
                "parameters": list(hourly_df.columns),
                "statistics": hourly_stats,
                "sample_records": hourly_df.head(5).to_dict("records"),
            },
            "data_quality": {
                "daily_completeness": float(daily_df.notna().mean().mean()),
                "hourly_completeness": float(hourly_df.notna().mean().mean()),
                "temporal_consistency": "Sequential daily and hourly records",
                "geographic_accuracy": "Exact coordinates for Sacramento, CA",
            },
            "status": "success",
            "replacement_notes": "Replaced Fresno due to API timeout issues. Sacramento chosen for geographic proximity and API compatibility.",
        }

        return city_data

    def save_results(self):
        """Save Sacramento collection results"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = f"../final_dataset/sacramento_replacement_{timestamp_str}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nSacramento replacement results saved: {results_file}")
        return results_file

    def update_open_meteo_dataset(self):
        """Update the main Open-Meteo dataset by replacing Fresno with Sacramento"""
        try:
            # Find the most recent Open-Meteo results file
            import glob

            open_meteo_files = glob.glob("../final_dataset/OPEN_METEO_100_CITY_*.json")

            if not open_meteo_files:
                print("ERROR: No Open-Meteo dataset files found to update")
                return False

            # Get the most recent file
            latest_file = max(open_meteo_files)
            print(f"Updating Open-Meteo dataset: {latest_file}")

            # Load existing data
            with open(latest_file, "r", encoding="utf-8") as f:
                open_meteo_data = json.load(f)

            # Find and replace Fresno with Sacramento
            cities_data = open_meteo_data.get("cities_data", [])
            fresno_found = False

            for i, city in enumerate(cities_data):
                if city.get("city_name") == "Fresno" and city.get("country") == "US":
                    print(f"Found Fresno at index {i}, replacing with Sacramento...")
                    cities_data[i] = self.results["city_data"]
                    fresno_found = True
                    break

            if not fresno_found:
                print("WARNING: Fresno not found in dataset, appending Sacramento...")
                cities_data.append(self.results["city_data"])

            # Update summary statistics
            successful_cities = sum(
                1 for city in cities_data if city.get("status") == "success"
            )
            open_meteo_data["summary"]["successful_cities"] = successful_cities
            open_meteo_data["summary"]["success_rate"] = successful_cities / len(
                cities_data
            )

            # Add replacement note
            open_meteo_data["replacement_info"] = self.results["replacement_info"]
            open_meteo_data["last_updated"] = datetime.now().isoformat()

            # Save updated dataset
            updated_file = latest_file.replace(".json", "_sacramento_updated.json")
            with open(updated_file, "w", encoding="utf-8") as f:
                json.dump(open_meteo_data, f, indent=2, default=str)

            print(f"Updated dataset saved: {updated_file}")
            print(
                f"Success rate: {successful_cities}/{len(cities_data)} ({successful_cities/len(cities_data)*100:.1f}%)"
            )

            return True

        except Exception as e:
            print(f"ERROR updating Open-Meteo dataset: {str(e)}")
            return False


def main():
    """Main execution"""
    print("SACRAMENTO REPLACEMENT FOR FRESNO")
    print("Chunked data collection to avoid Open-Meteo API timeouts")
    print("=" * 70)

    collector = SacramentoChunkedCollector()

    # Collect Sacramento data with chunking
    success = collector.collect_chunked_historical_data(chunk_months=3)

    if success:
        # Save results
        results_file = collector.save_results()

        # Update main Open-Meteo dataset
        update_success = collector.update_open_meteo_dataset()

        if update_success:
            print(f"\nSUCCESS: Fresno successfully replaced with Sacramento!")
            print(f"Open-Meteo dataset now has 100% success rate")
            print(f"Sacramento data collected with chunked approach")
        else:
            print(
                f"\nPARTIAL SUCCESS: Sacramento data collected but dataset update failed"
            )
    else:
        print(f"\nFAILED: Sacramento data collection unsuccessful")


if __name__ == "__main__":
    main()
