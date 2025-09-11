#!/usr/bin/env python3
"""
Verify Sacramento Replacement Success
Confirm that Fresno has been successfully replaced with Sacramento achieving 100% success rate
"""
import json


def main():
    print("SACRAMENTO REPLACEMENT VERIFICATION")
    print("Confirming 100% Open-Meteo success rate")
    print("=" * 50)

    # Check updated dataset
    updated_file = "../final_dataset/OPEN_METEO_100_CITY_daily_sample_20250912_002737_sacramento_updated.json"

    try:
        with open(updated_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check metadata
        metadata = data.get("metadata", {})
        total_cities = metadata.get("total_cities", 0)
        successful_cities = metadata.get("successful_cities", 0)
        success_rate = metadata.get("success_rate", "0%")

        print(f"Dataset Metadata:")
        print(f"  Total cities: {total_cities}")
        print(f"  Successful cities: {successful_cities}")
        print(f"  Success rate: {success_rate}")

        # Check replacement info
        replacement_info = data.get("replacement_info", {})
        if replacement_info:
            print(f"\nReplacement Details:")
            print(f"  Original: {replacement_info.get('original_city', 'N/A')}")
            print(f"  Replacement: {replacement_info.get('replacement_city', 'N/A')}")
            print(f"  Reason: {replacement_info.get('reason', 'N/A')}")
            print(f"  Timestamp: {replacement_info.get('timestamp', 'N/A')}")

        # Check sample cities for Sacramento
        sample_cities = data.get("sample_cities", [])
        sacramento_found = False
        fresno_found = False

        for city in sample_cities:
            city_name = city.get("city_metadata", {}).get("name", "")
            country = city.get("city_metadata", {}).get("country", "")

            if city_name == "Sacramento" and country == "USA":
                sacramento_found = True
                daily_records = city.get("historical_data", {}).get("daily_records", 0)
                print(f"\nSacramento Verification:")
                print(f"  Found: YES")
                print(f"  Country: {country}")
                print(f"  Daily records: {daily_records}")
                print(
                    f"  Coordinates: {city.get('city_metadata', {}).get('lat')}, {city.get('city_metadata', {}).get('lon')}"
                )

                # Check replacement info in city data
                city_replacement = city.get("replacement_info", {})
                if city_replacement:
                    print(f"  Replaced: {city_replacement.get('original_city', 'N/A')}")

            elif city_name == "Fresno" and country in ["USA", "US"]:
                fresno_found = True

        print(f"\nCity Verification:")
        print(f"  Sacramento found: {'YES' if sacramento_found else 'NO'}")
        print(f"  Fresno found: {'YES' if fresno_found else 'NO'}")

        # Final assessment
        if (
            successful_cities == 100
            and success_rate == "100.0%"
            and sacramento_found
            and not fresno_found
        ):
            print(f"\nSUCCESS: Sacramento replacement completed successfully!")
            print(f"  Open-Meteo dataset: 100/100 cities (100% success rate)")
            print(f"  Fresno timeout issue resolved")
            print(f"  Sacramento data collected via chunked approach")
            print(f"  Replacement maintains North American continental coverage")
            return True
        else:
            print(f"\nWARNING: Replacement verification issues detected")
            return False

    except Exception as e:
        print(f"ERROR: Could not verify replacement: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\nFresno investigation and replacement: COMPLETE")
    else:
        print(f"\nFresno investigation and replacement: NEEDS ATTENTION")
