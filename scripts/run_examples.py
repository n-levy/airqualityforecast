#!/usr/bin/env python3
"""
Example usage scripts for NOAA GEFS-Aerosols data collection.
Demonstrates various collection scenarios across different platforms.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display the result."""
    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("OK Command completed successfully")
    else:
        print(f"ERROR Command failed with exit code: {result.returncode}")

    return result.returncode == 0


def main():
    script_dir = Path(__file__).parent
    orchestrator = script_dir / "orchestrate_gefs_https.py"

    print("=== NOAA GEFS-Aerosols Collection Examples ===")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    examples = [
        {
            "name": "Environment Check",
            "cmd": [sys.executable, str(script_dir / "setup_environment.py")],
            "description": "Verify dependencies and setup",
        },
        {
            "name": "Quick Smoke Test",
            "cmd": [
                sys.executable,
                str(orchestrator),
                "--dry-run",
                "--start-date",
                "2024-01-12",
                "--end-date",
                "2024-01-12",
                "--cycles",
                "00",
                "--fhours",
                "24:24:24",
                "--bbox",
                "5,16,47,56",
            ],
            "description": "Test 1 day, 1 cycle, 1 forecast hour (dry run)",
        },
        {
            "name": "Small Region Test",
            "cmd": [
                sys.executable,
                str(orchestrator),
                "--dry-run",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-07",
                "--cycles",
                "00,12",
                "--fhours",
                "0:12:48",
                "--bbox",
                "10,15,50,55",  # Small European region
                "--pollutants",
                "PM25,PM10",
            ],
            "description": "Test 1 week, 2 cycles, 2 pollutants (dry run)",
        },
    ]

    for i, example in enumerate(examples, 1):
        success = run_command(example["cmd"], f"{i}. {example['name']}")
        if not success:
            print(f"\nExample {i} failed. Check your environment setup.")
            break

    print(f"\n=== Examples Complete ===")
    print("To run actual data collection (not dry run), remove --dry-run flag")
    print("For full 2-year collection, add --force flag to override safety limits")


if __name__ == "__main__":
    main()
