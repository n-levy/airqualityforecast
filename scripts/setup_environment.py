#!/usr/bin/env python3
"""
Environment setup script for NOAA GEFS-Aerosols data collection.
Cross-platform Python environment validator and setup helper.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required, found {version.major}.{version.minor}")
        return False

    print(f"OK Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name):
    """Check if a package is available."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_requirements():
    """Install requirements using pip."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False

    print(f"Installing requirements from {requirements_file}")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("OK Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install requirements - {e}")
        return False


def check_data_directory():
    """Check and create data directory."""
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        # Use platform-neutral default location
        data_root = Path.home() / "gefs_data"

    data_root = Path(data_root)

    try:
        # Create directory structure
        raw_root = data_root / "raw" / "gefs_chem"
        curated_root = data_root / "curated" / "gefs_chem" / "parquet"

        raw_root.mkdir(parents=True, exist_ok=True)
        curated_root.mkdir(parents=True, exist_ok=True)

        print(f"OK Data directory: {data_root}")
        return True

    except Exception as e:
        print(f"ERROR: Cannot create data directory - {e}")
        return False


def main():
    print("=== NOAA GEFS-Aerosols Environment Setup ===")
    print(f"Platform: {sys.platform}")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check required packages
    required_packages = ["requests", "pandas", "xarray", "cfgrib", "pyarrow"]
    missing_packages = [pkg for pkg in required_packages if not check_package(pkg)]

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")

        install = input("Install missing packages? (y/N): ").lower().strip()
        if install in ["y", "yes"]:
            if not install_requirements():
                sys.exit(1)
        else:
            print("Manual installation required:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
    else:
        print("OK All required packages available")

    # Check data directory
    if not check_data_directory():
        sys.exit(1)

    print("\n=== Environment Ready ===")
    print("You can now run:")
    print("python scripts/orchestrate_gefs_https.py --help")

    # Show example commands
    print("\nExample commands:")
    print("# Smoke test (1 day)")
    print(
        "python scripts/orchestrate_gefs_https.py --start-date 2024-01-12 --end-date 2024-01-12 --cycles 00 --fhours 24:24:24"
    )
    print("\n# Full collection (2 years)")
    print(
        "python scripts/orchestrate_gefs_https.py --start-date 2022-01-01 --end-date 2024-01-01 --force"
    )


if __name__ == "__main__":
    main()
