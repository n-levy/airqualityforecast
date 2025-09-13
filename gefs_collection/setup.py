#!/usr/bin/env python3
"""
Setup script for NOAA GEFS-Aerosols Data Collection System.
Makes the project easy to install and distribute to servers.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "NOAA GEFS-Aerosols Data Collection System"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "xarray>=2022.6.0",
        "cfgrib>=0.9.10",
        "eccodes>=1.4.0",
        "pyarrow>=10.0.0",
    ]

setup(
    name="gefs-collection",
    version="1.0.0",
    description="NOAA GEFS-Aerosols Data Collection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GEFS Collection Team",
    author_email="gefs-collection@example.com",
    url="https://github.com/your-org/gefs-collection",
    # Package structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Include data files
    package_data={
        "gefs_collection": ["config/*.yaml", "config/*.json"],
    },
    # Dependencies
    python_requires=">=3.8",
    install_requires=requirements,
    # Optional dependencies
    extras_require={
        "dev": ["pytest>=7.0.0", "flake8>=5.0.0", "black>=22.0.0"],
        "notebook": ["jupyter>=1.0.0", "matplotlib>=3.5.0", "seaborn>=0.11.0"],
    },
    # Console scripts
    entry_points={
        "console_scripts": [
            "gefs-collect=gefs_collection.cli:main",
            "gefs-setup=gefs_collection.setup:main",
            "gefs-examples=gefs_collection.examples:main",
        ],
    },
    # Metadata
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="atmospheric-science weather-data grib parquet noaa",
    project_urls={
        "Bug Reports": "https://github.com/your-org/gefs-collection/issues",
        "Source": "https://github.com/your-org/gefs-collection",
        "Documentation": "https://github.com/your-org/gefs-collection/blob/main/README.md",
    },
)
