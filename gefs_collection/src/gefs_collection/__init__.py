"""
NOAA GEFS-Aerosols Data Collection System

Universal Python system for downloading and processing NOAA GEFS-Aerosols
pollutant data via HTTPS. Designed for easy deployment to any server or laptop.
"""

__version__ = "1.0.0"
__author__ = "GEFS Collection Team"
__email__ = "gefs-collection@example.com"

from .config import GefsConfig
from .downloader import GefsDownloader
from .extractor import PollutantExtractor
from .orchestrator import GefsOrchestrator

__all__ = [
    "GefsOrchestrator",
    "GefsDownloader",
    "PollutantExtractor",
    "GefsConfig",
]
