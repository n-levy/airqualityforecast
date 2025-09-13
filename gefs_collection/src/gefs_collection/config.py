#!/usr/bin/env python3
"""
Configuration management for GEFS Collection System.
Handles settings, paths, and deployment-specific configurations.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GefsConfig:
    """Configuration for GEFS data collection."""

    # Data source settings
    base_url: str = "https://noaa-gefs-pds.s3.amazonaws.com"
    cycles: List[str] = None
    fhours: str = "0:6:120"

    # Geographic settings
    bbox: List[float] = None  # [lon_min, lon_max, lat_min, lat_max]
    pollutants: List[str] = None

    # System settings
    data_root: str = ""
    workers: int = 4
    max_planned_files: int = 50000

    # HTTP settings
    timeout: int = 30
    retries: int = 3
    retry_delay: int = 2

    # Processing settings
    chunk_size: int = 8192
    temp_dir: str = ""

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.cycles is None:
            self.cycles = ["00", "12"]
        if self.bbox is None:
            self.bbox = [5.0, 16.0, 47.0, 56.0]  # Germany
        if self.pollutants is None:
            self.pollutants = ["PM25", "PM10", "NO2", "SO2", "CO", "O3"]
        if not self.data_root:
            self.data_root = str(Path.home() / "gefs_data")
        if not self.temp_dir:
            self.temp_dir = str(Path(self.data_root) / "temp")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GefsConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: Path) -> "GefsConfig":
        """Load configuration from file (YAML or JSON)."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_environment(cls) -> "GefsConfig":
        """Create config from environment variables."""
        config_dict = {}

        # Map environment variables to config fields
        env_mapping = {
            "GEFS_DATA_ROOT": "data_root",
            "GEFS_WORKERS": "workers",
            "GEFS_TIMEOUT": "timeout",
            "GEFS_RETRIES": "retries",
            "GEFS_MAX_FILES": "max_planned_files",
            "GEFS_BBOX": "bbox",
            "GEFS_CYCLES": "cycles",
            "GEFS_POLLUTANTS": "pollutants",
        }

        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                # Parse based on expected type
                if config_key in ["workers", "timeout", "retries", "max_planned_files"]:
                    config_dict[config_key] = int(value)
                elif config_key == "bbox":
                    config_dict[config_key] = [
                        float(x.strip()) for x in value.split(",")
                    ]
                elif config_key in ["cycles", "pollutants"]:
                    config_dict[config_key] = [x.strip() for x in value.split(",")]
                else:
                    config_dict[config_key] = value

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_file(self, config_path: Path, format: str = "yaml"):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(config_path, "w") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data directory paths."""
        data_root = Path(self.data_root)

        return {
            "data_root": data_root,
            "raw_root": data_root / "raw" / "gefs_chem",
            "curated_root": data_root / "curated" / "gefs_chem" / "parquet",
            "manifest_dir": data_root / "raw" / "gefs_chem" / "_manifests",
            "temp_dir": Path(self.temp_dir),
            "logs_dir": data_root / "logs",
        }

    def create_directories(self):
        """Create all required directories."""
        paths = self.get_data_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate bbox
        if len(self.bbox) != 4:
            issues.append(
                "bbox must have exactly 4 values: [lon_min, lon_max, lat_min, lat_max]"
            )
        elif not (-180 <= self.bbox[0] < self.bbox[1] <= 180):
            issues.append("Invalid longitude range in bbox")
        elif not (-90 <= self.bbox[2] < self.bbox[3] <= 90):
            issues.append("Invalid latitude range in bbox")

        # Validate cycles
        valid_cycles = {"00", "06", "12", "18"}
        for cycle in self.cycles:
            if cycle not in valid_cycles:
                issues.append(f"Invalid cycle: {cycle}. Must be one of: {valid_cycles}")

        # Validate pollutants
        valid_pollutants = {"PM25", "PM10", "NO2", "SO2", "CO", "O3"}
        for pollutant in self.pollutants:
            if pollutant not in valid_pollutants:
                issues.append(
                    f"Invalid pollutant: {pollutant}. Must be one of: {valid_pollutants}"
                )

        # Validate numeric ranges
        if self.workers < 1:
            issues.append("workers must be >= 1")
        if self.timeout < 1:
            issues.append("timeout must be >= 1")
        if self.retries < 0:
            issues.append("retries must be >= 0")

        return issues


def load_config(
    config_path: Optional[Path] = None, env_override: bool = True
) -> GefsConfig:
    """
    Load configuration with priority: file -> environment -> defaults.

    Args:
        config_path: Path to config file (optional)
        env_override: Whether to override with environment variables

    Returns:
        GefsConfig instance
    """
    # Start with defaults
    config = GefsConfig()

    # Load from file if provided
    if config_path and Path(config_path).exists():
        try:
            config = GefsConfig.from_file(config_path)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using defaults...")

    # Override with environment variables
    if env_override:
        try:
            env_config = GefsConfig.from_environment()
            # Merge environment config into loaded config
            for key, value in env_config.to_dict().items():
                if value:  # Only override non-empty values
                    setattr(config, key, value)
        except Exception as e:
            print(f"Warning: Failed to load environment config: {e}")

    return config
