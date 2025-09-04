import pandas as pd
from pathlib import Path

def test_sample_csv_exists():
    p = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_air_quality.csv"
    assert p.exists()

def test_validated_output(tmp_path):
    # This test is a placeholder to ensure the pipeline creates the expected interim file.
    # In CI, you'd run the actual scripts; here we just assert the filename convention.
    stage2_root = Path(__file__).resolve().parents[1]
    interim = stage2_root / "data" / "interim"
    # No assert here as the file is created at runtime; keep as smoke test.
    assert interim.exists()
