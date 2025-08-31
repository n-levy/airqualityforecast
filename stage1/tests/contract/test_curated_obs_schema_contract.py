import os
import json
from pathlib import Path
import pandas as pd
from jsonschema import validate

def test_curated_schema_and_dtype():
    # 1) Locate data root (tests read OS env, not .env)
    data_root = os.getenv("DATA_ROOT") or str(Path.home() / "stage1_data")

    base = Path(data_root) / "curated" / "obs" / "berlin" / "pm25"
    files = sorted(base.rglob("data.parquet"))
    assert files, f"No curated parquet found under: {base}. Generate with OBS_FAKE=1 and run obs_pm25.py."

    # 2) Load one file
    df = pd.read_parquet(files[0])
    assert {"city","valid_time","value","unit"}.issubset(df.columns), \
        f"Missing columns in curated file; got: {list(df.columns)}"

    # 3) valid_time must be tz-aware UTC
    dtype = df["valid_time"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype), "valid_time must be timezone-aware"
    assert "UTC" in str(dtype), f"valid_time tz must be UTC, got: {dtype}"

    # 4) JSON-schema validation (records/list-of-objects)
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schemas" / "curated" / "observations_pm25.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Convert timestamps to RFC3339 strings for jsonschema
    recs = (
        df[["city","valid_time","value","unit"]]
        .assign(valid_time=lambda s: s["valid_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        .to_dict(orient="records")
    )
    validate(instance=recs, schema=schema)
