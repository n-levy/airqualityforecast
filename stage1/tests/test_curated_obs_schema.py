import os, glob, json
from pathlib import Path
import pandas as pd
from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]  # .../stage1

def load_schema(kind: str) -> dict:
    p = ROOT / "config" / "schemas" / ("raw" if kind=="raw" else "curated") / "observations_pm25.json"
    return json.loads(Path(p).read_text(encoding="utf-8"))

def test_curated_parquet_has_required_columns():
    data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "stage1_data"))
    files = glob.glob(str(data_root / "curated" / "obs" / "berlin" / "pm25" / "date=*" / "*.parquet"))
    assert files, "No curated obs Parquet found for Berlin"
    df = pd.read_parquet(files[0])

    # JSON schema expects strings; create a JSON-like payload copying df but casting the datetime to ISO strings
    payload = df.copy()
    payload["valid_time"] = payload["valid_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    validate(instance=payload.to_dict(orient="records"), schema=load_schema("curated"))

    # In storage we keep tz-aware datetimes
    assert str(df["valid_time"].dtype) == "datetime64[ns, UTC]"
