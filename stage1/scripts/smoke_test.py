from pathlib import Path
import os
from subprocess import check_call

os.environ.setdefault("OBS_FAKE", "1")
data_root = Path(os.environ.get("DATA_ROOT", Path.home() / "stage1_data"))
data_root.mkdir(parents=True, exist_ok=True)
check_call(["python", "apps/etl/obs_pm25.py", "--city", "berlin", "--since", "2025-07-01", "--until", "2025-07-01"])
print("Smoke OK, wrote to", data_root)
