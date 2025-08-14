import os, sys
from stage1_forecast import env  # loads .env via python-dotenv
def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)
data_root = os.getenv("DATA_ROOT")
models_root = os.getenv("MODELS_ROOT")
print(f"DATA_ROOT: {data_root}")
print(f"MODELS_ROOT: {models_root}")

def bad(v):
    return (v is None) or (v.strip().lower() in {"", "none", "null"})

missing = [k for k, v in [("DATA_ROOT", data_root), ("MODELS_ROOT", models_root)] if bad(v)]
if missing:
    fail("Missing or empty: " + ", ".join(missing))

print("Smoke test OK.")
