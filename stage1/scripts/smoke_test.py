# scripts/smoke_test.py
# Purpose: Validate that environment variables are discoverable and the package imports.

import os
import sys

try:
    from dotenv import find_dotenv, load_dotenv
except Exception as e:
    print("ERROR: python-dotenv is not installed. Install deps first (run 'scripts\\run.ps1 bootstrap').")
    sys.exit(1)

# 1) Load .env from the repo working directory (or nearest parent)
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    print("WARNING: No .env file found near the working directory. "
          "Create one by copying config\\env\\.env.example to .env")
    # We'll still continue to show what's set in the process env.

# 2) Read required paths
DATA_ROOT = os.getenv("DATA_ROOT")
MODELS_ROOT = os.getenv("MODELS_ROOT")
CACHE_ROOT = os.getenv("CACHE_ROOT")

# 3) Validate presence
missing = [k for k, v in [("DATA_ROOT", DATA_ROOT), ("MODELS_ROOT", MODELS_ROOT), ("CACHE_ROOT", CACHE_ROOT)] if not v]
if missing:
    print("ERROR: Missing required env vars:", ", ".join(missing))
    print("Please edit your .env and set absolute Windows paths, e.g.:")
    print("  DATA_ROOT=C:\\aqf311\\data")
    print("  MODELS_ROOT=C:\\aqf311\\models")
    print("  CACHE_ROOT=C:\\aqf311\\.cache")
    sys.exit(2)

# 4) Echo resolved paths for human verification
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"MODELS_ROOT: {MODELS_ROOT}")
print(f"CACHE_ROOT: {CACHE_ROOT}")

# 5) (Optional) Light existence checks; non-fatal if missing
def _exists_or_warn(path, label):
    try:
        if not os.path.exists(path):
            print(f"WARNING: {label} does not exist on disk yet: {path}")
    except Exception as e:
        print(f"WARNING: Could not check {label} at {path}: {e}")

_exists_or_warn(DATA_ROOT, "DATA_ROOT")
_exists_or_warn(MODELS_ROOT, "MODELS_ROOT")
_exists_or_warn(CACHE_ROOT, "CACHE_ROOT")

# 6) Package import smoke (adjust if your package/module name differs)
try:
    import stage1_forecast  # noqa: F401
except Exception as e:
    print("ERROR: Cannot import 'stage1_forecast' package. Did you install the project or configure PYTHONPATH?")
    print("Tip: Use editable install with your venv:  C:\\aqf311\\.venv\\Scripts\\python.exe -m pip install -e .")
    raise

print("Smoke test OK.")
