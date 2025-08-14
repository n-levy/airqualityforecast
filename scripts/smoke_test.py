import os, sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Please run: pip install python-dotenv", file=sys.stderr)
    sys.exit(2)

# Load .env next to repo root (current working directory when you run the script)
load_dotenv(dotenv_path=Path(".env"))

REQUIRED = ["DATA_ROOT", "MODELS_ROOT"]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    print(f"Missing env vars: {missing}")
    sys.exit(1)

data_root = Path(os.getenv("DATA_ROOT"))
models_root = Path(os.getenv("MODELS_ROOT"))

errors = []
for p in [data_root, models_root]:
    if not p.exists():
        errors.append(f"Path does not exist: {p}")

if errors:
    for e in errors:
        print(e)
    sys.exit(1)

print("Smoke test OK.")
