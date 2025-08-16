#!/usr/bin/env python
import sys, json, pathlib
import yaml  # pip install pyyaml
from jsonschema import validate  # pip install jsonschema

HERE = pathlib.Path(__file__).resolve()

def find_stage1_root(start: pathlib.Path) -> pathlib.Path | None:
    """
    Walk up to find either:
      - <root>/config/schemas
      - <root>/stage1/config/schemas
    Return the root that has a 'config/schemas' under it (stage1 root).
    """
    for p in [start, *start.parents]:
        # case A: already inside stage1 (preferred)
        if (p / "config" / "schemas").exists():
            return p
        # case B: repo root contains 'stage1'
        if (p / "stage1" / "config" / "schemas").exists():
            return p / "stage1"
    return None

ROOT = find_stage1_root(HERE)
if ROOT is None:
    print("ERROR: Could not locate stage1 root. "
          "Ensure you have a 'config/schemas' folder under stage1.")
    sys.exit(2)

schema_path = ROOT / "config" / "schemas" / "cities.json"
cities_dir  = ROOT / "config" / "cities"

if not schema_path.exists():
    print(f"ERROR: Schema file not found: {schema_path}")
    sys.exit(2)
if not cities_dir.exists():
    print(f"ERROR: Cities folder not found: {cities_dir}")
    sys.exit(2)

schema = json.loads(schema_path.read_text(encoding="utf-8"))

ok = True
for yml in sorted(cities_dir.glob("*.yml")):
    data = yaml.safe_load(yml.read_text(encoding="utf-8"))
    try:
        validate(instance=data, schema=schema)
        print(f"[OK] {yml.name}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {yml.name}: {e}")

sys.exit(0 if ok else 1)
