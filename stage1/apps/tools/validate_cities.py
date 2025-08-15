"""
Validate city configuration files against config/schemas/cities.json.

Usage:
  C:\aqf311\.venv\Scripts\python.exe apps\tools\validate_cities.py
  C:\aqf311\.venv\Scripts\python.exe apps\tools\validate_cities.py --cities-dir config/cities --schema config/schemas/cities.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator

def main() -> int:
    ap = argparse.ArgumentParser(description="Validate city configs.")
    ap.add_argument("--cities-dir", default="config/cities")
    ap.add_argument("--schema", default="config/schemas/cities.json")
    args = ap.parse_args()

    cities_dir = Path(args.cities_dir)
    schema_path = Path(args.schema)

    if not cities_dir.is_dir():
        print(f"ERROR: cities dir not found: {cities_dir}", file=sys.stderr); return 2
    if not schema_path.is_file():
        print(f"ERROR: schema not found: {schema_path}", file=sys.stderr); return 2

    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    validator = Draft202012Validator(schema)

    ymls = sorted(cities_dir.glob("*.yml"))
    if not ymls:
        print(f"WARNING: no .yml files found in {cities_dir}"); return 0

    ok = fail = 0
    for y in ymls:
        try:
            data = yaml.safe_load(y.read_text(encoding="utf-8-sig"))
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                print(f"FAIL: {y.name}")
                for e in errors:
                    loc = ".".join([str(p) for p in e.path]) or "<root>"
                    print(f"  - {loc}: {e.message}")
                fail += 1
            else:
                print(f"OK:   {y.name}"); ok += 1
        except Exception as ex:
            print(f"ERROR: {y.name}: {ex}", file=sys.stderr); fail += 1
    print(f"Summary: {ok} OK, {fail} fail")
    return 1 if fail else 0

if __name__ == "__main__":
    raise SystemExit(main())

