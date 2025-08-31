"""
Validate city configuration files against config/schemas/cities.json.

Usage (Windows PowerShell):
  C:\aqf311\.venv\Scripts\python.exe apps\tools\validate_cities.py
  C:\aqf311\.venv\Scripts\python.exe apps\tools\validate_cities.py --cities-dir config\cities --schema config\schemas\cities.json
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

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    cities_dir = Path(args.cities_dir)
    ok = 0; fail = 0
    for y in sorted(cities_dir.glob("*.yml")):
        try:
            data = yaml.safe_load(y.read_text(encoding="utf-8"))
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
