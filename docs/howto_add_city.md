# How to Add a City (Stage-1)

1) Create a city file:
   - Path: `stage1/config/cities/<cityname>.yml`
   - Example keys:
     ```
     name: Berlin
     country: DE
     lat: 52.5200
     lon: 13.4050
     tz: Europe/Berlin
     population: 3769000
     ```
2) Validate the schema:
   - Command (example): `python stage1/apps/tools/validate_cities.py`
   - Expect: "All city files valid".

3) Confirm it is listed:
   - Update `docs/cities.md` if this is part of the Stage-1 canonical set.

4) Smoke test downstream:
   - Ensure any ETL referencing cities resolves paths and produces at least a small output table.
