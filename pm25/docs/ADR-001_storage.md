# ADR 001 – Data Storage Format and Access

## Status
Accepted

## Context
We need to store raw, curated, and feature data in a way that is efficient, cloud-migratable, and supports both local and cloud execution.

## Decision
We will use **Apache Parquet** files stored via **fsspec** paths.
- Local MVP: `file://` paths in user directory.
- Cloud: `s3://` paths with the same interface.

## Consequences
- Easy migration from local to cloud by changing environment variables.
- Columnar storage for efficient analytics.
- Supports partitioning by city, date, and data type.

---
**Last updated:** YYYY-MM-DD
