# ADR 001 – Data Storage Format and Access

## Status
Accepted

## Context
We need to store raw, curated, and feature data in a way that is efficient, cloud-migratable, and supports both local and cloud execution.

## Decision
We will use **Apache Parquet** files stored via **fsspec** paths.
- Local Stage 1: `file://` paths in user directory.
- Cloud: `s3://` paths with the same interface.

## Consequences
- Easy migration from local to cloud by changing environment variables.
- Columnar storage for efficient analytics.
- Supports partitioning by city, date, and data type.

---
**Last updated:** YYYY-MM-DD


---
**Plain English Summary:**
This decision record explains what choice was made, why it was made, and its impact on the project.
It is intended to help both technical and non-technical contributors understand the reasoning.
