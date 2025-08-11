# ADR 003 – Job Scheduling

## Status
Accepted

## Context
We need to run ingestion, feature building, inference, verification, and publishing on a regular schedule.

## Decision
- Local Stage 1: user-space cron jobs or manual runs.
- Cloud: scheduled container jobs (e.g., Cloud Run Jobs, Fly.io Machines).

## Consequences
- Minimal setup locally.
- Easy migration to cloud-native scheduling.
- Decoupled job definitions from infrastructure.

---
**Last updated:** 2025-08-11


---
**Plain English Summary:**
This decision record explains what choice was made, why it was made, and its impact on the project.
It is intended to help both technical and non-technical contributors understand the reasoning.
