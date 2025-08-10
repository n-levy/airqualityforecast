# ADR 003 – Job Scheduling

## Status
Accepted

## Context
We need to run ingestion, feature building, inference, verification, and publishing on a regular schedule.

## Decision
- Local MVP: user-space cron jobs or manual runs.
- Cloud: scheduled container jobs (e.g., Cloud Run Jobs, Fly.io Machines).

## Consequences
- Minimal setup locally.
- Easy migration to cloud-native scheduling.
- Decoupled job definitions from infrastructure.

---
**Last updated:** YYYY-MM-DD
