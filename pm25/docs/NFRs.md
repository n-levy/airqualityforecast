# Non-Functional Requirements (NFRs) – PM2.5 Forecasting Project

## Performance & Accuracy
- Predictions ready within 3 minutes of data availability.
- Verification results updated within 90 minutes of observation availability.
- Day-ahead MAE reduced by ≥15% vs. benchmark.

## Reliability
- ≥99% pipeline success rate per month.
- Automated retries on transient failures.

## Scalability
- Architecture supports adding cities and pollutants without redesign.
- Able to scale to cloud deployment with minimal config changes.

## Transparency
- Publish raw and processed data outputs for verification.
- Document methodology and evaluation metrics.

## Security
- No personal data collected.
- Store secrets in environment variables.
- Use least-privilege access for cloud resources.

## Cost
- Local MVP: zero cloud costs.
- Cloud target: <€10/month at MVP scale.

---
**Last updated:** YYYY-MM-DD
