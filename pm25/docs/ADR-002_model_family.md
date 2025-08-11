# ADR 002 – Model Family

## Status
Accepted

## Context
We need a model that can handle tabular, temporal features, work with limited historical data, and run quickly for multiple lead times.

## Decision
We will use **gradient-boosted trees** (XGBoost) as a bias-correcting downscaler over public forecasts and meteorological inputs.

## Consequences
- Strong performance for tabular features.
- Interpretability through feature importance.
- Fast retraining and inference.
- Easy to implement in Python with open-source libraries.

---
**Last updated:** YYYY-MM-DD


---
**Plain English Summary:**
This decision record explains what choice was made, why it was made, and its impact on the project.
It is intended to help both technical and non-technical contributors understand the reasoning.
