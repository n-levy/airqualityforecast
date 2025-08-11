ADR: Evaluation Metrics for AQI Forecasting

Status: Accepted
Context: AQI is an ordinal categorical variable with public health implications. Metrics must capture both exact matches and near misses, as well as pollutant-level contributions to overall AQI accuracy.
Decision:
Example metrics are listed in `EVAL_METRICS.md`. These are **examples only**; the final set will be decided during Stage 1 evaluation.
Consequences:
1. Balanced view of forecast skill at both AQI and pollutant levels.
2. Ability to communicate results effectively to both technical and public audiences.
3. Supports targeted model improvements by pollutant.

### Additional Details on Metric Selection Criteria
The choice of final evaluation metrics will balance several priorities:
1. **Public Health Relevance** – Metrics should capture the model's ability to predict hazardous pollution levels accurately.
2. **Robustness Across Conditions** – The chosen metrics must produce stable and meaningful results across seasons and varying meteorological conditions.
3. **Transparency & Reproducibility** – Calculations will be open, reproducible, and well-documented to allow external verification.
4. **Operational Actionability** – Metrics should guide both model retraining schedules and real-time forecast adjustments.
5. **Trade-off Awareness** – In cases where higher exact match accuracy conflicts with broader tolerance for near misses, these trade-offs will be explicitly evaluated and documented.