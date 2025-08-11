ADR: Evaluation Metrics for AQI Forecasting

Status: Accepted
Context: AQI is an ordinal categorical variable with public health implications. Metrics must capture both exact matches and near misses, as well as pollutant-level contributions to overall AQI accuracy.
Decision:
Example metrics are listed in `EVAL_METRICS.md`; final set TBD. These are examples only.
Example metrics are listed in `EVAL_METRICS.md`. These are **examples only**; the final set will be decided during Stage 1 evaluation.
Consequences:
1. Balanced view of forecast skill at both AQI and pollutant levels.
2. Ability to communicate results effectively to both technical and public audiences.
3. Supports targeted model improvements by pollutant.

---
**Plain English Summary:**
This decision record explains what choice was made, why it was made, and its impact on the project.
It is intended to help both technical and non-technical contributors understand the reasoning.


### Additional Details on Metric Selection
The final metric set will be chosen based on:
1. **Public health relevance** – Metrics must reflect how well the system predicts harmful pollution levels.
2. **Robustness** – Metrics should remain stable across different seasons and weather conditions.
3. **Transparency** – All calculations will be reproducible and clearly documented.
4. **Actionability** – Metrics should help improve decision-making for both public dashboards and internal model tuning.
Where trade-offs exist (e.g., high exact match rate vs. higher tolerance for near misses), these will be explicitly documented and justified.

### Additional Details on Metric Selection Criteria
The choice of final evaluation metrics will balance several priorities:
1. **Public Health Relevance** – Metrics should reflect the accuracy of predictions in detecting harmful pollution episodes.
2. **Model Robustness** – Metrics must remain stable across seasons and varying weather patterns.
3. **Transparency & Reproducibility** – All metric calculations will be open and fully documented.
4. **Operational Actionability** – Metrics should guide both model retraining priorities and real-time alerting systems.
5. **Trade-off Awareness** – Decisions about prioritizing exact matches vs. tolerating near misses will be documented explicitly.


### Additional Details on Metric Selection Criteria
The choice of final evaluation metrics will balance several priorities:
1. **Public Health Relevance** – Metrics should capture the model's ability to predict hazardous pollution levels accurately.
2. **Robustness Across Conditions** – The chosen metrics must produce stable and meaningful results across seasons and varying meteorological conditions.
3. **Transparency & Reproducibility** – Calculations will be open, reproducible, and well-documented to allow external verification.
4. **Operational Actionability** – Metrics should guide both model retraining schedules and real-time forecast adjustments.
5. **Trade-off Awareness** – In cases where higher exact match accuracy conflicts with broader tolerance for near misses, these trade-offs will be explicitly evaluated and documented.
