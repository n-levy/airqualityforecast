# FEATURES

## Calendar Features
- Day of week
- Month
- Holiday flag (per city)

## Lag Features
- Lagged observations: t-1d, t-2d (no leakage)
- Lagged provider forecasts: previous cycle, same lead time (no leakage)

These features are designed to capture temporal dependencies.


**Note on 'No Leakage':**
Data leakage occurs when information from the future is used in training or forecasting.
To prevent this, lag features are constructed only from data available before the forecast issue time.
