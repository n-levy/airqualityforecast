"""
Stage 0 smoke test
- Loads environment via stage1_forecast.env (robust).
- Prints DATA_ROOT and MODELS_ROOT.
- Ends with a clear success message.
"""
from stage1_forecast import env

cfg = env.load_and_validate()
print("Smoke test OK.")
