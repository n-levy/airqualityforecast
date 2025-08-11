#!/usr/bin/env python
from pm25_forecast.env import *  # noqa: F401

if __name__ == "__main__":
    import os
    print("DATA_ROOT:", os.getenv("DATA_ROOT"))
    print("MODELS_ROOT:", os.getenv("MODELS_ROOT"))
