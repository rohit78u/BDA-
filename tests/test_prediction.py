"""
Unit tests for model loading, feature order alignment, and inference.
"""
import pytest
import joblib
import pandas as pd
import numpy as np
from src.common.config import config, PROJECT_ROOT
from src.training.train_model import FEATURE_COLS

def test_model_artifact_exists_and_loads():
    model_path = PROJECT_ROOT / config["modeling"]["model_path"]
    assert model_path.exists(), f"Model file not found at {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict")

def test_inference_shape_and_features():
    model_path = PROJECT_ROOT / config["modeling"]["model_path"]
    model = joblib.load(model_path)

    # Synthetic single sample matching exact feature column schema
    dummy_input = pd.DataFrame([{
        "current_speed": 60.0,
        "speed_lag_1": 61.0,
        "speed_lag_2": 62.0,
        "speed_lag_3": 63.0,
        "speed_mean_15m": 61.0,
        "speed_mean_1h": 60.5,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "historical_mean_speed": 59.0
    }])

    pred = model.predict(dummy_input[FEATURE_COLS])
    assert len(pred) == 1
    # Predicted speed should be a reasonable highway speed
    assert 20.0 <= pred[0] <= 80.0
