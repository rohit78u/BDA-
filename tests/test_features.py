"""
Unit tests for feature engineering, lags, and rolling averages.
Verifies strictly causal computation without target leakage.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.build_features import create_sensor_features

def test_feature_creation_and_lags():
    series = pd.Series([50.0, 52.0, 54.0, 56.0, 58.0], index=pd.date_range("2012-03-01", periods=5, freq="5min"))
    feats = create_sensor_features(series, lags=[1, 2, 3], rolling_windows=[3])

    # Check lag alignment
    # At index 3 (value 56.0): lag_1 should be 54.0, lag_2 should be 52.0, lag_3 should be 50.0
    assert feats.iloc[3]["current_speed"] == 56.0
    assert feats.iloc[3]["speed_lag_1"] == 54.0
    assert feats.iloc[3]["speed_lag_2"] == 52.0
    assert feats.iloc[3]["speed_lag_3"] == 50.0

    # Target speed next at index 3 should be 58.0 (1 step into future)
    assert feats.iloc[3]["target_speed_next"] == 58.0

    # Target at last step must be NaN (future value unknown)
    assert pd.isna(feats.iloc[-1]["target_speed_next"])

def test_no_future_leakage_in_features():
    series = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.date_range("2012-03-01", periods=4, freq="5min"))
    feats = create_sensor_features(series, lags=[1, 2])

    # Features at step 0 must NOT know about step 1, 2, or 3
    assert pd.isna(feats.iloc[0]["speed_lag_1"])
    assert pd.isna(feats.iloc[0]["speed_lag_2"])
