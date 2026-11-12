"""
Feature engineering module for traffic speed prediction.
Computes lag features, rolling statistics, calendar indicators,
and sensor-specific historical hour-of-week averages.
Guarantees zero data leakage by computing historical statistics solely on training data.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np

def create_sensor_features(
    sensor_series: pd.Series,
    lags: List[int] = [1, 2, 3],
    rolling_windows: List[int] = [3, 12]
) -> pd.DataFrame:
    """
    Constructs lag and rolling statistics for a single sensor time series.
    All calculations strictly use backward-looking windows (no future leakage).
    """
    df_feat = pd.DataFrame(index=sensor_series.index)
    df_feat["current_speed"] = sensor_series.values

    # Lag features (t-1, t-2, t-3)
    for lag in lags:
        df_feat[f"speed_lag_{lag}"] = sensor_series.shift(lag).values

    # Rolling window averages (e.g. 3 steps = 15 min, 12 steps = 60 min)
    for win in rolling_windows:
        col_name = f"speed_mean_{win*5}m" if win < 12 else "speed_mean_1h"
        # closed='left' or shift(0) on current+past
        df_feat[col_name] = sensor_series.rolling(window=win, min_periods=1).mean().values

    # Target: 1-step ahead speed (t+1, 5 minutes into the future)
    df_feat["target_speed_next"] = sensor_series.shift(-1).values

    return df_feat

def compute_historical_profiles(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average speed per (sensor_id, hour_of_day, day_of_week)
    strictly from training observations to avoid test data leakage.
    """
    profile = train_df.groupby(["sensor_id", "hour_of_day", "day_of_week"])["current_speed"].mean().reset_index()
    profile.rename(columns={"current_speed": "historical_mean_speed"}, inplace=True)
    return profile
