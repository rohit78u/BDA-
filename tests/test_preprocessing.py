"""
Unit tests for data preprocessing and missing value imputation.
"""
import pytest
import pandas as pd
import numpy as np

def test_causal_imputation_no_future_leak():
    # Construct a series with leading zero, middle zeros, and trailing zeros
    speeds = [60.0, 0.0, 0.0, 50.0, 0.0, 55.0]
    s = pd.Series(speeds)

    # Causal imputation: replace 0 with NaN, forward fill up to 3 steps
    s_clean = s.replace(0.0, np.nan).ffill(limit=3)

    assert s_clean.iloc[0] == 60.0
    assert s_clean.iloc[1] == 60.0 # forward filled from past
    assert s_clean.iloc[2] == 60.0 # forward filled from past
    assert s_clean.iloc[3] == 50.0
    assert s_clean.iloc[4] == 50.0 # forward filled from past
    assert s_clean.iloc[5] == 55.0

def test_chronological_split_ordering():
    total_len = 1000
    dates = pd.date_range("2012-03-01", periods=total_len, freq="5min")
    df = pd.DataFrame({"speed": np.random.rand(total_len)}, index=dates)

    train_ratio, val_ratio = 0.70, 0.15
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # Assert chronological non-overlapping boundaries
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
