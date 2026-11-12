"""
Unit tests for Pydantic schema validation and congestion thresholds.
"""
import pytest
from pydantic import ValidationError
from src.common.schemas import TrafficObservation, TrafficPrediction, get_congestion_level

def test_valid_traffic_observation():
    payload = {
        "timestamp": "2012-03-01T08:00:00",
        "sensor_id": "773869",
        "current_speed": 62.5,
        "speed_lag_1": 63.0,
        "speed_lag_2": 64.1,
        "speed_lag_3": 65.0,
        "speed_mean_15m": 63.2,
        "speed_mean_1h": 61.0,
        "hour_of_day": 8,
        "day_of_week": 3,
        "is_weekend": 0
    }
    obs = TrafficObservation(**payload)
    assert obs.sensor_id == "773869"
    assert obs.current_speed == 62.5

def test_invalid_traffic_observation_speed():
    # Negative speed or speed > 130 mph should raise ValidationError
    payload = {
        "timestamp": "2012-03-01T08:00:00",
        "sensor_id": "773869",
        "current_speed": -10.0,
        "speed_lag_1": 63.0,
        "speed_lag_2": 64.1,
        "speed_lag_3": 65.0,
        "speed_mean_15m": 63.2,
        "speed_mean_1h": 61.0,
        "hour_of_day": 8,
        "day_of_week": 3,
        "is_weekend": 0
    }
    with pytest.raises(ValidationError):
        TrafficObservation(**payload)

def test_congestion_level_boundaries():
    assert get_congestion_level(65.0) == "LOW"
    assert get_congestion_level(50.1) == "LOW"
    assert get_congestion_level(50.0) == "MODERATE"
    assert get_congestion_level(35.1) == "MODERATE"
    assert get_congestion_level(35.0) == "HIGH"
    assert get_congestion_level(20.1) == "HIGH"
    assert get_congestion_level(20.0) == "SEVERE"
    assert get_congestion_level(5.0) == "SEVERE"
