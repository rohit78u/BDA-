"""
Pydantic data schemas and PySpark StructType definitions for traffic observations
and predictions.
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Pydantic schema for raw streaming observations (traffic.raw)
class TrafficObservation(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp of the observation")
    sensor_id: str = Field(..., description="Unique sensor identifier (e.g. S773869)")
    current_speed: float = Field(..., ge=0.0, le=130.0, description="Observed traffic speed in mph")
    speed_lag_1: float = Field(..., ge=0.0, le=130.0, description="Speed at t-1 (5 min ago)")
    speed_lag_2: float = Field(..., ge=0.0, le=130.0, description="Speed at t-2 (10 min ago)")
    speed_lag_3: float = Field(..., ge=0.0, le=130.0, description="Speed at t-3 (15 min ago)")
    speed_mean_15m: float = Field(..., ge=0.0, le=130.0, description="Rolling 15-min average speed")
    speed_mean_1h: float = Field(..., ge=0.0, le=130.0, description="Rolling 1-hour average speed")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Binary flag: 1 if Saturday or Sunday, else 0")

# Pydantic schema for predicted traffic events (traffic.predictions)
class TrafficPrediction(BaseModel):
    timestamp: str = Field(..., description="Event timestamp of the traffic state")
    sensor_id: str = Field(..., description="Sensor identifier")
    observed_speed: float = Field(..., description="Actual speed observed at event time (mph)")
    predicted_speed: float = Field(..., description="Predicted future speed (5 min ahead, mph)")
    congestion_level: str = Field(..., description="Derived congestion: LOW, MODERATE, HIGH, SEVERE")
    prediction_timestamp: str = Field(..., description="Timestamp when prediction was calculated")
    model_name: str = Field(default="random_forest_v1", description="Model identifier used for inference")
    forecast_horizon_minutes: int = Field(default=5, description="Forecast horizon in minutes")

def get_congestion_level(speed: float) -> str:
    """
    Derives congestion classification label from traffic speed in mph.
    Thresholds:
      - > 50.0 mph: LOW (Free Flow)
      - 35.0 to 50.0 mph: MODERATE (Normal / Slight Delay)
      - 20.0 to 35.0 mph: HIGH (Heavy Congestion)
      - <= 20.0 mph: SEVERE (Severe Congestion / Gridlock)
    """
    if speed > 50.0:
        return "LOW"
    elif speed > 35.0:
        return "MODERATE"
    elif speed > 20.0:
        return "HIGH"
    else:
        return "SEVERE"

def get_spark_raw_schema():
    """Returns PySpark StructType schema for deserializing JSON from traffic.raw."""
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType
    )
    return StructType([
        StructField("timestamp", StringType(), False),
        StructField("sensor_id", StringType(), False),
        StructField("current_speed", DoubleType(), False),
        StructField("speed_lag_1", DoubleType(), False),
        StructField("speed_lag_2", DoubleType(), False),
        StructField("speed_lag_3", DoubleType(), False),
        StructField("speed_mean_15m", DoubleType(), False),
        StructField("speed_mean_1h", DoubleType(), False),
        StructField("hour_of_day", IntegerType(), False),
        StructField("day_of_week", IntegerType(), False),
        StructField("is_weekend", IntegerType(), False),
    ])
