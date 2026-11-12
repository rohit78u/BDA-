"""
Model training module.
Trains historical persistence baseline and Random Forest Regressor
on chronological training data. Evaluates against validation split.
Saves model to models/random_forest_traffic.joblib and metadata.
"""
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger

logger = setup_logger("train_model")

# Feature columns used for training and streaming inference
FEATURE_COLS = [
    "current_speed",
    "speed_lag_1",
    "speed_lag_2",
    "speed_lag_3",
    "speed_mean_15m",
    "speed_mean_1h",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "historical_mean_speed"
]

TARGET_COL = "target_speed_next"

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, MAPE (for valid speeds > 5 mph), and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Avoid division by zero in MAPE
    mask = y_true > 5.0
    if np.sum(mask) > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = None

    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "mape_percent": round(mape, 2) if mape is not None else None,
        "r2": round(float(r2), 4)
    }

def train():
    train_path = PROJECT_ROOT / "data/processed/train_features.parquet"
    val_path = PROJECT_ROOT / "data/processed/val_features.parquet"
    model_save_path = PROJECT_ROOT / config["modeling"]["model_path"]
    metadata_save_path = PROJECT_ROOT / config["modeling"]["metadata_path"]

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 1. Historical Persistence Baseline: predict speed at t+1 = current_speed at t
    logger.info("Evaluating Historical Persistence Baseline (speed[t+1] = speed[t])...")
    val_pred_baseline = val_df["current_speed"].values
    baseline_metrics = compute_metrics(y_val.values, val_pred_baseline)
    logger.info(f"Persistence Baseline Validation Metrics: MAE={baseline_metrics['mae']} mph, "
                f"RMSE={baseline_metrics['rmse']} mph, MAPE={baseline_metrics['mape_percent']}%, R2={baseline_metrics['r2']}")

    # 2. Random Forest Regressor
    rf_params = config["modeling"]["random_forest"]
    logger.info(f"Training Random Forest Regressor with params: {rf_params}...")
    start_time = time.time()
    
    rf = RandomForestRegressor(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        n_jobs=rf_params["n_jobs"],
        random_state=rf_params["random_state"]
    )
    rf.fit(X_train, y_train)
    training_duration = time.time() - start_time
    logger.info(f"Random Forest trained in {training_duration:.2f} seconds.")

    # Validation evaluation
    val_pred_rf = rf.predict(X_val)
    rf_metrics = compute_metrics(y_val.values, val_pred_rf)
    logger.info(f"Random Forest Validation Metrics: MAE={rf_metrics['mae']} mph, "
                f"RMSE={rf_metrics['rmse']} mph, MAPE={rf_metrics['mape_percent']}%, R2={rf_metrics['r2']}")

    # Compute Feature Importances
    importances = dict(zip(FEATURE_COLS, [round(float(x), 4) for x in rf.feature_importances_]))
    logger.info(f"Top feature importances: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]}")

    # Save model artifact
    joblib.dump(rf, model_save_path)
    logger.info(f"Saved trained Random Forest model to {model_save_path}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "training_duration_seconds": round(training_duration, 2),
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "forecast_horizon_minutes": config["dataset"]["forecast_horizon_minutes"],
        "hyperparameters": rf_params,
        "validation_metrics": {
            "baseline": baseline_metrics,
            "random_forest": rf_metrics
        },
        "feature_importances": importances
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_save_path}")

    return metadata

if __name__ == "__main__":
    train()
