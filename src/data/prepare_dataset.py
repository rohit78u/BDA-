"""
Dataset preparation module.
Reads METR-LA H5, handles missing/zero values using strictly causal imputation,
constructs chronological train/val/test splits, generates features,
and writes Parquet feature tables for training and JSONL for Kafka replay.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger
from src.data.build_features import create_sensor_features, compute_historical_profiles

logger = setup_logger("prepare_dataset")

def prepare_data(num_sensors: int = 25, sample_demo_records: int = 2000):
    """
    Preprocesses raw traffic speeds, generates features, and splits chronologically.
    
    Args:
        num_sensors: Number of sensors to process (default: 25 key sensors for fast,
                     practical student training; pass 0 or 207 for all sensors).
        sample_demo_records: Number of test records to save as JSONL for Kafka replay.
    """
    h5_path = PROJECT_ROOT / config["dataset"]["raw_file"]
    processed_dir = PROJECT_ROOT / "data/processed"
    samples_dir = PROJECT_ROOT / "data/samples"
    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw speed matrix from {h5_path}...")
    df_raw = pd.read_hdf(h5_path)

    # Convert column names to clean strings
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Select sensors if requested
    if 0 < num_sensors < len(df_raw.columns):
        selected_sensors = list(df_raw.columns[:num_sensors])
        logger.info(f"Selecting {num_sensors} representative sensors for training/evaluation: {selected_sensors[:5]}...")
        df_raw = df_raw[selected_sensors]
    else:
        logger.info(f"Processing all {len(df_raw.columns)} sensors...")

    # Impute missing/zero values causally:
    # 1. Replace 0.0 with NaN
    # 2. Forward-fill up to 3 steps (15 mins) - strictly historical, no future leakage
    # 3. For any remaining NaNs, fill with the sensor's expanding median
    logger.info("Applying causal missing value imputation (forward fill, zero leakage)...")
    df_clean = df_raw.replace(0.0, np.nan)
    df_clean = df_clean.ffill(limit=3)
    # Fill remaining NaNs with sensor's column median
    df_clean = df_clean.fillna(df_clean.median())

    # Chronological Split
    total_steps = len(df_clean)
    train_ratio = config["dataset"]["chronological_split"]["train_ratio"]
    val_ratio = config["dataset"]["chronological_split"]["val_ratio"]
    
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))

    df_train_raw = df_clean.iloc[:train_end]
    df_val_raw = df_clean.iloc[train_end:val_end]
    df_test_raw = df_clean.iloc[val_end:]

    logger.info(f"Chronological split points: Train={len(df_train_raw)} steps ({df_train_raw.index[0]} to {df_train_raw.index[-1]}), "
                f"Val={len(df_val_raw)} steps, Test={len(df_test_raw)} steps ({df_test_raw.index[0]} to {df_test_raw.index[-1]})")

    # Build feature tables per sensor
    def process_split_features(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        records = []
        for sid in split_df.columns:
            s_series = split_df[sid]
            feat_df = create_sensor_features(s_series)
            feat_df["sensor_id"] = sid
            feat_df["timestamp"] = feat_df.index.strftime("%Y-%m-%dT%H:%M:%S")
            feat_df["hour_of_day"] = feat_df.index.hour
            feat_df["day_of_week"] = feat_df.index.dayofweek
            feat_df["is_weekend"] = (feat_df.index.dayofweek >= 5).astype(int)
            feat_df.reset_index(drop=True, inplace=True)
            records.append(feat_df)
        
        combined = pd.concat(records, axis=0, ignore_index=True)
        # Drop rows where lag or target is NaN
        combined.dropna(inplace=True)
        return combined

    logger.info("Building feature tables for Train, Validation, and Test splits...")
    train_feat = process_split_features(df_train_raw, "train")
    val_feat = process_split_features(df_val_raw, "val")
    test_feat = process_split_features(df_test_raw, "test")

    # Compute historical speed profile strictly from training set
    logger.info("Computing historical profiles from training set...")
    profile = compute_historical_profiles(train_feat)

    # Merge profile into train, val, and test
    def attach_profile(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(profile, on=["sensor_id", "hour_of_day", "day_of_week"], how="left")
        # In case of unseen (sensor, hour, day), fallback to overall mean
        merged["historical_mean_speed"] = merged["historical_mean_speed"].fillna(train_feat["current_speed"].mean())
        return merged

    train_feat = attach_profile(train_feat)
    val_feat = attach_profile(val_feat)
    test_feat = attach_profile(test_feat)

    # Save splits as Parquet
    train_path = processed_dir / "train_features.parquet"
    val_path = processed_dir / "val_features.parquet"
    test_path = processed_dir / "test_features.parquet"

    train_feat.to_parquet(train_path, index=False)
    val_feat.to_parquet(val_path, index=False)
    test_feat.to_parquet(test_path, index=False)

    logger.info(f"Saved Train features: {train_path} ({len(train_feat):,} rows)")
    logger.info(f"Saved Val features: {val_path} ({len(val_feat):,} rows)")
    logger.info(f"Saved Test features: {test_path} ({len(test_feat):,} rows)")

    # Save sample JSONL replay stream for Kafka producer demonstration
    replay_path = samples_dir / "replay_stream_sample.jsonl"
    demo_sample = test_feat.head(sample_demo_records)
    demo_sample.to_json(replay_path, orient="records", lines=True)
    logger.info(f"Saved sample Kafka replay stream ({len(demo_sample):,} records) to {replay_path}")

    # Also save a sample for unit testing
    unit_test_sample_path = samples_dir / "sample_features.parquet"
    test_feat.head(100).to_parquet(unit_test_sample_path, index=False)
    logger.info(f"Saved unit test sample to {unit_test_sample_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and build features.")
    parser.add_argument("--num-sensors", type=int, default=25, help="Number of sensors to include (default: 25)")
    args = parser.parse_args()
    prepare_data(num_sensors=args.num_sensors)
