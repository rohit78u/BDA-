"""
Dataset acquisition module for METR-LA traffic speed dataset.
Downloads metr-la.h5 and graph_sensor_locations.csv from verified research repositories.
Provides offline fallback sample generator for test environments.
"""
import os
import sys
import argparse
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger

logger = setup_logger("download_dataset")

METR_LA_URL = config["dataset"]["data_url"]
SENSOR_LOCATIONS_URL = config["dataset"]["sensor_locations_url"]

def download_file(url: str, dest_path: Path, min_expected_bytes: int = 1000) -> bool:
    """Downloads a file from url with streaming and basic validation."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size >= min_expected_bytes:
        logger.info(f"File already exists: {dest_path} ({dest_path.stat().st_size:,} bytes). Skipping download.")
        return True

    logger.info(f"Downloading from {url} to {dest_path}...")
    try:
        # Stream download with progress logging
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024 # 1 MB chunks
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (5 * 1024 * 1024) < chunk_size: # log every ~5MB
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")

        logger.info(f"Successfully downloaded {dest_path} ({dest_path.stat().st_size:,} bytes).")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink() # remove partial file
        return False

def generate_sample_dataset(h5_dest: Path, csv_dest: Path, num_sensors: int = 20, num_steps: int = 288 * 14):
    """
    Generates a realistic synthetic sample dataset matching METR-LA schema
    (5-minute intervals, diurnal speed cycle, 20 sensors over 14 days).
    Used as fallback when network is unavailable or for fast unit testing.
    """
    logger.info(f"Generating synthetic fallback dataset ({num_sensors} sensors, {num_steps} time steps)...")
    h5_dest.parent.mkdir(parents=True, exist_ok=True)
    csv_dest.parent.mkdir(parents=True, exist_ok=True)

    # 5-minute frequency starting at 2012-03-01 00:00:00 (matching METR-LA start)
    timestamps = pd.date_range("2012-03-01 00:00:00", periods=num_steps, freq="5min")
    sensor_ids = [f"7738{i:02d}" for i in range(num_sensors)]

    # Generate diurnal speed curve (typical highway: ~65 mph night, dips to ~30-40 mph morning/evening rush)
    np.random.seed(42)
    hour_float = timestamps.hour + timestamps.minute / 60.0

    # Base diurnal traffic pattern
    morning_rush = np.exp(-((hour_float - 8.5) ** 2) / (2 * 1.5 ** 2))
    evening_rush = np.exp(-((hour_float - 17.5) ** 2) / (2 * 2.0 ** 2))
    base_speed = 63.0 - (22.0 * morning_rush) - (25.0 * evening_rush)

    data = {}
    lat_base, lon_base = 34.0522, -118.2437 # Los Angeles coordinates
    coords = []

    for idx, sid in enumerate(sensor_ids):
        # Add sensor-specific offset and random noise
        sensor_noise = np.random.normal(0, 3.5, size=num_steps)
        # Occasionally inject sensor downtime / missing values (recorded as 0 in PeMS)
        downtime_mask = np.random.rand(num_steps) < 0.02
        sensor_speed = np.clip(base_speed + sensor_noise + np.random.uniform(-4, 4), 5.0, 75.0)
        sensor_speed[downtime_mask] = 0.0 # PeMS missing convention
        data[sid] = sensor_speed.astype(np.float32)

        coords.append({
            "sensor_id": sid,
            "latitude": lat_base + np.random.uniform(-0.15, 0.15),
            "longitude": lon_base + np.random.uniform(-0.15, 0.15)
        })

    df = pd.DataFrame(data, index=timestamps)
    # Save as HDF5 PyTables format exactly like original METR-LA
    df.to_hdf(h5_dest, key="df", mode="w")
    logger.info(f"Saved synthetic METR-LA H5 to {h5_dest} ({h5_dest.stat().st_size:,} bytes).")

    # Save sensor coordinates
    coords_df = pd.DataFrame(coords)
    coords_df.to_csv(csv_dest, index=False)
    logger.info(f"Saved synthetic sensor locations to {csv_dest}.")

def main():
    parser = argparse.ArgumentParser(description="Acquire METR-LA dataset.")
    parser.add_argument("--fallback-sample", action="store_true", help="Generate synthetic sample dataset instead of downloading")
    args = parser.parse_args()

    h5_path = PROJECT_ROOT / config["dataset"]["raw_file"]
    coords_path = PROJECT_ROOT / config["dataset"]["locations_file"]

    if args.fallback_sample:
        generate_sample_dataset(h5_path, coords_path)
        return

    logger.info("Starting METR-LA dataset acquisition...")
    # 1. Download sensor coordinates
    coords_success = download_file(SENSOR_LOCATIONS_URL, coords_path, min_expected_bytes=1000)
    
    # 2. Download metr-la.h5 (57 MB)
    data_success = download_file(METR_LA_URL, h5_path, min_expected_bytes=50_000_000)

    if not data_success or not coords_success:
        logger.warning("Network download encountered an issue. Falling back to synthetic METR-LA sample dataset for local development.")
        generate_sample_dataset(h5_path, coords_path)
    else:
        logger.info("METR-LA dataset acquisition completed successfully.")

if __name__ == "__main__":
    main()
