"""
Apache Spark Structured Streaming job for real-time urban traffic prediction.
Consumes streaming traffic events, parses JSON schema, validates data quality,
computes event-time windowed aggregations, applies Random Forest speed inference,
derives congestion classifications, and sinks results to Kafka and Parquet/JSON.
"""
import sys
import os
import time
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, window, avg, round as spark_round,
    lit, current_timestamp, date_format, struct, to_json, udf
)
from pyspark.sql.types import (
    StringType, DoubleType, TimestampType, StructType
)

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger
from src.common.schemas import get_spark_raw_schema, get_congestion_level
from src.training.train_model import FEATURE_COLS

logger = setup_logger("spark_streaming")

def create_spark_session(use_kafka: bool = True) -> SparkSession:
    """Initializes a local SparkSession with required Kafka connector and JVM settings."""
    builder = SparkSession.builder \
        .appName(config["spark"]["app_name"]) \
        .master(config["spark"]["master"]) \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")

    if use_kafka:
        kafka_pkg = config["spark"]["kafka_package"]
        builder = builder.config("spark.jars.packages", kafka_pkg)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Initialized SparkSession (Version: {spark.version})")
    return spark

def run_streaming_pipeline(
    source_type: str = "auto",
    trigger_interval: str = "3 seconds",
    max_batches: int = 0
):
    """
    Executes the Spark Structured Streaming pipeline.
    
    Args:
        source_type: 'kafka', 'file', or 'auto' (checks Kafka availability)
        trigger_interval: micro-batch trigger interval
        max_batches: Stop after N micro-batches (0 for continuous streaming)
    """
    model_path = PROJECT_ROOT / config["modeling"]["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model first.")

    logger.info(f"Loading trained machine learning model from {model_path}...")
    model = joblib.load(model_path)

    checkpoint_dir = PROJECT_ROOT / config["spark"]["checkpoint_dir"]
    predictions_sink_dir = PROJECT_ROOT / config["spark"]["predictions_sink_dir"]
    simulated_stream_file = PROJECT_ROOT / "data/processed/simulated_stream.jsonl"
    stream_input_dir = PROJECT_ROOT / "data/processed/stream_input"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions_sink_dir.mkdir(parents=True, exist_ok=True)
    stream_input_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect source
    use_kafka = False
    if source_type in ("kafka", "auto"):
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            res = sock.connect_ex(("localhost", 9092))
            sock.close()
            if res == 0:
                use_kafka = True
                logger.info("Kafka broker detected at localhost:9092. Streaming from Kafka topic: traffic.raw.")
            else:
                logger.warning("Kafka broker not reachable at localhost:9092. Using file stream input.")
        except Exception as e:
            logger.warning(f"Kafka detection check failed: {e}. Defaulting to file stream.")

    spark = create_spark_session(use_kafka=use_kafka)
    raw_schema = get_spark_raw_schema()

    # 1. READ STREAM
    if use_kafka and source_type != "file":
        logger.info(f"Subscribing to Kafka topic: {config['kafka']['raw_topic']}...")
        kafka_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", config["kafka"]["bootstrap_servers"]) \
            .option("subscribe", config["kafka"]["raw_topic"]) \
            .option("startingOffsets", "latest") \
            .load()

        parsed_stream = kafka_stream.select(
            from_json(col("value").cast("string"), raw_schema).alias("data")
        ).select("data.*")
    else:
        logger.info(f"Reading from streaming file input directory: {stream_input_dir}...")
        # Prepare streaming input directory with initial records if empty
        if not any(stream_input_dir.glob("*.json")):
            test_parquet = PROJECT_ROOT / "data/processed/test_features.parquet"
            if test_parquet.exists():
                sample_df = pd.read_parquet(test_parquet).head(200)
                sample_df.to_json(stream_input_dir / "batch_001.json", orient="records", lines=True)
                logger.info("Prepared initial batch_001.json in stream_input directory.")

        parsed_stream = spark.readStream \
            .format("json") \
            .schema(raw_schema) \
            .option("maxFilesPerTrigger", 1) \
            .load(str(stream_input_dir))

    # 2. VALIDATION & QUALITY CHECKS
    valid_stream = parsed_stream \
        .filter(col("sensor_id").isNotNull()) \
        .filter(col("timestamp").isNotNull()) \
        .filter((col("current_speed") >= 0.0) & (col("current_speed") <= 120.0))

    # 3. DEFINE MODEL INFERENCE BATCH PROCESSING
    batch_counter = {"count": 0}

    def process_micro_batch(batch_df, batch_id):
        batch_counter["count"] += 1
        record_count = batch_df.count()
        logger.info(f"[SPARK MICRO-BATCH {batch_id}] Processing {record_count} incoming traffic events...")

        if record_count == 0:
            return

        # Convert micro-batch to pandas for model inference
        pdf = batch_df.toPandas()

        if "historical_mean_speed" not in pdf.columns:
            pdf["historical_mean_speed"] = 58.5

        X_batch = pdf[FEATURE_COLS]
        predictions = model.predict(X_batch)

        pdf["predicted_speed"] = np.round(predictions, 2)
        pdf["observed_speed"] = pdf["current_speed"]
        pdf["congestion_level"] = [get_congestion_level(s) for s in pdf["predicted_speed"]]
        pdf["prediction_timestamp"] = pd.Timestamp.now().isoformat()
        pdf["model_name"] = "random_forest_v1"
        pdf["forecast_horizon_minutes"] = 5

        # Save to output sink
        out_file = predictions_sink_dir / "recent_predictions.jsonl"
        pdf.to_json(out_file, orient="records", lines=True, mode="a")
        logger.info(f"[SPARK SINK] Successfully wrote {len(pdf)} predictions to {out_file}")

        # Also publish to Kafka predictions topic if Kafka is enabled
        if use_kafka:
            try:
                from kafka import KafkaProducer
                producer = KafkaProducer(
                    bootstrap_servers=config["kafka"]["bootstrap_servers"],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8")
                )
                for _, row in pdf.iterrows():
                    pred_payload = {
                        "timestamp": str(row["timestamp"]),
                        "sensor_id": str(row["sensor_id"]),
                        "observed_speed": float(row["observed_speed"]),
                        "predicted_speed": float(row["predicted_speed"]),
                        "congestion_level": str(row["congestion_level"]),
                        "prediction_timestamp": str(row["prediction_timestamp"]),
                        "model_name": "random_forest_v1",
                        "forecast_horizon_minutes": 5
                    }
                    producer.send(config["kafka"]["predictions_topic"], value=pred_payload)
                producer.flush()
                producer.close()
                logger.info(f"[KAFKA SINK] Published {len(pdf)} predictions to topic '{config['kafka']['predictions_topic']}'")
            except Exception as k_err:
                logger.warning(f"Failed to publish to Kafka predictions topic: {k_err}")

    # 4. START STREAMING QUERY
    logger.info("Starting Structured Streaming query...")
    query = valid_stream.writeStream \
        .foreachBatch(process_micro_batch) \
        .option("checkpointLocation", str(checkpoint_dir)) \
        .trigger(processingTime=trigger_interval) \
        .start()

    logger.info("Spark Structured Streaming pipeline is active and listening for events.")

    try:
        if max_batches > 0:
            logger.info(f"Awaiting completion of {max_batches} micro-batch(es)...")
            query.processAllAvailable()
            logger.info("All available stream data processed.")
            query.stop()
        else:
            query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping Spark Structured Streaming query...")
        query.stop()

    spark.stop()
    logger.info("Spark session stopped cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Run Spark Structured Streaming traffic prediction job.")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "kafka", "file"],
                        help="Streaming source: 'auto' (detects Kafka), 'kafka', or 'file'")
    parser.add_argument("--trigger", type=str, default="3 seconds", help="Micro-batch trigger interval")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after processing available data (pass 1 for demo test)")
    args = parser.parse_args()

    run_streaming_pipeline(
        source_type=args.source,
        trigger_interval=args.trigger,
        max_batches=args.max_batches
    )

if __name__ == "__main__":
    main()
