"""
Kafka producer module for simulating real-time traffic data streams.
Replays preprocessed historical test observations into Kafka topic: traffic.raw.
Supports configurable replay rate, record limits, graceful shutdown,
and offline simulation buffer fallback when Kafka broker is offline.
"""
import sys
import time
import json
import signal
import argparse
from pathlib import Path
import pandas as pd

from src.common.config import config, PROJECT_ROOT
from src.common.logging_config import setup_logger
from src.common.schemas import TrafficObservation

logger = setup_logger("kafka_producer")

class TrafficKafkaProducer:
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic: str = None,
        rate_per_sec: float = 10.0,
        limit: int = 0,
        offline_buffer: bool = False
    ):
        self.bootstrap_servers = bootstrap_servers or config["kafka"]["bootstrap_servers"]
        self.topic = topic or config["kafka"]["raw_topic"]
        self.rate_per_sec = rate_per_sec
        self.delay = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.05
        self.limit = limit
        self.offline_buffer = offline_buffer
        self.running = True
        self.producer = None
        self.buffer_file = PROJECT_ROOT / "data/processed/simulated_stream.jsonl"

        # Signal handling for graceful termination
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("\n[SHUTDOWN] Interruption signal received. Gracefully stopping producer...")
        self.running = False

    def connect(self) -> bool:
        """Attempts to connect to Apache Kafka broker."""
        if self.offline_buffer:
            logger.info(f"[MODE: OFFLINE] Operating in offline simulation mode. Messages will be written to {self.buffer_file}")
            self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
            # Clear or prepare buffer file
            with open(self.buffer_file, "w", encoding="utf-8") as f:
                pass
            return True

        logger.info(f"Connecting to Apache Kafka broker at {self.bootstrap_servers}...")
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=1, # Leader acknowledge
                retries=3,
                request_timeout_ms=10000,
                max_block_ms=5000
            )
            # Test connectivity
            self.producer.bootstrap_connected()
            logger.info(f"Connected to Kafka broker! Target topic: '{self.topic}'.")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to Kafka at {self.bootstrap_servers}: {e}")
            logger.warning("Tip: Run 'docker compose up -d' to start the local Kafka broker.")
            logger.warning("Falling back automatically to offline simulation buffer mode (--offline-buffer).")
            self.offline_buffer = True
            self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.buffer_file, "w", encoding="utf-8") as f:
                pass
            return True

    def replay_from_parquet(self, parquet_path: Path):
        """Streams records sequentially from a Parquet file."""
        if not parquet_path.exists():
            raise FileNotFoundError(f"Source file not found at: {parquet_path}")

        logger.info(f"Loading stream replay records from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        logger.info(f"Total available records in replay file: {len(df):,}")

        if not self.connect():
            return

        sent_count = 0
        start_time = time.time()
        logger.info(f"Beginning stream replay (Rate: {self.rate_per_sec} records/sec, Limit: {'Unlimited' if self.limit <= 0 else self.limit})...")

        for idx, row in df.iterrows():
            if not self.running:
                break
            if self.limit > 0 and sent_count >= self.limit:
                logger.info(f"Demo limit of {self.limit} records reached.")
                break

            # Construct validated observation dictionary
            record_dict = {
                "timestamp": str(row["timestamp"]),
                "sensor_id": str(row["sensor_id"]),
                "current_speed": round(float(row["current_speed"]), 2),
                "speed_lag_1": round(float(row["speed_lag_1"]), 2),
                "speed_lag_2": round(float(row["speed_lag_2"]), 2),
                "speed_lag_3": round(float(row["speed_lag_3"]), 2),
                "speed_mean_15m": round(float(row["speed_mean_15m"]), 2),
                "speed_mean_1h": round(float(row["speed_mean_1h"]), 2),
                "hour_of_day": int(row["hour_of_day"]),
                "day_of_week": int(row["day_of_week"]),
                "is_weekend": int(row["is_weekend"])
            }

            # Schema validation check
            try:
                TrafficObservation(**record_dict)
            except Exception as val_err:
                logger.warning(f"Skipping invalid record: {val_err}")
                continue

            # Publish to Kafka or write to offline simulation sink
            if self.producer and not self.offline_buffer:
                try:
                    self.producer.send(self.topic, key=record_dict["sensor_id"], value=record_dict)
                except Exception as send_err:
                    logger.error(f"Error publishing record to Kafka: {send_err}")
            else:
                with open(self.buffer_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record_dict) + "\n")

            sent_count += 1
            if sent_count % 50 == 0 or sent_count == 1:
                elapsed = time.time() - start_time
                actual_rate = sent_count / elapsed if elapsed > 0 else 0
                logger.info(f"[PRODUCER STREAM] Sent {sent_count:,} records | Sensor: {record_dict['sensor_id']} | Speed: {record_dict['current_speed']} mph | Rate: {actual_rate:.1f} rec/s")

            time.sleep(self.delay)

        total_elapsed = time.time() - start_time
        avg_rate = sent_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Stream replay completed. Total sent: {sent_count:,} records in {total_elapsed:.2f}s ({avg_rate:.1f} records/sec).")

        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer flushed and closed gracefully.")

def main():
    parser = argparse.ArgumentParser(description="Replay traffic sensor records into Kafka.")
    parser.add_argument("--rate", type=float, default=20.0, help="Replay rate in records per second (default: 20)")
    parser.add_argument("--limit", type=int, default=200, help="Limit number of records to send (default: 200, 0 for all)")
    parser.add_argument("--offline-buffer", action="store_true", help="Force offline simulation buffer mode")
    parser.add_argument("--source", type=str, default="data/processed/test_features.parquet", help="Path to input parquet")
    args = parser.parse_args()

    producer = TrafficKafkaProducer(
        rate_per_sec=args.rate,
        limit=args.limit,
        offline_buffer=args.offline_buffer
    )
    producer.replay_from_parquet(PROJECT_ROOT / args.source)

if __name__ == "__main__":
    main()
