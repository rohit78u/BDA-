# System Architecture Specification

## 1. Architectural Overview

The **Real-Time Urban Traffic Congestion Prediction and Analysis** system is designed as a distributed, stream-first Big Data pipeline that continuously digests sensor readings, performs event-time data cleaning, aggregates traffic statistics, forecasts short-term speed via a machine learning model, and publishes live intelligence to a web dashboard.

```
+─────────────────────────────────────────────────────────────────────────────+
|                            1. DATA INGESTION TIER                           |
+─────────────────────────────────────────────────────────────────────────────+
| METR-LA Historical Dataset (207 Freeway Loop Detectors in Los Angeles)      |
|                              │                                              |
|                              ▼                                              |
| Python Stream Replay Producer (Replay Multiplier, Schema Validation)        |
|                              │                                              |
|                              ▼                                              |
| Apache Kafka Broker (KRaft Mode) ───► Topic: traffic.raw                    |
+──────────────────────────────┼──────────────────────────────────────────────+
                               │ (JSON Stream over TCP Port 9092)
                               ▼
+─────────────────────────────────────────────────────────────────────────────+
|                       2. STREAM PROCESSING TIER                             |
+─────────────────────────────────────────────────────────────────────────────+
| Apache Spark Structured Streaming Engine (Local / Cluster Mode)             |
|   ├─ Step A: Explicit StructType Deserialization & Null Filtering           |
|   ├─ Step B: Domain Range Validation (0.0 <= speed <= 120.0 mph)            |
|   ├─ Step C: Event-Time Assignment & Watermarking                           |
|   ├─ Step D: Micro-Batch Feature Vector Construction                        |
|   ├─ Step E: Pre-Trained Random Forest Inference (foreachBatch Execution)   |
|   └─ Step F: Rule-Based Congestion Categorization (LOW, MODERATE, HIGH, SEVERE)|
+──────────────────────────────┼──────────────────────────────────────────────+
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
+──────────────────────────────+ +────────────────────────────────────────────+
|     3. STORAGE SINK TIER     | |            4. PRESENTATION TIER            |
+──────────────────────────────+ +────────────────────────────────────────────+
| a) Kafka Predictions Topic   | | Streamlit Interactive Analytics Dashboard  |
|    Topic: traffic.predictions| |   ├─ Real-Time Traffic Stream Monitor      |
| b) Persistent Storage        | |   ├─ Geospatial Plotly Map (LA Coordinates)|
|    - recent_predictions.jsonl| |   ├─ Hourly Speed & Congestion Profiles    |
|    - Structured Parquet      | |   ├─ Offline Baseline vs ML Evaluation     |
|    - Spark Recovery CP dir   | |   └─ Streaming Throughput & Latency Gauges |
+──────────────────────────────+ +────────────────────────────────────────────+
```

---

## 2. Component Breakdown

### A. Data Source: METR-LA
- **Origin:** Caltrans Performance Measurement System (PeMS).
- **Physical Sensing:** Inductive loop detectors embedded in pavement across major Los Angeles County freeways (I-5, I-10, I-405, US-101, SR-110).
- **Sampling Frequency:** Observations aggregated into 5-minute intervals.
- **Data Points:** 207 sensors $\times$ 34,272 time intervals = 7,094,304 total traffic readings.

### B. Ingestion Tier: Apache Kafka
- **Broker Architecture:** Apache Kafka 3.7.0 operating in **KRaft (Kafka Raft Metadata)** mode. This modern architecture removes the legacy dependency on Apache ZooKeeper, simplifying deployment, reducing memory footprint, and accelerating controller failover.
- **Message Schema:** Strongly typed JSON payload complying with `TrafficObservation` schema.
- **Replay Mechanism:** A Python-based replay producer (`kafka_producer.py`) reads historical test timestamps and streams them at calibrated frequencies (e.g., 20 events/second) to emulate a live freeway sensor network.

### C. Stream Processing Tier: Apache Spark Structured Streaming
- **Processing Paradigm:** Micro-batch processing triggered every 3 to 5 seconds.
- **Data Quality Pipeline:**
  1. Filters malformed records and null sensor IDs.
  2. Enforces physical traffic limits ($0.0 \le \text{current\_speed} \le 120.0$ mph).
  3. Integrates causal lag features ($t-1, t-2, t-3$) and rolling temporal statistics.
- **Model Inference:** Employs a pre-trained Random Forest Regressor serialized using `joblib`. In local prototype mode, micro-batches are evaluated via driver-side vectorized execution (`foreachBatch`). For distributed cluster deployments, model inference scales across worker executors via PySpark `pandas_udf`.
- **Fault Tolerance:** Spark maintains transactional checkpoint logs (`data/processed/checkpoints/`) guaranteeing *at-least-once* stream processing semantics.

### D. Presentation Tier: Streamlit & Plotly
- **Real-Time Visualization:** Connects directly to the streaming storage sink and re-renders live traffic feeds without re-executing data pipelines.
- **Geographic Network View:** Renders an interactive Plotly scatter mapbox leveraging verified latitude and longitude coordinates for all 207 freeway sensors, dynamically colored by derived congestion levels.
