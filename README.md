# Real-Time Urban Traffic Congestion Prediction and Analysis Using Big Data

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-orange.svg)](https://spark.apache.org/)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-3.7.0%20(KRaft)-black.svg)](https://kafka.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Academic-green.svg)](LICENSE)

A complete, beginner-friendly, locally runnable B.Tech Computer Science and Data Science Big Data Analytics semester project.

---

## 1. Problem Statement

Urban road networks face severe traffic congestion, leading to lost productivity, increased greenhouse gas emissions, and elevated accident risks. Traditional traffic monitoring relies on batch analysis of historical sensor logs, preventing traffic operators and commuters from taking proactive actions.

This project designs and implements an **end-to-end, stream-first Big Data pipeline** that continuously ingests traffic sensor telemetry, performs real-time distributed stream processing with **Apache Spark Structured Streaming**, forecasts short-term vehicle speeds using a **machine learning model (Random Forest Regressor)**, and surfaces actionable traffic intelligence via an interactive **Streamlit dashboard**.

---

## 2. Objectives

1. Ingest real-world traffic sensor observations as an asynchronous real-time event stream via **Apache Kafka**.
2. Perform event-time data cleaning, validation, and micro-batch aggregation using **Apache Spark Structured Streaming**.
3. Establish a rigorous machine learning benchmark comparing a **historical persistence baseline** against a **Random Forest Regressor** using a strict **chronological train/validation/test split**.
4. Derive rule-based congestion classifications (`LOW`, `MODERATE`, `HIGH`, `SEVERE`) from predicted continuous speeds.
5. Visualize live stream telemetry, spatial Los Angeles highway network maps, and model performance metrics in a **Streamlit analytics dashboard**.
6. Provide complete academic documentation, architecture diagrams, and viva examination preparation materials.

---

## 3. Scope and Exclusions

### In Scope:
- Ingestion of the benchmark **METR-LA** public freeway traffic dataset.
- Real-time event replay simulating live streaming sensors using Kafka broker topic `traffic.raw`.
- Spark Structured Streaming consumer with explicit schema enforcement and data quality checks.
- Zero-leakage chronological split (Train 70%, Val 15%, Test 15%) for machine learning evaluation.
- Local reproducible deployment on Windows 11 (PowerShell) and Linux/WSL2.

### Exclusions:
- Complex deep Graph Neural Networks (GNNs) requiring GPUs.
- Paid cloud infrastructure (e.g., AWS Kinesis, Snowflake).
- Physical IoT hardware deployment on city roads.

---

## 4. System Architecture

```
[ Public METR-LA Dataset (HDF5) ]
            │
            ▼
[ Dataset Adapter & Preprocessing ] ────► Causal Missing Value Imputation (Zero Leakage)
            │
            ▼
[ Feature Engineering ] ───────────────► Speed Lags (5, 10, 15 min), Rolling Means, Time Features
            │                                    │
            │                                    ▼
            │                        [ Model Training & Evaluation ]
            │                        ├─ Persistence Baseline
            │                        └─ Random Forest Regressor (joblib)
            ▼
[ Configurable Kafka Producer ] ───────► Kafka Topic: traffic.raw
            │
            ▼
[ Apache Spark Structured Streaming ]
  ├─ Strict JSON Schema Validation
  ├─ Outlier & Speed Filtering (0.0 to 120.0 mph)
  ├─ Vectorized Model Inference UDF
  └─ Derived Congestion Classification (LOW, MODERATE, HIGH, SEVERE)
            │
            ├──► Kafka Topic: traffic.predictions
            └──► Storage Sink: recent_predictions.jsonl / Parquet
                        │
                        ▼
            [ Streamlit Analytics Dashboard ]
```

---

## 5. Technology Stack & Design Decisions

| Component | Technology | Version | Rationale |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11.15 | Optimal compatibility with PySpark 3.5.1 and modern ML libraries. |
| **Package Manager**| `uv` | 0.10.9 | Extremely fast virtual environment and dependency management. |
| **Message Broker** | Apache Kafka | 3.7.0 (KRaft) | High-throughput distributed event log; KRaft mode eliminates ZooKeeper dependency. |
| **Stream Engine** | Apache Spark | 3.5.1 | Industry-standard distributed processing with event-time micro-batching and fault-tolerant checkpointing. |
| **Machine Learning**| Scikit-learn | 1.5.2 | Non-parametric Random Forest Regressor handles non-linear trends with high explainability. |
| **Data Engine** | Pandas & PyArrow | 2.2.3 / 17.0 | Tabular feature preparation and high-performance Parquet serialization. |
| **Dashboard** | Streamlit | 1.36.0 | Reactive, beginner-friendly web UI with automatic stream polling. |
| **Mapping & Plots**| Plotly | 7.1.0 | Interactive Mapbox sensor maps and residual distribution charts. |
| **Unit Testing** | pytest | 9.1.1 | Automated verification of schemas, preprocessing, and model inference. |

---

## 6. Dataset Source & Licensing

- **Dataset:** METR-LA (Traffic Speed Benchmark)
- **Source:** Caltrans Performance Measurement System (PeMS), Los Angeles County, California.
- **Physical Detectors:** 207 inductive loop detectors across key freeways (I-5, I-10, I-405, US-101, SR-110).
- **Time Window:** March 1, 2012 to June 27, 2012 (34,272 time steps at 5-minute sampling intervals).
- **Total Records:** 7,094,304 physical speed observations.
- **License:** Open California Department of Transportation (Caltrans) research benchmark.

---

## 7. Data Schemas

### Ingestion Schema (`traffic.raw`)
```json
{
  "timestamp": "2012-06-10T08:30:00",
  "sensor_id": "773869",
  "current_speed": 62.45,
  "speed_lag_1": 63.10,
  "speed_lag_2": 64.50,
  "speed_lag_3": 65.00,
  "speed_mean_15m": 63.35,
  "speed_mean_1h": 61.20,
  "hour_of_day": 8,
  "day_of_week": 6,
  "is_weekend": 1
}
```

### Prediction Output Schema (`traffic.predictions`)
```json
{
  "timestamp": "2012-06-10T08:30:00",
  "sensor_id": "773869",
  "observed_speed": 62.45,
  "predicted_speed": 62.10,
  "congestion_level": "LOW",
  "prediction_timestamp": "2026-10-22T13:31:54.474524",
  "model_name": "random_forest_v1",
  "forecast_horizon_minutes": 5
}
```

---

## 8. Installation & Prerequisites

### Prerequisites
- Windows 11 / Linux (Ubuntu) / macOS
- Java 8 JRE/JDK (pre-configured to `C:\Program Files\Java\jre1.8.0_461`)
- Docker Desktop (for Apache Kafka broker and Kafka UI)
- `uv` Python package manager

### Environment Setup

```powershell
# 1. Clone repository and navigate to project folder
cd C:\Users\rohit\.gemini\antigravity\scratch\traffic-bda-project

# 2. Create isolated Python 3.11 virtual environment using uv
uv venv .venv --python 3.11

# 3. Install pinned dependencies
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

---

## 9. Step-by-Step Execution Workflow

### Step 1: Download & Inspect Dataset
```powershell
# Download official METR-LA HDF5 dataset and sensor coordinates
.\.venv\Scripts\python.exe -m src.data.download_dataset

# Inspect dataset dimensions, sampling frequency, and missing values
.\.venv\Scripts\python.exe -m src.data.inspect_dataset
```

### Step 2: Feature Engineering & Preprocessing
```powershell
# Preprocess speeds, compute causal lag features, and split chronologically
.\.venv\Scripts\python.exe -m src.data.prepare_dataset --num-sensors 25
```

### Step 3: Train & Benchmark Prediction Model
```powershell
# Train Persistence Baseline and Random Forest Regressor
.\.venv\Scripts\python.exe -m src.training.train_model

# Evaluate models on the held-out test split (June 10 to June 27, 2012)
.\.venv\Scripts\python.exe -m src.training.evaluate_model
```

### Step 4: Run Automated Tests
```powershell
# Execute pytest suite (9 tests covering preprocessing, features, schemas, models)
.\.venv\Scripts\pytest.exe tests/ -v
```

### Step 5: Start Infrastructure & Streaming Pipeline

**Terminal 1 — Launch Local Kafka Broker (Optional, via Docker):**
```powershell
docker compose up -d
# Kafka UI available at http://localhost:8080
```

**Terminal 2 — Start Spark Structured Streaming Job:**
```powershell
# Ingests events, performs model inference, and writes to prediction sink
.\.venv\Scripts\python.exe -m src.streaming.spark_streaming_job
```

**Terminal 3 — Replay Simulated Traffic Stream:**
```powershell
# Publishes 500 events at 20 records/second to Kafka (or offline buffer)
.\.venv\Scripts\python.exe -m src.ingestion.kafka_producer --limit 500 --rate 20
```

**Terminal 4 — Launch Analytics Dashboard:**
```powershell
# Launch the Streamlit web dashboard
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 10. Empirical Evaluation Results

Evaluated on **128,425 held-out chronological test observations** (June 10 to June 27, 2012):

| Metric | Persistence Baseline | Random Forest Regressor | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 2.442 mph | **2.126 mph** | **▼ 12.9% reduction** |
| **Root Mean Squared Error (RMSE)**| 4.144 mph | **3.673 mph** | **▼ 11.4% reduction** |
| **Mean Absolute Percentage Error (MAPE)**| 5.54% | **5.10%** | **▼ 0.44% lower error** |
| **Coefficient of Determination ($R^2$)**| 0.9055 | **0.9258** | **▲ +0.0203 variance** |

---

## 11. Known Limitations & Academic Disclaimers

1. **Simulated Real-Time Replay:** Historical data is replayed sequentially through Kafka. While it accurately simulates streaming ingestion mechanics and micro-batch processing, it cannot dynamically alter physical traffic light cycles or road conditions.
2. **Freeway vs. Arterial Traffic:** METR-LA measures controlled-access freeways. These patterns do not generalize directly to mixed-vehicle urban city streets in developing countries without local calibration.
3. **Sensor Downtime:** PeMS records sensor outages as `0.0 mph`. In real production systems, hardware telemetry flags must distinguish zero-speed traffic jams from hardware sensor dropouts.

---

## 12. Project Directory Structure

```
traffic-bda-project/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- docker-compose.yml
|-- .env.example
|-- .gitignore
|
|-- config/
|   `-- project.yaml
|
|-- data/
|   |-- raw/
|   |   |-- metr-la.h5
|   |   `-- graph_sensor_locations.csv
|   |-- processed/
|   |   |-- train_features.parquet
|   |   |-- val_features.parquet
|   |   |-- test_features.parquet
|   |   `-- live_predictions/
|   `-- samples/
|
|-- models/
|   |-- random_forest_traffic.joblib
|   `-- model_metadata.json
|
|-- src/
|   |-- common/
|   |   |-- schemas.py
|   |   |-- config.py
|   |   `-- logging_config.py
|   |-- data/
|   |   |-- download_dataset.py
|   |   |-- inspect_dataset.py
|   |   |-- prepare_dataset.py
|   |   `-- build_features.py
|   |-- training/
|   |   |-- train_model.py
|   |   `-- evaluate_model.py
|   |-- ingestion/
|   |   `-- kafka_producer.py
|   |-- streaming/
|   |   `-- spark_streaming_job.py
|   `-- dashboard/
|       `-- app.py
|
|-- tests/
|   |-- test_preprocessing.py
|   |-- test_features.py
|   |-- test_schema_validation.py
|   `-- test_prediction.py
|
|-- scripts/
|   |-- run_pipeline.ps1
|   |-- start_kafka.ps1
|   |-- start_services.sh
|   `-- run_pipeline.sh
|
|-- outputs/
|   |-- figures/
|   `-- reports/
|
`-- docs/
    |-- architecture.md
    |-- data_dictionary.md
    |-- experiment_protocol.md
    |-- troubleshooting.md
    `-- viva_questions.md
```
