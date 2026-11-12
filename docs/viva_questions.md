# B.Tech Project Viva Examination Preparation Guide

## 25+ Essential Viva Questions & Comprehensive Technical Answers

---

### Part 1: Big Data & Architecture Fundamentals

#### Q1. What does "Big Data" mean in the context of this traffic project?
**Answer:** Big Data in this project is characterized by:
- **Volume:** The METR-LA dataset contains over 7 million physical sensor observations across 207 highway stations over 4 months. In production, thousands of city sensors generate gigabytes of daily telemetry.
- **Velocity:** Traffic sensors report continuous telemetry every few minutes. The pipeline must ingest, validate, aggregate, and infer predictions with low latency (under 5 seconds) before traffic conditions change.
- **Variety:** Data arrives as semi-structured JSON streaming events with temporal metadata, continuous speeds, and categorical sensor spatial IDs.
- **Value:** Transforming raw speed telemetry into actionable 5-minute ahead forecasts allows navigation engines, traffic authorities, and commuters to proactively mitigate congestion bottlenecks.

#### Q2. Why did you choose Apache Kafka instead of RabbitMQ or a traditional REST API?
**Answer:**
- **Persistent Distributed Commit Log:** Unlike traditional message brokers (like RabbitMQ) that delete messages once acknowledged, Kafka retains immutable message logs on disk with configurable retention. This allows multiple downstream consumers (Spark, archiving, analytics) to replay data independently at their own pace.
- **High Throughput & Partitioning:** Kafka achieves hundreds of thousands of messages per second by utilizing sequential disk I/O, OS page caching, and zero-copy network transfer (`sendfile`). Partitions enable horizontal scaling.
- **Decoupling:** Decouples sensor ingestion rate from Spark stream processing rate, acting as a shock absorber during peak traffic surges.

#### Q3. What is Kafka KRaft mode, and why is it preferred over ZooKeeper?
**Answer:**
- Prior to Kafka 3.0, Kafka required a separate Apache ZooKeeper ensemble to manage cluster metadata, broker registration, and topic partitions.
- **KRaft (Kafka Raft Metadata)** integrates consensus directly inside Kafka using an event-driven Raft protocol.
- **Benefits:** Eliminates ZooKeeper operational complexity, reduces memory and CPU overhead, speeds up controller recovery from minutes to milliseconds, and allows unified security management.

#### Q4. Why did you use Apache Spark Structured Streaming rather than legacy Spark DStreams or simple Python consumers?
**Answer:**
- **Event-Time Processing & Watermarking:** Structured Streaming treats streams as unbounded append tables with native support for event-time timestamps rather than processing arrival time, handling out-of-order and late-arriving events.
- **Catalyst Optimizer & Tungsten Engine:** Optimizes streaming SQL queries, generates efficient JVM bytecode, and minimizes garbage collection overhead.
- **Fault-Tolerant Checkpointing:** Maintains offset checkpoints and state metadata to provide transactional *at-least-once* or *exactly-once* end-to-end processing guarantees.
- **Unified API:** Uses the exact same DataFrame/Dataset API as batch processing, eliminating dual codebase maintenance.

#### Q5. What is the difference between Event Time and Processing Time?
**Answer:**
- **Event Time:** The timestamp embedded inside the record when the physical traffic event actually occurred at the roadside sensor (e.g., `2012-06-10T08:30:00`).
- **Processing Time:** The local clock time of the Spark executor machine when the event is being processed (e.g., `2026-10-22T13:31:54`).
- Real-time traffic analysis **must** rely on Event Time because network jitter or communication delays could cause night readings to be processed during morning rush hour if processing time were used.

---

### Part 2: Machine Learning & Feature Engineering

#### Q6. What is your primary prediction target and forecast horizon?
**Answer:**
- **Target:** Continuous vehicle speed in miles per hour (mph) at the next interval ($t+1$).
- **Forecast Horizon:** **5 minutes ahead**, corresponding to the discrete sampling interval of Caltrans PeMS sensors.
- Speed is chosen as the primary target because it is an objective, continuous physical measurement. Categorical congestion levels (LOW, MODERATE, HIGH, SEVERE) are derived from the predicted speed.

#### Q7. Why did you compare against a Persistence Baseline?
**Answer:**
- The **Persistence Baseline** ($\hat{y}_{t+1} = y_t$) predicts that traffic speed 5 minutes from now will be identical to current speed.
- In traffic engineering, persistence is a notoriously strong baseline because vehicle speeds on freeways do not drop instantaneously; physical inertia and gradual congestion buildup create high autocorrelation over 5-minute windows.
- Any legitimate machine learning model **must beat the persistence baseline** to justify its computational complexity.

#### Q8. Why did you use a Chronological Split instead of random $k$-fold cross-validation?
**Answer:**
- In time-series data, observations are serially correlated.
- If you use standard random cross-validation, observations from time $t+1$ and $t+2$ will leak into the training set, while evaluating on time $t$.
- This creates **data leakage** and produces deceptively optimistic error metrics that fail completely in real-world deployment.
- A **chronological split** (Train: first 70%, Val: next 15%, Test: last 15%) strictly respects temporal causality—the model is trained only on the past and tested on the unseen future.

#### Q9. Why did you choose a Random Forest Regressor over deep Graph Neural Networks (GNNs)?
**Answer:**
- **Explainability & Student Accessibility:** Random Forest provides explicit feature importances (e.g., current speed, 15-min rolling mean) that can be easily inspected, explained, and validated during an examination.
- **Execution Efficiency:** Does not require expensive GPUs, gigabytes of CUDA libraries, or hours of training time. It trains in under 2 minutes on CPU.
- **Robustness:** Decision tree ensembles are non-parametric, handle non-linear speed relationships naturally, and are resilient to feature scaling issues and outliers.
- While spatial GNNs (like DCRNN or STGCN) model road topology, their high training complexity makes them impractical for lightweight local demonstration without providing proportionally higher 5-minute forecast accuracy.

#### Q10. What features were engineered and why?
**Answer:**
- `current_speed`: The latest observed speed at $t$.
- `speed_lag_1`, `speed_lag_2`, `speed_lag_3`: Speeds at $t-1$ (5 min ago), $t-2$ (10 min ago), and $t-3$ (15 min ago) to capture acceleration/deceleration trends.
- `speed_mean_15m` and `speed_mean_1h`: Short-term and medium-term rolling averages to smooth temporary detector noise.
- `hour_of_day`, `day_of_week`, `is_weekend`: Calendar features to capture diurnal commute rhythms and weekend vs. weekday traffic profiles.
- `historical_mean_speed`: Long-term expected speed for that specific sensor on that day and hour, calculated **strictly from training data** to avoid leakage.

---

### Part 3: Model Evaluation & Metrics

#### Q11. What metrics did you use to evaluate your model, and what were the actual results?
**Answer:**
Evaluated on **128,425 held-out chronological test observations**:
1. **MAE (Mean Absolute Error):**
   - Baseline: `2.442 mph` | Random Forest: `2.126 mph` (**▼ 12.9% improvement**)
2. **RMSE (Root Mean Squared Error):**
   - Baseline: `4.144 mph` | Random Forest: `3.673 mph` (**▼ 11.4% improvement**)
3. **MAPE (Mean Absolute Percentage Error):**
   - Baseline: `5.54%` | Random Forest: `5.10%` (**▼ 0.44% lower relative error**)
4. **$R^2$ (Coefficient of Determination):**
   - Baseline: `0.9055` | Random Forest: `0.9258` (**▲ +0.0203 variance explained**)

#### Q12. What is the difference between MAE and RMSE, and why report both?
**Answer:**
- **MAE** treats all errors linearly ($\frac{1}{n}\sum |y - \hat{y}|$), providing an intuitive average error in miles per hour that represents typical performance.
- **RMSE** squares errors before taking the root ($\sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$), heavily penalizing large outliers.
- In traffic management, missing a sudden congestion collapse by 25 mph is much more severe than having a 2 mph error during free flow. Reporting both reveals whether a model avoids large catastrophic forecasting errors.

#### Q13. Why must MAPE be calculated with caution on traffic datasets?
**Answer:**
- $\text{MAPE} = \frac{100\%}{n} \sum \left| \frac{y - \hat{y}}{y} \right|$.
- When traffic is in severe gridlock ($y \approx 0$ or $y < 5$ mph), the denominator approaches zero, causing MAPE to explode towards infinity even for tiny 2 mph absolute errors.
- Therefore, MAPE was evaluated specifically on non-zero valid speeds ($y > 5$ mph) to maintain mathematical validity.

#### Q14. Where does your model provide the largest advantage over the baseline?
**Answer:**
- During **Free Flow (> 50 mph)**, persistence is already very accurate (MAE 1.83 mph), and Random Forest improves slightly to 1.67 mph.
- During **Severe Gridlock ($\le$ 20 mph)**, persistence fails significantly (MAE 7.92 mph), whereas Random Forest achieves **6.10 mph (a 1.81 mph error reduction)**.
- Random Forest anticipates congestion onset using lag trends, whereas persistence only reacts after speed has already crashed.

---

### Part 4: Practical & Urban Traffic Realities

#### Q15. Can these results from Los Angeles freeways generalize directly to Indian urban traffic?
**Answer:**
**No, they cannot directly generalize without adaptation.**
1. **Homogeneous vs. Heterogeneous Traffic:** METR-LA measures controlled-access freeways with homogeneous vehicle types (cars, trucks) adhering to lane discipline. Indian urban roads feature highly mixed traffic (two-wheelers, auto-rickshaws, buses, pedestrians) with non-lane-based flow.
2. **Sensor Infrastructure:** PeMS uses inductive loop detectors embedded in highway asphalt. In India, video cameras, toll FASTag sensors, and GPS vehicle tracking (e.g. city buses or ride-hailing probes) are more prevalent.
3. **Speed Dynamics:** US freeway speeds range from 20 to 70 mph; Indian city streets typically operate between 10 and 40 km/h with frequent signalized interruptions, jaywalking, and road obstructions.

#### Q16. What are the limitations of replaying historical data through Kafka?
**Answer:**
- **Simulated Real-Time:** Replaying historical records mimics streaming ingestion mechanics (Kafka queues, Spark micro-batches, real-time deserialization), but the pipeline cannot influence the physical world or receive dynamic downstream feedback.
- **Missing Value Artifacts:** The PeMS dataset records sensor downtime as `0.0 mph`, requiring causal forward-fill preprocessing rather than true real-time sensor diagnostics.

#### Q17. How does the dashboard prevent unnecessary pipeline re-execution on refresh?
**Answer:**
- Streamlit utilizes `@st.cache_data` decorators on static assets (sensor coordinates, dataset inspection metadata, offline evaluation reports).
- The dashboard only re-reads the latest appended lines from `recent_predictions.jsonl` upon auto-refresh, completely separating the visualization loop from the Spark/Kafka processing engine.
