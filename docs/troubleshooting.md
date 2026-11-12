# Troubleshooting & Operational Guide

This document catalogs technical issues encountered when running Apache Kafka and Apache Spark on Windows development environments, along with verified diagnostic steps and resolutions.

---

## 1. Spark on Windows: `UnsatisfiedLinkError: NativeIO$Windows.access0`

### Symptom:
```
java.lang.UnsatisfiedLinkError: org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(Ljava/lang/String;I)Z
at org.apache.hadoop.fs.RawLocalFileSystem.listStatus(RawLocalFileSystem.java:601)
```

### Cause:
Apache Spark relies on Apache Hadoop client libraries for file system operations. On Windows, Hadoop requires native C++ helper binaries (`winutils.exe` and `hadoop.dll`) to interface with Windows NTFS file permissions.

### Resolution:
1. The project includes verified Hadoop 3.3.5 binaries inside `hadoop/bin/` (`hadoop.dll` and `winutils.exe`).
2. `src/common/config.py` automatically sets `HADOOP_HOME` to the `hadoop/` directory and prepends `hadoop/bin/` to the Windows `PATH`.
3. To manually set in PowerShell:
   ```powershell
   $env:HADOOP_HOME = "$PWD\hadoop"
   $env:PATH = "$env:HADOOP_HOME\bin;" + $env:PATH
   ```

---

## 2. Invalid Spark URL / Hostname with Underscore (`my_pc`)

### Symptom:
```
org.apache.spark.SparkException: Invalid Spark URL: spark://HeartbeatReceiver@my_pc:61102
```

### Cause:
RFC 2396 and Java `URI` parsing strictly forbid underscores (`_`) in hostnames. When a Windows computer name contains an underscore (such as `my_pc`), Spark driver RPC initialization fails.

### Resolution:
Bind the Spark driver explicitly to IPv4 localhost:
```python
builder.config("spark.driver.host", "127.0.0.1") \
       .config("spark.driver.bindAddress", "127.0.0.1")
```
This is configured automatically in `src/streaming/spark_streaming_job.py`.

---

## 3. Java 24 vs Java 8/17 SecurityManager Exception

### Symptom:
```
java.lang.UnsupportedOperationException: getSubject is not supported
at java.base/javax.security.auth.Subject.getSubject(Subject.java:277)
```

### Cause:
Java 24 completely removed the legacy Java SecurityManager and deprecated `Subject.getSubject()`, which Hadoop 3.3.4 calls. Spark 3.5 requires Java 8, 11, or 17.

### Resolution:
The system's installed Java 8 JRE (`C:\Program Files\Java\jre1.8.0_461`) is configured automatically in `src/common/config.py`:
```python
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jre1.8.0_461"
```

---

## 4. Docker Desktop Daemon Not Responding

### Symptom:
```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

### Cause:
Docker Desktop application is installed but the background engine service is not active.

### Resolution:
1. Open **Docker Desktop** from the Windows Start Menu or taskbar.
2. Wait 30 seconds until the engine icon in the system tray turns steady green.
3. Verify in PowerShell:
   ```powershell
   docker ps
   docker compose up -d
   ```
4. **Fallback Option:** If Docker Desktop is unavailable, both `kafka_producer.py` and `spark_streaming_job.py` include automatic offline simulation fallback modes (`--offline-buffer` and `--source file`) to allow full demonstration and testing without Docker!

---

## 5. PyArrow and NumPy 1.26 Compatibility

### Symptom:
```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

### Cause:
PyArrow 26+ was compiled against NumPy 2.0 ABI, which is incompatible with PySpark 3.5's requirement for NumPy 1.x.

### Resolution:
PyArrow is pinned to `pyarrow>=16.0.0,<18.0.0` (installed `pyarrow==17.0.0`), which has full dual compatibility with NumPy 1.26.4 and pandas 2.2.
