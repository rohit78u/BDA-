"""
Central configuration loader and environment manager.
Loads config/project.yaml and configures Java/PySpark environment paths,
including HADOOP_HOME and winutils for Windows execution.
"""
from pathlib import Path
import os
import sys
import yaml
from dotenv import load_dotenv

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

def _setup_runtime_environment():
    """Ensure Java, Hadoop, and Python paths are configured for PySpark execution."""
    # Ensure PYSPARK_PYTHON points to the current Python executable
    if "PYSPARK_PYTHON" not in os.environ:
        os.environ["PYSPARK_PYTHON"] = sys.executable
    if "PYSPARK_DRIVER_PYTHON" not in os.environ:
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Configure HADOOP_HOME for Windows winutils and native I/O
    hadoop_dir = PROJECT_ROOT / "hadoop"
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        hadoop_bin = str(hadoop_dir / "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

    # Check JAVA_HOME
    java_home = os.environ.get("JAVA_HOME")
    if not java_home or not Path(java_home).exists():
        candidate_jre = Path(r"C:\Program Files\Java\jre1.8.0_461")
        if candidate_jre.exists():
            os.environ["JAVA_HOME"] = str(candidate_jre)
            bin_dir = str(candidate_jre / "bin")
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

_setup_runtime_environment()

def load_config(config_path: str = None) -> dict:
    """Load project YAML configuration with resolved absolute paths."""
    if config_path is None:
        config_path = os.environ.get("PROJECT_CONFIG_PATH", "config/project.yaml")
    
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

# Global config instance for convenience
config = load_config()
