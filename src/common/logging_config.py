"""
Standardized logging configuration for the traffic streaming project.
"""
import logging
import sys

def setup_logger(name: str = "traffic_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures a clear, formatted console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if re-called
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
