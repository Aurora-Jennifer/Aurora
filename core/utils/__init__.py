# Core utilities for the trading system
import logging
import os

def setup_logging(
    log_file: str = "trading.log",
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Use default format if none provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)

def ensure_directories(*dirs):
    """Ensure directories exist, creating them if necessary"""
    import os
    for directory in dirs:
        if directory:
            os.makedirs(directory, exist_ok=True)

__all__ = ['setup_logging', 'ensure_directories']
