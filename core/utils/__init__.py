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

def _safe_len(x) -> int:
    """
    Safely get the length of an object, handling None and objects without __len__.
    
    Args:
        x: Object to get length of
        
    Returns:
        Length of object, or 0 if object is None or has no length
    """
    if x is None:
        return 0
    if hasattr(x, "__len__"):
        return len(x)
    return 0

def _last(x):
    """
    Safely get the last element of a sequence, handling empty sequences.
    
    Args:
        x: Sequence to get last element from
        
    Returns:
        Last element, or None if sequence is empty or None
    """
    if x is None or _safe_len(x) == 0:
        return None
    if hasattr(x, "iloc"):
        return x.iloc[-1]
    return x[-1]

__all__ = ['setup_logging', 'ensure_directories', '_safe_len', '_last']
