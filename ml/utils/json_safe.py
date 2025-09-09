"""
JSON-safe serialization utilities for numpy/pandas types
"""
import numpy as np
import pandas as pd
import datetime as dt
from typing import Any


def json_safe(o: Any) -> Any:
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    
    Args:
        o: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(o, (np.generic,)):  # np.float64, np.int64, etc.
        return o.item()
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Series):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict('records')
    return str(o)
