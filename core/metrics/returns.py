from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def percent_returns(close: pd.Series) -> pd.Series:
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    r = close.pct_change()
    return r.fillna(0.0)


def log_returns(close: pd.Series) -> pd.Series:
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    r = np.log(close / close.shift(1))
    return r.fillna(0.0)


def diff_returns(close: pd.Series) -> pd.Series:
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    r = close.diff()
    return r.fillna(0.0)


def get_returns(close: pd.Series, kind: Literal["percent", "log", "diff"] = "percent") -> pd.Series:
    # Structured one-time log for shape and selection
    try:
        logger.debug(
            "returns_selection",
            extra={"kind": kind, "n": int(len(close) if close is not None else 0)},
        )
    except Exception:
        pass
    if kind == "log":
        return log_returns(close)
    if kind == "diff":
        return diff_returns(close)
    return percent_returns(close)
