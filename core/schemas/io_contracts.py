from __future__ import annotations
from pydantic import BaseModel
import pandas as pd
import numpy as np

class PriceFrame(BaseModel):
    @staticmethod
    def validate_df(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        if df.index.has_duplicates:
            raise ValueError("Index contains duplicates")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Index must be monotonic increasing")
        required = ["Close"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")
        numeric_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        for c in numeric_cols:
            if not np.isfinite(df[c]).all():
                raise ValueError(f"Non-finite values in {c}")


