"""
OHLC Consistency Rule for DataSanity
Validates that OHLC data follows proper relationships.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class OHLCConsistencyRuleConfig:
    strict: bool = True  # If False, allow minor violations


class OHLCConsistencyRule:
    def __init__(self, cfg: OHLCConsistencyRuleConfig, ohlc_cols: Tuple[str, ...] = ("Open", "High", "Low", "Close")):
        self.cfg = cfg
        self.ohlc_cols = ohlc_cols

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC consistency relationships."""
        required = set(self.ohlc_cols) & set(df.columns)
        
        if len(required) != 4:
            # Not all OHLC columns present, skip validation
            return df
        
        open_col, high_col, low_col, close_col = self.ohlc_cols
        
        # Check OHLC relationships
        bad = (
            (df[low_col] > df[[open_col, close_col]].min(axis=1)) |  # Low > min(Open, Close)
            (df[high_col] < df[[open_col, close_col]].max(axis=1)) |  # High < max(Open, Close)
            (df[low_col] > df[high_col])  # Low > High
        )
        
        if bad.any():
            bad_count = int(bad.sum())
            bad_indices = df.index[bad].tolist()[:5]  # First 5 for debugging
            raise ValueError(
                f"DataSanity: OHLC inconsistent rows={bad_count} "
                f"(indices: {bad_indices})"
            )
        
        return df

    def validate_and_repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and optionally repair OHLC inconsistencies."""
        try:
            return self.validate(df), []
        except ValueError as e:
            if not self.cfg.strict:
                # In non-strict mode, we could attempt repairs
                # For now, just raise the error
                raise
            else:
                raise


def create_ohlc_consistency_rule(config: dict) -> OHLCConsistencyRule:
    """Factory function to create OHLCConsistencyRule from config dict."""
    cfg = OHLCConsistencyRuleConfig(
        strict=config.get("strict", True)
    )
    ohlc_cols = config.get("cols", ("Open", "High", "Low", "Close"))
    return OHLCConsistencyRule(cfg, ohlc_cols)
