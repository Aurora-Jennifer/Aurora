"""
Finite Numbers Rule for DataSanity
Validates that numeric columns contain only finite values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FiniteNumbersRuleConfig:
    allow_nan: bool = False
    allow_inf: bool = False
    allow_complex: bool = False


class FiniteNumbersRule:
    def __init__(self, cfg: FiniteNumbersRuleConfig, cols: tuple[str, ...] = None):
        self.cfg = cfg
        self.cols = cols

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that specified columns contain only finite values."""
        if self.cols is None:
            # Check all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in self.cols if col in df.columns]

        bad_cols = []
        total_bad = 0

        for col in numeric_cols:
            series = df[col]

            # Check for NaN
            if not self.cfg.allow_nan and series.isna().any():
                bad_cols.append(f"{col}(NaN)")
                total_bad += series.isna().sum()

            # Check for infinite values
            if not self.cfg.allow_inf and np.isinf(series).any():
                bad_cols.append(f"{col}(inf)")
                total_bad += np.isinf(series).sum()

            # Check for complex numbers
            if not self.cfg.allow_complex and np.iscomplexobj(series):
                bad_cols.append(f"{col}(complex)")
                total_bad += len(series)

        if bad_cols:
            details = ", ".join(bad_cols)
            raise ValueError(
                f"DataSanity: non-finite values detected ({details}), "
                f"total_bad={total_bad}"
            )

        return df

    def validate_and_repair(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Validate and optionally repair non-finite values."""
        try:
            return self.validate(df), []
        except ValueError:
            # For now, just raise the error
            # In the future, we could implement repair strategies
            raise


def create_finite_numbers_rule(config: dict) -> FiniteNumbersRule:
    """Factory function to create FiniteNumbersRule from config dict."""
    cfg = FiniteNumbersRuleConfig(
        allow_nan=config.get("allow_nan", False),
        allow_inf=config.get("allow_inf", False),
        allow_complex=config.get("allow_complex", False)
    )
    cols = config.get("cols")
    return FiniteNumbersRule(cfg, cols)
