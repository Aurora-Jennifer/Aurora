"""
Price Positivity Rule for DataSanity
Validates that price data contains only positive values.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class PricePositivityRuleConfig:
    allow_negative_prices: bool = False
    allow_zero_prices: bool = False  # new: zero leads to -inf log returns


class PricePositivityRule:
    def __init__(self, cfg: PricePositivityRuleConfig, price_cols: tuple[str, ...] = ("Open", "High", "Low", "Close")):
        self.cfg = cfg
        self.price_cols = price_cols

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that all price columns contain only positive values."""
        bad = {}
        for col in self.price_cols:
            if col in df.columns:
                if not self.cfg.allow_zero_prices:
                    mask = df[col] <= 0
                elif not self.cfg.allow_negative_prices:
                    mask = df[col] < 0
                else:
                    mask = df[col] < float("-inf")  # never true, keeps structure

                if mask.any():
                    bad[col] = df.loc[mask, col].index.tolist()

        if bad:
            # Raise with first few offenders for debuggability
            details = ", ".join(f"{col}={len(idxs)}" for col, idxs in bad.items())
            raise ValueError(f"DataSanity: non-positive prices detected ({details}). Rejecting batch.")

        return df

    def validate_and_repair(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Validate and optionally repair by dropping bad rows."""
        try:
            return self.validate(df), []
        except ValueError:
            if self.cfg.allow_negative_prices and self.cfg.allow_zero_prices:
                # If both are allowed, this shouldn't happen
                raise

            # Drop rows with non-positive prices
            mask = pd.Series(True, index=df.index)
            for col in self.price_cols:
                if col in df.columns:
                    if not self.cfg.allow_zero_prices:
                        mask &= df[col] > 0
                    elif not self.cfg.allow_negative_prices:
                        mask &= df[col] >= 0

            cleaned_df = df.loc[mask].copy()
            dropped_count = len(df) - len(cleaned_df)

            if dropped_count > 0:
                return cleaned_df, [f"dropped_{dropped_count}_non_positive_prices"]
            # Shouldn't happen, but just in case
            raise ValueError("Failed to repair non-positive prices")


def create_price_positivity_rule(config: dict) -> PricePositivityRule:
    """Factory function to create PricePositivityRule from config dict."""
    cfg = PricePositivityRuleConfig(
        allow_negative_prices=config.get("allow_negative_prices", False),
        allow_zero_prices=config.get("allow_zero_prices", False)
    )
    price_cols = config.get("price_cols", ("Open", "High", "Low", "Close"))
    return PricePositivityRule(cfg, price_cols)
