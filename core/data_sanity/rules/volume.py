"""
Volume validation rule for DataSanity staged pipeline.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class VolumeRuleConfig:
    """Configuration for volume validation rule."""
    allow_negative_volume: bool = False
    allow_zero_volume: bool = False
    max_volume: float = 1e12  # 1T shares
    repair_mode: str = "warn"  # warn, drop, fail, clip


class VolumeRule:
    """Volume validation rule for staged pipeline."""

    def __init__(self, config: VolumeRuleConfig):
        self.config = config

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate volume data without repairs."""
        volume_col = self._get_volume_column(df)
        if volume_col not in df.columns:
            return df

        volume = df[volume_col]

        # Check for negative volume
        if not self.config.allow_negative_volume:
            negative_volume = volume < 0
            if negative_volume.any():
                raise ValueError(f"Negative volume values detected: {negative_volume.sum()} rows")

        # Check for zero volume
        if not self.config.allow_zero_volume:
            zero_volume = volume == 0
            if zero_volume.any():
                raise ValueError(f"Zero volume values detected: {zero_volume.sum()} rows")

        # Check for excessive volume
        excessive_volume = volume > self.config.max_volume
        if excessive_volume.any():
            raise ValueError(f"Excessive volume values detected: {excessive_volume.sum()} rows > {self.config.max_volume}")

        return df

    def validate_and_repair(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Validate and repair volume data."""
        repairs = []
        volume_col = self._get_volume_column(df)

        if volume_col not in df.columns:
            return df, repairs

        volume = df[volume_col]

        # Handle negative volume
        if not self.config.allow_negative_volume:
            negative_volume = volume < 0
            if negative_volume.any():
                if self.config.repair_mode == "fail":
                    raise ValueError(f"Negative volume values detected: {negative_volume.sum()} rows")
                if self.config.repair_mode == "drop":
                    df = df.loc[~negative_volume]
                    repairs.append("dropped_negative_volume")
                else:  # clip or warn
                    df.loc[negative_volume, volume_col] = volume[negative_volume].abs()
                    repairs.append("made_negative_volume_positive")

        # Handle zero volume
        if not self.config.allow_zero_volume:
            zero_volume = volume == 0
            if zero_volume.any():
                if self.config.repair_mode == "fail":
                    raise ValueError(f"Zero volume values detected: {zero_volume.sum()} rows")
                if self.config.repair_mode == "drop":
                    df = df.loc[~zero_volume]
                    repairs.append("dropped_zero_volume")
                else:  # clip or warn
                    # Replace with median volume
                    median_volume = volume[volume > 0].median()
                    if pd.isna(median_volume):
                        median_volume = 1000000  # Default
                    df.loc[zero_volume, volume_col] = median_volume
                    repairs.append("replaced_zero_volume_with_median")

        # Handle excessive volume
        excessive_volume = volume > self.config.max_volume
        if excessive_volume.any():
            if self.config.repair_mode == "fail":
                raise ValueError(f"Excessive volume values detected: {excessive_volume.sum()} rows > {self.config.max_volume}")
            if self.config.repair_mode == "drop":
                df = df.loc[~excessive_volume]
                repairs.append("dropped_excessive_volume")
            else:  # clip or warn
                df.loc[excessive_volume, volume_col] = self.config.max_volume
                repairs.append("capped_excessive_volume")

        return df, repairs

    def _get_volume_column(self, df: pd.DataFrame) -> str:
        """Get the volume column name."""
        volume_cols = ["Volume", "volume", "VOLUME"]
        for col in volume_cols:
            if col in df.columns:
                return col
        return "Volume"  # Default


def create_volume_rule(config: dict) -> VolumeRule:
    """Create a volume validation rule from config."""
    rule_config = VolumeRuleConfig(
        allow_negative_volume=config.get("allow_negative_volume", False),
        allow_zero_volume=config.get("allow_zero_volume", False),
        max_volume=config.get("max_volume", 1e12),
        repair_mode=config.get("repair_mode", "warn")
    )
    return VolumeRule(rule_config)
