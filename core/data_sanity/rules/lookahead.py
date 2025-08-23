"""
Lookahead Contamination Rule for DataSanity
Detects future information leakage in data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ..lookahead_detector import detect_lookahead_with_context


@dataclass
class LookaheadContaminationRuleConfig:
    fail: bool = True  # If True, raise error; if False, just flag
    eps_zero_return: float = 0.0  # Epsilon for considering returns "zero"
    min_zero_run: int = 2  # Minimum consecutive zero returns to consider a "stable run"
    max_suspicious_rate: float = 0.001  # Maximum allowed suspicious match rate


class LookaheadContaminationRule:
    def __init__(self, cfg: LookaheadContaminationRuleConfig):
        self.cfg = cfg

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect potential lookahead contamination using improved algorithm."""
        if "Returns" in df.columns and len(df) > 1:
            returns = df["Returns"]
            close_prices = df.get("Close", None)
            
            # Use the improved lookahead detector
            result = detect_lookahead_with_context(
                returns=returns,
                close_prices=close_prices,
                eps=self.cfg.eps_zero_return,
                min_run=self.cfg.min_zero_run
            )
            
            # Check if the suspicious rate exceeds our threshold
            if result["suspicious_match_rate"] > self.cfg.max_suspicious_rate:
                if self.cfg.fail:
                    raise ValueError(
                        f"DataSanity: lookahead contamination detected "
                        f"(suspicious rate: {result['suspicious_match_rate']:.4f}, "
                        f"threshold: {self.cfg.max_suspicious_rate:.4f}, "
                        f"suspicious: {result['n_suspicious']}/{result['n_total']})"
                    )
                else:
                    # Just flag it
                    return df
        
        return df

    def validate_and_repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and optionally repair lookahead contamination."""
        try:
            return self.validate(df), []
        except ValueError as e:
            if self.cfg.fail:
                raise
            else:
                # In non-fail mode, we could attempt repairs
                # For now, just return the data as-is with a flag
                return df, ["lookahead_contamination_detected"]


def create_lookahead_contamination_rule(config: dict) -> LookaheadContaminationRule:
    """Factory function to create LookaheadContaminationRule from config dict."""
    cfg = LookaheadContaminationRuleConfig(
        fail=config.get("fail", True),
        eps_zero_return=config.get("eps_zero_return", 0.0),
        min_zero_run=config.get("min_zero_run", 2),
        max_suspicious_rate=config.get("max_suspicious_rate", 0.001)
    )
    return LookaheadContaminationRule(cfg)
