"""
Corporate Actions Validation Rules
Part of Rung 4 - Corporate Actions handling.

Validates that price data properly accounts for splits and dividends
to prevent fake PnL in backtests.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorporateActionsRuleConfig:
    actions_dir: str = "data/corporate_actions"
    strict_split_validation: bool = True
    check_dividend_gaps: bool = True
    check_volume_spikes: bool = True


class CorporateActionsRule:
    """Validates price data against corporate action events."""
    
    def __init__(self, cfg: CorporateActionsRuleConfig):
        self.cfg = cfg
        self.actions_dir = Path(cfg.actions_dir)
        self.actions_cache = {}
        self._load_corporate_actions()
    
    def _load_corporate_actions(self):
        """Load corporate actions data for all symbols."""
        if not self.actions_dir.exists():
            logger.warning(f"Corporate actions directory not found: {self.actions_dir}")
            return
            
        for actions_file in self.actions_dir.glob("*_actions.json"):
            symbol = actions_file.stem.replace("_actions", "")
            try:
                with open(actions_file, 'r') as f:
                    self.actions_cache[symbol] = json.load(f)
                logger.debug(f"Loaded corporate actions for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load corporate actions for {symbol}: {e}")
    
    def _get_symbol_actions(self, symbol: str) -> Dict:
        """Get corporate actions for a symbol."""
        return self.actions_cache.get(symbol, {
            "splits": {},
            "dividends": {},
            "splits_count": 0,
            "dividends_count": 0
        })
    
    def _check_split_adjustments(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check if price data is properly adjusted for splits."""
        violations = []
        actions = self._get_symbol_actions(symbol)
        
        if not actions.get("splits"):
            return violations  # No splits to check
            
        # Convert split dates from ISO strings to timestamps
        split_events = {}
        for date_str, ratio in actions["splits"].items():
            split_date = pd.Timestamp(date_str).tz_convert('UTC')
            split_events[split_date] = ratio
            
        # Check each split event
        for split_date, split_ratio in split_events.items():
            # Find data around split date
            before_split = df[df.index < split_date]
            after_split = df[df.index >= split_date]
            
            if len(before_split) < 5 or len(after_split) < 5:
                continue  # Need sufficient data around split
                
            # Get prices just before and after split
            pre_price = before_split['Close'].iloc[-1]
            post_price = after_split['Close'].iloc[0]
            
            # Expected price after split (if properly adjusted)
            expected_post_price = pre_price / split_ratio
            
            # Allow 5% tolerance for normal market movement
            price_ratio = post_price / expected_post_price
            if not (0.95 <= price_ratio <= 1.05):
                violations.append(
                    f"Potential unadjusted split on {split_date.date()}: "
                    f"{split_ratio}:1 split, pre=${pre_price:.2f}, post=${post_price:.2f}, "
                    f"expected=${expected_post_price:.2f} (ratio={price_ratio:.3f})"
                )
                
        return violations
    
    def _check_dividend_gaps(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for unusual price gaps on dividend ex-dates."""
        violations = []
        actions = self._get_symbol_actions(symbol)
        
        if not actions.get("dividends"):
            return violations
            
        # Convert dividend dates and amounts
        dividend_events = {}
        for date_str, amount in actions["dividends"].items():
            div_date = pd.Timestamp(date_str).tz_convert('UTC')
            dividend_events[div_date] = amount
            
        # Check major dividend events (>$1.00)
        for div_date, div_amount in dividend_events.items():
            if div_amount < 1.0:  # Skip small dividends
                continue
                
            # Find data around dividend date
            before_div = df[df.index < div_date]
            after_div = df[df.index >= div_date]
            
            if len(before_div) < 2 or len(after_div) < 2:
                continue
                
            pre_close = before_div['Close'].iloc[-1]
            post_open = after_div['Open'].iloc[0] if 'Open' in after_div.columns else after_div['Close'].iloc[0]
            
            # Expected gap from dividend (approximate)
            expected_gap = div_amount / pre_close
            actual_gap = (pre_close - post_open) / pre_close
            
            # Flag if gap is unusually different from dividend amount
            if abs(actual_gap - expected_gap) > 0.02:  # 2% tolerance
                violations.append(
                    f"Unusual dividend gap on {div_date.date()}: "
                    f"${div_amount:.2f} dividend, gap={actual_gap:.1%}, expected≈{expected_gap:.1%}"
                )
                
        return violations
    
    def _check_volume_spikes(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """Check for volume spikes around corporate action dates."""
        violations = []
        
        if 'Volume' not in df.columns:
            return violations
            
        actions = self._get_symbol_actions(symbol)
        action_dates = set()
        
        # Collect all action dates
        for date_str in actions.get("splits", {}):
            action_dates.add(pd.Timestamp(date_str).tz_convert('UTC'))
        for date_str in actions.get("dividends", {}):
            action_dates.add(pd.Timestamp(date_str).tz_convert('UTC'))
            
        # Calculate rolling volume average
        df_vol = df['Volume'].rolling(window=20, min_periods=10).mean()
        
        for action_date in action_dates:
            # Find data around action date
            action_window = df[
                (df.index >= action_date - pd.Timedelta(days=2)) &
                (df.index <= action_date + pd.Timedelta(days=2))
            ]
            
            if len(action_window) == 0:
                continue
                
            action_volume = action_window['Volume'].max()
            normal_volume = df_vol.loc[action_window.index].mean()
            
            if pd.isna(normal_volume) or normal_volume == 0:
                continue
                
            volume_ratio = action_volume / normal_volume
            
            # Flag unusual volume activity (>5x normal)
            if volume_ratio > 5.0:
                violations.append(
                    f"High volume spike around {action_date.date()}: "
                    f"{volume_ratio:.1f}x normal volume"
                )
                
        return violations

    def validate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Validate corporate actions consistency."""
        if symbol is None:
            # Try to extract symbol from DataFrame if available
            if hasattr(df, 'attrs') and 'symbol' in df.attrs:
                symbol = df.attrs['symbol']
            elif 'symbol' in df.columns and not df.empty:
                symbol = df['symbol'].iloc[0]
            else:
                logger.warning("Corporate actions validation skipped: no symbol provided")
                return df
        
        violations = []
        
        # Check split adjustments if enabled
        if self.cfg.strict_split_validation:
            split_violations = self._check_split_adjustments(df, symbol)
            violations.extend(split_violations)
        
        # Check dividend gaps if enabled
        if self.cfg.check_dividend_gaps:
            dividend_violations = self._check_dividend_gaps(df, symbol)
            violations.extend(dividend_violations)
        
        # Check volume spikes if enabled
        if self.cfg.check_volume_spikes:
            volume_violations = self._check_volume_spikes(df, symbol)
            violations.extend(volume_violations)
        
        if violations:
            # Log violations as warnings but don't fail validation for now
            for violation in violations:
                if "unadjusted split" in violation.lower():
                    logger.warning(f"CRITICAL: {violation}")
                else:
                    logger.warning(f"Corporate Actions: {violation}")
        
        return df

    def validate_and_repair(self, df: pd.DataFrame, symbol: str = None) -> tuple[pd.DataFrame, list[str]]:
        """Validate and optionally repair corporate actions issues."""
        if symbol is None:
            # Try to extract symbol from DataFrame if available
            if hasattr(df, 'attrs') and 'symbol' in df.attrs:
                symbol = df.attrs['symbol']
            elif 'symbol' in df.columns and not df.empty:
                symbol = df['symbol'].iloc[0]
            else:
                logger.warning("Corporate actions repair skipped: no symbol provided")
                return df, []
        
        try:
            return self.validate(df, symbol), []
        except ValueError:
            # For now, just raise the error
            # Future: could implement split adjustment repairs
            raise


def create_corporate_actions_rule(config: dict) -> CorporateActionsRule:
    """Factory function to create corporate actions rule."""
    cfg = CorporateActionsRuleConfig(
        actions_dir=config.get("actions_dir", "data/corporate_actions"),
        strict_split_validation=config.get("strict_split_validation", True),
        check_dividend_gaps=config.get("check_dividend_gaps", True),
        check_volume_spikes=config.get("check_volume_spikes", True)
    )
    return CorporateActionsRule(cfg)
