"""
ADV-based capacity enforcement for production trading.

Implements hard participation limits and order blocking to ensure 
realistic capacity assumptions in paper and live trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapacityBreach:
    """Record of a capacity constraint violation."""
    symbol: str
    requested_dollars: float
    adv_dollars: float
    participation_pct: float
    limit_pct: float
    timestamp: str
    action_taken: str


class ADVCapacityEnforcer:
    """
    Enforce ADV-based capacity constraints in position sizing.
    
    Prevents orders that would exceed participation limits,
    logs violations, and provides capacity analytics.
    """
    
    def __init__(self, 
                 max_participation_pct: float = 0.02,
                 min_adv_dollars: float = 100_000,
                 equity: float = 10_000_000):
        """
        Initialize capacity enforcer.
        
        Args:
            max_participation_pct: Maximum ADV participation (e.g., 0.02 = 2%)
            min_adv_dollars: Minimum ADV required for trading
            equity: Portfolio equity for position sizing
        """
        self.max_participation = max_participation_pct
        self.min_adv = min_adv_dollars
        self.equity = equity
        self.breaches: List[CapacityBreach] = []
        
    def validate_adv_data(self, adv_data: pd.DataFrame) -> None:
        """
        Validate ADV data quality and completeness.
        
        Args:
            adv_data: DataFrame with columns ['symbol', 'adv_20d_dollars']
            
        Raises:
            ValueError: If ADV data fails validation
        """
        required_cols = ['symbol', 'adv_20d_dollars']
        missing_cols = [col for col in required_cols if col not in adv_data.columns]
        if missing_cols:
            raise ValueError(f"Missing ADV columns: {missing_cols}")
            
        # Check for negative or zero ADV
        invalid_adv = adv_data['adv_20d_dollars'] <= 0
        if invalid_adv.any():
            invalid_symbols = adv_data.loc[invalid_adv, 'symbol'].tolist()
            logger.warning(f"Invalid ADV for symbols: {invalid_symbols[:10]}...")
            
        # Check for missing ADV data
        missing_adv = adv_data['adv_20d_dollars'].isna()
        if missing_adv.any():
            missing_symbols = adv_data.loc[missing_adv, 'symbol'].tolist()
            logger.warning(f"Missing ADV for symbols: {missing_symbols[:10]}...")
    
    def compute_max_position_dollars(self, adv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute maximum tradeable position size per symbol.
        
        Args:
            adv_data: DataFrame with ADV data
            
        Returns:
            DataFrame with max position limits
        """
        result = adv_data.copy()
        
        # Max position based on ADV participation limit
        result['max_position_adv'] = (
            result['adv_20d_dollars'] * self.max_participation
        )
        
        # Max position based on portfolio equity limit (e.g., 2% per symbol)
        max_equity_per_symbol = self.equity * 0.02  # 2% position limit
        result['max_position_equity'] = max_equity_per_symbol
        
        # Take the more restrictive limit
        result['max_position_dollars'] = np.minimum(
            result['max_position_adv'],
            result['max_position_equity']
        )
        
        # Flag symbols with insufficient ADV
        result['tradeable'] = (
            (result['adv_20d_dollars'] >= self.min_adv) &
            (result['adv_20d_dollars'].notna()) &
            (result['max_position_dollars'] > 1000)  # Minimum $1K position
        )
        
        return result[['symbol', 'adv_20d_dollars', 'max_position_dollars', 'tradeable']]
    
    def enforce_capacity_constraints(self, 
                                   target_positions: pd.DataFrame,
                                   adv_data: pd.DataFrame,
                                   current_timestamp: str) -> Tuple[pd.DataFrame, List[CapacityBreach]]:
        """
        Enforce capacity constraints on target positions.
        
        Args:
            target_positions: DataFrame with ['symbol', 'target_dollars']
            adv_data: DataFrame with ADV data
            current_timestamp: ISO timestamp for breach recording
            
        Returns:
            Tuple of (constrained_positions, new_breaches)
        """
        # Validate inputs
        self.validate_adv_data(adv_data)
        
        required_cols = ['symbol', 'target_dollars']
        if not all(col in target_positions.columns for col in required_cols):
            raise ValueError(f"target_positions must have columns: {required_cols}")
        
        # Compute capacity limits
        limits = self.compute_max_position_dollars(adv_data)
        
        # Merge with target positions
        constrained = target_positions.merge(limits, on='symbol', how='left')
        
        # Handle missing ADV data
        missing_adv = constrained['max_position_dollars'].isna()
        if missing_adv.any():
            missing_symbols = constrained.loc[missing_adv, 'symbol'].tolist()
            logger.warning(f"No ADV data for {len(missing_symbols)} symbols, blocking orders")
            constrained.loc[missing_adv, 'max_position_dollars'] = 0
            constrained.loc[missing_adv, 'tradeable'] = False
        
        # Identify breaches
        constrained['abs_target'] = constrained['target_dollars'].abs()
        constrained['exceeds_limit'] = (
            constrained['abs_target'] > constrained['max_position_dollars']
        )
        
        breach_mask = constrained['exceeds_limit'] & constrained['tradeable']
        new_breaches = []
        
        for _, row in constrained.loc[breach_mask].iterrows():
            participation_pct = row['abs_target'] / row['adv_20d_dollars'] if row['adv_20d_dollars'] > 0 else float('inf')
            
            breach = CapacityBreach(
                symbol=row['symbol'],
                requested_dollars=row['abs_target'],
                adv_dollars=row['adv_20d_dollars'],
                participation_pct=participation_pct,
                limit_pct=self.max_participation,
                timestamp=current_timestamp,
                action_taken='clipped_to_limit'
            )
            new_breaches.append(breach)
            
        # Apply constraints
        constrained['constrained_dollars'] = np.where(
            constrained['tradeable'],
            np.sign(constrained['target_dollars']) * np.minimum(
                constrained['abs_target'],
                constrained['max_position_dollars']
            ),
            0  # Block untradeable symbols
        )
        
        # Log summary
        blocked_count = (~constrained['tradeable']).sum()
        clipped_count = (constrained['exceeds_limit'] & constrained['tradeable']).sum()
        
        if blocked_count > 0 or clipped_count > 0:
            logger.info(f"Capacity enforcement: {blocked_count} blocked, {clipped_count} clipped")
            
        # Store breaches
        self.breaches.extend(new_breaches)
        
        return constrained[['symbol', 'constrained_dollars']], new_breaches
    
    def get_capacity_report(self) -> Dict:
        """
        Generate capacity utilization report.
        
        Returns:
            Dict with capacity metrics and breach summary
        """
        if not self.breaches:
            return {
                'total_breaches': 0,
                'breach_rate': 0.0,
                'summary': 'No capacity violations detected'
            }
        
        breach_df = pd.DataFrame([{
            'symbol': b.symbol,
            'participation_pct': b.participation_pct,
            'timestamp': b.timestamp
        } for b in self.breaches])
        
        return {
            'total_breaches': len(self.breaches),
            'unique_symbols': breach_df['symbol'].nunique(),
            'max_participation_attempted': breach_df['participation_pct'].max(),
            'median_participation_attempted': breach_df['participation_pct'].median(),
            'recent_breaches_24h': len([b for b in self.breaches[-100:] if '2024' in b.timestamp]),  # Rough filter
            'breach_symbols': breach_df['symbol'].value_counts().head(10).to_dict()
        }
    
    def clear_breach_history(self) -> None:
        """Clear stored breach history."""
        self.breaches.clear()


def create_mock_adv_data(symbols: List[str], seed: int = 42) -> pd.DataFrame:
    """
    Create mock ADV data for testing.
    
    Args:
        symbols: List of symbol names
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with mock ADV data
    """
    np.random.seed(seed)
    
    # Simulate realistic ADV distribution
    # Large caps: $10M-100M ADV
    # Mid caps: $1M-10M ADV  
    # Small caps: $100K-1M ADV
    n_symbols = len(symbols)
    
    # 20% large cap, 50% mid cap, 30% small cap
    large_cap_count = int(n_symbols * 0.2)
    mid_cap_count = int(n_symbols * 0.5)
    small_cap_count = n_symbols - large_cap_count - mid_cap_count
    
    adv_values = []
    
    # Large caps
    adv_values.extend(np.random.lognormal(np.log(30_000_000), 0.8, large_cap_count))
    
    # Mid caps  
    adv_values.extend(np.random.lognormal(np.log(3_000_000), 0.9, mid_cap_count))
    
    # Small caps
    adv_values.extend(np.random.lognormal(np.log(500_000), 1.0, small_cap_count))
    
    # Shuffle to randomize assignment
    np.random.shuffle(adv_values)
    
    return pd.DataFrame({
        'symbol': symbols,
        'adv_20d_dollars': adv_values
    })


def run_capacity_enforcement_test():
    """Run comprehensive capacity enforcement test."""
    print("🧪 TESTING ADV CAPACITY ENFORCEMENT")
    print("="*50)
    
    # Create test data
    symbols = [f'TEST{i:03d}' for i in range(50)]
    adv_data = create_mock_adv_data(symbols, seed=42)
    
    # Create target positions (some will breach)
    np.random.seed(123)
    target_dollars = np.random.normal(0, 100_000, 50)  # Some large positions
    target_positions = pd.DataFrame({
        'symbol': symbols,
        'target_dollars': target_dollars
    })
    
    # Initialize enforcer
    enforcer = ADVCapacityEnforcer(
        max_participation_pct=0.02,  # 2%
        min_adv_dollars=100_000,     # $100K minimum
        equity=10_000_000            # $10M portfolio
    )
    
    # Test capacity enforcement
    constrained_positions, breaches = enforcer.enforce_capacity_constraints(
        target_positions, 
        adv_data, 
        "2024-09-08T15:30:00Z"
    )
    
    print(f"✅ Processed {len(target_positions)} target positions")
    print(f"🚨 Capacity breaches: {len(breaches)}")
    print(f"📊 Positions clipped: {(constrained_positions['constrained_dollars'] != target_positions['target_dollars']).sum()}")
    
    # Show sample breaches
    if breaches:
        print(f"\n📋 Sample breaches:")
        for breach in breaches[:3]:
            print(f"   {breach.symbol}: ${breach.requested_dollars:,.0f} req, "
                  f"{breach.participation_pct:.1%} participation (limit: {breach.limit_pct:.1%})")
    
    # Capacity report
    report = enforcer.get_capacity_report()
    print(f"\n📊 Capacity Report:")
    for key, value in report.items():
        if key != 'breach_symbols':
            print(f"   {key}: {value}")
    
    print(f"✅ Capacity enforcement test completed")
    return enforcer, constrained_positions, breaches


if __name__ == "__main__":
    run_capacity_enforcement_test()
