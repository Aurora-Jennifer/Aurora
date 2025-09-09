"""
Volume-dependent slippage and market impact models.

Replaces fixed basis point costs with participation-dependent impact functions.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings


class MarketImpactModel:
    """
    Volume-dependent market impact model for realistic trading costs.
    
    Implements square-root impact model: impact = a * sqrt(participation) + b
    where participation = order_size / ADV
    """
    
    def __init__(self, 
                 sqrt_coeff: float = 10.0,     # Coefficient for sqrt(participation) term
                 linear_coeff: float = 2.0,    # Fixed component (bps)
                 max_impact_bps: float = 50.0, # Cap on total impact
                 min_impact_bps: float = 1.0): # Floor on impact
        """
        Initialize impact model parameters.
        
        Args:
            sqrt_coeff: Coefficient for sqrt term (scales with participation)
            linear_coeff: Fixed cost component in bps
            max_impact_bps: Maximum impact in basis points (circuit breaker)
            min_impact_bps: Minimum impact in basis points
        """
        self.sqrt_coeff = sqrt_coeff
        self.linear_coeff = linear_coeff
        self.max_impact_bps = max_impact_bps
        self.min_impact_bps = min_impact_bps
    
    def compute_impact(self, participation_pct: float) -> float:
        """
        Compute market impact in basis points.
        
        Args:
            participation_pct: Participation rate (e.g., 0.02 = 2% of ADV)
            
        Returns:
            Impact in basis points
        """
        if participation_pct <= 0:
            return self.min_impact_bps
            
        # Square-root impact model
        sqrt_term = self.sqrt_coeff * np.sqrt(participation_pct)
        total_impact = self.linear_coeff + sqrt_term
        
        # Apply bounds
        return np.clip(total_impact, self.min_impact_bps, self.max_impact_bps)
    
    def compute_batch_impact(self, participations: np.ndarray) -> np.ndarray:
        """
        Compute impact for multiple orders efficiently.
        
        Args:
            participations: Array of participation rates
            
        Returns:
            Array of impacts in basis points
        """
        # Vectorized computation
        safe_participations = np.maximum(participations, 1e-10)  # Avoid sqrt(0)
        sqrt_terms = self.sqrt_coeff * np.sqrt(safe_participations)
        total_impacts = self.linear_coeff + sqrt_terms
        
        return np.clip(total_impacts, self.min_impact_bps, self.max_impact_bps)
    
    def estimate_capacity(self, max_impact_bps: float = 20.0) -> float:
        """
        Estimate maximum participation before hitting impact limit.
        
        Args:
            max_impact_bps: Maximum acceptable impact
            
        Returns:
            Maximum participation rate
        """
        # Solve: max_impact = linear_coeff + sqrt_coeff * sqrt(participation)
        if max_impact_bps <= self.linear_coeff:
            return 0.0
            
        residual = max_impact_bps - self.linear_coeff
        max_participation = (residual / self.sqrt_coeff) ** 2
        
        return max_participation


class RealisticCostModel:
    """
    Comprehensive cost model combining spread, fees, and market impact.
    """
    
    def __init__(self,
                 spread_model: Optional[MarketImpactModel] = None,
                 impact_model: Optional[MarketImpactModel] = None,
                 fixed_fee_bps: float = 1.0):
        """
        Initialize cost model.
        
        Args:
            spread_model: Model for bid-ask spread costs
            impact_model: Model for market impact costs  
            fixed_fee_bps: Fixed commission/fees in bps
        """
        # Default spread model (less participation-sensitive)
        self.spread_model = spread_model or MarketImpactModel(
            sqrt_coeff=5.0, linear_coeff=2.0, max_impact_bps=15.0
        )
        
        # Default impact model (more participation-sensitive)
        self.impact_model = impact_model or MarketImpactModel(
            sqrt_coeff=15.0, linear_coeff=1.0, max_impact_bps=40.0
        )
        
        self.fixed_fee_bps = fixed_fee_bps
    
    def compute_total_cost(self, order_dollars: float, adv_dollars: float) -> Dict[str, float]:
        """
        Compute total trading cost breakdown.
        
        Args:
            order_dollars: Order size in dollars (absolute value)
            adv_dollars: Average daily volume in dollars
            
        Returns:
            Dict with cost breakdown in basis points
        """
        # Calculate participation
        participation = order_dollars / adv_dollars if adv_dollars > 0 else 0.0
        
        # Compute cost components
        spread_cost = self.spread_model.compute_impact(participation)
        impact_cost = self.impact_model.compute_impact(participation)
        fee_cost = self.fixed_fee_bps
        
        total_cost = spread_cost + impact_cost + fee_cost
        
        return {
            'spread_bps': spread_cost,
            'impact_bps': impact_cost, 
            'fee_bps': fee_cost,
            'total_bps': total_cost,
            'participation_pct': participation * 100
        }
    
    def estimate_effective_cost(self, target_dollars: Dict[str, float],
                               adv_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Estimate portfolio-level effective cost.
        
        Args:
            target_dollars: Dict of symbol -> target_position_dollars
            adv_data: DataFrame with columns ['symbol', 'adv_20d_dollars']
            
        Returns:
            Tuple of (weighted_average_cost_bps, detailed_breakdown)
        """
        total_dollars = sum(abs(v) for v in target_dollars.values())
        
        if total_dollars == 0:
            return 0.0, {}
            
        # Merge with ADV data
        adv_map = dict(zip(adv_data['symbol'], adv_data['adv_20d_dollars']))
        
        weighted_cost = 0.0
        breakdown = {}
        
        for symbol, dollars in target_dollars.items():
            abs_dollars = abs(dollars)
            if abs_dollars == 0:
                continue
                
            adv = adv_map.get(symbol, 1_000_000)  # Default ADV if missing
            cost_info = self.compute_total_cost(abs_dollars, adv)
            
            # Weight by dollar amount
            weight = abs_dollars / total_dollars
            weighted_cost += weight * cost_info['total_bps']
            
            breakdown[symbol] = cost_info
        
        return weighted_cost, breakdown


def create_cost_sensitivity_ladder(base_model: RealisticCostModel,
                                  multipliers: list = [1.0, 2.0, 3.0]) -> Dict[str, RealisticCostModel]:
    """
    Create cost sensitivity test models.
    
    Args:
        base_model: Base cost model
        multipliers: Cost multipliers to test
        
    Returns:
        Dict of label -> cost_model
    """
    models = {}
    
    for mult in multipliers:
        # Scale impact coefficients
        scaled_spread = MarketImpactModel(
            sqrt_coeff=base_model.spread_model.sqrt_coeff * mult,
            linear_coeff=base_model.spread_model.linear_coeff * mult,
            max_impact_bps=base_model.spread_model.max_impact_bps * mult
        )
        
        scaled_impact = MarketImpactModel(
            sqrt_coeff=base_model.impact_model.sqrt_coeff * mult,
            linear_coeff=base_model.impact_model.linear_coeff * mult,
            max_impact_bps=base_model.impact_model.max_impact_bps * mult
        )
        
        scaled_model = RealisticCostModel(
            spread_model=scaled_spread,
            impact_model=scaled_impact,
            fixed_fee_bps=base_model.fixed_fee_bps * mult
        )
        
        models[f"{mult:.0f}x"] = scaled_model
    
    return models


def run_impact_model_test():
    """Test impact model functionality."""
    print("🧪 TESTING VOLUME-DEPENDENT IMPACT MODEL")
    print("="*50)
    
    # Create base model
    model = RealisticCostModel()
    
    # Test different participation levels
    participations = [0.005, 0.01, 0.02, 0.05, 0.10]  # 0.5% to 10%
    
    print("📊 Impact vs Participation:")
    print("   Participation | Spread | Impact | Fee | Total")
    print("   -------------|--------|--------|-----|-------")
    
    for participation in participations:
        spread = model.spread_model.compute_impact(participation)
        impact = model.impact_model.compute_impact(participation)
        fee = model.fixed_fee_bps
        total = spread + impact + fee
        
        print(f"   {participation*100:>11.1f}% | {spread:>6.1f} | {impact:>6.1f} | {fee:>3.0f} | {total:>5.1f}")
    
    # Test portfolio-level costs
    print(f"\n📊 Portfolio Cost Example:")
    target_positions = {
        'AAPL': 100_000,  # $100K position
        'MSFT': 50_000,   # $50K position  
        'GOOGL': 75_000   # $75K position
    }
    
    # Mock ADV data
    adv_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'adv_20d_dollars': [50_000_000, 30_000_000, 20_000_000]  # $50M, $30M, $20M ADV
    })
    
    avg_cost, breakdown = model.estimate_effective_cost(target_positions, adv_data)
    
    print(f"   Portfolio average cost: {avg_cost:.1f} bps")
    
    for symbol, cost_info in breakdown.items():
        print(f"   {symbol}: {cost_info['total_bps']:.1f} bps "
              f"({cost_info['participation_pct']:.2f}% ADV)")
    
    # Test cost sensitivity
    print(f"\n📊 Cost Sensitivity Analysis:")
    sensitivity_models = create_cost_sensitivity_ladder(model, [1.0, 2.0, 3.0])
    
    print("   Multiplier | Avg Cost (bps)")
    print("   ----------|---------------")
    
    for label, sens_model in sensitivity_models.items():
        sens_cost, _ = sens_model.estimate_effective_cost(target_positions, adv_data)
        print(f"   {label:>9} | {sens_cost:>13.1f}")
    
    print(f"\n✅ Volume-dependent impact model test completed")


if __name__ == "__main__":
    run_impact_model_test()
