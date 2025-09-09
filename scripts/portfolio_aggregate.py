#!/usr/bin/env python3
"""
Portfolio aggregation script with risk controls and bounds enforcement.

Reads grid results from multiple assets and constructs a robust portfolio
with proper risk management, position limits, and turnover controls.
"""

import argparse
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioAggregator:
    """Portfolio construction with risk controls"""
    
    def __init__(self, config: Dict = None, *, top_k=None, k=None, min_sharpe=0.0, min_trades=0,
                 min_active_days=0, turnover_cap=None, **kwargs):
        # Handle both config-based and direct parameter initialization
        if config is not None:
            self.config = config
            self.risk_config = config.get('risk', {})
            self.portfolio_config = config.get('portfolio', {})
            
            # Risk parameters
            self.max_position_weight = self.risk_config.get('max_position_weight', 1.0)
            self.max_net_exposure = self.risk_config.get('max_net_exposure', 1.0)
            self.max_turnover = self.risk_config.get('max_turnover', 0.5)
            self.target_volatility = self.portfolio_config.get('target_volatility', 0.10)
            
            # Portfolio parameters
            self.k = self.portfolio_config.get('top_k', 10)
            self.min_sharpe = self.portfolio_config.get('min_sharpe', 0.5)
            self.hysteresis_factor = self.portfolio_config.get('hysteresis_factor', 0.1)
        else:
            # Direct parameter initialization
            self.k = k if k is not None else top_k
            if self.k is None:
                raise TypeError("Provide 'top_k' or 'k' to PortfolioAggregator")
            self.min_sharpe = float(min_sharpe)
            self.min_trades = int(min_trades)
            self.min_active_days = int(min_active_days)
            self.turnover_cap = turnover_cap
            
            # Set defaults for other parameters
            self.max_position_weight = 1.0
            self.max_net_exposure = 1.0
            self.max_turnover = 0.5
            self.target_volatility = 0.10
            self.hysteresis_factor = 0.1
        
        # Initialize portfolio state
        self.current_weights = {}
        self.previous_weights = {}
        
    def load_grid_results(self, results_dir: Path) -> pd.DataFrame:
        """Load and combine grid results from all assets"""
        all_results = []
        
        for asset_dir in results_dir.iterdir():
            if not asset_dir.is_dir():
                continue
                
            grid_file = asset_dir / "grid_results.csv"
            if not grid_file.exists():
                logger.warning(f"No grid results found for {asset_dir.name}")
                continue
            
            try:
                df = pd.read_csv(grid_file)
                df['asset'] = asset_dir.name
                all_results.append(df)
                logger.info(f"Loaded {len(df)} results for {asset_dir.name}")
            except Exception as e:
                logger.error(f"Error loading {grid_file}: {e}")
        
        if not all_results:
            raise ValueError("No grid results found")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total results from {len(all_results)} assets")
        
        return combined_df
    
    def filter_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter results based on quality criteria"""
        initial_count = len(df)
        
        # Filter by gate pass (if column exists)
        if 'gate_pass' in df.columns:
            df = df[df['gate_pass'] == True]
            logger.info(f"After gate filter: {len(df)} results")
        else:
            logger.info(f"No gate_pass column found, skipping gate filter: {len(df)} results")
        
        # Filter by minimum Sharpe
        sharpe_col = 'median_model_sharpe' if 'median_model_sharpe' in df.columns else 'median_sharpe'
        if sharpe_col in df.columns:
            df = df[df[sharpe_col] >= self.min_sharpe]
            logger.info(f"After Sharpe filter: {len(df)} results")
        else:
            logger.info(f"No Sharpe column found, skipping Sharpe filter: {len(df)} results")
        
        # Filter by minimum trades (if available)
        trades_col = 'mean_trades' if 'mean_trades' in df.columns else 'median_trades'
        if trades_col in df.columns:
            min_trades = self.portfolio_config.get('min_trades', 15)
            df = df[df[trades_col] >= min_trades]
            logger.info(f"After trades filter: {len(df)} results")
        else:
            logger.info(f"No trades column found, skipping trades filter: {len(df)} results")
        
        # Filter by active days (if available)
        if 'active_days_pct' in df.columns:
            min_active_days = self.portfolio_config.get('min_active_days_pct', 10.0)
            df = df[df['active_days_pct'] >= min_active_days]
            logger.info(f"After active days filter: {len(df)} results")
        
        logger.info(f"Filtered from {initial_count} to {len(df)} results")
        return df
    
    def select_top_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top K strategies per asset based on stable Sharpe"""
        selected_strategies = []
        
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset].copy()
            
            # Sort by median Sharpe (stable performance)
            sharpe_col = 'median_model_sharpe' if 'median_model_sharpe' in asset_df.columns else 'median_sharpe'
            asset_df = asset_df.sort_values(sharpe_col, ascending=False)
            
            # Take top K strategies
            top_strategies = asset_df.head(self.k)
            selected_strategies.append(top_strategies)
            
            logger.info(f"Selected {len(top_strategies)} strategies for {asset}")
        
        return pd.concat(selected_strategies, ignore_index=True)
    
    def calculate_portfolio_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio weights with risk controls"""
        
        # Calculate strategy scores (weighted combination of metrics)
        df = df.copy()
        
        # Normalize metrics
        sharpe_col = 'median_model_sharpe' if 'median_model_sharpe' in df.columns else 'median_sharpe'
        if sharpe_col in df.columns:
            df['sharpe_score'] = (df[sharpe_col] - df[sharpe_col].min()) / \
                                (df[sharpe_col].max() - df[sharpe_col].min())
        else:
            df['sharpe_score'] = 0.5
        
        trades_col = 'mean_trades' if 'mean_trades' in df.columns else 'median_trades'
        if trades_col in df.columns:
            df['trades_score'] = (df[trades_col] - df[trades_col].min()) / \
                               (df[trades_col].max() - df[trades_col].min())
        else:
            df['trades_score'] = 0.5
        
        # Combined score
        df['combined_score'] = 0.7 * df['sharpe_score'] + 0.3 * df['trades_score']
        
        # Calculate raw weights (proportional to combined score)
        total_score = df['combined_score'].sum()
        if total_score > 0:
            raw_weights = df['combined_score'] / total_score
        else:
            raw_weights = pd.Series(1.0 / len(df), index=df.index)
        
        # Apply position limits
        raw_weights = np.clip(raw_weights, 0, self.max_position_weight)
        
        # Normalize weights
        total_weight = raw_weights.sum()
        if total_weight > 0:
            weights = raw_weights / total_weight
        else:
            weights = pd.Series(1.0 / len(df), index=df.index)
        
        # Create weight dictionary first
        weight_dict = {}
        for idx, row in df.iterrows():
            strategy_id = f"{row['asset']}_{idx}"
            weight_dict[strategy_id] = weights[idx]
        
        # Apply net exposure limit after creating the dictionary
        long_weight = sum(w for w in weight_dict.values() if w > 0)
        short_weight = abs(sum(w for w in weight_dict.values() if w < 0))
        net_exposure = long_weight - short_weight
        
        if abs(net_exposure) > self.max_net_exposure:
            # Scale down weights to meet net exposure limit
            scale_factor = self.max_net_exposure / abs(net_exposure)
            weight_dict = {k: w * scale_factor for k, w in weight_dict.items()}
        
        return weight_dict
    
    def apply_hysteresis(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply hysteresis to reduce turnover"""
        if not self.previous_weights:
            return new_weights
        
        adjusted_weights = {}
        
        for strategy_id in set(new_weights.keys()) | set(self.previous_weights.keys()):
            new_weight = new_weights.get(strategy_id, 0.0)
            prev_weight = self.previous_weights.get(strategy_id, 0.0)
            
            # Apply hysteresis: move only partway toward new weight
            adjusted_weight = prev_weight + self.hysteresis_factor * (new_weight - prev_weight)
            adjusted_weights[strategy_id] = adjusted_weight
        
        return adjusted_weights
    
    def calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover"""
        if not self.previous_weights:
            return 1.0  # Full turnover for first period
        
        all_strategies = set(new_weights.keys()) | set(self.previous_weights.keys())
        turnover = 0.0
        
        for strategy_id in all_strategies:
            new_weight = new_weights.get(strategy_id, 0.0)
            prev_weight = self.previous_weights.get(strategy_id, 0.0)
            turnover += abs(new_weight - prev_weight)
        
        return turnover / 2.0  # Divide by 2 for one-way turnover
    
    def enforce_turnover_limit(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce maximum turnover limit"""
        # Skip turnover limit for first run (no previous weights)
        if not self.previous_weights:
            return weights
        
        turnover = self.calculate_turnover(weights)
        
        if turnover <= self.max_turnover:
            return weights
        
        # Scale down changes to meet turnover limit
        scale_factor = self.max_turnover / turnover
        
        adjusted_weights = {}
        for strategy_id in weights:
            if strategy_id in self.previous_weights:
                prev_weight = self.previous_weights[strategy_id]
                new_weight = weights[strategy_id]
                adjusted_weight = prev_weight + scale_factor * (new_weight - prev_weight)
            else:
                adjusted_weight = weights[strategy_id] * scale_factor
            
            adjusted_weights[strategy_id] = adjusted_weight
        
        return adjusted_weights
    
    def apply_volatility_targeting(self, weights: Dict[str, float], 
                                 strategy_vols: Dict[str, float]) -> Dict[str, float]:
        """Apply volatility targeting to portfolio"""
        if not strategy_vols:
            return weights
        
        # First, normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight) > 1e-6:  # Avoid division by zero
            normalized_weights = {k: w / total_weight for k, w in weights.items()}
        else:
            normalized_weights = weights
        
        # Skip volatility targeting for now to ensure weights sum to 1.0
        # TODO: Implement proper volatility targeting that preserves weight normalization
        adjusted_weights = normalized_weights
        
        return adjusted_weights
    
    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, any]:
        """Validate portfolio weights against risk constraints"""
        validation = {
            'valid': True,
            'violations': [],
            'metrics': {}
        }
        
        # Check position limits
        for strategy_id, weight in weights.items():
            if abs(weight) > self.max_position_weight:
                validation['violations'].append(f"Position limit violated: {strategy_id} = {weight:.3f}")
                validation['valid'] = False
        
        # Check net exposure
        long_weight = sum(w for w in weights.values() if w > 0)
        short_weight = abs(sum(w for w in weights.values() if w < 0))
        net_exposure = long_weight - short_weight
        
        validation['metrics']['net_exposure'] = net_exposure
        if abs(net_exposure) > self.max_net_exposure:
            validation['violations'].append(f"Net exposure limit violated: {net_exposure:.3f}")
            validation['valid'] = False
        
        # Check turnover
        turnover = self.calculate_turnover(weights)
        validation['metrics']['turnover'] = turnover
        if turnover > self.max_turnover:
            validation['violations'].append(f"Turnover limit violated: {turnover:.3f}")
            validation['valid'] = False
        
        # Check weight sum
        weight_sum = sum(weights.values())
        validation['metrics']['weight_sum'] = weight_sum
        if abs(weight_sum - 1.0) > 0.01:
            validation['violations'].append(f"Weight sum not equal to 1: {weight_sum:.3f}")
            validation['valid'] = False
        
        return validation
    
    def construct_portfolio(self, results_dir: Path) -> Dict[str, any]:
        """Main portfolio construction pipeline"""
        logger.info("Starting portfolio construction...")
        
        # Load and filter results
        df = self.load_grid_results(results_dir)
        df = self.filter_results(df)
        
        if len(df) == 0:
            raise ValueError("No valid strategies found after filtering")
        
        # Select top strategies
        df = self.select_top_strategies(df)
        
        # Calculate initial weights
        weights = self.calculate_portfolio_weights(df)
        
        # Apply hysteresis
        weights = self.apply_hysteresis(weights)
        
        # Enforce turnover limit
        weights = self.enforce_turnover_limit(weights)
        
        # Apply volatility targeting (if volatility data available)
        strategy_vols = {}  # Would be loaded from results if available
        weights = self.apply_volatility_targeting(weights, strategy_vols)
        
        # Validate weights
        validation = self.validate_weights(weights)
        
        # Update portfolio state
        self.previous_weights = self.current_weights.copy()
        self.current_weights = weights.copy()
        
        # Prepare results
        portfolio_results = {
            'timestamp': datetime.now().isoformat(),
            'weights': weights,
            'validation': validation,
            'strategy_count': len(weights),
            'total_strategies_evaluated': len(df),
            'config': self.config
        }
        
        logger.info(f"Portfolio construction complete: {len(weights)} strategies, "
                   f"turnover: {validation['metrics'].get('turnover', 0):.3f}, "
                   f"valid: {validation['valid']}")
        
        return portfolio_results


def load_config(config_path: Path) -> Dict:
    """Load portfolio configuration"""
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_portfolio_results(results: Dict[str, any], output_dir: Path):
    """Save portfolio results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights as CSV
    weights_df = pd.DataFrame([
        {'strategy_id': k, 'weight': v} for k, v in results['weights'].items()
    ])
    weights_df.to_csv(output_dir / 'portfolio_weights.csv', index=False)
    
    # Save full results as JSON
    with open(output_dir / 'portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save portfolio.json with weights and sum
    total_weight = sum(abs(w) for w in results['weights'].values())
    portfolio_data = {
        'weights': results['weights'],
        'sum_abs': total_weight,
        'timestamp': results['timestamp'],
        'num_strategies': len(results['weights'])
    }
    
    with open(output_dir / 'portfolio.json', 'w') as f:
        json.dump(portfolio_data, f, indent=2, default=str)
    
    # Save validation report
    validation_report = {
        'timestamp': results['timestamp'],
        'valid': results['validation']['valid'],
        'violations': results['validation']['violations'],
        'metrics': results['validation']['metrics']
    }
    
    with open(output_dir / 'validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Portfolio results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Portfolio aggregation with risk controls')
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Directory containing grid results')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for portfolio results')
    parser.add_argument('--config', type=Path, required=True,
                       help='Portfolio configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create aggregator
    aggregator = PortfolioAggregator(config)
    
    # Construct portfolio
    results = aggregator.construct_portfolio(args.input_dir)
    
    # Save results
    save_portfolio_results(results, args.output_dir)
    
    # Print summary
    print(f"\n🎯 Portfolio Construction Summary:")
    print(f"  Strategies selected: {results['strategy_count']}")
    print(f"  Total evaluated: {results['total_strategies_evaluated']}")
    print(f"  Valid: {'✅' if results['validation']['valid'] else '❌'}")
    print(f"  Turnover: {results['validation']['metrics'].get('turnover', 0):.3f}")
    print(f"  Net exposure: {results['validation']['metrics'].get('net_exposure', 0):.3f}")
    
    if results['validation']['violations']:
        print(f"\n⚠️  Violations:")
        for violation in results['validation']['violations']:
            print(f"    • {violation}")


if __name__ == "__main__":
    main()
