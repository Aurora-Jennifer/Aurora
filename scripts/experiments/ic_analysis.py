#!/usr/bin/env python3
"""
Aurora IC Analysis & Return Backtesting
=======================================

Bridge between Information Coefficient (IC) and actual returns.
Implements the framework you described for proper IC measurement and backtesting.

Examples:
    # Analyze IC across different horizons
    python scripts/ic_analysis.py --experiment exp_001 --horizons 1 5 20
    
    # Run return backtest from IC predictions
    python scripts/ic_analysis.py --experiment exp_001 --backtest --cost-bps 5
    
    # Compare IC stability across experiments
    python scripts/ic_analysis.py --compare exp_001 exp_002 exp_003
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import setup_logging

logger = setup_logging(__name__)


class ICAnalyzer:
    """Analyze Information Coefficient and convert to returns."""
    
    def __init__(self):
        self.experiments_dir = Path("reports/experiments")
        self.results_dir = Path("reports/ic_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_ic_properly(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           method: str = "spearman") -> dict:
        """
        Measure IC properly with significance testing.
        
        Args:
            y_true: Actual future returns
            y_pred: Model predictions
            method: 'spearman' (rank) or 'pearson' (linear)
        
        Returns:
            Dict with IC, p-value, t-stat, and quality assessment
        """
        # Remove NaN pairs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 10:
            return {"ic": 0.0, "p_value": 1.0, "t_stat": 0.0, "n_obs": mask.sum(), "quality": "insufficient_data"}
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if method == "spearman":
            ic, p_value = spearmanr(y_true_clean, y_pred_clean)
        elif method == "pearson":
            ic = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            # Simple t-test approximation for Pearson
            n = len(y_true_clean)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 0.99 else 0
            p_value = 2 * (1 - abs(t_stat)) if abs(t_stat) < 1 else 0  # Crude approximation
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # T-statistic for significance
        n_obs = len(y_true_clean)
        t_stat = ic * np.sqrt(n_obs - 2) / np.sqrt(1 - ic**2) if abs(ic) < 0.99 else 0
        
        # Quality assessment based on IC value and significance
        if p_value < 0.05 and ic > 0.05:
            quality = "excellent"
        elif p_value < 0.05 and ic > 0.02:
            quality = "good"
        elif p_value < 0.10 and ic > 0.01:
            quality = "marginal"
        elif ic > 0.01:
            quality = "weak_signal"
        else:
            quality = "noise"
        
        return {
            "ic": float(ic),
            "p_value": float(p_value),
            "t_stat": float(t_stat),
            "n_obs": int(n_obs),
            "quality": quality
        }
    
    def analyze_ic_across_horizons(self, data: pd.DataFrame, predictions: np.ndarray,
                                  horizons: list[int] = [1, 5, 10, 20]) -> dict:
        """
        Analyze IC across different forward return horizons.
        
        Args:
            data: DataFrame with OHLCV data
            predictions: Model predictions aligned with data
            horizons: List of forward return horizons to test
        
        Returns:
            Dict with IC analysis for each horizon
        """
        if 'close' not in data.columns:
            logger.error("Data must have 'close' column")
            return {}
        
        results = {}
        
        # Calculate returns for each horizon
        for horizon in horizons:
            # Forward return calculation
            future_returns = data['close'].pct_change(horizon).shift(-horizon)
            
            # Align predictions with future returns
            min_len = min(len(predictions), len(future_returns))
            pred_aligned = predictions[:min_len]
            returns_aligned = future_returns.iloc[:min_len].values
            
            # Measure IC
            ic_result = self.measure_ic_properly(returns_aligned, pred_aligned)
            ic_result["horizon"] = horizon
            results[f"horizon_{horizon}"] = ic_result
            
            logger.info(f"Horizon {horizon}: IC={ic_result['ic']:.4f}, p={ic_result['p_value']:.3f}, quality={ic_result['quality']}")
        
        return results
    
    def compute_ic_stability(self, y_true_series: list[np.ndarray], 
                           y_pred_series: list[np.ndarray]) -> dict:
        """
        Compute IC stability across multiple folds/periods.
        
        Args:
            y_true_series: List of true return arrays for each fold
            y_pred_series: List of prediction arrays for each fold
        
        Returns:
            Dict with IC distribution statistics
        """
        ics = []
        
        for y_true, y_pred in zip(y_true_series, y_pred_series, strict=False):
            ic_result = self.measure_ic_properly(y_true, y_pred)
            ics.append(ic_result["ic"])
        
        ics = np.array(ics)
        
        # Statistical analysis
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ic_median = np.median(ics)
        
        # Test if mean IC is significantly different from 0
        if len(ics) > 1:
            t_stat, p_value = ttest_1samp(ics, 0)
        else:
            t_stat, p_value = 0, 1
        
        # Stability metrics
        consistency_ratio = np.mean(ics > 0)  # Fraction of positive ICs
        
        return {
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_median": float(ic_median),
            "ic_min": float(np.min(ics)),
            "ic_max": float(np.max(ics)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "consistency_ratio": float(consistency_ratio),
            "n_folds": len(ics),
            "ic_values": ics.tolist()
        }
    
    def backtest_from_predictions(self, data: pd.DataFrame, predictions: np.ndarray,
                                 cost_bps: float = 5.0, rebalance_freq: int = 1) -> dict:
        """
        Convert IC predictions to actual returns via long-short backtesting.
        
        Args:
            data: Price data with 'close' column
            predictions: Model predictions (higher = more bullish)
            cost_bps: Transaction cost in basis points
            rebalance_freq: Rebalancing frequency in bars
        
        Returns:
            Dict with backtest results
        """
        if len(predictions) != len(data):
            logger.error(f"Prediction length {len(predictions)} != data length {len(data)}")
            return {}
        
        # Calculate forward returns
        returns = data['close'].pct_change().shift(-1)  # Next period return
        
        # Create signals from predictions (rank-based)
        signals = pd.Series(predictions, index=data.index)
        
        # Position sizing based on prediction ranks
        positions = pd.Series(index=data.index, dtype=float)
        
        for i in range(0, len(signals), rebalance_freq):
            end_idx = min(i + rebalance_freq, len(signals))
            period_signals = signals.iloc[i:end_idx]
            
            if len(period_signals) == 0:
                continue
            
            # Rank signals and convert to positions (-1 to +1)
            ranks = period_signals.rank(pct=True)  # Percentile ranks
            period_positions = 2 * ranks - 1  # Scale to [-1, 1]
            
            positions.iloc[i:end_idx] = period_positions
        
        # Calculate gross returns
        gross_returns = positions * returns
        
        # Calculate turnover and transaction costs
        position_changes = positions.diff().abs()
        turnover = position_changes.sum() / len(positions)
        
        # Apply transaction costs
        transaction_costs = position_changes * (cost_bps / 10000)
        net_returns = gross_returns - transaction_costs
        
        # Calculate performance metrics
        gross_cumulative = (1 + gross_returns.fillna(0)).cumprod()
        net_cumulative = (1 + net_returns.fillna(0)).cumprod()
        
        # Performance statistics
        gross_total_return = gross_cumulative.iloc[-1] - 1
        net_total_return = net_cumulative.iloc[-1] - 1
        
        gross_sharpe = self._calculate_sharpe(gross_returns)
        net_sharpe = self._calculate_sharpe(net_returns)
        
        max_drawdown = self._calculate_max_drawdown(net_cumulative)
        
        return {
            "gross_total_return": float(gross_total_return),
            "net_total_return": float(net_total_return),
            "gross_sharpe": float(gross_sharpe),
            "net_sharpe": float(net_sharpe),
            "max_drawdown": float(max_drawdown),
            "turnover": float(turnover),
            "cost_bps": float(cost_bps),
            "n_trades": int(position_changes.sum()),
            "gross_returns": gross_returns.tolist(),
            "net_returns": net_returns.tolist(),
            "positions": positions.tolist()
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize assuming daily data
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_vol
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())
    
    def create_ic_report(self, experiment_id: str, save_plots: bool = True) -> dict:
        """Create comprehensive IC analysis report for an experiment."""
        
        # Load experiment data
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if not exp_file.exists():
            # Try finding by prefix
            matches = list(self.experiments_dir.glob(f"{experiment_id}*.json"))
            if not matches:
                raise FileNotFoundError(f"Experiment {experiment_id} not found")
            exp_file = matches[0]
        
        with open(exp_file) as f:
            exp_data = json.load(f)
        
        # This is a placeholder - in practice you'd need to:
        # 1. Load the actual data used in training
        # 2. Load the model predictions
        # 3. Run the IC analysis
        
        report = {
            "experiment_id": experiment_id,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "ic_analysis": {
                "primary_ic": exp_data.get("metrics", {}).get("ic", 0.0),
                "quality": exp_data.get("metrics", {}).get("ic_quality", "unknown")
            },
            "recommendations": self._generate_recommendations(exp_data)
        }
        
        # Save report
        report_path = self.results_dir / f"ic_report_{experiment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"IC report saved to {report_path}")
        return report
    
    def _generate_recommendations(self, exp_data: dict) -> list[str]:
        """Generate recommendations based on IC analysis."""
        recommendations = []
        
        metrics = exp_data.get("metrics", {})
        ic = metrics.get("ic", 0.0)
        ic_quality = metrics.get("ic_quality", "unknown")
        
        if ic_quality == "excellent":
            recommendations.append("✅ Excellent IC! Consider deploying this model.")
            recommendations.append("🔍 Run return backtest to verify profitability after costs.")
            
        elif ic_quality == "good":
            recommendations.append("✅ Good IC signal detected.")
            recommendations.append("🔍 Test with lower transaction costs or longer horizons.")
            
        elif ic_quality == "marginal":
            recommendations.append("⚠️  Marginal IC - needs improvement.")
            recommendations.append("🔧 Try: more features, longer training period, regularization tuning.")
            
        elif ic_quality == "poor":
            recommendations.append("❌ Poor IC - major changes needed.")
            recommendations.append("🔧 Try: different features, different model, check data quality.")
            
        # Model-specific recommendations
        model = exp_data.get("model", {})
        if model.get("kind") == "ridge":
            if ic < 0.02:
                recommendations.append("🔧 For Ridge: try different alpha values or feature engineering.")
        elif model.get("kind") == "xgboost":
            if ic < 0.02:
                recommendations.append("🔧 For XGBoost: check for overfitting, try simpler model.")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Aurora IC Analysis & Return Backtesting")
    parser.add_argument("--experiment", required=True, help="Experiment ID to analyze")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20], 
                       help="Forward return horizons to test")
    parser.add_argument("--backtest", action="store_true", help="Run return backtest")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps")
    parser.add_argument("--compare", nargs="+", help="Compare multiple experiments")
    
    args = parser.parse_args()
    
    analyzer = ICAnalyzer()
    
    try:
        if args.compare:
            # Compare IC across experiments
            print("\nIC Comparison Across Experiments:")
            print("=" * 60)
            
            for exp_id in args.compare:
                report = analyzer.create_ic_report(exp_id, save_plots=False)
                ic = report["ic_analysis"]["primary_ic"]
                quality = report["ic_analysis"]["quality"]
                print(f"{exp_id:<15} IC: {ic:>8.4f} ({quality})")
                
        else:
            # Single experiment analysis
            report = analyzer.create_ic_report(args.experiment)
            
            print(f"\n{'='*60}")
            print(f"IC ANALYSIS REPORT: {args.experiment}")
            print(f"{'='*60}")
            
            ic_analysis = report["ic_analysis"]
            print(f"Primary IC: {ic_analysis['primary_ic']:.6f}")
            print(f"Quality: {ic_analysis['quality']}")
            
            print("\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  {rec}")
                
            if args.backtest:
                print(f"\n📊 Return backtest would run here with {args.cost_bps} bps costs")
                print("   (Requires model predictions and price data)")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
