#!/usr/bin/env python3
"""
Baseline Comparison Script
Compares strategy performance against simple baseline strategies
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import sys
import os
import json

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from baseline_strategies import run_baseline_strategies, compare_with_baselines
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch market data for baseline comparison."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        
        # Rename columns to match expected format
        data = data.rename(columns={'Close': 'close'})
        
        return data[['close', 'returns']].dropna()
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise


def load_strategy_results(results_file: str) -> Dict:
    """Load strategy results from CSV file."""
    try:
        df = pd.read_csv(results_file)
        
        # Extract key metrics
        results = {
            'sharpe_ratio': df['median_model_sharpe'].median(),
            'annualized_return': df['median_model_sharpe'].median() * 0.2,  # Rough estimate
            'volatility': 0.2,  # Rough estimate
            'max_drawdown': df['mean_mdd'].median(),
            'trades': df['mean_trades'].median()
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading strategy results: {e}")
        return {}


def generate_baseline_report(strategy_results: Dict, baseline_results: Dict, 
                           symbol: str, output_dir: str) -> None:
    """Generate baseline comparison report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    
    # Add strategy results
    comparison_data.append({
        'strategy': 'ML Strategy',
        'type': 'main',
        'sharpe_ratio': strategy_results.get('sharpe_ratio', np.nan),
        'annualized_return': strategy_results.get('annualized_return', np.nan),
        'volatility': strategy_results.get('volatility', np.nan),
        'max_drawdown': strategy_results.get('max_drawdown', np.nan),
        'trades': strategy_results.get('trades', 0)
    })
    
    # Add baseline results
    for strategy_name, results in baseline_results.items():
        if 'error' not in results:
            comparison_data.append({
                'strategy': strategy_name,
                'type': 'baseline',
                'sharpe_ratio': results.get('sharpe_ratio', np.nan),
                'annualized_return': results.get('annualized_return', np.nan),
                'volatility': results.get('volatility', np.nan),
                'max_drawdown': results.get('max_drawdown', np.nan),
                'trades': results.get('trades', 0)
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate relative performance
    if len(df) > 0:
        buy_hold_sharpe = df[df['strategy'] == 'Buy and Hold']['sharpe_ratio'].iloc[0] if 'Buy and Hold' in df['strategy'].values else np.nan
        best_baseline_sharpe = df[df['type'] == 'baseline']['sharpe_ratio'].max()
        
        df['vs_buy_hold'] = df['sharpe_ratio'] - buy_hold_sharpe
        df['vs_best_baseline'] = df['sharpe_ratio'] - best_baseline_sharpe
    
    # Save results
    df.to_csv(output_path / f"baseline_comparison_{symbol}.csv", index=False)
    
    # Generate markdown report
    report_content = generate_markdown_report(df, symbol, strategy_results, baseline_results)
    with open(output_path / f"baseline_report_{symbol}.md", 'w') as f:
        f.write(report_content)
    
    logger.info(f"Baseline comparison report saved to {output_path}")


def generate_markdown_report(df: pd.DataFrame, symbol: str, 
                           strategy_results: Dict, baseline_results: Dict) -> str:
    """Generate markdown baseline comparison report."""
    
    ml_sharpe = strategy_results.get('sharpe_ratio', np.nan)
    buy_hold_sharpe = df[df['strategy'] == 'Buy and Hold']['sharpe_ratio'].iloc[0] if 'Buy and Hold' in df['strategy'].values else np.nan
    best_baseline_sharpe = df[df['type'] == 'baseline']['sharpe_ratio'].max()
    
    # Determine performance status
    if not np.isnan(ml_sharpe) and not np.isnan(buy_hold_sharpe):
        if ml_sharpe > buy_hold_sharpe + 0.1:
            status = "🟢 **OUTPERFORMING**"
        elif ml_sharpe > buy_hold_sharpe:
            status = "🟡 **SLIGHTLY BETTER**"
        else:
            status = "🔴 **UNDERPERFORMING**"
    else:
        status = "❓ **UNKNOWN**"
    
    report = f"""# Baseline Strategy Comparison Report

## Symbol: {symbol}

## Performance Status: {status}

## Summary

- **ML Strategy Sharpe**: {ml_sharpe:.3f}
- **Buy & Hold Sharpe**: {buy_hold_sharpe:.3f}
- **Best Baseline Sharpe**: {best_baseline_sharpe:.3f}
- **Alpha vs Buy & Hold**: {ml_sharpe - buy_hold_sharpe:.3f}
- **Alpha vs Best Baseline**: {ml_sharpe - best_baseline_sharpe:.3f}

## Strategy Comparison

| Strategy | Type | Sharpe | Return | Volatility | Max DD | Trades | vs Buy&Hold | vs Best |
|----------|------|--------|--------|------------|--------|--------|-------------|---------|
"""
    
    for _, row in df.iterrows():
        type_icon = "🤖" if row['type'] == 'main' else "📊"
        report += f"| {type_icon} {row['strategy']} | {row['type']} | {row['sharpe_ratio']:.3f} | {row['annualized_return']:.3f} | {row['volatility']:.3f} | {row['max_drawdown']:.3f} | {row['trades']:.0f} | {row['vs_buy_hold']:.3f} | {row['vs_best_baseline']:.3f} |\n"
    
    report += f"""
## Interpretation

### Performance Analysis
- **ML Strategy**: {ml_sharpe:.3f} Sharpe ratio
- **Buy & Hold**: {buy_hold_sharpe:.3f} Sharpe ratio
- **Best Baseline**: {best_baseline_sharpe:.3f} Sharpe ratio

### Alpha Assessment
- **vs Buy & Hold**: {ml_sharpe - buy_hold_sharpe:+.3f} Sharpe difference
- **vs Best Baseline**: {ml_sharpe - best_baseline_sharpe:+.3f} Sharpe difference

### Recommendations

"""
    
    if ml_sharpe > best_baseline_sharpe + 0.1:
        report += """✅ **STRONG ALPHA**: ML strategy significantly outperforms all baselines
- Strategy shows genuine alpha generation
- Consider scaling up allocation
- Monitor for consistency across different market regimes
"""
    elif ml_sharpe > buy_hold_sharpe:
        report += """⚠️ **MARGINAL ALPHA**: ML strategy slightly better than buy & hold
- Strategy shows some alpha but may not justify complexity
- Consider simplifying approach or improving features
- Monitor transaction costs carefully
"""
    else:
        report += """❌ **NO ALPHA**: ML strategy underperforms simple baselines
- Strategy complexity not justified by performance
- Consider fundamental strategy review
- Focus on feature engineering or model improvements
"""
    
    report += f"""
## Baseline Strategy Details

"""
    
    for strategy_name, results in baseline_results.items():
        if 'error' not in results:
            report += f"""### {strategy_name}
- **Sharpe Ratio**: {results['sharpe_ratio']:.3f}
- **Annualized Return**: {results['annualized_return']:.3f}
- **Volatility**: {results['volatility']:.3f}
- **Max Drawdown**: {results['max_drawdown']:.3f}
- **Total Trades**: {results['trades']:.0f}

"""
    
    report += """---
*Generated by baseline_comparison.py*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare strategy performance with baseline strategies')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--strategy-results', required=True, help='Path to strategy results CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory for reports')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for data')
    parser.add_argument('--end-date', default='2024-01-01', help='End date for data')
    parser.add_argument('--costs-bps', type=float, default=3.0, help='Transaction costs in basis points')
    
    args = parser.parse_args()
    
    try:
        # Fetch market data
        logger.info(f"Fetching data for {args.symbol}")
        data = fetch_market_data(args.symbol, args.start_date, args.end_date)
        logger.info(f"Loaded {len(data)} days of data")
        
        # Load strategy results
        logger.info(f"Loading strategy results from {args.strategy_results}")
        strategy_results = load_strategy_results(args.strategy_results)
        
        # Stamp context for reproducibility
        context = {
            "symbol": args.symbol,
            "start": str(data.index.min()) if len(data) else None,
            "end": str(data.index.max()) if len(data) else None,
            "costs_bps": float(args.costs_bps),
            "annualization": "252d",
            "data_source": "yfinance",
            "auto_adjust": True  # yfinance default changed
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "baseline_context.json"), "w") as f:
            json.dump(context, f, indent=2)
        
        # Run baseline strategies
        logger.info("Running baseline strategies")
        baseline_results = run_baseline_strategies(data, costs_bps=args.costs_bps)
        
        # Generate comparison report
        logger.info("Generating baseline comparison report")
        generate_baseline_report(strategy_results, baseline_results, args.symbol, args.output_dir)
        
        logger.info("Baseline comparison complete!")
        
    except Exception as e:
        logger.error(f"Error in baseline comparison: {e}")
        raise


if __name__ == "__main__":
    main()
