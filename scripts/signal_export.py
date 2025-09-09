#!/usr/bin/env python3
"""
Signal Export for Paper Trading

Exports trading signals from ensemble results in a standardized format
for paper trading and live deployment.
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
import logging
import json
from typing import Dict, List, Optional
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble_results(pred_dir: Path) -> Dict:
    """Load ensemble results"""
    ensemble_file = pred_dir / "demo_results.json"
    if not ensemble_file.exists():
        raise FileNotFoundError(f"Ensemble results not found: {ensemble_file}")
    
    with open(ensemble_file, 'r') as f:
        return json.load(f)


def load_leaderboard(results_dir: Path) -> pd.DataFrame:
    """Load leaderboard results"""
    leaderboard_file = results_dir / "leaderboard.csv"
    if not leaderboard_file.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_file}")
    
    return pd.read_csv(leaderboard_file)


def export_signals(ensemble_results: Dict, leaderboard: pd.DataFrame, 
                  out_dir: Path, schema_path: Optional[str] = None) -> Dict:
    """Export trading signals in standardized format"""
    
    # Filter to gate-passing symbols
    gate_pass_symbols = leaderboard[leaderboard['gate_pass'] == True]['ticker'].tolist()
    
    if len(gate_pass_symbols) == 0:
        raise ValueError("No symbols passed gate criteria")
    
    logger.info(f"Exporting signals for {len(gate_pass_symbols)} gate-passing symbols")
    
    # Create signal export structure
    signal_export = {
        "metadata": {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "total_symbols": len(gate_pass_symbols),
            "ensemble_weights": ensemble_results.get("weights", {}),
            "schema_version": "1.0"
        },
        "signals": {}
    }
    
    # For each symbol, create signal entry
    for symbol in gate_pass_symbols:
        symbol_data = leaderboard[leaderboard['ticker'] == symbol].iloc[0]
        
        # Create signal entry
        signal_entry = {
            "symbol": symbol,
            "signal_strength": min(symbol_data['best_median_sharpe'] / 2.0, 1.0),  # Normalize to [0,1]
            "confidence": min(symbol_data['best_median_sharpe'] / 3.0, 1.0),  # Confidence based on Sharpe
            "expected_return": symbol_data['best_median_sharpe'] * 0.1,  # Rough annualized return
            "risk_score": 1.0 / (symbol_data['best_median_sharpe'] + 0.1),  # Inverse of Sharpe
            "turnover": symbol_data.get('median_turnover', 0.1),
            "trades_per_month": symbol_data.get('median_trades', 10),
            "last_updated": pd.Timestamp.now().isoformat(),
            "status": "active"
        }
        
        signal_export["signals"][symbol] = signal_entry
    
    # Save signals
    signals_file = out_dir / "signals.json"
    with open(signals_file, 'w') as f:
        json.dump(signal_export, f, indent=2)
    
    # Create CSV for easy consumption
    signals_df = pd.DataFrame.from_dict(signal_export["signals"], orient='index')
    signals_df.to_csv(out_dir / "signals.csv", index_label="symbol")
    
    # Create summary
    summary = {
        "total_signals": len(gate_pass_symbols),
        "avg_signal_strength": signals_df['signal_strength'].mean(),
        "avg_confidence": signals_df['confidence'].mean(),
        "avg_expected_return": signals_df['expected_return'].mean(),
        "high_confidence_signals": len(signals_df[signals_df['confidence'] > 0.7]),
        "export_files": ["signals.json", "signals.csv"]
    }
    
    with open(out_dir / "export_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export trading signals for paper trading")
    parser.add_argument("--pred-dir", required=True, help="Directory containing ensemble predictions")
    parser.add_argument("--out-dir", required=True, help="Output directory for signals")
    parser.add_argument("--schema", help="Path to signal export schema documentation")
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading ensemble results from: {pred_dir}")
    logger.info(f"Exporting signals to: {out_dir}")
    
    try:
        # Load ensemble results
        ensemble_results = load_ensemble_results(pred_dir)
        
        # Load leaderboard (assume it's in parent directory)
        results_dir = pred_dir.parent
        leaderboard = load_leaderboard(results_dir)
        
        # Export signals
        summary = export_signals(ensemble_results, leaderboard, out_dir, args.schema)
        
        # Print summary
        print(f"\n📊 Signal Export Summary:")
        print(f"Total signals: {summary['total_signals']}")
        print(f"Average signal strength: {summary['avg_signal_strength']:.3f}")
        print(f"Average confidence: {summary['avg_confidence']:.3f}")
        print(f"High confidence signals: {summary['high_confidence_signals']}")
        print(f"Average expected return: {summary['avg_expected_return']:.3f}")
        
        print(f"\n📁 Files exported:")
        for file in summary['export_files']:
            print(f"  - {out_dir / file}")
        
        print(f"\n✅ Signal export completed successfully!")
        
    except Exception as e:
        logger.error(f"Signal export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
