#!/usr/bin/env python3
"""
Trust-but-Verify Production Results Validator

Validates production run results to catch bugs and suspicious metrics.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse


def validate_leaderboard(leaderboard_path: str) -> dict:
    """Validate leaderboard results for suspicious patterns"""
    issues = []
    
    try:
        df = pd.read_csv(leaderboard_path)
    except Exception as e:
        return {"status": "FAIL", "error": f"Cannot read leaderboard: {e}"}
    
    # Check 1: best_vs_BH should not equal best_median_sharpe
    if 'best_vs_BH' in df.columns and 'best_median_sharpe' in df.columns:
        identical_vs_bh = df['best_vs_BH'] == df['best_median_sharpe']
        if identical_vs_bh.any():
            issues.append(f"best_vs_BH equals best_median_sharpe for {identical_vs_bh.sum()} symbols")
    
    # Check 2: Missing median_trades/turnover with gate_pass=True
    if 'gate_pass' in df.columns:
        gate_pass = df['gate_pass'] == True
        if 'median_trades' in df.columns:
            missing_trades = gate_pass & df['median_trades'].isna()
            if missing_trades.any():
                issues.append(f"gate_pass=True but median_trades is NaN for {missing_trades.sum()} symbols")
        
        if 'median_turnover' in df.columns:
            missing_turnover = gate_pass & df['median_turnover'].isna()
            if missing_turnover.any():
                issues.append(f"gate_pass=True but median_turnover is NaN for {missing_turnover.sum()} symbols")
    
    # Check 3: Suspiciously high Sharpe ratios (but allow for cross-sectional models)
    if 'best_median_sharpe' in df.columns:
        high_sharpe = df['best_median_sharpe'] > 8.0  # Very high threshold for cross-sectional models
        if high_sharpe.any():
            issues.append(f"Extremely high Sharpe (>8.0) for {high_sharpe.sum()} symbols: {df[high_sharpe]['ticker'].tolist()}")
    
    # Check 4: Unrealistic runtime for number of configs
    if 'runtime_sec' in df.columns and 'num_configs' in df.columns:
        fast_runs = (df['runtime_sec'] < 5) & (df['num_configs'] >= 20)
        if fast_runs.any():
            issues.append(f"Unrealistically fast runs (<5s for >=20 configs) for {fast_runs.sum()} symbols")
    
    # Check 5: All successful runs should have num_configs > 0
    if 'num_configs' in df.columns:
        zero_configs = df['num_configs'] == 0
        if zero_configs.any():
            issues.append(f"Zero configs for {zero_configs.sum()} symbols")
    
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "total_symbols": len(df),
        "gate_passes": df.get('gate_pass', pd.Series()).sum() if 'gate_pass' in df.columns else 0
    }



def validate_symbol_results(symbol: str, results_dir: Path) -> dict:
    """Validate individual symbol results"""
    issues = []
    
    # Check for cross-sectional mode first (panel_predictions.parquet exists)
    panel_file = results_dir / "panel_predictions.parquet"
    if panel_file.exists():
        # In cross-sectional mode, validate that symbol exists in panel
        try:
            import pandas as pd
            panel = pd.read_parquet(panel_file)
            if symbol not in panel['symbol'].values:
                issues.append(f"Symbol {symbol} not found in panel predictions")
                return {"status": "FAIL", "issues": issues}
            
            # Check for reasonable metrics in leaderboard
            leaderboard_file = results_dir / "leaderboard.csv"
            if leaderboard_file.exists():
                leaderboard = pd.read_csv(leaderboard_file)
                symbol_data = leaderboard[leaderboard['ticker'] == symbol]
                if len(symbol_data) == 0:
                    issues.append(f"Symbol {symbol} not found in leaderboard")
                    return {"status": "FAIL", "issues": issues}
                
                symbol_row = symbol_data.iloc[0]
                if pd.isna(symbol_row['best_median_sharpe']):
                    issues.append("Missing Sharpe ratio")
                if pd.isna(symbol_row['median_trades']) or symbol_row['median_trades'] == 0:
                    issues.append("No trades recorded")
                if pd.isna(symbol_row['median_turnover']):
                    issues.append("Missing turnover data")
            
            return {
                "status": "PASS" if not issues else "FAIL",
                "issues": issues,
                "total_configs": 1,  # Cross-sectional mode has 1 config per symbol
                "successful_configs": 1 if not issues else 0
            }
        except Exception as e:
            issues.append(f"Error validating cross-sectional results: {e}")
            return {"status": "FAIL", "issues": issues}
    
    # Check for flat structure (symbol_grid.csv in results dir)
    grid_csv = results_dir / f"{symbol}_grid.csv"
    if not grid_csv.exists():
        # Check for nested structure (symbol/symbol_grid.csv)
        symbol_path = results_dir / symbol
        if symbol_path.exists():
            grid_csv = symbol_path / f"{symbol}_grid.csv"
        else:
            issues.append("Missing grid CSV file")
            return {"status": "FAIL", "issues": issues}
    
    if not grid_csv.exists():
        issues.append("Missing grid CSV file")
        return {"status": "FAIL", "issues": issues}
    
    try:
        df = pd.read_csv(grid_csv)
    except Exception as e:
        issues.append(f"Cannot read grid CSV: {e}")
        return {"status": "FAIL", "issues": issues}
    
    # Check for successful runs
    if 'success' in df.columns:
        successful = df['success'].sum()
        if successful == 0:
            issues.append("No successful configurations")
        elif successful < len(df) * 0.5:
            issues.append(f"Low success rate: {successful}/{len(df)}")
    
    # Check for reasonable Sharpe ratios
    if 'median_model_sharpe' in df.columns:
        sharpe_vals = df['median_model_sharpe'].dropna()
        if len(sharpe_vals) > 0:
            if sharpe_vals.max() > 5.0:
                issues.append(f"Extremely high Sharpe: {sharpe_vals.max():.3f}")
            if sharpe_vals.std() == 0 and len(sharpe_vals) > 1:
                issues.append("All Sharpe ratios identical (suspicious)")
    
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "total_configs": len(df),
        "successful_configs": df.get('success', pd.Series()).sum() if 'success' in df.columns else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Validate production results")
    parser.add_argument("--results-dir", required=True, help="Results directory to validate")
    parser.add_argument("--symbol", help="Validate specific symbol only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"🔍 Validating results in: {results_dir}")
    
    # Validate leaderboard
    leaderboard_path = results_dir / "leaderboard.csv"
    if leaderboard_path.exists():
        print(f"\n📊 Validating leaderboard...")
        leaderboard_result = validate_leaderboard(str(leaderboard_path))
        
        if leaderboard_result["status"] == "PASS":
            print(f"✅ Leaderboard validation PASSED")
            print(f"   Total symbols: {leaderboard_result['total_symbols']}")
            print(f"   Gate passes: {leaderboard_result['gate_passes']}")
        else:
            print(f"❌ Leaderboard validation FAILED")
            for issue in leaderboard_result["issues"]:
                print(f"   - {issue}")
    else:
        print(f"⚠️  No leaderboard.csv found")
    
    # Validate individual symbols - drive from leaderboard.csv
    symbols_to_check = []
    missing = []
    
    if leaderboard_path.exists():
        import csv
        with leaderboard_path.open() as f:
            reader = csv.DictReader(f)
            tickers = [row["ticker"] for row in reader]
        
        for t in tickers:
            # Check for flat structure first (symbol_grid.csv in results dir)
            grid_file = results_dir / f"{t}_grid.csv"
            if grid_file.exists():
                symbols_to_check.append(t)
                continue
            
            # Check for nested structure (symbol/symbol_grid.csv)
            tdir = results_dir / t
            if tdir.exists() and (tdir / f"{t}_grid.csv").exists():
                symbols_to_check.append(t)
                continue
            
            # Check for cross-sectional mode (panel_predictions.parquet exists)
            panel_file = results_dir / "panel_predictions.parquet"
            if panel_file.exists():
                symbols_to_check.append(t)  # In CS mode, all symbols are in the panel
                continue
            
            missing.append(t)
    else:
        # Fallback: check for any _grid.csv files
        grid_files = list(results_dir.glob("*_grid.csv"))
        symbols_to_check = [f.stem.replace("_grid", "") for f in grid_files[:5]]
    
    print(f"\n🔍 Validating {len(symbols_to_check)} symbols...")
    
    if len(symbols_to_check) == 0:
        print("❌ No per-symbol results found; failing hard.")
        if missing:
            print(f"   Missing artifacts for: {', '.join(missing)}")
        sys.exit(2)
    
    if missing:
        print(f"⚠️  Missing artifacts for: {', '.join(missing)}")
    
    symbol_issues = []
    for symbol in symbols_to_check:
        result = validate_symbol_results(symbol, results_dir)
        
        if result["status"] == "PASS":
            print(f"✅ {symbol}: PASS ({result['successful_configs']}/{result['total_configs']} configs)")
        else:
            print(f"❌ {symbol}: FAIL")
            if "issues" in result:
                for issue in result["issues"]:
                    print(f"   - {issue}")
                symbol_issues.extend(result["issues"])
            else:
                print(f"   - {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n📋 Summary:")
    if leaderboard_result.get("status") == "PASS" and not symbol_issues:
        print(f"✅ All validations PASSED")
        sys.exit(0)
    else:
        print(f"❌ Validations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
