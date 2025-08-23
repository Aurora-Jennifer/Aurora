#!/usr/bin/env python3
"""
Check for non-positive prices in data files.
Used as a pre-gate to ensure data quality before processing.
"""

import sys
import pandas as pd
import argparse
from pathlib import Path


def check_non_positive_prices(file_path: str, price_cols: list = None) -> int:
    """
    Check for non-positive prices in a data file.
    
    Args:
        file_path: Path to the data file (parquet, csv, etc.)
        price_cols: List of price columns to check
        
    Returns:
        0 if no non-positive prices found, 1 if found
    """
    if price_cols is None:
        price_cols = ["Open", "High", "Low", "Close"]
    
    try:
        # Load data
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"Unsupported file format: {file_path}", file=sys.stderr)
            return 1
        
        # Check for non-positive prices
        bad_counts = {}
        total_bad = 0
        
        for col in price_cols:
            if col in df.columns:
                bad_mask = df[col] <= 0
                bad_count = bad_mask.sum()
                if bad_count > 0:
                    bad_counts[col] = bad_count
                    total_bad += bad_count
        
        if total_bad > 0:
            details = ", ".join(f"{col}={count}" for col, count in bad_counts.items())
            print(f"FATAL: found {total_bad} non-positive prices ({details})", file=sys.stderr)
            return 1
        else:
            print("OK: No non-positive prices found")
            return 0
            
    except Exception as e:
        print(f"Error checking file {file_path}: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Check for non-positive prices in data files")
    parser.add_argument("file", help="Path to data file to check")
    parser.add_argument("--cols", nargs="+", default=["Open", "High", "Low", "Close"], 
                       help="Price columns to check")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    exit_code = check_non_positive_prices(args.file, args.cols)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
