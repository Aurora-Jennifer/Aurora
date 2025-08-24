#!/usr/bin/env python3
"""
Trace Bad Prices Script
Analyzes non-positive prices to identify their source and patterns.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def analyze_bad_prices(file_path: str) -> dict:
    """
    Analyze bad prices in a data file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load data
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"Unsupported file format: {file_path}", file=sys.stderr)
            return {}

        # Find bad prices
        price_cols = ["Open", "High", "Low", "Close"]
        bad_rows = pd.DataFrame()

        for col in price_cols:
            if col in df.columns:
                bad_mask = df[col] <= 0
                if bad_mask.any():
                    bad_rows = pd.concat([bad_rows, df[bad_mask]])

        if bad_rows.empty:
            return {"status": "clean", "message": "No bad prices found"}

        # Remove duplicates
        bad_rows = bad_rows.drop_duplicates()

        # Analysis
        results = {
            "status": "bad_prices_found",
            "total_bad_rows": len(bad_rows),
            "bad_by_column": {},
            "bad_by_date": {},
            "bad_by_symbol": {},
            "sample_bad_rows": []
        }

        # Analyze by column
        for col in price_cols:
            if col in df.columns:
                bad_count = (df[col] <= 0).sum()
                if bad_count > 0:
                    results["bad_by_column"][col] = int(bad_count)

        # Analyze by date
        if isinstance(df.index, pd.DatetimeIndex):
            bad_by_date = bad_rows.groupby(bad_rows.index.date).size().sort_values(ascending=False)
            results["bad_by_date"] = bad_by_date.head(10).to_dict()

        # Analyze by symbol (if multi-index)
        if isinstance(df.index, pd.MultiIndex) and 'symbol' in df.index.names:
            symbol_level = df.index.names.index('symbol')
            bad_by_symbol = bad_rows.groupby(level=symbol_level).size().sort_values(ascending=False)
            results["bad_by_symbol"] = bad_by_symbol.to_dict()

        # Sample bad rows
        results["sample_bad_rows"] = bad_rows.head(5).to_dict('records')

        return results

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Trace and analyze bad prices in data files")
    parser.add_argument("file", help="Path to data file to analyze")
    parser.add_argument("--output", "-o", help="Output file for detailed report")

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    results = analyze_bad_prices(args.file)

    if results["status"] == "clean":
        print("✅ No bad prices found")
        return

    if results["status"] == "error":
        print(f"❌ Error: {results['message']}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"🔍 Found {results['total_bad_rows']} rows with bad prices")
    print()

    print("📊 Bad prices by column:")
    for col, count in results["bad_by_column"].items():
        print(f"  {col}: {count}")
    print()

    if results["bad_by_date"]:
        print("📅 Top 10 dates with bad prices:")
        for date, count in results["bad_by_date"].items():
            print(f"  {date}: {count}")
        print()

    if results["bad_by_symbol"]:
        print("🏷️ Bad prices by symbol:")
        for symbol, count in results["bad_by_symbol"].items():
            print(f"  {symbol}: {count}")
        print()

    print("📋 Sample bad rows:")
    for i, row in enumerate(results["sample_bad_rows"], 1):
        print(f"  Row {i}: {row}")

    # Save detailed report if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
