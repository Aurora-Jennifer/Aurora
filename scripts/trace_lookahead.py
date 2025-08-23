#!/usr/bin/env python3
"""
Trace Lookahead Contamination Script
Analyzes data to identify the source of lookahead contamination.
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def analyze_lookahead_contamination(file_path: str) -> dict:
    """
    Analyze lookahead contamination in a data file.
    
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
        
        results = {
            "status": "analysis_complete",
            "total_rows": len(df),
            "lookahead_issues": [],
            "returns_analysis": {},
            "feature_analysis": {}
        }
        
        # Check if Returns column exists
        if "Returns" in df.columns:
            returns = df["Returns"]
            results["returns_analysis"]["exists"] = True
            results["returns_analysis"]["total_values"] = len(returns)
            results["returns_analysis"]["finite_values"] = returns.notna().sum()
            results["returns_analysis"]["unique_values"] = returns.nunique()
            
            # Check for exact matches with future values
            if len(returns) > 1:
                # Check offset 1
                future_1 = returns.shift(-1)
                exact_matches_1 = (returns == future_1) & returns.notna() & future_1.notna()
                
                if exact_matches_1.any():
                    match_count_1 = exact_matches_1.sum()
                    match_indices_1 = exact_matches_1[exact_matches_1].index.tolist()
                    results["lookahead_issues"].append({
                        "type": "returns_exact_match_offset_1",
                        "count": int(match_count_1),
                        "indices": match_indices_1[:10],  # First 10
                        "sample_values": returns[exact_matches_1].head(5).tolist()
                    })
                
                # Check offset 2
                future_2 = returns.shift(-2)
                exact_matches_2 = (returns == future_2) & returns.notna() & future_2.notna()
                
                if exact_matches_2.any():
                    match_count_2 = exact_matches_2.sum()
                    match_indices_2 = exact_matches_2[exact_matches_2].index.tolist()
                    results["lookahead_issues"].append({
                        "type": "returns_exact_match_offset_2",
                        "count": int(match_count_2),
                        "indices": match_indices_2[:10],  # First 10
                        "sample_values": returns[exact_matches_2].head(5).tolist()
                    })
            
            # Check for suspicious patterns
            if len(returns) > 10:
                # Check for consecutive identical values
                consecutive_identical = (returns == returns.shift(1)) & returns.notna()
                if consecutive_identical.any():
                    results["lookahead_issues"].append({
                        "type": "consecutive_identical_returns",
                        "count": int(consecutive_identical.sum()),
                        "description": "Returns values that are identical to previous values"
                    })
                
                # Check for zero returns (might indicate data issues)
                zero_returns = (returns == 0) & returns.notna()
                if zero_returns.any():
                    results["lookahead_issues"].append({
                        "type": "zero_returns",
                        "count": int(zero_returns.sum()),
                        "description": "Returns values that are exactly zero"
                    })
        else:
            results["returns_analysis"]["exists"] = False
        
        # Analyze price columns for potential lookahead
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                series = df[col]
                if len(series) > 1:
                    # Check for consecutive identical prices
                    consecutive_identical = (series == series.shift(1)) & series.notna()
                    if consecutive_identical.any():
                        results["lookahead_issues"].append({
                            "type": f"consecutive_identical_{col.lower()}",
                            "count": int(consecutive_identical.sum()),
                            "description": f"{col} values that are identical to previous values"
                        })
        
        return results
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Trace and analyze lookahead contamination in data files")
    parser.add_argument("file", help="Path to data file to analyze")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    results = analyze_lookahead_contamination(args.file)
    
    if results["status"] == "error":
        print(f"❌ Error: {results['message']}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print(f"🔍 Lookahead Contamination Analysis")
    print(f"📊 Total rows: {results['total_rows']}")
    print()
    
    if results["returns_analysis"]["exists"]:
        print("📈 Returns Analysis:")
        print(f"  Total values: {results['returns_analysis']['total_values']}")
        print(f"  Finite values: {results['returns_analysis']['finite_values']}")
        print(f"  Unique values: {results['returns_analysis']['unique_values']}")
        print()
    
    if results["lookahead_issues"]:
        print("⚠️  Lookahead Issues Found:")
        for issue in results["lookahead_issues"]:
            print(f"  {issue['type']}: {issue['count']} occurrences")
            if "description" in issue:
                print(f"    Description: {issue['description']}")
            if "sample_values" in issue:
                print(f"    Sample values: {issue['sample_values']}")
            if "indices" in issue:
                print(f"    First few indices: {issue['indices']}")
            print()
    else:
        print("✅ No lookahead issues detected")
    
    # Save detailed report if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
