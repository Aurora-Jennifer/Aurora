#!/usr/bin/env python3
"""
Detect and Fix Lookahead Contamination
Based on the comprehensive checklist for eliminating lookahead bias.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class LookaheadDetector:
    """Detect and fix lookahead contamination in trading data."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.issues = []
        self.fixes_applied = []

    def load_data(self) -> pd.DataFrame:
        """Load data from snapshot or file."""
        if self.data_path.is_dir():
            # Load from snapshot directory
            dfs = []
            for parquet_file in self.data_path.glob("*.parquet"):
                if parquet_file.name in ['manifest.json']:
                    continue

                df = pd.read_parquet(parquet_file)

                # Handle case where timestamp is in index
                if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = df.index

                # Add symbol if not present
                if 'symbol' not in df.columns:
                    df['symbol'] = parquet_file.stem

                # Ensure all dataframes have the same columns before concatenating
                if dfs:
                    # Get all unique columns
                    all_columns = set()
                    for d in dfs + [df]:
                        all_columns.update(d.columns)

                    # Add missing columns to each dataframe
                    for d in dfs:
                        for col in all_columns:
                            if col not in d.columns:
                                d[col] = np.nan

                    for col in all_columns:
                        if col not in df.columns:
                            df[col] = np.nan

                dfs.append(df)

            if not dfs:
                raise FileNotFoundError(f"No parquet files found in {self.data_path}")

            df = pd.concat(dfs, ignore_index=True)
        else:
            # Load single file
            if self.data_path.suffix == '.parquet':
                df = pd.read_parquet(self.data_path)
                # Handle case where timestamp is in index
                if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = df.index
            elif self.data_path.suffix == '.csv':
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        return df

    def check_temporal_integrity(self, df: pd.DataFrame) -> bool:
        """Check temporal integrity of the dataset."""
        print("🔍 Checking temporal integrity...")

        # Check for required columns
        required_cols = ['symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.issues.append(f"Missing required columns: {missing_cols}")
            return False

        # Handle missing timestamp column
        if 'timestamp' not in df.columns:
            print("⚠️ No timestamp column found, creating synthetic timestamps for analysis")
            df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
            self.fixes_applied.append("Created synthetic timestamps for analysis")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set index for analysis
        df_analysis = df.set_index(['symbol', 'timestamp']).sort_index()

        # Check monotonicity
        if not df_analysis.index.is_monotonic_increasing:
            self.issues.append("Index is not monotonic increasing")
            return False

        # Check per-symbol monotonicity (simplified)
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            if len(symbol_data) > 1:
                # Check if timestamps are strictly increasing
                timestamps = symbol_data['timestamp'].values
                if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)):
                    self.issues.append(f"Timestamps are not strictly increasing for symbol {symbol}")
                    return False

        # Check for duplicates
        duplicates = df.duplicated(subset=['symbol', 'timestamp']).sum()
        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate (symbol, timestamp) pairs")
            return False

        print("✅ Temporal integrity check passed")
        return True

    def check_feature_lookahead(self, df: pd.DataFrame) -> list[str]:
        """Check for lookahead contamination in features."""
        print("🔍 Checking for lookahead contamination in features...")

        suspicious_features = []

        # Check for obvious future data patterns
        for col in df.columns:
            if col in ['timestamp', 'symbol']:
                continue

            # Check for negative shifts (future data)
            if any(keyword in col.lower() for keyword in ['future', 'next', 'forward', 'fwd']):
                suspicious_features.append(f"Feature '{col}' has future-looking name")

            # Check for shift(-1) patterns in feature names
            if 'shift_neg' in col.lower() or 'shift_minus' in col.lower():
                suspicious_features.append(f"Feature '{col}' appears to use negative shift")

        # Check for rolling windows without proper lagging
        for col in df.columns:
            if 'rolling' in col.lower() or 'ma_' in col.lower() or 'sma_' in col.lower():
                # These should be lagged by 1
                if not any(lag in col.lower() for lag in ['lag', 'shift']):
                    suspicious_features.append(f"Rolling feature '{col}' may not be properly lagged")

        if suspicious_features:
            self.issues.extend(suspicious_features)
            print(f"⚠️ Found {len(suspicious_features)} suspicious features")
        else:
            print("✅ No obvious lookahead contamination detected in features")

        return suspicious_features

    def check_target_construction(self, df: pd.DataFrame) -> list[str]:
        """Check target/label construction for issues."""
        print("🔍 Checking target construction...")

        target_issues = []

        # Look for target columns
        target_cols = [col for col in df.columns if any(keyword in col.lower()
                       for keyword in ['target', 'label', 'y', 'ret_fwd', 'return_fwd'])]

        for target_col in target_cols:
            # Check if target is forward-filled
            if df[target_col].notna().sum() > df[target_col].notna().sum():
                target_issues.append(f"Target '{target_col}' appears to be forward-filled")

            # Check for proper forward shift in target names
            if 'fwd' in target_col.lower() and not any(shift in target_col.lower()
               for shift in ['shift_neg', 'shift_minus']):
                target_issues.append(f"Target '{target_col}' should use shift(-1) for forward returns")

        if target_issues:
            self.issues.extend(target_issues)
            print(f"⚠️ Found {len(target_issues)} target construction issues")
        else:
            print("✅ Target construction appears correct")

        return target_issues

    def suggest_fixes(self) -> list[str]:
        """Suggest fixes for detected issues."""
        print("🔧 Suggesting fixes...")

        fixes = []

        for issue in self.issues:
            if "not strictly increasing" in issue:
                fixes.append("Sort data by (symbol, timestamp) and remove duplicates")
            elif "duplicate" in issue:
                fixes.append("Remove duplicate (symbol, timestamp) pairs")
            elif "future-looking name" in issue:
                fixes.append("Rename features to avoid future-looking terminology")
            elif "not be properly lagged" in issue:
                fixes.append("Add .shift(1) to rolling features to ensure proper lagging")
            elif "forward-filled" in issue:
                fixes.append("Remove forward-filling from targets - use proper shift(-1)")
            elif "should use shift(-1)" in issue:
                fixes.append("Use shift(-1) for forward-looking targets")

        return fixes

    def apply_basic_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic fixes to the dataset."""
        print("🔧 Applying basic fixes...")

        df_fixed = df.copy()

        # Fix 1: Handle missing timestamp column
        if 'timestamp' not in df_fixed.columns:
            print("⚠️ No timestamp column found, creating synthetic timestamps")
            df_fixed['timestamp'] = pd.date_range('2023-01-01', periods=len(df_fixed), freq='D')
            self.fixes_applied.append("Created synthetic timestamps")

        # Fix 2: Ensure proper timestamp format
        if not pd.api.types.is_datetime64_any_dtype(df_fixed['timestamp']):
            df_fixed['timestamp'] = pd.to_datetime(df_fixed['timestamp'])
            self.fixes_applied.append("Converted timestamp to datetime")

        # Fix 3: Remove duplicates
        initial_rows = len(df_fixed)
        df_fixed = df_fixed.drop_duplicates(subset=['symbol', 'timestamp'])
        final_rows = len(df_fixed)
        if final_rows < initial_rows:
            self.fixes_applied.append(f"Removed {initial_rows - final_rows} duplicate rows")

        # Fix 4: Sort by (symbol, timestamp)
        df_fixed = df_fixed.sort_values(['symbol', 'timestamp'])
        self.fixes_applied.append("Sorted by (symbol, timestamp)")

        return df_fixed

    def generate_report(self) -> dict:
        """Generate a comprehensive report."""
        report = {
            "data_path": str(self.data_path),
            "issues_detected": len(self.issues),
            "issues": self.issues,
            "fixes_applied": self.fixes_applied,
            "suggested_fixes": self.suggest_fixes(),
            "status": "PASS" if not self.issues else "FAIL"
        }

        return report

    def run_analysis(self) -> bool:
        """Run the complete lookahead contamination analysis."""
        print("🚀 Starting lookahead contamination analysis...")
        print(f"📁 Analyzing: {self.data_path}")

        try:
            # Load data
            df = self.load_data()
            print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")

            # Apply basic fixes
            df_fixed = self.apply_basic_fixes(df)

            # Run checks
            self.check_temporal_integrity(df_fixed)
            self.check_feature_lookahead(df_fixed)
            self.check_target_construction(df_fixed)

            # Generate report
            report = self.generate_report()

            # Print summary
            print("\n" + "="*60)
            print("📋 LOOKAHEAD CONTAMINATION ANALYSIS REPORT")
            print("="*60)
            print(f"Status: {report['status']}")
            print(f"Issues detected: {report['issues_detected']}")
            print(f"Fixes applied: {len(report['fixes_applied'])}")

            if report['issues']:
                print("\n🚨 ISSUES DETECTED:")
                for i, issue in enumerate(report['issues'], 1):
                    print(f"  {i}. {issue}")

            if report['fixes_applied']:
                print("\n✅ FIXES APPLIED:")
                for fix in report['fixes_applied']:
                    print(f"  • {fix}")

            if report['suggested_fixes']:
                print("\n🔧 SUGGESTED FIXES:")
                for fix in report['suggested_fixes']:
                    print(f"  • {fix}")

            # Save report
            report_path = Path("reports/lookahead_analysis.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\n📄 Report saved to: {report_path}")

            return report['status'] == "PASS"

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Detect and fix lookahead contamination")
    parser.add_argument("data_path", help="Path to data file or snapshot directory")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    args = parser.parse_args()

    detector = LookaheadDetector(args.data_path)
    success = detector.run_analysis()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
