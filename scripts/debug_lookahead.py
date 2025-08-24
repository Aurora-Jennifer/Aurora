#!/usr/bin/env python3
"""
Debug Lookahead Contamination Script
Traces lookahead contamination during DataSanity processing.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sanity.main import DataSanityValidator


def debug_lookahead_contamination():
    """Debug lookahead contamination step by step."""

    # Load the data
    data_path = "artifacts/snapshots/golden_ml_v1/SPY.parquet"
    print(f"📁 Loading data from: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"📊 Original data shape: {df.shape}")
    print(f"📋 Original columns: {list(df.columns)}")
    print()

    # Check if Returns column exists in original data
    if "Returns" in df.columns:
        print("⚠️  Returns column already exists in original data!")
        returns = df["Returns"]
        print(f"   Returns stats: min={returns.min()}, max={returns.max()}, unique={returns.nunique()}")

        # Check for exact matches with future values
        if len(returns) > 1:
            future_1 = returns.shift(-1)
            exact_matches_1 = (returns == future_1) & returns.notna() & future_1.notna()
            if exact_matches_1.any():
                print(f"   ⚠️  Found {exact_matches_1.sum()} exact matches with offset 1")
                print(f"   Sample matches: {returns[exact_matches_1].head(3).tolist()}")
    else:
        print("✅ No Returns column in original data")

    print()

    # Initialize DataSanity validator
    print("🔧 Initializing DataSanity validator...")
    validator = DataSanityValidator()

    # Run validation using validate_and_repair
    print("📥 Running validation...")
    try:
        validated_df, validation_result = validator.validate_and_repair(df, symbol="SPY")
        print(f"✅ Validation passed: {len(validated_df)} rows")
        print(f"🔧 Repairs: {validation_result.repairs}")
        print(f"🚩 Flags: {validation_result.flags}")

        # Check if Returns column was created during validation
        if "Returns" in validated_df.columns:
            print("📈 Returns column created during validation")
            returns = validated_df["Returns"]
            print(f"   Returns stats: min={returns.min()}, max={returns.max()}, unique={returns.nunique()}")

            # Check for exact matches with future values
            if len(returns) > 1:
                future_1 = returns.shift(-1)
                exact_matches_1 = (returns == future_1) & returns.notna() & future_1.notna()
                if exact_matches_1.any():
                    print(f"   ⚠️  Found {exact_matches_1.sum()} exact matches with offset 1")
                    print(f"   Sample matches: {returns[exact_matches_1].head(3).tolist()}")
        else:
            print("❌ No Returns column after validation")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return

    print()

    # Run post-adjust validation (if needed)
    print("🔧 Running post-adjust validation...")
    try:
        final_df = validator.validate_post_adjust(validated_df, symbol="SPY")
        print(f"✅ Post-adjust validation passed: {len(final_df)} rows")

        # Check if Returns column exists after post-adjust
        if "Returns" in final_df.columns:
            print("📈 Returns column exists after post-adjust")
            returns = final_df["Returns"]
            print(f"   Returns stats: min={returns.min()}, max={returns.max()}, unique={returns.nunique()}")

            # Check for exact matches with future values
            if len(returns) > 1:
                future_1 = returns.shift(-1)
                exact_matches_1 = (returns == future_1) & returns.notna() & future_1.notna()
                if exact_matches_1.any():
                    print(f"   ⚠️  Found {exact_matches_1.sum()} exact matches with offset 1")
                    print(f"   Sample matches: {returns[exact_matches_1].head(3).tolist()}")

                    # Show the actual data around these matches
                    print("   📋 Sample data around matches:")
                    for i, idx in enumerate(exact_matches_1[exact_matches_1].index[:3]):
                        if i < len(final_df):
                            row = final_df.loc[idx]
                            next_row = final_df.loc[final_df.index[final_df.index.get_loc(idx) + 1]] if final_df.index.get_loc(idx) + 1 < len(final_df) else None
                            print(f"     Row {idx}: Returns={row.get('Returns', 'N/A')}")
                            if next_row is not None:
                                print(f"     Next: Returns={next_row.get('Returns', 'N/A')}")
        else:
            print("❌ No Returns column after post-adjust")

    except Exception as e:
        print(f"❌ Post-adjust validation failed: {e}")
        return

    print()
    print("🔍 Debug complete!")


if __name__ == "__main__":
    debug_lookahead_contamination()
