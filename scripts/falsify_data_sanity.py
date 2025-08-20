#!/usr/bin/env python3
"""
DataSanity Falsification Script

This script applies various corruption scenarios to test the robustness
of the DataSanity validation layer. It should reliably turn the build RED
when protections are removed or weakened.
"""

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from importlib import import_module

get_data_sanity_wrapper = import_module("core.data_sanity").get_data_sanity_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSanityFalsifier:
    """Apply corruption scenarios to test DataSanity robustness."""

    def __init__(self, config_path: str = "config/data_sanity.yaml", profile: str = "strict"):
        """Initialize falsifier with configuration."""
        self.config_path = config_path
        self.profile = profile
        self.wrapper = get_data_sanity_wrapper(config_path, profile)
        self.results = []

    def create_base_data(self, rows: int = 100) -> pd.DataFrame:
        """Create base test data."""
        dates = pd.date_range("2023-01-01", periods=rows, freq="D", tz=UTC)

        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, rows)
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.002, rows)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 0.5, rows),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data

    def apply_corruption(self, data: pd.DataFrame, corruption_type: str, **kwargs) -> pd.DataFrame:
        """Apply specific corruption to data."""
        corrupted = data.copy()

        if corruption_type == "extreme_prices":
            # Add extreme prices
            indices = kwargs.get("indices", [10, 20, 30])
            for idx in indices:
                if idx < len(corrupted):
                    corrupted.loc[corrupted.index[idx], "Close"] = 1e11
                    corrupted.loc[corrupted.index[idx], "Open"] = 1e11
                    corrupted.loc[corrupted.index[idx], "High"] = 1e11
                    corrupted.loc[corrupted.index[idx], "Low"] = 1e11

        elif corruption_type == "negative_prices":
            # Add negative prices
            indices = kwargs.get("indices", [15, 25])
            for idx in indices:
                if idx < len(corrupted):
                    corrupted.loc[corrupted.index[idx], "Close"] = -50.0
                    corrupted.loc[corrupted.index[idx], "Open"] = -50.0
                    corrupted.loc[corrupted.index[idx], "High"] = -48.0
                    corrupted.loc[corrupted.index[idx], "Low"] = -52.0

        elif corruption_type == "nan_burst":
            # Add NaN values
            indices = kwargs.get("indices", [5, 15, 25, 35])
            columns = kwargs.get("columns", ["Close", "Open", "High", "Low"])
            for idx in indices:
                if idx < len(corrupted):
                    for col in columns:
                        if col in corrupted.columns:
                            corrupted.loc[corrupted.index[idx], col] = np.nan

        elif corruption_type == "ohlc_violations":
            # Create OHLC violations
            indices = kwargs.get("indices", [10, 20, 30])
            for idx in indices:
                if idx < len(corrupted):
                    # High < Low
                    corrupted.loc[corrupted.index[idx], "High"] = (
                        corrupted.loc[corrupted.index[idx], "Low"] - 1
                    )
                    # Open > High
                    corrupted.loc[corrupted.index[idx], "Open"] = (
                        corrupted.loc[corrupted.index[idx], "High"] + 5
                    )

        elif corruption_type == "negative_volume":
            # Add negative volume
            indices = kwargs.get("indices", [10, 20, 30, 40])
            for idx in indices:
                if idx < len(corrupted):
                    corrupted.loc[corrupted.index[idx], "Volume"] = -1000000

        elif corruption_type == "duplicate_timestamps":
            # Create duplicate timestamps
            if len(corrupted) > 1:
                corrupted.index = corrupted.index.repeat(2)[::2]  # Create duplicates

        elif corruption_type == "non_monotonic":
            # Create non-monotonic index
            corrupted.index = corrupted.index[::-1]  # Reverse

        elif corruption_type == "timezone_mixing":
            # Mix timezone-aware and naive timestamps
            naive_dates = pd.date_range("2023-01-01", periods=len(corrupted), freq="D")
            aware_dates = pd.date_range("2023-01-01", periods=len(corrupted), freq="D", tz=UTC)
            mixed_dates = []
            for i in range(len(corrupted)):
                if i % 2 == 0:
                    mixed_dates.append(naive_dates[i])
                else:
                    mixed_dates.append(aware_dates[i])
            corrupted.index = mixed_dates

        elif corruption_type == "missing_columns":
            # Remove required columns
            columns_to_remove = kwargs.get("columns", ["Close"])
            for col in columns_to_remove:
                if col in corrupted.columns:
                    corrupted = corrupted.drop(col, axis=1)

        elif corruption_type == "wrong_dtypes":
            # Change data types
            columns_to_change = kwargs.get("columns", ["Close"])
            for col in columns_to_change:
                if col in corrupted.columns:
                    corrupted[col] = corrupted[col].astype(str)

        elif corruption_type == "lookahead_contamination":
            # Add lookahead contamination
            if "Returns" not in corrupted.columns:
                corrupted["Returns"] = np.log(corrupted["Close"] / corrupted["Close"].shift(1))

            # Add future information
            indices = kwargs.get("indices", [5, 10, 15])
            for idx in indices:
                if idx + 1 < len(corrupted):
                    corrupted.loc[corrupted.index[idx], "Returns"] = corrupted.loc[
                        corrupted.index[idx + 1], "Returns"
                    ]

        elif corruption_type == "rolling_window_misalignment":
            # Create rolling window misalignment
            if len(corrupted) > 10:
                # Shift some data to create misalignment
                corrupted.iloc[5:10] = corrupted.iloc[0:5].values

        elif corruption_type == "inconsistent_adj_close":
            # Add inconsistent Adj Close
            corrupted["Adj Close"] = corrupted["Close"] * 1.5  # Simulate split
            # But make some inconsistent
            indices = kwargs.get("indices", [10, 20, 30])
            for idx in indices:
                if idx < len(corrupted):
                    corrupted.loc[corrupted.index[idx], "Adj Close"] = (
                        corrupted.loc[corrupted.index[idx], "Close"] * 10
                    )  # Inconsistent

        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        return corrupted

    def test_corruption_scenario(self, corruption_type: str, **kwargs) -> dict[str, Any]:
        """Test a specific corruption scenario."""
        logger.info(f"Testing corruption scenario: {corruption_type}")

        # Create base data
        data = self.create_base_data(100)

        # Apply corruption
        corrupted_data = self.apply_corruption(data, corruption_type, **kwargs)

        # Test validation
        start_time = time.time()
        try:
            clean_data = self.wrapper.validate_dataframe(
                corrupted_data, f"FALSIFY_{corruption_type.upper()}"
            )
            validation_time = time.time() - start_time

            result = {
                "scenario": corruption_type,
                "status": "PASSED",
                "validation_time": validation_time,
                "original_rows": len(data),
                "clean_rows": len(clean_data),
                "error": None,
                "error_type": None,
            }

            logger.error(f"❌ Corruption scenario {corruption_type} PASSED but should have failed!")

        except Exception as e:
            validation_time = time.time() - start_time

            result = {
                "scenario": corruption_type,
                "status": "FAILED",
                "validation_time": validation_time,
                "original_rows": len(data),
                "clean_rows": len(corrupted_data),
                "error": str(e),
                "error_type": type(e).__name__,
            }

            logger.info(
                f"✅ Corruption scenario {corruption_type} correctly FAILED: {type(e).__name__}"
            )

        return result

    def run_falsification_battery(self) -> list[dict[str, Any]]:
        """Run the complete falsification battery."""
        logger.info("🚀 Starting DataSanity Falsification Battery")

        corruption_scenarios = [
            ("extreme_prices", {}),
            ("negative_prices", {}),
            ("nan_burst", {}),
            ("ohlc_violations", {}),
            ("negative_volume", {}),
            ("duplicate_timestamps", {}),
            ("non_monotonic", {}),
            ("timezone_mixing", {}),
            ("missing_columns", {"columns": ["Close"]}),
            ("wrong_dtypes", {"columns": ["Close"]}),
            ("lookahead_contamination", {}),
            ("rolling_window_misalignment", {}),
            ("inconsistent_adj_close", {}),
        ]

        results = []

        for corruption_type, kwargs in corruption_scenarios:
            try:
                result = self.test_corruption_scenario(corruption_type, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error testing {corruption_type}: {e}")
                results.append(
                    {
                        "scenario": corruption_type,
                        "status": "ERROR",
                        "validation_time": 0,
                        "original_rows": 0,
                        "clean_rows": 0,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate a comprehensive report."""
        if not self.results:
            return "No results to report"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DATASANITY FALSIFICATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Total scenarios: {len(self.results)}")

        # Summary
        passed = sum(1 for r in self.results if r["status"] == "PASSED")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        errors = sum(1 for r in self.results if r["status"] == "ERROR")

        report_lines.append(f"Passed: {passed} (should be 0)")
        report_lines.append(f"Failed: {failed} (should be {len(self.results)})")
        report_lines.append(f"Errors: {errors}")

        report_lines.append("\n" + "-" * 60)
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 60)

        for result in self.results:
            status_icon = (
                "✅"
                if result["status"] == "FAILED"
                else "❌"
                if result["status"] == "PASSED"
                else "⚠️"
            )
            report_lines.append(f"{status_icon} {result['scenario']}: {result['status']}")

            if result["error"]:
                report_lines.append(f"   Error: {result['error_type']}: {result['error'][:100]}...")

            report_lines.append(f"   Time: {result['validation_time']:.3f}s")
            report_lines.append(f"   Rows: {result['original_rows']} → {result['clean_rows']}")
            report_lines.append("")

        # Performance summary
        avg_time = np.mean([r["validation_time"] for r in self.results if r["validation_time"] > 0])
        report_lines.append(f"Average validation time: {avg_time:.3f}s")

        # Recommendations
        report_lines.append("\n" + "-" * 60)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 60)

        if passed > 0:
            report_lines.append("❌ CRITICAL: Some corruption scenarios passed validation!")
            report_lines.append("   This indicates DataSanity protections are insufficient.")
            report_lines.append("   Review and strengthen validation rules.")
        else:
            report_lines.append("✅ All corruption scenarios correctly failed validation.")
            report_lines.append("   DataSanity protections are working correctly.")

        if avg_time > 1.0:
            report_lines.append("⚠️  Validation performance may be too slow for production.")
            report_lines.append("   Consider optimizing validation logic.")

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    def save_results(self, output_path: str = "results/falsification_results.json"):
        """Save results to file."""
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # Convert numpy types to native Python types
            for key, value in serializable_result.items():
                if isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
            serializable_results.append(serializable_result)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "results": serializable_results,
                },
                f,
                indent=2,
            )

        logger.info(f"Results saved to {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DataSanity Falsification Script")
    parser.add_argument("--config", default="config/data_sanity.yaml", help="Config file path")
    parser.add_argument(
        "--output",
        default="results/falsification_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results",
    )

    args = parser.parse_args()

    if args.report_only:
        # Load existing results and generate report
        import json

        with open(args.output) as f:
            data = json.load(f)

        falsifier = DataSanityFalsifier(args.config)
        falsifier.results = data["results"]
        print(falsifier.generate_report())
        return

    # Run falsification battery
    falsifier = DataSanityFalsifier(args.config)
    results = falsifier.run_falsification_battery()

    # Generate and print report
    report = falsifier.generate_report()
    print(report)

    # Save results
    falsifier.save_results(args.output)

    # Exit with error code if any scenarios passed (should all fail)
    passed_scenarios = sum(1 for r in results if r["status"] == "PASSED")
    if passed_scenarios > 0:
        logger.error(f"❌ {passed_scenarios} corruption scenarios passed validation!")
        sys.exit(1)
    else:
        logger.info("✅ All corruption scenarios correctly failed validation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
