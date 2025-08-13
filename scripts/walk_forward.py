#!/usr/bin/env python3
import subprocess

QUARTERS = [
    ("2023-10-01", "2023-12-31"),
    ("2024-01-01", "2024-03-31"),
    ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"),
    ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-08-12"),
]

SYMBOL_SETS = [["SPY", "QQQ"], ["AAPL", "MSFT", "NVDA"]]


def run_backtest(start_date, end_date, symbols):
    """Run a single backtest and return success status."""
    cmd = [
        "python",
        "backtest.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--symbols",
    ] + symbols

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Success")
        return True
    else:
        print(f"❌ Failed: {result.stderr}")
        return False


def main():
    """Run walk-forward tests for all quarters and symbol sets."""
    results = []

    for start_date, end_date in QUARTERS:
        for symbols in SYMBOL_SETS:
            symbol_str = " ".join(symbols)
            print(f"\n{'='*60}")
            print(f"Testing: {symbol_str} from {start_date} to {end_date}")
            print(f"{'='*60}")

            success = run_backtest(start_date, end_date, symbols)
            results.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbols": symbol_str,
                    "success": success,
                }
            )

    # Summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Successful runs: {successful}/{total}")

    if successful < total:
        print("\nFailed runs:")
        for r in results:
            if not r["success"]:
                print(f"  {r['symbols']} {r['start_date']} to {r['end_date']}")


if __name__ == "__main__":
    main()
