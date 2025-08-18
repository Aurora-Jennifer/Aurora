#!/usr/bin/env python3
"""
Performance Test Script for Walkforward Framework

This script tests the walkforward framework with different configurations
to demonstrate performance improvements and help identify optimal settings
for longer periods.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging

import numpy as np

from core.walk.folds import gen_walkforward
from scripts.walkforward_framework import (
    LeakageProofPipeline,
    build_feature_table,
    walkforward_run,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_performance_configurations():
    """Test different walkforward configurations and measure performance."""

    # Test configurations
    configs = [
        {
            "name": "Short Period (1 year)",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "train_len": 252,
            "test_len": 63,
            "stride": 63,
            "expected_folds": 4,
        },
        {
            "name": "Medium Period (3 years)",
            "start_date": "2021-01-01",
            "end_date": "2023-12-31",
            "train_len": 252,
            "test_len": 63,
            "stride": 63,
            "expected_folds": 12,
        },
        {
            "name": "Long Period (5 years)",
            "start_date": "2019-01-01",
            "end_date": "2023-12-31",
            "train_len": 252,
            "test_len": 63,
            "stride": 63,
            "expected_folds": 20,
        },
        {
            "name": "Very Long Period (10 years)",
            "start_date": "2014-01-01",
            "end_date": "2023-12-31",
            "train_len": 252,
            "test_len": 63,
            "stride": 126,  # Larger stride for performance
            "expected_folds": 20,
        },
    ]

    results = []

    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"{'='*60}")

        try:
            # Load data
            import yfinance as yf

            logger.info(
                f"Loading SPY data: {config['start_date']} to {config['end_date']}"
            )

            start_time = time.time()
            data = yf.download(
                "SPY",
                start=config["start_date"],
                end=config["end_date"],
                progress=False,
            )
            load_time = time.time() - start_time

            logger.info(f"Loaded {len(data)} days in {load_time:.2f}s")

            if len(data) < 100:
                logger.warning(f"Very little data loaded ({len(data)} days)")
                continue

            # Build features
            logger.info("Building feature table...")
            start_time = time.time()
            X, y, prices = build_feature_table(data)
            feature_time = time.time() - start_time

            logger.info(f"Features built in {feature_time:.2f}s - Shape: {X.shape}")

            # Generate folds
            logger.info("Generating folds...")
            folds = list(
                gen_walkforward(
                    n=len(X),
                    train_len=config["train_len"],
                    test_len=config["test_len"],
                    stride=config["stride"],
                    warmup=0,
                    anchored=False,
                )
            )

            logger.info(f"Generated {len(folds)} folds")

            # Create pipeline
            pipeline = LeakageProofPipeline(X, y)

            # Test with DataSanity disabled (fast)
            logger.info("Running walkforward (DataSanity disabled)...")
            start_time = time.time()
            results_fast = walkforward_run(
                pipeline,
                folds,
                prices,
                validate_data=False,  # Fast mode
                performance_mode="RELAXED",
            )
            fast_time = time.time() - start_time

            # Test with DataSanity enabled (slow)
            logger.info("Running walkforward (DataSanity enabled)...")
            start_time = time.time()
            try:
                results_slow = walkforward_run(
                    pipeline,
                    folds,
                    prices,
                    validate_data=True,  # Slow mode
                    performance_mode="RELAXED",
                )
                slow_time = time.time() - start_time
                slow_success = True
            except Exception as e:
                logger.error(f"DataSanity validation failed: {e}")
                slow_time = float("inf")
                slow_success = False

            # Calculate metrics
            total_trades = sum(len(trades) for _, _, trades in results_fast)
            avg_sharpe = np.mean(
                [metrics["sharpe_nw"] for _, metrics, _ in results_fast]
            )

            # Store results
            result = {
                "config": config,
                "data_days": len(data),
                "load_time": load_time,
                "feature_time": feature_time,
                "num_folds": len(folds),
                "fast_time": fast_time,
                "slow_time": slow_time if slow_success else None,
                "speedup": slow_time / fast_time if slow_success else float("inf"),
                "total_trades": total_trades,
                "avg_sharpe": avg_sharpe,
                "success": True,
            }

            results.append(result)

            # Print summary
            logger.info(f"\nResults for {config['name']}:")
            logger.info(f"  Data loading: {load_time:.2f}s")
            logger.info(f"  Feature building: {feature_time:.2f}s")
            logger.info(f"  Walkforward (fast): {fast_time:.2f}s")
            if slow_success:
                logger.info(f"  Walkforward (slow): {slow_time:.2f}s")
                logger.info(f"  Speedup: {slow_time/fast_time:.1f}x")
            logger.info(f"  Total trades: {total_trades}")
            logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")

        except Exception as e:
            logger.error(f"Error testing {config['name']}: {e}")
            results.append({"config": config, "success": False, "error": str(e)})

    # Print final summary
    logger.info(f"\n{'='*80}")
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info(f"{'='*80}")

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        logger.info(f"\nSuccessful tests: {len(successful_results)}/{len(configs)}")

        for result in successful_results:
            config = result["config"]
            logger.info(f"\n{config['name']}:")
            logger.info(f"  Period: {config['start_date']} to {config['end_date']}")
            logger.info(f"  Data: {result['data_days']} days")
            logger.info(f"  Folds: {result['num_folds']}")
            logger.info(f"  Fast time: {result['fast_time']:.2f}s")
            if result["slow_time"]:
                logger.info(f"  Slow time: {result['slow_time']:.2f}s")
                logger.info(f"  Speedup: {result['speedup']:.1f}x")
            logger.info(f"  Trades: {result['total_trades']}")
            logger.info(f"  Sharpe: {result['avg_sharpe']:.3f}")

        # Performance recommendations
        logger.info(f"\n{'='*60}")
        logger.info("PERFORMANCE RECOMMENDATIONS")
        logger.info(f"{'='*60}")

        avg_speedup = np.mean(
            [r["speedup"] for r in successful_results if r["speedup"] != float("inf")]
        )
        logger.info(f"Average speedup from disabling DataSanity: {avg_speedup:.1f}x")

        logger.info("\nFor long periods (>6 months):")
        logger.info("1. Use --validate-data=False (default)")
        logger.info("2. Increase stride to reduce number of folds")
        logger.info("3. Use RELAXED performance mode")
        logger.info("4. Consider reducing train/test window sizes")

    else:
        logger.error("No successful tests completed!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test walkforward performance with different configurations"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only short and medium tests"
    )

    args = parser.parse_args()

    logger.info("🚀 Walkforward Performance Test")
    logger.info("=" * 50)

    results = test_performance_configurations()

    logger.info("\n✅ Performance test completed!")
    logger.info(
        f"Results: {len([r for r in results if r['success']])}/{len(results)} successful"
    )
