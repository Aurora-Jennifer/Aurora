#!/usr/bin/env python3
"""
Metrics Golden Check - Verify metrics compliance and stability.

This script validates that metrics collection:
1. Produces valid output matching the contract
2. Remains stable across runs (within tolerance)
3. Passes all invariant checks
"""

import json
import sys
from pathlib import Path
from typing import Any


class MetricsContractError(Exception):
    """Raised when metrics violate the contract."""


def validate_schema(metrics: dict[str, Any]) -> None:
    """Validate metrics against the contract schema."""
    required_fields = {
        "run_id": str,
        "timestamp": str,
        "runtime_seconds": (int, float),
    }

    for field, expected_type in required_fields.items():
        if field not in metrics:
            raise MetricsContractError(f"Missing required field: {field}")
        if not isinstance(metrics[field], expected_type):
            raise MetricsContractError(f"Field {field} has wrong type: {type(metrics[field])}")

    # Validate latency_ms structure
    if "latency_ms" in metrics:
        latency = metrics["latency_ms"]
        for subfield in ["avg", "p95", "max"]:
            if subfield not in latency:
                raise MetricsContractError(f"Missing latency_ms.{subfield}")
            if not isinstance(latency[subfield], (int, float)):
                raise MetricsContractError(f"latency_ms.{subfield} must be numeric")

    # Validate memory_peak_mb structure
    if "memory_mb" in metrics:
        memory = metrics["memory_mb"]
        if "peak" not in memory:
            raise MetricsContractError("Missing memory_mb.peak")
        if not isinstance(memory["peak"], (int, float)):
            raise MetricsContractError("memory_mb.peak must be numeric")

    # Validate trading structure
    if "trading" in metrics:
        trading = metrics["trading"]
        for field in ["orders_sent", "fills_received", "rejections"]:
            if field not in trading:
                raise MetricsContractError(f"Missing trading.{field}")
            if not isinstance(trading[field], int):
                raise MetricsContractError(f"trading.{field} must be integer")


def validate_invariants(metrics: dict[str, Any]) -> None:
    """Validate logical invariants."""
    # Latency invariants
    if "latency_ms" in metrics:
        lat = metrics["latency_ms"]
        if lat["p95"] < lat["avg"]:
            raise MetricsContractError(f"latency_ms.p95 ({lat['p95']}) < avg ({lat['avg']})")
        if lat["max"] < lat["p95"]:
            raise MetricsContractError(f"latency_ms.max ({lat['max']}) < p95 ({lat['p95']})")
        if lat["avg"] < 0:
            raise MetricsContractError(f"latency_ms.avg ({lat['avg']}) < 0")

    # Memory invariants
    if "memory_mb" in metrics and "peak" in metrics["memory_mb"]:
        peak = metrics["memory_mb"]["peak"]
        if peak < 0:
            raise MetricsContractError(f"memory_mb.peak ({peak}) < 0")

    # Trading invariants
    if "trading" in metrics:
        trading = metrics["trading"]
        if trading["fills_received"] > trading["orders_sent"]:
            raise MetricsContractError(
                f"fills_received ({trading['fills_received']}) > orders_sent ({trading['orders_sent']})"
            )
        if trading["orders_sent"] < 0:
            raise MetricsContractError(f"orders_sent ({trading['orders_sent']}) < 0")
        if trading["fills_received"] < 0:
            raise MetricsContractError(f"fills_received ({trading['fills_received']}) < 0")
        if trading["rejections"] < 0:
            raise MetricsContractError(f"rejections ({trading['rejections']}) < 0")

    # IC invariants (if present and not null)
    if "ic_spearman" in metrics and metrics["ic_spearman"] is not None:
        ic = metrics["ic_spearman"]
        if isinstance(ic, dict) and "value" in ic:
            ic_val = ic["value"]
        else:
            ic_val = ic

        if ic_val is not None and abs(ic_val) > 1:
            raise MetricsContractError(f"|ic_spearman| ({abs(ic_val)}) > 1")

    # Fill rate invariants (if present and not null)
    if "fill_rate" in metrics and metrics["fill_rate"] is not None:
        fill_rate = metrics["fill_rate"]
        if isinstance(fill_rate, dict) and "value" in fill_rate:
            fr_val = fill_rate["value"]
        else:
            fr_val = fill_rate

        if fr_val is not None and (fr_val < 0 or fr_val > 1):
            raise MetricsContractError(f"fill_rate ({fr_val}) not in [0,1]")


def check_stability(metrics: dict[str, Any], golden_path: Path) -> None:
    """Check stability against golden reference."""
    if not golden_path.exists():
        print(f"WARNING: Golden reference {golden_path} not found, skipping stability check")
        return

    with open(golden_path) as f:
        golden = json.load(f)

    # Define tolerances
    tolerances = {
        ("latency_ms", "avg"): 0.10,  # ±10%
        ("latency_ms", "p95"): 0.10,  # ±10%
        ("memory_mb", "peak"): 0.15,  # ±15%
    }

    absolute_tolerances = {
        ("ic_spearman", "value"): 0.02,  # ±0.02 absolute
        ("turnover", "value"): 0.05,    # ±0.05 absolute
        ("fill_rate", "value"): 0.02,   # ±0.02 absolute
    }

    # Check relative tolerances
    for (field1, field2), tolerance in tolerances.items():
        if field1 in metrics and field2 in metrics[field1]:
            current = metrics[field1][field2]
            if field1 in golden and field2 in golden[field1]:
                reference = golden[field1][field2]
                relative_diff = abs(current - reference) / reference
                if relative_diff > tolerance:
                    raise MetricsContractError(
                        f"{field1}.{field2} stability breach: {current} vs {reference} "
                        f"(diff: {relative_diff:.1%} > {tolerance:.1%})"
                    )

    # Check absolute tolerances
    for (field1, field2), tolerance in absolute_tolerances.items():
        if field1 in metrics:
            current_field = metrics[field1]
            if isinstance(current_field, dict) and field2 in current_field:
                current = current_field[field2]
            else:
                current = current_field

            if field1 in golden:
                golden_field = golden[field1]
                if isinstance(golden_field, dict) and field2 in golden_field:
                    reference = golden_field[field2]
                else:
                    reference = golden_field

                if current is not None and reference is not None:
                    absolute_diff = abs(current - reference)
                    if absolute_diff > tolerance:
                        raise MetricsContractError(
                            f"{field1}.{field2} stability breach: {current} vs {reference} "
                            f"(diff: {absolute_diff:.3f} > {tolerance:.3f})"
                        )


def main() -> int:
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/ci/metrics_golden_check.py <metrics_file> [golden_file]")
        return 1

    metrics_file = Path(sys.argv[1])
    golden_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/goldens/metrics_e2d.json")

    try:
        # Check file exists
        if not metrics_file.exists():
            raise MetricsContractError(f"Metrics file not found: {metrics_file}")

        # Load and parse JSON
        with open(metrics_file) as f:
            metrics = json.load(f)

        print(f"✅ Metrics file found: {metrics_file}")
        print("✅ JSON parsed successfully")

        # Validate schema
        validate_schema(metrics)
        print("✅ Schema validation passed")

        # Validate invariants
        validate_invariants(metrics)
        print("✅ Invariant checks passed")

        # Check stability
        check_stability(metrics, golden_file)
        print("✅ Stability check passed")

        print("🎉 Metrics contract compliance: PASSED")
        return 0

    except MetricsContractError as e:
        print(f"❌ Metrics contract violation: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
