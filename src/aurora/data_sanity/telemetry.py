"""
Telemetry module for DataSanity validation runs.

Emits structured JSON logs for monitoring validation behavior,
error budgets, and performance metrics.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import get_cfg


def emit_validation_telemetry(
    symbol: str,
    profile: str,
    result: Any,
    run_id: str | None = None
) -> None:
    """
    Emit one-line JSON telemetry for validation run.

    Args:
        symbol: Symbol being validated
        profile: DataSanity profile used
        result: Validation result (DataFrame or ValidationResult)
        run_id: Optional run identifier for correlation
    """
    if not get_cfg("datasanity.telemetry.enabled", True):
        return

    output_dir = get_cfg("datasanity.telemetry.output_dir", "artifacts/ds_runs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract validation status and flags
    passed = True
    flags = {}
    error_code = None
    error_message = None

    if hasattr(result, 'flags'):
        flags = result.flags
        passed = not bool(flags)
    elif hasattr(result, 'error_code'):
        error_code = result.error_code
        error_message = str(result)
        passed = False

    summary = {
        "timestamp": time.time(),
        "run_id": run_id,
        "symbol": symbol,
        "profile": profile,
        "passed": passed,
        "flags": flags,
        "error_code": error_code,
        "error_message": error_message,
    }

    # Write to JSONL file
    telemetry_file = f"{output_dir}/validation_telemetry.jsonl"
    with open(telemetry_file, "a") as f:
        f.write(json.dumps(summary) + "\n")


def get_telemetry_stats(output_dir: str | None = None) -> dict[str, Any]:
    """
    Get aggregated telemetry statistics.

    Args:
        output_dir: Directory containing telemetry files

    Returns:
        Dictionary with aggregated stats
    """
    if output_dir is None:
        output_dir = get_cfg("datasanity.telemetry.output_dir", "artifacts/ds_runs")

    telemetry_file = f"{output_dir}/validation_telemetry.jsonl"
    if not os.path.exists(telemetry_file):
        return {"total_runs": 0, "pass_rate": 0.0}

    stats = {
        "total_runs": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "by_profile": {},
        "by_symbol": {},
        "error_codes": {}
    }

    with open(telemetry_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                stats["total_runs"] += 1

                if entry.get("passed", True):
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
                    error_code = entry.get("error_code")
                    if error_code:
                        stats["error_codes"][error_code] = stats["error_codes"].get(error_code, 0) + 1

                # Profile stats
                profile = entry.get("profile", "unknown")
                if profile not in stats["by_profile"]:
                    stats["by_profile"][profile] = {"total": 0, "passed": 0}
                stats["by_profile"][profile]["total"] += 1
                if entry.get("passed", True):
                    stats["by_profile"][profile]["passed"] += 1

                # Symbol stats
                symbol = entry.get("symbol", "unknown")
                if symbol not in stats["by_symbol"]:
                    stats["by_symbol"][symbol] = {"total": 0, "passed": 0}
                stats["by_symbol"][symbol]["total"] += 1
                if entry.get("passed", True):
                    stats["by_symbol"][symbol]["passed"] += 1

            except json.JSONDecodeError:
                continue

    if stats["total_runs"] > 0:
        stats["pass_rate"] = stats["passed"] / stats["total_runs"]

        # Calculate pass rates for profiles and symbols
        for profile_stats in stats["by_profile"].values():
            if profile_stats["total"] > 0:
                profile_stats["pass_rate"] = profile_stats["passed"] / profile_stats["total"]

        for symbol_stats in stats["by_symbol"].values():
            if symbol_stats["total"] > 0:
                symbol_stats["pass_rate"] = symbol_stats["passed"] / symbol_stats["total"]

    return stats
