"""
Metrics module for DataSanity validation.

Tracks counters for validation checks, error codes, and severity levels
to monitor validation behavior and error budgets.
"""

import time
from collections import Counter
from typing import Any

# Global counter for validation metrics
COUNTS = Counter()


def bump(code: str, level: str = "info") -> None:
    """
    Bump counter for validation check and severity.

    Args:
        code: Error code or check identifier
        level: Severity level (info, warn, error, fail)
    """
    COUNTS[(code, level)] += 1


def get_metrics() -> dict[str, Any]:
    """
    Get current validation metrics.

    Returns:
        Dictionary with aggregated metrics
    """
    metrics = {
        "total_checks": sum(COUNTS.values()),
        "by_code": {},
        "by_level": {},
        "by_code_level": {}
    }

    for (code, level), count in COUNTS.items():
        # By code
        if code not in metrics["by_code"]:
            metrics["by_code"][code] = 0
        metrics["by_code"][code] += count

        # By level
        if level not in metrics["by_level"]:
            metrics["by_level"][level] = 0
        metrics["by_level"][level] += count

        # By code and level
        key = f"{code}:{level}"
        metrics["by_code_level"][key] = count

    return metrics


def reset_metrics() -> None:
    """Reset all validation metrics."""
    COUNTS.clear()


def get_error_budget(code: str, window_hours: int = 24) -> dict[str, Any]:
    """
    Get error budget for a specific code.

    Args:
        code: Error code to check
        window_hours: Time window in hours

    Returns:
        Dictionary with error budget information
    """
    # This is a simplified implementation
    # In production, you'd want to use time-series data
    total_errors = sum(count for (c, level), count in COUNTS.items()
                      if c == code and level in ("error", "fail"))
    total_checks = sum(count for (c, _), count in COUNTS.items() if c == code)

    error_rate = total_errors / total_checks if total_checks > 0 else 0.0

    return {
        "code": code,
        "total_checks": total_checks,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "window_hours": window_hours
    }


def export_metrics() -> dict[str, Any]:
    """
    Export metrics for external monitoring.

    Returns:
        Dictionary with metrics suitable for external systems
    """
    return {
        "timestamp": time.time(),
        "metrics": get_metrics(),
        "error_budgets": {
            code: get_error_budget(code)
            for code in {c for c, _ in COUNTS}
        }
    }
