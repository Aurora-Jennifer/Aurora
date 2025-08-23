"""
Comprehensive Metrics Collection
Collects IC, turnover, fill_rate, latency, and memory metrics for paper trading observability.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class ComprehensiveMetrics:
    """Comprehensive metrics collection for paper trading observability."""

    def __init__(self, run_id: str, output_dir: str = "reports/metrics"):
        """
        Initialize comprehensive metrics collection.

        Args:
            run_id: Unique run identifier
            output_dir: Output directory for metrics
        """
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.memory_history: List[float] = []
        
        # Trading metrics
        self.orders_sent = 0
        self.fills_received = 0
        self.rejections = 0
        
        # IC and turnover tracking
        self.ic_predictions: List[float] = []
        self.ic_returns: List[float] = []
        self.positions_history: List[Dict[str, float]] = []
        
        logger.info(f"Initialized comprehensive metrics for run {run_id}")

    def calculate_ic(self, predictions: np.ndarray, returns: np.ndarray) -> Optional[float]:
        """
        Calculate Information Coefficient (Spearman correlation) per contract.
        
        Args:
            predictions: Model predictions (any numeric values)
            returns: Same-bar future returns (1 period forward)
            
        Returns:
            Spearman correlation coefficient or None if insufficient data
        """
        if len(predictions) < 10 or len(returns) < 10:
            return None
            
        try:
            # Drop pairs where either value is NaN (contract requirement)
            mask = ~(np.isnan(predictions) | np.isnan(returns))
            if np.sum(mask) < 10:
                return None
                
            correlation, _ = spearmanr(predictions[mask], returns[mask])
            return float(correlation) if not np.isnan(correlation) else None
        except Exception as e:
            logger.warning(f"IC calculation failed: {e}")
            return None

    def calculate_turnover(self, positions: pd.Series) -> float:
        """
        Calculate portfolio turnover per contract.
        
        Args:
            positions: Position weights over time
            
        Returns:
            Average turnover per period: 0.5 * sum(|w_t - w_{t-1}|)
        """
        if len(positions) < 2:
            return 0.0
            
        try:
            # Contract-specified method: 0.5 * sum(|w_t - w_{t-1}|) per period
            position_changes = positions.diff().abs()
            return float(0.5 * position_changes.mean()) if not position_changes.empty else 0.0
        except Exception as e:
            logger.warning(f"Turnover calculation failed: {e}")
            return 0.0

    def calculate_fill_rate(self) -> float:
        """
        Calculate fill rate (fills / orders).
        
        Returns:
            Fill rate as percentage
        """
        if self.orders_sent == 0:
            return 0.0
        return (self.fills_received / self.orders_sent) * 100.0

    def record_latency(self, latency_ms: float) -> None:
        """
        Record latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_history.append(latency_ms)

    def record_memory_usage(self) -> float:
        """
        Record current memory usage.
        
        Returns:
            Memory usage in MB
        """
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_history.append(memory_mb)
        return memory_mb

    def record_order(self, filled: bool = False) -> None:
        """
        Record order sent.
        
        Args:
            filled: Whether order was filled
        """
        self.orders_sent += 1
        if filled:
            self.fills_received += 1

    def record_rejection(self) -> None:
        """Record order rejection."""
        self.rejections += 1

    def record_ic_data(self, predictions: List[float], returns: List[float]) -> None:
        """
        Record IC prediction/return pairs.
        
        Args:
            predictions: Model predictions
            returns: Corresponding future returns
        """
        self.ic_predictions.extend(predictions)
        self.ic_returns.extend(returns)

    def record_positions(self, positions: Dict[str, float]) -> None:
        """
        Record position snapshot.
        
        Args:
            positions: Dict of symbol -> position weight
        """
        self.positions_history.append(positions.copy())

    def get_current_ic(self) -> Optional[float]:
        """Get current IC from recorded data."""
        if not self.ic_predictions or not self.ic_returns:
            return None
        return self.calculate_ic(np.array(self.ic_predictions), np.array(self.ic_returns))

    def get_current_turnover(self) -> float:
        """Get current turnover from recorded positions."""
        if len(self.positions_history) < 2:
            return 0.0
        
        # Convert position history to series for main symbol (simplified)
        # In practice, this would aggregate across all symbols
        main_positions = []
        for pos_dict in self.positions_history:
            # Sum absolute positions as proxy for total exposure
            total_exposure = sum(abs(v) for v in pos_dict.values())
            main_positions.append(total_exposure)
        
        return self.calculate_turnover(pd.Series(main_positions))

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current comprehensive metrics in contract-compliant format.
        
        Returns:
            Dictionary of current metrics following docs/metrics_contract.md
        """
        current_time = time.perf_counter()
        runtime_seconds = current_time - self.start_time
        memory_mb = self.record_memory_usage()
        
        # Calculate latency statistics
        avg_latency = float(np.mean(self.latency_history)) if self.latency_history else 0.0
        p95_latency = float(np.percentile(self.latency_history, 95)) if self.latency_history else 0.0
        max_latency = float(np.max(self.latency_history)) if self.latency_history else 0.0
        
        # Calculate memory statistics
        max_memory = float(np.max(self.memory_history)) if self.memory_history else memory_mb
        memory_peak = max_memory
        
        # Calculate fill rate (contract-compliant)
        fill_rate = None if self.orders_sent == 0 else (self.fills_received / self.orders_sent)
        
        # Get IC and turnover
        current_ic = self.get_current_ic()
        current_turnover = self.get_current_turnover()
        
        # Contract-compliant metrics structure
        metrics = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": float(runtime_seconds),
            
            # Contract: IC Spearman
            "ic_spearman": {
                "value": current_ic,
                "method": "scipy.stats.spearmanr on same-bar future return target",
                "horizon": "1 bar forward",
                "nan_handling": "drop pairwise",
                "min_pairs": 10,
                "pairs_used": len(self.ic_predictions) if self.ic_predictions else 0
            },
            
            # Contract: Turnover
            "turnover": {
                "value": current_turnover,
                "method": "0.5 * sum(|w_t - w_{t-1}|) per period",
                "denominator_includes": "all position changes",
                "units": "fraction of portfolio value"
            },
            
            # Contract: latency_ms with avg, p95, max
            "latency_ms": {
                "avg": avg_latency,
                "p95": p95_latency,
                "max": max_latency
            },
            
            # Contract: memory_peak_mb
            "memory_mb": {
                "current": float(memory_mb),
                "peak": memory_peak,
                "start": float(self.start_memory / 1024 / 1024)
            },
            
            # Contract: trading metrics  
            "trading": {
                "orders_sent": self.orders_sent,
                "fills_received": self.fills_received,
                "rejections": self.rejections
            },
            
            # Contract: fill_rate (null when zero orders)
            "fill_rate": {
                "value": fill_rate,
                "method": "fills_received / orders_submitted",
                "denominator_includes": "submitted orders, excludes cancels",
                "zero_orders_convention": "null"
            }
        }
        
        return metrics

    def log_metrics(self, additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log current metrics to file.
        
        Args:
            additional_metrics: Additional metrics to include
        """
        metrics = self.get_current_metrics()
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Write to file
        metrics_file = self.output_dir / f"{self.run_id}_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        logger.info(f"Logged metrics: latency={metrics['latency_ms']['avg']:.1f}ms, "
                   f"memory={metrics['memory_mb']['current']:.1f}MB, "
                   f"fill_rate={metrics['fill_rate']['value'] or 0:.1%}")

    def save_summary(self) -> None:
        """Save metrics summary."""
        if not self.metrics_history:
            logger.warning("No metrics history to save")
            return
        
        # Calculate summary statistics
        latencies = [m["latency_ms"]["avg"] for m in self.metrics_history]
        memories = [m["memory_mb"]["current"] for m in self.metrics_history]
        fill_rates = [m["fill_rate"]["value"] or 0.0 for m in self.metrics_history]
        
        summary = {
            "run_id": self.run_id,
            "start_time": self.metrics_history[0]["timestamp"],
            "end_time": self.metrics_history[-1]["timestamp"],
            "total_measurements": len(self.metrics_history),
            
            "latency_summary": {
                "mean_ms": float(np.mean(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "max_ms": float(np.max(latencies)),
                "min_ms": float(np.min(latencies))
            },
            
            "memory_summary": {
                "mean_mb": float(np.mean(memories)),
                "max_mb": float(np.max(memories)),
                "min_mb": float(np.min(memories)),
                "peak_mb": float(np.max(memories))
            },
            
            "trading_summary": {
                "total_orders": self.orders_sent,
                "total_fills": self.fills_received,
                "total_rejections": self.rejections,
                "avg_fill_rate_pct": float(np.mean(fill_rates))
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"{self.run_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved metrics summary: {summary_file}")


def create_metrics_collector(run_id: str, output_dir: str = "reports/metrics") -> ComprehensiveMetrics:
    """
    Create a comprehensive metrics collector.
    
    Args:
        run_id: Unique run identifier
        output_dir: Output directory for metrics
        
    Returns:
        ComprehensiveMetrics instance
    """
    return ComprehensiveMetrics(run_id, output_dir)
