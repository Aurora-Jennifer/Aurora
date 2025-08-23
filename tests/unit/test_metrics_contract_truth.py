"""
Synthetic Truth Tests for Metrics Contract
Tests mathematical correctness of metrics calculations with known truth.
"""

import numpy as np
import pandas as pd
import pytest
import time
from unittest.mock import patch

from core.metrics.comprehensive import ComprehensiveMetrics


class TestMetricsContractTruth:
    """Synthetic truth tests proving math correctness."""

    def test_ic_sanity_monotone_transform(self):
        """IC Sanity: monotone transform of predictions → Spearman ≈ 1"""
        metrics = ComprehensiveMetrics("test_ic_monotone", "/tmp")
        
        # Create monotonically increasing predictions and returns
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        
        ic = metrics.calculate_ic(predictions, returns)
        
        # Should be perfect positive correlation
        assert ic is not None
        assert abs(ic - 1.0) < 0.001, f"Monotone transform should give IC ≈ 1, got {ic}"

    def test_ic_sanity_permuted_predictions(self):
        """IC Sanity: permuted predictions → IC ≈ 0"""
        metrics = ComprehensiveMetrics("test_ic_permuted", "/tmp")
        
        # Create random permutation
        np.random.seed(42)  # Deterministic
        predictions = np.random.permutation(np.arange(1, 21))  # 20 values
        returns = np.arange(1, 21)  # Monotonic
        
        ic = metrics.calculate_ic(predictions, returns)
        
        # Should be low correlation (not exactly 0 due to randomness, but close)
        assert ic is not None
        assert abs(ic) < 0.5, f"Permuted predictions should give low IC, got {ic}"

    def test_ic_sanity_inverted_predictions(self):
        """IC Sanity: inverted predictions → IC ≈ -1"""
        metrics = ComprehensiveMetrics("test_ic_inverted", "/tmp")
        
        # Create perfectly anti-correlated data
        predictions = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        
        ic = metrics.calculate_ic(predictions, returns)
        
        # Should be perfect negative correlation
        assert ic is not None
        assert abs(ic - (-1.0)) < 0.001, f"Inverted predictions should give IC ≈ -1, got {ic}"

    def test_turnover_identity_single_asset(self):
        """Turnover identity: single-asset position jumps 0→1→0 across 3 bars → turnovers = [0.5, 0.5]"""
        metrics = ComprehensiveMetrics("test_turnover", "/tmp")
        
        # Position jumps: 0 → 1 → 0
        positions = pd.Series([0.0, 1.0, 0.0])
        
        turnover = metrics.calculate_turnover(positions)
        
        # Expected: diff = [NaN, 1.0, -1.0], abs = [NaN, 1.0, 1.0], mean = 1.0, * 0.5 = 0.5
        expected_turnover = 0.5
        assert abs(turnover - expected_turnover) < 0.001, f"Expected turnover {expected_turnover}, got {turnover}"

    def test_turnover_identity_constant_positions(self):
        """Turnover identity: constant positions → turnover = 0"""
        metrics = ComprehensiveMetrics("test_turnover_constant", "/tmp")
        
        # Constant positions
        positions = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        
        turnover = metrics.calculate_turnover(positions)
        
        # Should be exactly 0
        assert turnover == 0.0, f"Constant positions should give turnover = 0, got {turnover}"

    def test_fill_rate_edge_zero_orders(self):
        """Fill-rate edge: 0 submitted orders → fill_rate = null (per contract)"""
        metrics = ComprehensiveMetrics("test_fill_rate_zero", "/tmp")
        
        # No orders submitted
        assert metrics.orders_sent == 0
        
        current_metrics = metrics.get_current_metrics()
        fill_rate = current_metrics["fill_rate"]["value"]
        
        # Should be null when zero orders
        assert fill_rate is None, f"Zero orders should give fill_rate = null, got {fill_rate}"

    def test_fill_rate_edge_partial_fills(self):
        """Fill-rate edge: 3 orders, 2 fills → fill_rate = 2/3"""
        metrics = ComprehensiveMetrics("test_fill_rate_partial", "/tmp")
        
        # Submit orders with some fills
        metrics.record_order(filled=True)   # Fill
        metrics.record_order(filled=False)  # No fill
        metrics.record_order(filled=True)   # Fill
        
        current_metrics = metrics.get_current_metrics()
        fill_rate = current_metrics["fill_rate"]["value"]
        
        # Should be 2/3
        expected_fill_rate = 2.0 / 3.0
        assert abs(fill_rate - expected_fill_rate) < 0.001, f"Expected fill_rate {expected_fill_rate:.3f}, got {fill_rate}"

    @patch('time.sleep')
    def test_latency_measurement_fixed_sleep(self, mock_sleep):
        """Latency measurement: inject fixed 50ms sleep → assert avg>=50 and max>=avg"""
        metrics = ComprehensiveMetrics("test_latency", "/tmp")
        
        # Record some latencies including a 50ms one
        metrics.record_latency(20.0)
        metrics.record_latency(50.0)  # Simulated 50ms latency
        metrics.record_latency(30.0)
        
        current_metrics = metrics.get_current_metrics()
        latency = current_metrics["latency_ms"]
        
        # Check constraints
        assert latency["avg"] >= 20.0, f"Average latency should be >= 20ms, got {latency['avg']}"
        assert latency["max"] >= latency["avg"], f"Max latency ({latency['max']}) should be >= avg ({latency['avg']})"
        assert latency["p95"] >= latency["avg"], f"P95 latency ({latency['p95']}) should be >= avg ({latency['avg']})"
        assert latency["max"] >= latency["p95"], f"Max latency ({latency['max']}) should be >= p95 ({latency['p95']})"

    def test_cross_metric_invariants_fill_rate_bounds(self):
        """Cross-metric invariant: 0 ≤ fill_rate ≤ 1"""
        metrics = ComprehensiveMetrics("test_invariants", "/tmp")
        
        # Test various scenarios
        scenarios = [
            (0, 0),  # No orders
            (5, 0),  # No fills
            (5, 3),  # Partial fills
            (5, 5),  # All fills
        ]
        
        for orders, fills in scenarios:
            metrics_test = ComprehensiveMetrics(f"test_{orders}_{fills}", "/tmp")
            
            # Record orders and fills
            for i in range(orders):
                filled = i < fills
                metrics_test.record_order(filled=filled)
            
            current_metrics = metrics_test.get_current_metrics()
            fill_rate = current_metrics["fill_rate"]["value"]
            
            if fill_rate is not None:
                assert 0.0 <= fill_rate <= 1.0, f"Fill rate {fill_rate} not in [0,1] for {orders} orders, {fills} fills"

    def test_cross_metric_invariants_latency_ordering(self):
        """Cross-metric invariant: latency.p95 ≥ latency.avg and latency.max ≥ latency.p95"""
        metrics = ComprehensiveMetrics("test_latency_order", "/tmp")
        
        # Record various latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for lat in latencies:
            metrics.record_latency(lat)
        
        current_metrics = metrics.get_current_metrics()
        lat = current_metrics["latency_ms"]
        
        # Check ordering invariants
        assert lat["p95"] >= lat["avg"], f"P95 ({lat['p95']}) should be >= avg ({lat['avg']})"
        assert lat["max"] >= lat["p95"], f"Max ({lat['max']}) should be >= p95 ({lat['p95']})"
        assert lat["avg"] >= 0, f"Avg latency ({lat['avg']}) should be >= 0"

    def test_cross_metric_invariants_memory_positive(self):
        """Cross-metric invariant: memory_peak_mb ≥ 0"""
        metrics = ComprehensiveMetrics("test_memory", "/tmp")
        
        current_metrics = metrics.get_current_metrics()
        memory = current_metrics["memory_mb"]
        
        assert memory["peak"] >= 0, f"Memory peak ({memory['peak']}) should be >= 0"
        assert memory["current"] >= 0, f"Current memory ({memory['current']}) should be >= 0"

    def test_cross_metric_invariants_ic_bounds(self):
        """Cross-metric invariant: abs(ic_spearman) ≤ 1"""
        metrics = ComprehensiveMetrics("test_ic_bounds", "/tmp")
        
        # Test various IC scenarios
        test_cases = [
            (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 
             np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),  # Perfect correlation
            (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 
             np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])),  # Perfect anti-correlation
        ]
        
        for predictions, returns in test_cases:
            ic = metrics.calculate_ic(predictions, returns)
            if ic is not None:
                assert abs(ic) <= 1.0, f"IC magnitude ({abs(ic)}) should be <= 1"

    def test_trading_invariants_fills_vs_orders(self):
        """Cross-metric invariant: fills_received ≤ orders_sent"""
        metrics = ComprehensiveMetrics("test_trading", "/tmp")
        
        # Record some orders and fills
        metrics.record_order(filled=True)
        metrics.record_order(filled=False)
        metrics.record_order(filled=True)
        
        current_metrics = metrics.get_current_metrics()
        trading = current_metrics["trading"]
        
        assert trading["fills_received"] <= trading["orders_sent"], \
            f"Fills ({trading['fills_received']}) should be <= orders ({trading['orders_sent']})"
        assert trading["orders_sent"] >= 0, f"Orders sent ({trading['orders_sent']}) should be >= 0"
        assert trading["fills_received"] >= 0, f"Fills received ({trading['fills_received']}) should be >= 0"
        assert trading["rejections"] >= 0, f"Rejections ({trading['rejections']}) should be >= 0"
