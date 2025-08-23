"""Unit tests for comprehensive metrics collection."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.metrics.comprehensive import ComprehensiveMetrics, create_metrics_collector


class TestComprehensiveMetrics:
    """Test comprehensive metrics collection."""

    def test_init(self):
        """Test metrics initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            assert metrics.run_id == "test_run"
            assert metrics.orders_sent == 0
            assert metrics.fills_received == 0
            assert metrics.rejections == 0
            assert len(metrics.latency_history) == 0
            assert len(metrics.memory_history) == 0

    def test_calculate_ic(self):
        """Test IC calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Test with correlated data (need >=10 pairs for contract)
            predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            returns = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
            ic = metrics.calculate_ic(predictions, returns)
            assert ic is not None and ic > 0.9  # Should be highly correlated
            
            # Test with anti-correlated data
            returns_anti = np.array([12.1, 11.1, 10.1, 9.1, 8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1])
            ic_anti = metrics.calculate_ic(predictions, returns_anti)
            assert ic_anti is not None and ic_anti < -0.9  # Should be highly anti-correlated
            
            # Test with insufficient data
            ic_short = metrics.calculate_ic(np.array([1]), np.array([1]))
            assert ic_short is None  # Contract: None when insufficient data

    def test_calculate_turnover(self):
        """Test turnover calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Test with position changes
            positions = pd.Series([0, 1, 0, 2, 1, 0])
            turnover = metrics.calculate_turnover(positions)
            assert turnover > 0  # Should have some turnover
            
            # Test with constant positions
            positions_constant = pd.Series([1, 1, 1, 1, 1])
            turnover_constant = metrics.calculate_turnover(positions_constant)
            assert turnover_constant == 0.0  # No turnover

    def test_calculate_fill_rate(self):
        """Test fill rate calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Test with no orders
            assert metrics.calculate_fill_rate() == 0.0
            
            # Test with some fills
            metrics.record_order(filled=True)
            metrics.record_order(filled=False)
            metrics.record_order(filled=True)
            assert abs(metrics.calculate_fill_rate() - 66.67) < 0.01  # 2/3 * 100

    def test_record_latency(self):
        """Test latency recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            metrics.record_latency(10.5)
            metrics.record_latency(20.3)
            
            assert len(metrics.latency_history) == 2
            assert metrics.latency_history[0] == 10.5
            assert metrics.latency_history[1] == 20.3

    @patch('psutil.Process')
    def test_record_memory_usage(self, mock_process):
        """Test memory usage recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Mock memory info
            mock_memory = type('MockMemory', (), {'rss': 1024 * 1024 * 100})()  # 100MB
            mock_process.return_value.memory_info.return_value = mock_memory
            
            memory_mb = metrics.record_memory_usage()
            assert memory_mb == 100.0
            assert len(metrics.memory_history) == 1
            assert metrics.memory_history[0] == 100.0

    def test_record_order(self):
        """Test order recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            metrics.record_order(filled=True)
            assert metrics.orders_sent == 1
            assert metrics.fills_received == 1
            
            metrics.record_order(filled=False)
            assert metrics.orders_sent == 2
            assert metrics.fills_received == 1

    def test_record_rejection(self):
        """Test rejection recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            metrics.record_rejection()
            assert metrics.rejections == 1
            
            metrics.record_rejection()
            assert metrics.rejections == 2

    def test_get_current_metrics(self):
        """Test current metrics retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Add some data
            metrics.record_latency(10.0)
            metrics.record_latency(20.0)
            metrics.record_order(filled=True)
            metrics.record_order(filled=False)
            metrics.record_rejection()
            
            current_metrics = metrics.get_current_metrics()
            
            assert current_metrics["run_id"] == "test_run"
            assert current_metrics["latency_ms"]["avg"] == 15.0  # Average
            assert current_metrics["trading"]["orders_sent"] == 2
            assert current_metrics["trading"]["fills_received"] == 1
            assert current_metrics["trading"]["rejections"] == 1
            assert current_metrics["fill_rate"]["value"] == 0.5  # 50% as decimal

    def test_log_metrics(self):
        """Test metrics logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Add some data
            metrics.record_latency(10.0)
            metrics.record_order(filled=True)
            
            # Log metrics
            metrics.log_metrics({"additional": "data"})
            
            # Check file was created
            metrics_file = Path(temp_dir) / "test_run_metrics.jsonl"
            assert metrics_file.exists()
            
            # Check content
            with open(metrics_file) as f:
                lines = f.readlines()
                assert len(lines) == 1
                
                logged_metrics = json.loads(lines[0])
                assert logged_metrics["run_id"] == "test_run"
                assert logged_metrics["additional"] == "data"

    def test_save_summary(self):
        """Test summary saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = ComprehensiveMetrics("test_run", temp_dir)
            
            # Add some data
            metrics.record_latency(10.0)
            metrics.record_latency(20.0)
            metrics.record_order(filled=True)
            metrics.record_order(filled=False)
            
            # Log metrics first
            metrics.log_metrics()
            metrics.log_metrics()
            
            # Save summary
            metrics.save_summary()
            
            # Check summary file
            summary_file = Path(temp_dir) / "test_run_summary.json"
            assert summary_file.exists()
            
            # Check content
            with open(summary_file) as f:
                summary = json.load(f)
                assert summary["run_id"] == "test_run"
                assert summary["total_measurements"] == 2
                assert summary["latency_summary"]["mean_ms"] == 15.0
                assert summary["trading_summary"]["total_orders"] == 2

    def test_create_metrics_collector(self):
        """Test metrics collector factory function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = create_metrics_collector("test_run", temp_dir)
            
            assert isinstance(metrics, ComprehensiveMetrics)
            assert metrics.run_id == "test_run"
            assert str(metrics.output_dir) == temp_dir
