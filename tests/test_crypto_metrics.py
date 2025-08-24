#!/usr/bin/env python3
"""
Tests for Crypto Model Evaluation Metrics

Validates IC, hit rate, and comprehensive crypto-specific metrics.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from core.crypto.metrics import (
    information_coefficient,
    hit_rate,
    crypto_specific_metrics,
    batch_evaluate_crypto_models,
    create_crypto_metrics_report,
    _calculate_quality_score
)
from core.crypto.determinism import DeterministicContext


@pytest.fixture
def synthetic_trading_data():
    """Create synthetic trading data for testing metrics."""
    with DeterministicContext(seed=42):
        n_samples = 500
        
        # Create true signal (alpha)
        true_alpha = np.random.randn(n_samples) * 0.02
        
        # Model predictions (correlated with alpha + noise)
        predictions = 0.7 * true_alpha + 0.3 * np.random.randn(n_samples) * 0.01
        
        # Actual returns (alpha + market noise)
        actuals = true_alpha + np.random.randn(n_samples) * 0.015
        
        # Create timestamps
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'true_alpha': true_alpha
        }


@pytest.fixture
def poor_quality_data():
    """Create poor quality data (random noise) for testing."""
    with DeterministicContext(seed=123):
        n_samples = 200
        
        # Random predictions and actuals (no correlation)
        predictions = np.random.randn(n_samples) * 0.02
        actuals = np.random.randn(n_samples) * 0.025
        
        return {
            'predictions': predictions,
            'actuals': actuals
        }


class TestInformationCoefficient:
    """Test Information Coefficient calculation."""
    
    def test_perfect_positive_correlation(self):
        """Test IC with perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = x * 2  # Perfect correlation
        
        result = information_coefficient(x, y, method='spearman')
        
        assert result['ic'] == pytest.approx(1.0, abs=1e-10)
        assert result['ic_abs'] == pytest.approx(1.0, abs=1e-10)
        assert result['p_value'] < 0.05
        assert bool(result['is_significant']) is True
        assert result['n_samples'] == 5
        assert result['method'] == 'spearman'
    
    def test_perfect_negative_correlation(self):
        """Test IC with perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = -x  # Perfect negative correlation
        
        result = information_coefficient(x, y, method='pearson')
        
        assert result['ic'] == pytest.approx(-1.0, abs=1e-10)
        assert result['ic_abs'] == pytest.approx(1.0, abs=1e-10)
        assert result['p_value'] < 0.05
        assert bool(result['is_significant']) is True
    
    def test_no_correlation(self):
        """Test IC with no correlation (random data)."""
        with DeterministicContext(seed=42):
            x = np.random.randn(100)
            y = np.random.randn(100)  # Independent
        
        result = information_coefficient(x, y)
        
        # Should be close to zero
        assert abs(result['ic']) < 0.2  # Allow some random variation
        assert result['n_samples'] == 100
    
    def test_nan_handling(self):
        """Test IC handles NaN values correctly."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, np.nan, 10])
        
        result = information_coefficient(x, y)
        
        # Should use only valid pairs: (1,2), (2,4), (5,10)
        assert result['n_samples'] == 3
        assert not np.isnan(result['ic'])
    
    def test_insufficient_data(self):
        """Test IC with insufficient data."""
        x = np.array([1, 2])
        y = np.array([2, 4])
        
        result = information_coefficient(x, y)
        
        assert result['ic'] == 0.0
        assert result['p_value'] == 1.0
        assert bool(result['is_significant']) is False


class TestHitRate:
    """Test hit rate calculation."""
    
    def test_perfect_hit_rate(self):
        """Test hit rate with perfect directional accuracy."""
        predictions = np.array([0.1, -0.1, 0.2, -0.2, 0.05])
        actuals = np.array([0.05, -0.05, 0.15, -0.15, 0.02])
        
        result = hit_rate(predictions, actuals)
        
        assert result['hit_rate'] == 1.0
        assert result['correct_predictions'] == 5
        assert result['total_predictions'] == 5
    
    def test_zero_hit_rate(self):
        """Test hit rate with zero directional accuracy."""
        predictions = np.array([0.1, -0.1, 0.2, -0.2])
        actuals = np.array([-0.05, 0.05, -0.15, 0.15])  # Opposite directions
        
        result = hit_rate(predictions, actuals)
        
        assert result['hit_rate'] == 0.0
        assert result['correct_predictions'] == 0
        assert result['total_predictions'] == 4
    
    def test_custom_threshold(self):
        """Test hit rate with custom threshold."""
        predictions = np.array([0.05, -0.05, 0.15, -0.15])
        actuals = np.array([0.02, -0.02, 0.12, -0.12])
        
        # With threshold 0.1, only last two are considered positive
        result = hit_rate(predictions, actuals, threshold=0.1)
        
        # pred: [False, False, True, False]
        # actual: [False, False, True, False]
        assert result['hit_rate'] == 1.0
        
    def test_nan_handling_hit_rate(self):
        """Test hit rate handles NaN values."""
        predictions = np.array([0.1, np.nan, 0.2, -0.1])
        actuals = np.array([0.05, 0.05, np.nan, -0.05])
        
        result = hit_rate(predictions, actuals)
        
        # Should use only valid pairs: (0.1, 0.05), (-0.1, -0.05)
        assert result['total_predictions'] == 2
        assert result['hit_rate'] == 1.0  # Both correct directions
    
    def test_baseline_calculations(self):
        """Test baseline accuracy calculations."""
        # 80% positive actuals
        predictions = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        actuals = np.array([0.1, 0.1, 0.1, 0.1, -0.1])  # 4 positive, 1 negative
        
        result = hit_rate(predictions, actuals)
        
        # Baseline should be 0.8 (better of always positive vs always negative)
        assert result['baseline_accuracy'] == 0.8
        assert result['positive_actual_rate'] == 0.8


class TestCryptoSpecificMetrics:
    """Test comprehensive crypto metrics."""
    
    def test_good_quality_data(self, synthetic_trading_data):
        """Test metrics with good quality synthetic data."""
        data = synthetic_trading_data
        
        metrics = crypto_specific_metrics(
            predictions=data['predictions'],
            actuals=data['actuals'],
            timestamps=data['timestamps'],
            symbol='BTCUSDT'
        )
        
        # Check structure
        assert 'symbol' in metrics
        assert 'data_quality' in metrics
        assert 'information_coefficient' in metrics
        assert 'hit_rate' in metrics
        assert 'statistical_measures' in metrics
        assert 'risk_adjusted' in metrics
        assert 'overall_quality_score' in metrics
        
        # Check data quality
        assert metrics['data_quality']['n_valid_samples'] == 500
        assert metrics['data_quality']['data_completeness'] == 1.0
        
        # Should have reasonable IC (data is correlated)
        assert abs(metrics['information_coefficient']['ic']) > 0.3
        assert bool(metrics['information_coefficient']['is_significant']) is True
        
        # Should have decent hit rate
        assert metrics['hit_rate']['hit_rate'] > 0.55
        
        # Should have good quality score
        assert metrics['overall_quality_score'] > 50
    
    def test_poor_quality_data(self, poor_quality_data):
        """Test metrics with poor quality (random) data."""
        data = poor_quality_data
        
        metrics = crypto_specific_metrics(
            predictions=data['predictions'],
            actuals=data['actuals'],
            symbol='RANDOMCOIN'
        )
        
        # Should have low IC
        assert abs(metrics['information_coefficient']['ic']) < 0.3
        
        # Hit rate should be around random (0.5)
        assert 0.4 < metrics['hit_rate']['hit_rate'] < 0.6
        
        # Should have low quality score
        assert metrics['overall_quality_score'] < 30
    
    def test_insufficient_data(self):
        """Test metrics with insufficient data."""
        predictions = np.array([0.1])  # Only 1 sample
        actuals = np.array([0.05])
        
        metrics = crypto_specific_metrics(predictions, actuals, symbol='TESTCOIN')
        
        # Should return empty/default metrics
        assert metrics['data_quality']['n_valid_samples'] == 1
        assert metrics['information_coefficient']['ic'] == 0.0
        assert metrics['overall_quality_score'] == 0.0
    
    def test_missing_data_handling(self):
        """Test handling of NaN values in data."""
        predictions = np.array([0.1, np.nan, 0.2, np.nan, 0.3])
        actuals = np.array([0.05, 0.1, np.nan, 0.15, np.nan])
        
        metrics = crypto_specific_metrics(predictions, actuals)
        
        # Should use only (0.1, 0.05) - only valid pair
        assert metrics['data_quality']['n_valid_samples'] == 1
        assert metrics['data_quality']['data_completeness'] == 0.2  # 1/5
    
    def test_time_analysis_integration(self, synthetic_trading_data):
        """Test time-based analysis with timestamps."""
        data = synthetic_trading_data
        
        metrics = crypto_specific_metrics(
            predictions=data['predictions'],
            actuals=data['actuals'],
            timestamps=data['timestamps'],
            symbol='BTCUSDT'
        )
        
        # Should have time analysis section
        assert 'time_analysis' in metrics
        
        # With enough data, should have some time metrics
        time_metrics = metrics['time_analysis']
        if len(data['predictions']) >= 60:
            # Might have monthly analysis
            assert isinstance(time_metrics, dict)


class TestQualityScore:
    """Test overall quality score calculation."""
    
    def test_perfect_score_components(self):
        """Test quality score with perfect metrics."""
        metrics = {
            'information_coefficient': {
                'ic_abs': 0.2,  # Max IC component
                'is_significant': True,
                'p_value': 0.001
            },
            'hit_rate': {
                'hit_rate': 0.65,
                'baseline_accuracy': 0.5  # 0.15 lift
            },
            'data_quality': {
                'data_completeness': 1.0,
                'n_valid_samples': 1000
            }
        }
        
        score = _calculate_quality_score(metrics)
        
        # Should be close to 100
        assert score >= 95
        assert score <= 100
    
    def test_poor_score_components(self):
        """Test quality score with poor metrics."""
        metrics = {
            'information_coefficient': {
                'ic_abs': 0.01,  # Very low IC
                'is_significant': False,
                'p_value': 0.8
            },
            'hit_rate': {
                'hit_rate': 0.48,  # Below baseline
                'baseline_accuracy': 0.5
            },
            'data_quality': {
                'data_completeness': 0.5,  # Missing data
                'n_valid_samples': 50  # Small sample
            }
        }
        
        score = _calculate_quality_score(metrics)
        
        # Should be low
        assert score <= 20


class TestBatchEvaluation:
    """Test batch evaluation of multiple symbols."""
    
    def test_batch_evaluation(self, synthetic_trading_data, poor_quality_data):
        """Test batch evaluation with multiple symbols."""
        # Prepare batch data
        batch_data = {
            'BTCUSDT': {
                'predictions': synthetic_trading_data['predictions'],
                'actuals': synthetic_trading_data['actuals'],
                'timestamps': synthetic_trading_data['timestamps']
            },
            'RANDOMCOIN': {
                'predictions': poor_quality_data['predictions'],
                'actuals': poor_quality_data['actuals']
            }
        }
        
        results = batch_evaluate_crypto_models(batch_data)
        
        # Should have results for both symbols plus summary
        assert 'BTCUSDT' in results
        assert 'RANDOMCOIN' in results
        assert '_summary' in results
        
        # Summary should have statistics
        summary = results['_summary']
        assert summary['n_symbols'] == 2
        assert 'ic_statistics' in summary
        assert 'hit_rate_statistics' in summary
        assert 'overall_quality' in summary
        
        # BTCUSDT should be better than RANDOMCOIN
        btc_quality = results['BTCUSDT']['overall_quality_score']
        random_quality = results['RANDOMCOIN']['overall_quality_score']
        assert btc_quality > random_quality
    
    def test_batch_with_missing_data(self):
        """Test batch evaluation with missing required data."""
        batch_data = {
            'VALIDCOIN': {
                'predictions': np.array([0.1, 0.2, 0.3]),
                'actuals': np.array([0.05, 0.15, 0.25])
            },
            'INVALIDCOIN': {
                'predictions': np.array([0.1, 0.2])
                # Missing 'actuals'
            }
        }
        
        results = batch_evaluate_crypto_models(batch_data)
        
        # Should handle invalid gracefully by skipping it
        assert 'VALIDCOIN' in results
        assert 'INVALIDCOIN' not in results  # Should be skipped due to missing data
        
        # Only valid symbols should be in results
        symbols = [k for k in results.keys() if not k.startswith('_')]
        assert len(symbols) == 1
        assert symbols[0] == 'VALIDCOIN'
    
    def test_batch_save_functionality(self, synthetic_trading_data):
        """Test saving batch results to file."""
        batch_data = {
            'TESTCOIN': {
                'predictions': synthetic_trading_data['predictions'][:100],
                'actuals': synthetic_trading_data['actuals'][:100]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "batch_results.json"
            
            results = batch_evaluate_crypto_models(batch_data, str(output_path))
            
            # File should exist
            assert output_path.exists()
            
            # Should be valid JSON
            with open(output_path) as f:
                loaded_results = json.load(f)
            
            assert 'TESTCOIN' in loaded_results
            assert '_summary' in loaded_results


class TestReportGeneration:
    """Test metrics report generation."""
    
    def test_single_symbol_report(self, synthetic_trading_data):
        """Test generating report for single symbol."""
        data = synthetic_trading_data
        
        metrics = crypto_specific_metrics(
            predictions=data['predictions'],
            actuals=data['actuals'],
            symbol='BTCUSDT'
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.md"
            
            create_crypto_metrics_report(metrics, str(report_path))
            
            # File should exist and have content
            assert report_path.exists()
            content = report_path.read_text()
            
            assert 'BTCUSDT Model Evaluation' in content
            assert 'Information Coefficient' in content
            assert 'Hit Rate' in content
            assert 'Quality Score' in content
    
    def test_batch_report(self, synthetic_trading_data, poor_quality_data):
        """Test generating report for batch evaluation."""
        batch_data = {
            'BTCUSDT': {
                'predictions': synthetic_trading_data['predictions'],
                'actuals': synthetic_trading_data['actuals']
            },
            'RANDOMCOIN': {
                'predictions': poor_quality_data['predictions'],
                'actuals': poor_quality_data['actuals']
            }
        }
        
        results = batch_evaluate_crypto_models(batch_data)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "batch_report.md"
            
            create_crypto_metrics_report(results, str(report_path))
            
            # File should exist and have batch content
            assert report_path.exists()
            content = report_path.read_text()
            
            assert 'Summary (2 symbols)' in content
            assert 'Individual Symbol Results' in content
            assert 'BTCUSDT' in content
            assert 'RANDOMCOIN' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
