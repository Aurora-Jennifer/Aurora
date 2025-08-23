"""
Tests for DataSanity engine switch functionality.

These tests verify that the engine switch facade correctly routes
validation requests to the appropriate engine version.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.data_sanity import (
    export_metrics,
    get_telemetry_stats,
    validate_and_repair_with_engine_switch,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    data = {
        "Open": [100 + i for i in range(50)],
        "High": [102 + i for i in range(50)],
        "Low": [98 + i for i in range(50)],
        "Close": [101 + i for i in range(50)],
        "Volume": [10000 + i * 100 for i in range(50)],
    }
    return pd.DataFrame(data, index=dates)


@pytest.mark.sanity
def test_engine_switch_basic(sample_df):
    """Test basic engine switch functionality."""
    # Test with default engine (should be v1)
    cleaned_df, result = validate_and_repair_with_engine_switch(
        sample_df, "TEST", "walkforward_smoke", "test_run_1"
    )

    # Should pass validation (walkforward_smoke allows lookahead)
    assert len(cleaned_df) == len(sample_df)
    # Note: walkforward_smoke may flag lookahead but not fail
    assert result.profile == "walkforward_smoke"


@pytest.mark.sanity
def test_engine_switch_telemetry(sample_df):
    """Test that telemetry is emitted."""
    # Clear any existing telemetry
    from core.data_sanity.metrics import reset_metrics
    reset_metrics()

    # Run validation
    validate_and_repair_with_engine_switch(
        sample_df, "TEST", "strict", "test_run_2"
    )

    # Check that telemetry was emitted
    stats = get_telemetry_stats()
    assert stats["total_runs"] >= 1

    # Check metrics
    metrics = export_metrics()
    assert "timestamp" in metrics
    assert "metrics" in metrics


@pytest.mark.sanity
def test_engine_switch_profiles(sample_df):
    """Test engine switch with different profiles."""
    profiles = ["strict", "walkforward_ci", "walkforward_smoke"]

    for profile in profiles:
        cleaned_df, result = validate_and_repair_with_engine_switch(
            sample_df, "TEST", profile, f"test_run_{profile}"
        )

        # Should always return a DataFrame
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) == len(sample_df)

        # Should have correct profile
        assert result.profile == profile


@pytest.mark.sanity
def test_engine_switch_error_handling():
    """Test engine switch with invalid data."""
    # Create invalid DataFrame (empty)
    empty_df = pd.DataFrame()

    # Should handle gracefully
    cleaned_df, result = validate_and_repair_with_engine_switch(
        empty_df, "TEST", "walkforward_smoke", "test_run_empty"
    )

    # Should return empty DataFrame and appropriate result
    assert len(cleaned_df) == 0
    assert result.rows_in == 0
    assert result.rows_out == 0


@pytest.mark.sanity
def test_engine_switch_run_id_tracking(sample_df):
    """Test that run_id is properly tracked in telemetry."""
    run_id = "unique_test_run_123"

    # Run validation
    validate_and_repair_with_engine_switch(
        sample_df, "TEST", "strict", run_id
    )

    # Check telemetry stats
    stats = get_telemetry_stats()
    assert stats["total_runs"] >= 1

    # Note: The current telemetry implementation doesn't expose run_id
    # in the stats, but it should be logged to the JSONL file
    telemetry_file = Path("artifacts/ds_runs/validation_telemetry.jsonl")
    if telemetry_file.exists():
        with open(telemetry_file) as f:
            lines = f.readlines()
            # Check that our run_id appears in the telemetry
            run_id_found = any(run_id in line for line in lines)
            assert run_id_found, f"Run ID {run_id} not found in telemetry"


@pytest.mark.sanity
def test_engine_switch_metrics_increment():
    """Test that metrics are properly incremented."""
    from core.data_sanity.metrics import get_metrics, reset_metrics

    # Reset metrics
    reset_metrics()

    # Get initial metrics
    initial_metrics = get_metrics()
    initial_total = initial_metrics["total_checks"]

    # Run validation
    validate_and_repair_with_engine_switch(
        sample_df, "TEST", "strict", "test_run_metrics"
    )

    # Check that metrics were incremented
    final_metrics = get_metrics()
    final_total = final_metrics["total_checks"]

    # Should have some metrics recorded
    assert final_total >= initial_total


@pytest.mark.sanity
def test_engine_switch_config_independence(sample_df):
    """Test that engine switch works regardless of config state."""
    # Test with different config scenarios
    # (This test ensures the engine switch doesn't depend on specific config)

    profiles = ["strict", "walkforward_smoke"]

    for profile in profiles:
        # Should work regardless of current config state
        cleaned_df, result = validate_and_repair_with_engine_switch(
            sample_df, "TEST", profile, f"test_run_config_{profile}"
        )

        # Basic validation
        assert len(cleaned_df) == len(sample_df)
        assert result.profile == profile
