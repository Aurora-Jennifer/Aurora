"""
Utility functions for end-to-end testing.
"""

import json
import hashlib
import subprocess
import socket
import contextlib
from pathlib import Path
from typing import List, Dict, Any
import jsonschema
import pandas as pd


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """
    Run a command and capture output.
    
    Args:
        cmd: Command to run as list of strings
        
    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    try:
        # Use current Python executable for subprocess
        if cmd[0] == "python":
            import sys
            cmd[0] = sys.executable
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env={"PYTHONPATH": str(Path.cwd())}
        )
        return result
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out: {' '.join(cmd)}")


def sha256_of(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash as hex string
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def assert_json_schema(file_path: Path, schema: Dict[str, Any]) -> None:
    """
    Assert that a JSON file conforms to a schema.
    
    Args:
        file_path: Path to JSON file
        schema: JSON schema dictionary
        
    Raises:
        AssertionError: If file doesn't conform to schema
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path) as f:
        data = json.load(f)
    
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise AssertionError(f"JSON schema validation failed: {e}")


@contextlib.contextmanager
def assert_no_network():
    """
    Context manager that blocks network access during execution.
    
    Raises:
        AssertionError: If any network calls are attempted
    """
    # Store original socket functions
    original_socket = socket.socket
    
    def blocked_socket(*args, **kwargs):
        raise AssertionError("Network access blocked during test")
    
    # Replace socket with blocked version
    socket.socket = blocked_socket
    
    try:
        yield
    finally:
        # Restore original socket
        socket.socket = original_socket


def validate_csv_structure(file_path: Path, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Validate CSV file structure and return DataFrame.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        DataFrame if valid
        
    Raises:
        AssertionError: If CSV is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise AssertionError(f"Failed to read CSV {file_path}: {e}")
    
    assert len(df) > 0, f"CSV file {file_path} is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise AssertionError(f"Missing required columns in {file_path}: {missing_cols}")
    
    return df


def check_timestamp_monotonicity(df: pd.DataFrame, timestamp_col: str = "timestamp") -> bool:
    """
    Check that timestamps are monotonically increasing.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        
    Returns:
        True if timestamps are monotonic
    """
    if timestamp_col not in df.columns:
        return True  # No timestamp column to check
    
    # Convert to datetime if needed
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    return df[timestamp_col].is_monotonic_increasing


def extract_metrics_from_report(report_path: Path) -> Dict[str, Any]:
    """
    Extract metrics from a smoke run report.
    
    Args:
        report_path: Path to smoke run JSON report
        
    Returns:
        Dictionary of metrics
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    
    with open(report_path) as f:
        data = json.load(f)
    
    return data.get("metrics", {})


def assert_metrics_reasonable(metrics: Dict[str, Any]) -> None:
    """
    Assert that metrics are within reasonable bounds for test data.
    
    Args:
        metrics: Dictionary of metrics
        
    Raises:
        AssertionError: If metrics are unreasonable
    """
    # IC should be reasonable for synthetic data
    if "ic" in metrics:
        ic = metrics["ic"]
        assert -0.5 <= ic <= 0.5, f"IC {ic} outside reasonable range [-0.5, 0.5]"
    
    # Hit rate should be valid probability
    if "hit_rate" in metrics:
        hit_rate = metrics["hit_rate"]
        assert 0.0 <= hit_rate <= 1.0, f"Hit rate {hit_rate} not in [0, 1]"
    
    # Trade count should be non-negative
    if "n_trades" in metrics:
        n_trades = metrics["n_trades"]
        assert n_trades >= 0, f"Trade count {n_trades} should be non-negative"
    
    # Turnover should be reasonable
    if "turnover" in metrics:
        turnover = metrics["turnover"]
        assert 0.0 <= turnover <= 10.0, f"Turnover {turnover} outside reasonable range [0, 10]"


def create_test_config(config_path: Path, symbols: List[str] = None) -> Path:
    """
    Create a test configuration file.
    
    Args:
        config_path: Path to save config
        symbols: List of symbols to include
        
    Returns:
        Path to created config file
    """
    if symbols is None:
        symbols = ["SPY", "TSLA"]
    
    config = {
        "engine": {
            "min_history_bars": 50,
            "max_na_fraction": 0.1,
            "rng_seed": 42
        },
        "walkforward": {
            "fold_length": 100,
            "step_size": 25,
            "allow_truncated_final_fold": True
        },
        "data": {
            "source": "local",
            "auto_adjust": False,
            "cache": False
        },
        "risk": {
            "pos_size_method": "vol_target",
            "vol_target": 0.15,
            "max_drawdown": 0.20,
            "daily_loss_limit": 0.03
        },
        "composer": {
            "use_composer": True,
            "regime_extractor": "basic_kpis",
            "blender": "softmax_blender",
            "min_history_bars": 50,
            "hold_on_nan": True,
            "params": {
                "temperature": 1.0,
                "trend_bias": 1.2,
                "chop_bias": 1.1,
                "min_confidence": 0.10
            }
        },
        "tickers": symbols
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path
