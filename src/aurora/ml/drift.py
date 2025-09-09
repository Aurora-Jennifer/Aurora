"""
Population Stability Index (PSI) and drift detection utilities.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change
    PSI > 0.25: Significant change

    Args:
        expected: Expected/reference distribution
        actual: Actual/current distribution
        bins: Number of bins for histogram

    Returns:
        PSI value (float)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float('inf')

    # Create bins based on expected distribution
    e_bins = np.quantile(expected, np.linspace(0, 1, bins + 1))

    # Ensure unique bin edges
    e_bins = np.unique(e_bins)
    if len(e_bins) < 2:
        return float('inf')

    # Calculate histograms
    e_hist, _ = np.histogram(expected, bins=e_bins)
    a_hist, _ = np.histogram(actual, bins=e_bins)

    # Normalize to probabilities
    e_prob = e_hist / max(e_hist.sum(), 1)
    a_prob = a_hist / max(a_hist.sum(), 1)

    # Clip to avoid log(0)
    e_prob = np.clip(e_prob, 1e-6, 1)
    a_prob = np.clip(a_prob, 1e-6, 1)

    # Calculate PSI
    return float(((a_prob - e_prob) * np.log(a_prob / e_prob)).sum())



def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test for distribution similarity.

    Args:
        expected: Expected/reference distribution
        actual: Actual/current distribution

    Returns:
        Tuple of (KS statistic, p-value)
    """
    from scipy.stats import ks_2samp

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float('inf'), 0.0

    try:
        ks_stat, p_value = ks_2samp(expected, actual)
        return float(ks_stat), float(p_value)
    except Exception:
        return float('inf'), 0.0


def detect_drift(
    expected: np.ndarray,
    actual: np.ndarray,
    psi_threshold: float = 0.25,
    ks_p_threshold: float = 0.05
) -> dict:
    """
    Detect drift between expected and actual distributions.

    Args:
        expected: Expected/reference distribution
        actual: Actual/current distribution
        psi_threshold: PSI threshold for drift detection
        ks_p_threshold: KS test p-value threshold

    Returns:
        Dictionary with drift detection results
    """
    psi_value = psi(expected, actual)
    ks_stat, ks_p = ks_test(expected, actual)

    # Determine drift severity
    if psi_value < 0.1:
        psi_severity = "none"
    elif psi_value < 0.25:
        psi_severity = "moderate"
    else:
        psi_severity = "significant"

    # Determine if drift detected
    psi_drift = psi_value > psi_threshold
    ks_drift = ks_p < ks_p_threshold

    return {
        "psi": {
            "value": psi_value,
            "threshold": psi_threshold,
            "drift_detected": psi_drift,
            "severity": psi_severity
        },
        "ks_test": {
            "statistic": ks_stat,
            "p_value": ks_p,
            "threshold": ks_p_threshold,
            "drift_detected": ks_drift
        },
        "overall_drift": psi_drift or ks_drift,
        "expected_samples": len(expected),
        "actual_samples": len(actual)
    }


def load_golden_predictions() -> np.ndarray | None:
    """
    Load golden/reference predictions for drift comparison.

    Returns:
        Golden predictions array or None if not found
    """
    try:
        # Try to load from golden snapshot
        golden_path = Path("artifacts/golden/predictions.npy")
        if golden_path.exists():
            return np.load(golden_path)

        # Try to load from OOF predictions
        oof_path = Path("artifacts/parity/preds_oof.parquet")
        if oof_path.exists():
            df = pd.read_parquet(oof_path)
            if "pred" in df.columns:
                return df["pred"].to_numpy()

        return None
    except Exception:
        return None


def save_golden_predictions(predictions: np.ndarray) -> bool:
    """
    Save predictions as golden reference for future drift detection.

    Args:
        predictions: Predictions to save as golden reference

    Returns:
        True if saved successfully
    """
    try:
        golden_dir = Path("artifacts/golden")
        golden_dir.mkdir(parents=True, exist_ok=True)

        golden_path = golden_dir / "predictions.npy"
        np.save(golden_path, predictions)

        # Also save metadata
        meta = {
            "created_at": pd.Timestamp.now().isoformat(),
            "n_samples": len(predictions),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions))
        }

        meta_path = golden_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return True
    except Exception:
        return False
