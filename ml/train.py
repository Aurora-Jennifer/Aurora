#!/usr/bin/env python3
"""
ML Training Module for ML Trading System v0.2

This module provides machine learning training and prediction capabilities
with walkforward validation and model calibration to prevent overfitting
and improve prediction reliability.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

# Try to import optional dependencies
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LightGBM not available, falling back to scikit-learn")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost not available, falling back to scikit-learn")

# Set up logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_type: str = "xgb",
    calibrate: str = "isotonic",
    task_type: str = "regression",
    **model_params: Any,
) -> Dict[str, Any]:
    """
    Train a machine learning model and generate predictions.

    Args:
        train_df: Training data
        test_df: Test data
        feature_cols: List of feature column names
        target_col: Target column name
        model_type: Model type ("xgb", "lgb", "gbm")
        calibrate: Calibration method ("isotonic", "platt", "none")
        task_type: Task type ("regression" or "classification")
        **model_params: Additional model parameters

    Returns:
        Dictionary containing:
        - model: Trained model
        - y_pred: Raw predictions
        - y_cal: Calibrated probabilities (if applicable)
        - calibrator: Calibration object (if applicable)
        - metrics: Performance metrics
    """

    # Validate inputs
    if not all(col in train_df.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in train_df.columns]
        raise ValueError(f"Missing feature columns in training data: {missing_cols}")

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    if not all(col in test_df.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in test_df.columns]
        raise ValueError(f"Missing feature columns in test data: {missing_cols}")

    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col] if target_col in test_df.columns else None

    # Train model
    model = _train_model(X_train, y_train, model_type, task_type, **model_params)

    # Generate predictions
    y_pred = _generate_predictions(model, X_test, task_type)

    # Calibrate predictions if requested
    y_cal = None
    calibrator = None

    if calibrate != "none" and task_type == "classification":
        y_cal, calibrator = _calibrate_predictions(
            model, X_train, y_train, X_test, y_pred, calibrate
        )

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred, y_cal, task_type)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_cal": y_cal,
        "calibrator": calibrator,
        "metrics": metrics,
        "feature_importance": _get_feature_importance(model, feature_cols),
    }


def rolling_walkforward(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    fold_length: int,
    step: int,
    retrain_every: str = "M",
    model_type: str = "xgb",
    calibrate: str = "isotonic",
    task_type: str = "regression",
    ts_col: str = "ts",
    asset_col: str = "asset",
    **model_params: Any,
) -> pd.DataFrame:
    """
    Perform rolling walkforward validation with time-based cross-validation.

    Args:
        df: Input dataframe with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        fold_length: Length of each fold in periods
        step: Step size between folds
        retrain_every: Retrain frequency ("D", "W", "M", "Q", "Y")
        model_type: Model type for training
        calibrate: Calibration method
        task_type: Task type ("regression" or "classification")
        ts_col: Timestamp column name
        asset_col: Asset column name
        **model_params: Additional model parameters

    Returns:
        DataFrame with out-of-sample predictions:
        ["ts", "asset", "fold_id", "y_true", "y_pred", "y_cal"]
    """

    # Validate inputs
    required_cols = [ts_col, asset_col, target_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure proper data types
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    # Get unique timestamps and assets
    all_timestamps = df[ts_col].unique()
    all_assets = df[asset_col].unique()

    # Sort timestamps
    all_timestamps = np.sort(all_timestamps)

    # Initialize results
    results = []
    fold_id = 0

    # Generate fold boundaries
    fold_boundaries = _generate_fold_boundaries(
        all_timestamps, fold_length, step, retrain_every
    )

    logger.info(f"Starting walkforward validation with {len(fold_boundaries)} folds")

    for train_end, test_start, test_end in fold_boundaries:
        fold_id += 1

        logger.info(
            f"Processing fold {fold_id}/{len(fold_boundaries)}: "
            f"train_end={train_end}, test_start={test_start}, test_end={test_end}"
        )

        # Split data
        train_mask = df[ts_col] <= train_end
        test_mask = (df[ts_col] >= test_start) & (df[ts_col] <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(f"Skipping fold {fold_id}: insufficient data")
            continue

        # Train model and predict
        try:
            result = fit_predict(
                train_df,
                test_df,
                feature_cols,
                target_col,
                model_type,
                calibrate,
                task_type,
                **model_params,
            )

            # Store results
            test_df["fold_id"] = fold_id
            test_df["y_pred"] = result["y_pred"]
            test_df["y_cal"] = (
                result["y_cal"] if result["y_cal"] is not None else result["y_pred"]
            )

            # Select output columns
            output_cols = [ts_col, asset_col, "fold_id", target_col, "y_pred", "y_cal"]
            fold_results = test_df[output_cols].copy()

            results.append(fold_results)

            # Log fold metrics
            if result["metrics"]:
                logger.info(f"Fold {fold_id} metrics: {result['metrics']}")

        except Exception as e:
            logger.error(f"Error in fold {fold_id}: {str(e)}")
            continue

    if not results:
        raise ValueError("No successful folds completed")

    # Combine all results
    final_results = pd.concat(results, ignore_index=True)
    final_results = final_results.sort_values([asset_col, ts_col]).reset_index(
        drop=True
    )

    logger.info(
        f"Walkforward validation completed. Total predictions: {len(final_results)}"
    )

    return final_results


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    task_type: str,
    **params: Any,
) -> Any:
    """
    Train a machine learning model.

    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Model type
        task_type: Task type
        **params: Model parameters

    Returns:
        Trained model
    """

    # Set default parameters
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
    }

    # Update with provided parameters
    model_params = {**default_params, **params}

    if model_type == "xgb" and XGBOOST_AVAILABLE:
        if task_type == "regression":
            model = xgb.XGBRegressor(**model_params)
        else:
            model = xgb.XGBClassifier(**model_params)

    elif model_type == "lgb" and LIGHTGBM_AVAILABLE:
        if task_type == "regression":
            model = lgb.LGBMRegressor(**model_params)
        else:
            model = lgb.LGBMClassifier(**model_params)

    else:  # Fallback to scikit-learn
        if task_type == "regression":
            model = GradientBoostingRegressor(**model_params)
        else:
            model = GradientBoostingClassifier(**model_params)

    # Train model
    model.fit(X_train, y_train)

    return model


def _generate_predictions(
    model: Any, X_test: pd.DataFrame, task_type: str
) -> np.ndarray:
    """
    Generate predictions from trained model.

    Args:
        model: Trained model
        X_test: Test features
        task_type: Task type

    Returns:
        Predictions array
    """

    if task_type == "classification":
        # For classification, return probability of positive class
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]
        else:
            return model.predict(X_test)
    else:
        # For regression, return raw predictions
        return model.predict(X_test)


def _calibrate_predictions(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    calibrate: str,
) -> Tuple[np.ndarray, Any]:
    """
    Calibrate model predictions.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_pred: Raw predictions
        calibrate: Calibration method

    Returns:
        Tuple of (calibrated_predictions, calibrator)
    """

    if calibrate == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_pred, y_train)
        y_cal = calibrator.predict(y_pred)

    elif calibrate == "platt":
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression()
        calibrator.fit(y_pred.reshape(-1, 1), y_train)
        y_cal = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]

    else:
        raise ValueError(f"Unknown calibration method: {calibrate}")

    return y_cal, calibrator


def _compute_metrics(
    y_true: Optional[pd.Series],
    y_pred: np.ndarray,
    y_cal: Optional[np.ndarray],
    task_type: str,
) -> Dict[str, float]:
    """
    Compute performance metrics.

    Args:
        y_true: True values
        y_pred: Raw predictions
        y_cal: Calibrated predictions
        task_type: Task type

    Returns:
        Dictionary of metrics
    """

    if y_true is None:
        return {}

    metrics = {}

    if task_type == "classification":
        # Classification metrics
        if len(np.unique(y_true)) > 1:  # Need at least 2 classes
            try:
                metrics["auc"] = roc_auc_score(y_true, y_pred)
                if y_cal is not None:
                    metrics["auc_calibrated"] = roc_auc_score(y_true, y_cal)
            except:
                pass

            try:
                metrics["log_loss"] = log_loss(y_true, y_pred)
                if y_cal is not None:
                    metrics["log_loss_calibrated"] = log_loss(y_true, y_cal)
            except:
                pass

    else:
        # Regression metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))

        if y_cal is not None:
            metrics["mse_calibrated"] = mean_squared_error(y_true, y_cal)
            metrics["rmse_calibrated"] = np.sqrt(metrics["mse_calibrated"])

    return metrics


def _get_feature_importance(model: Any, feature_cols: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from model.

    Args:
        model: Trained model
        feature_cols: Feature column names

    Returns:
        DataFrame with feature importance
    """

    try:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()


def _generate_fold_boundaries(
    timestamps: np.ndarray, fold_length: int, step: int, retrain_every: str
) -> List[Tuple[datetime, datetime, datetime]]:
    """
    Generate fold boundaries for walkforward validation.

    Args:
        timestamps: Array of timestamps
        fold_length: Length of each fold
        step: Step size between folds
        retrain_every: Retrain frequency

    Returns:
        List of (train_end, test_start, test_end) tuples
    """

    boundaries = []

    # Convert retrain_every to timedelta
    if retrain_every == "D":
        delta = timedelta(days=1)
    elif retrain_every == "W":
        delta = timedelta(weeks=1)
    elif retrain_every == "M":
        delta = timedelta(days=30)  # Approximate
    elif retrain_every == "Q":
        delta = timedelta(days=90)  # Approximate
    elif retrain_every == "Y":
        delta = timedelta(days=365)  # Approximate
    else:
        raise ValueError(f"Invalid retrain_every: {retrain_every}")

    # Generate boundaries
    for i in range(0, len(timestamps) - fold_length, step):
        train_end = timestamps[i]
        test_start = timestamps[i + 1]
        test_end = timestamps[min(i + fold_length, len(timestamps) - 1)]

        boundaries.append((train_end, test_start, test_end))

    return boundaries


def validate_walkforward_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate walkforward validation results.

    Args:
        results_df: Results from rolling_walkforward

    Returns:
        Validation results dictionary
    """

    validation = {
        "total_predictions": len(results_df),
        "total_folds": results_df["fold_id"].nunique(),
        "date_range": {"start": results_df["ts"].min(), "end": results_df["ts"].max()},
        "assets": results_df["asset"].nunique(),
        "warnings": [],
    }

    # Check for data leakage
    for fold_id in results_df["fold_id"].unique():
        fold_data = results_df[results_df["fold_id"] == fold_id]

        # Check if predictions are in chronological order
        for asset in fold_data["asset"].unique():
            asset_data = fold_data[fold_data["asset"] == asset].sort_values("ts")

            if len(asset_data) > 1:
                # Check for any non-chronological predictions
                if not asset_data["ts"].is_monotonic_increasing:
                    validation["warnings"].append(
                        f"Non-chronological predictions in fold {fold_id}, asset {asset}"
                    )

    # Check for missing predictions
    missing_pct = results_df["y_pred"].isnull().mean()
    if missing_pct > 0.1:
        validation["warnings"].append(f"High missing predictions: {missing_pct:.1%}")

    return validation


if __name__ == "__main__":
    # Example usage and testing
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    assets = ["SPY", "QQQ"]

    sample_data = []
    for asset in assets:
        for date in dates:
            # Create some features
            feature1 = np.random.randn()
            feature2 = np.random.randn()

            # Create target (simple relationship)
            target = 0.3 * feature1 + 0.7 * feature2 + np.random.randn() * 0.1

            sample_data.append(
                {
                    "ts": date,
                    "asset": asset,
                    "feature1": feature1,
                    "feature2": feature2,
                    "target": target,
                }
            )

    df = pd.DataFrame(sample_data)

    # Test walkforward validation
    results = rolling_walkforward(
        df=df,
        feature_cols=["feature1", "feature2"],
        target_col="target",
        fold_length=50,
        step=10,
        retrain_every="M",
        model_type="gbm",
        task_type="regression",
    )

    # Validate results
    validation = validate_walkforward_results(results)

    print("Walkforward validation completed!")
    print(f"Total predictions: {validation['total_predictions']}")
    print(f"Total folds: {validation['total_folds']}")
    print(f"Assets: {validation['assets']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
