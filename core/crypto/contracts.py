#!/usr/bin/env python3
"""
Crypto Data Contracts and Validation

Implements schema validation and data quality checks for crypto model training.
Ensures deterministic, leak-proof, and consistent data processing.
"""

import logging
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class CryptoDataContract:
    """Enforces data contracts for crypto model training."""
    
    def __init__(self, contract_path: str = "contracts/crypto_features.yaml"):
        """Initialize with contract specification."""
        self.contract_path = Path(contract_path)
        self.contract = self._load_contract()
        
    def _load_contract(self) -> dict:
        """Load contract specification."""
        try:
            if not self.contract_path.exists():
                raise FileNotFoundError(f"Contract not found: {self.contract_path}")
            
            with open(self.contract_path) as f:
                contract = yaml.safe_load(f)
            
            logger.info(f"Loaded crypto data contract: {contract.get('schema_name', 'unknown')}")
            return contract
            
        except Exception as e:
            logger.error(f"Failed to load contract: {e}")
            raise
    
    def validate_input_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate input data against contract schema.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            # Check required columns
            inputs = self.contract.get('inputs', {})
            required_columns = set(inputs.get('required_columns', []))
            actual_columns = set(df.columns)
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                violations.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            dtypes_spec = self.contract.get('dtypes', {})
            for col_name, expected_dtype in dtypes_spec.items():
                if col_name in df.columns:
                    violations.extend(self._validate_column_dtype(df[col_name], col_name, expected_dtype))
            
            # Check value constraints
            constraints = self.contract.get('constraints', {})
            for col_name, constraint_spec in constraints.items():
                if col_name in df.columns:
                    violations.extend(self._validate_column_constraints(df[col_name], col_name, constraint_spec))
            
            # Check data quality requirements
            quality_spec = self.contract.get('quality', {})
            violations.extend(self._validate_quality_requirements(df, quality_spec))
            
            # Check validation rules
            validation_spec = self.contract.get('validation', {})
            violations.extend(self._validate_rules(df, validation_spec))
            
            is_valid = len(violations) == 0
            
            if is_valid:
                logger.info("✅ Input data validation passed")
            else:
                logger.warning(f"❌ Input data validation failed: {len(violations)} violations")
                for violation in violations:
                    logger.warning(f"  - {violation}")
            
            return is_valid, violations
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False, [f"Validation error: {e}"]
    
    def _validate_column_dtype(self, series: pd.Series, col_name: str, expected_dtype: str) -> list[str]:
        """Validate column data type."""
        violations = []
        
        # Map expected dtype strings to pandas dtypes
        dtype_mapping = {
            'float64': 'float64',
            'int64': 'int64', 
            'string': 'object',  # pandas uses object for strings
            'datetime64[ns]': 'datetime64[ns]'
        }
        
        expected_pandas_dtype = dtype_mapping.get(expected_dtype, expected_dtype)
        actual_dtype = str(series.dtype)
        
        # Handle datetime types specially
        if expected_dtype.startswith('datetime64'):
            if not pd.api.types.is_datetime64_any_dtype(series):
                violations.append(f"{col_name}: expected datetime type, got {actual_dtype}")
        elif expected_dtype == 'string':
            if not pd.api.types.is_object_dtype(series):
                violations.append(f"{col_name}: expected string/object type, got {actual_dtype}")
        else:
            if actual_dtype != expected_pandas_dtype:
                violations.append(f"{col_name}: dtype mismatch (expected {expected_dtype}, got {actual_dtype})")
        
        return violations
    
    def _validate_column_constraints(self, series: pd.Series, col_name: str, constraint_spec: dict) -> list[str]:
        """Validate column value constraints."""
        violations = []
        
        logger.debug(f"Validating constraints for column '{col_name}' with dtype {series.dtype}")
        
        # Only validate numeric columns
        if not pd.api.types.is_numeric_dtype(series):
            logger.debug(f"Skipping non-numeric column '{col_name}'")
            return violations
        
        # Check min/max constraints
        if 'min' in constraint_spec:
            min_val = constraint_spec['min']
            logger.debug(f"Checking min constraint for '{col_name}': {min_val}")
            if series.min() < min_val:
                violations.append(f"{col_name}: value {series.min():.6f} below minimum {min_val}")
        
        if 'max' in constraint_spec:
            max_val = constraint_spec['max']
            logger.debug(f"Checking max constraint for '{col_name}': {max_val}")
            if series.max() > max_val:
                violations.append(f"{col_name}: value {series.max():.6f} above maximum {max_val}")
        
        return violations
    
    def _validate_quality_requirements(self, df: pd.DataFrame, quality_spec: dict) -> list[str]:
        """Validate data quality requirements."""
        violations = []
        
        logger.debug(f"Validating quality requirements on columns: {list(df.columns)}")
        logger.debug(f"Data types: {df.dtypes.to_dict()}")
        
        # Check no missing data requirements
        no_missing = quality_spec.get('no_missing_data', {})
        if no_missing:
            required_columns = no_missing.get('columns', [])
            tolerance = no_missing.get('tolerance', 0.0)
            
            for col in required_columns:
                if col in df.columns:
                    missing_ratio = df[col].isna().sum() / len(df)
                    if missing_ratio > tolerance:
                        violations.append(f"{col}: missing data ratio {missing_ratio:.3f} exceeds tolerance {tolerance}")
        
        # Check monotonic time
        monotonic_time = quality_spec.get('monotonic_time', {})
        if monotonic_time:
            time_col = monotonic_time.get('column')
            ascending = monotonic_time.get('ascending', True)
            
            if time_col in df.columns:
                if ascending and not df[time_col].is_monotonic_increasing:
                    violations.append(f"{time_col}: not monotonically increasing")
                elif not ascending and not df[time_col].is_monotonic_decreasing:
                    violations.append(f"{time_col}: not monotonically decreasing")
        
        # Check unique timestamps
        unique_timestamps = quality_spec.get('unique_timestamps', {})
        if unique_timestamps:
            timestamp_cols = unique_timestamps.get('columns', [])
            if len(timestamp_cols) >= 2:
                col1, col2 = timestamp_cols[0], timestamp_cols[1]
                if col1 in df.columns and col2 in df.columns:
                    duplicates = df.duplicated(subset=[col1, col2]).sum()
                    if duplicates > 0:
                        violations.append(f"Found {duplicates} duplicate timestamp-symbol combinations")
        
        # Check reasonable price changes (only on numeric columns)
        reasonable_changes = quality_spec.get('reasonable_price_changes', {})
        if reasonable_changes:
            col = reasonable_changes.get('column')
            max_abs = reasonable_changes.get('max_abs')
            
            if col in df.columns and max_abs is not None:
                # Only validate numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    extreme_changes = (df[col].abs() > max_abs).sum()
                    if extreme_changes > 0:
                        violations.append(f"{col}: {extreme_changes} values exceed maximum absolute value {max_abs}")
        
        return violations
    
    def _validate_rules(self, df: pd.DataFrame, validation_spec: dict) -> list[str]:
        """Validate general validation rules."""
        violations = []
        
        # Check minimum samples
        min_samples = validation_spec.get('min_samples', 0)
        if len(df) < min_samples:
            violations.append(f"Insufficient samples: {len(df)} < {min_samples}")
        
        # Check maximum missing ratio
        max_missing_ratio = validation_spec.get('max_missing_ratio', 1.0)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        if missing_ratio > max_missing_ratio:
            violations.append(f"Missing data ratio {missing_ratio:.3f} exceeds maximum {max_missing_ratio}")
        
        # Check correlation between target and predictions
        correlation_checks = validation_spec.get('correlation_checks', {})
        if correlation_checks:
            min_corr = correlation_checks.get('target_prediction_min', 0.0)
            
            if 'target_1h' in df.columns and 'predictions' in df.columns:
                # Remove NaN pairs for correlation
                clean_data = df[['target_1h', 'predictions']].dropna()
                if len(clean_data) > 10:  # Need sufficient data
                    corr = clean_data['target_1h'].corr(clean_data['predictions'])
                    if pd.notna(corr) and abs(corr) < min_corr:
                        violations.append(f"Target-prediction correlation {corr:.4f} below minimum {min_corr}")
        
        return violations
    
    def _validate_data_quality(self, df: pd.DataFrame) -> list[str]:
        """Validate overall data quality."""
        violations = []
        
        # Check minimum samples
        min_samples = self.contract.get('training_requirements', {}).get('min_samples', 0)
        if len(df) < min_samples:
            violations.append(f"Insufficient samples: {len(df)} < {min_samples}")
        
        # Check OHLC relationships (only on numeric columns)
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_cols):
            # Ensure all OHLC columns are numeric
            numeric_ohlc = all(pd.api.types.is_numeric_dtype(df[col]) for col in ohlc_cols)
            if numeric_ohlc:
                # High >= max(Open, Close)
                high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
                if high_violations > 0:
                    violations.append(f"high < max(open, close) in {high_violations} rows")
                
                # Low <= min(Open, Close) 
                low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
                if low_violations > 0:
                    violations.append(f"low > min(open, close) in {low_violations} rows")
        
        # Check symbol validity for crypto
        if 'symbol' in df.columns:
            valid_crypto_patterns = ['USDT', 'USD', 'BTC', 'ETH']
            invalid_symbols = []
            for symbol in df['symbol'].unique():
                if not any(pattern in str(symbol).upper() for pattern in valid_crypto_patterns):
                    invalid_symbols.append(symbol)
            
            if invalid_symbols:
                violations.append(f"Invalid crypto symbols: {invalid_symbols}")
        
        return violations
    
    def validate_features(self, X: pd.DataFrame, y: pd.Series) -> tuple[bool, list[str]]:
        """Validate feature matrix and labels against contract."""
        violations = []
        
        # Check feature schema
        feature_schema = self.contract.get('feature_schema', {})
        expected_shape = feature_schema.get('shape', [None, None])
        
        if expected_shape[1] is not None and X.shape[1] != expected_shape[1]:
            violations.append(f"Feature count mismatch: expected {expected_shape[1]}, got {X.shape[1]}")
        
        # Check feature columns
        expected_columns = list(feature_schema.get('columns', {}).keys())
        if expected_columns and list(X.columns) != expected_columns:
            violations.append(f"Feature columns mismatch: expected {expected_columns}, got {list(X.columns)}")
        
        # Check no NaN values
        if X.isna().any().any():
            violations.append("Features contain NaN values")
        
        if y.isna().any():
            violations.append("Labels contain NaN values")
        
        # Check finite values
        if not np.isfinite(X.values).all():
            violations.append("Features contain infinite values")
        
        if not np.isfinite(y.values).all():
            violations.append("Labels contain infinite values")
        
        # Check label schema
        label_schema = self.contract.get('label_schema', {})
        expected_name = label_schema.get('name')
        if expected_name and y.name != expected_name:
            violations.append(f"Label name mismatch: expected {expected_name}, got {y.name}")
        
        is_valid = len(violations) == 0
        
        if is_valid:
            logger.info("✅ Feature validation passed")
        else:
            logger.warning(f"❌ Feature validation failed: {len(violations)} violations")
        
        return is_valid, violations
    
    def create_feature_manifest(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Create feature manifest for artifact tracking."""
        
        # Calculate feature hash for reproducibility
        feature_str = '|'.join(X.columns) + '|' + str(X.dtypes.values.tolist())
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:12]
        
        manifest = {
            'schema_version': self.contract.get('version', '1.0'),
            'feature_hash': feature_hash,
            'feature_names': list(X.columns),
            'feature_dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
            'feature_count': X.shape[1],
            'sample_count': X.shape[0],
            'label_name': y.name,
            'label_dtype': str(y.dtype),
            'date_range': {
                'start': str(X.index.min()) if hasattr(X.index, 'min') else None,
                'end': str(X.index.max()) if hasattr(X.index, 'max') else None,
            },
            'data_quality': {
                'nan_features': int(X.isna().sum().sum()),
                'nan_labels': int(y.isna().sum()),
                'infinite_features': int((~np.isfinite(X.values)).sum()),
                'infinite_labels': int((~np.isfinite(y.values)).sum()),
            }
        }
        
        return manifest


def enforce_determinism(seed: int = 42) -> None:
    """Enforce deterministic execution for reproducible results."""
    import os
    import random
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Set sklearn random state (will be passed to estimators)
    logger.info(f"✅ Determinism enforced: seed={seed}")


def create_data_lineage(df: pd.DataFrame, processing_steps: list[str]) -> dict:
    """Create data lineage tracking."""
    
    # Calculate data hash for tracking
    data_str = str(df.shape) + str(df.columns.tolist()) + str(df.index.min()) + str(df.index.max())
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:12]
    
    lineage = {
        'data_hash': data_hash,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'date_range': {
            'start': str(df.index.min()) if hasattr(df.index, 'min') else None,
            'end': str(df.index.max()) if hasattr(df.index, 'max') else None,
        },
        'processing_steps': processing_steps,
        'symbols': df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
    }
    
    return lineage
