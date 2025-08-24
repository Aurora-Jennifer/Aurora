#!/usr/bin/env python3
"""
Time Series Cross-Validation with Purging

Implements leak-proof time series cross-validation for financial data.
Prevents forward-looking bias by purging overlapping periods between train/test.
"""

import logging
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time Series Cross-Validation with purging to prevent leakage.
    
    This implementation ensures that:
    1. Training data is always before test data
    2. There's a purge gap between train and test to prevent overlap
    3. Splits respect the temporal order of the data
    """
    
    def __init__(self, n_splits: int = 3, purge_gap: int = 1, test_size: float = 0.2):
        """
        Initialize Purged Time Series Split.
        
        Args:
            n_splits: Number of splits to generate
            purge_gap: Number of periods to purge between train and test
            test_size: Fraction of data to use for testing in each split
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.test_size = test_size
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Features (with time-based index)
            y: Labels (optional)
            groups: Group labels (not used)
            
        Yields:
            train_indices, test_indices
        """
        n_samples = len(X)
        
        if n_samples < self.n_splits * 2:
            raise ValueError(f"Not enough samples ({n_samples}) for {self.n_splits} splits")
        
        # Calculate test size in samples
        test_size_samples = max(1, int(n_samples * self.test_size))
        
        # Calculate the step size between splits
        total_used_samples = test_size_samples * self.n_splits + self.purge_gap * self.n_splits
        remaining_samples = n_samples - total_used_samples
        
        if remaining_samples < test_size_samples:
            logger.warning(f"Limited training data: {remaining_samples} samples for initial training")
        
        # Generate splits
        for split_idx in range(self.n_splits):
            # Calculate test start and end
            test_end = n_samples - split_idx * (test_size_samples + self.purge_gap)
            test_start = test_end - test_size_samples
            
            # Calculate train end (with purge gap)
            train_end = test_start - self.purge_gap
            train_start = 0  # Use all available training data
            
            # Ensure valid indices
            if train_end <= train_start or test_start >= n_samples:
                logger.warning(f"Skip split {split_idx}: insufficient data")
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            # Log split info
            train_size = len(train_indices)
            test_size = len(test_indices)
            gap_size = test_start - train_end
            
            logger.debug(
                f"Split {split_idx}: train={train_size} samples "
                f"[{train_start}:{train_end}], gap={gap_size}, "
                f"test={test_size} samples [{test_start}:{test_end}]"
            )
            
            yield train_indices, test_indices


class WalkForwardSplit(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation for time series.
    
    Incrementally grows the training set while maintaining fixed test size.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = None, expanding: bool = True):
        """
        Initialize Walk-Forward Split.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (samples). If None, uses 1/n_splits of data
            expanding: If True, training set grows; if False, sliding window
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.expanding = expanding
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        n_samples = len(X)
        
        # Determine test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if test_size * self.n_splits >= n_samples:
            raise ValueError("Not enough samples for the specified number of splits")
        
        for i in range(self.n_splits):
            # Test set: fixed size, moving forward
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            
            # Training set
            if self.expanding:
                # Expanding window: use all data before test
                train_start = 0
                train_end = test_start
            else:
                # Sliding window: fixed size training set
                train_size = test_start // 2  # Use half of available data
                train_start = max(0, test_start - train_size)
                train_end = test_start
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            logger.debug(
                f"Walk-forward {i}: train={len(train_indices)} "
                f"[{train_start}:{train_end}], test={len(test_indices)} "
                f"[{test_start}:{test_end}]"
            )
            
            yield train_indices, test_indices


def validate_time_series_split(X: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    """
    Validate that time series split doesn't have temporal leakage.
    
    Args:
        X: DataFrame with time-based index
        train_idx: Training indices
        test_idx: Test indices
        
    Returns:
        True if split is valid (no leakage)
    """
    if not hasattr(X.index, 'min'):
        logger.warning("Cannot validate split: X must have time-based index")
        return True
    
    # Get actual timestamps
    train_times = X.index[train_idx]
    test_times = X.index[test_idx]
    
    # Check temporal ordering
    max_train_time = train_times.max()
    min_test_time = test_times.min()
    
    if max_train_time >= min_test_time:
        logger.error(
            f"Temporal leakage detected: max train time {max_train_time} "
            f">= min test time {min_test_time}"
        )
        return False
    
    # Check for overlapping indices
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        logger.error(f"Index overlap detected: {len(overlap)} samples")
        return False
    
    logger.debug(f"✅ Split validation passed: {len(train_idx)} train, {len(test_idx)} test")
    return True
