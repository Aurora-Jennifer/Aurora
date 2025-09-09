from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Fold:
    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int
    fold_id: int


def gen_walkforward(
    n: int,
    train_len: int,
    test_len: int,
    stride: int,
    warmup: int = 0,
    anchored: bool = False,
    allow_truncated_final_fold: bool = False,
) -> Iterator[Fold]:
    """
    Generate walk-forward folds with short fold handling.

    Args:
        n: Total number of bars
        train_len: Training window length
        test_len: Test window length
        stride: Stride between folds
        warmup: Warmup period
        anchored: Whether to use anchored training
        allow_truncated_final_fold: Whether to allow truncated final fold
    """
    import logging

    logger = logging.getLogger(__name__)

    fold_id = 0
    t0 = warmup + (0 if anchored else train_len)
    while True:
        train_hi = t0 - 1
        train_lo = warmup if anchored else train_hi - train_len + 1
        test_lo = t0
        test_hi = min(test_lo + test_len - 1, n - 1)

        if test_lo >= n or train_lo < warmup:
            break

        # Check for short test window
        test_window_size = test_hi - test_lo + 1
        if test_window_size < test_len:
            if allow_truncated_final_fold:
                # Adjust stride to match test window size
                stride = test_window_size
                logger.info(
                    f"Fold {fold_id}: Using truncated final fold with test_len={test_window_size}"
                )
            else:
                # Skip the fold
                logger.info(
                    f"Fold {fold_id}: Skipping final fold with test_len={test_window_size} < {test_len}"
                )
                break

        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride
