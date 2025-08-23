"""
Walkforward fold integrity tests.
Validate fold boundaries, counts, coverage, and no lookahead leakage.
"""

import pytest
from scripts.walkforward_framework import Fold, gen_walkforward


def test_window_boundaries_strictly_increasing():
    """Test that train_end < test_start for all folds."""
    # Small synthetic config
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))

    for fold in folds:
        assert fold.train_hi < fold.test_lo, (
            f"WF_BOUNDARY: train_hi ({fold.train_hi}) must be < test_lo ({fold.test_lo})"
        )
        assert fold.train_lo <= fold.train_hi, (
            f"WF_BOUNDARY: train_lo ({fold.train_lo}) must be <= train_hi ({fold.train_hi})"
        )
        assert fold.test_lo <= fold.test_hi, (
            f"WF_BOUNDARY: test_lo ({fold.test_lo}) must be <= test_hi ({fold.test_hi})"
        )


def test_correct_fold_count_for_known_input():
    """Test fold count for specific configurations."""
    # N=40, train=10, test=5, step=5 -> expect 6 folds
    # Train windows: [0-9], [5-14], [10-19], [15-24], [20-29], [25-34]
    # Test windows: [10-14], [15-19], [20-24], [25-29], [30-34], [35-39]
    folds = list(gen_walkforward(n=40, train_len=10, test_len=5, stride=5))
    assert len(folds) == 6, f"WF_COUNT: Expected 6 folds, got {len(folds)}"

    # Verify fold IDs are sequential
    for i, fold in enumerate(folds):
        assert fold.fold_id == i, f"WF_ID: Expected fold_id {i}, got {fold.fold_id}"


def test_full_coverage_of_eval_range():
    """Test that test windows cover the evaluation range."""
    n = 100
    train_len = 20
    test_len = 10
    stride = 5

    folds = list(gen_walkforward(n=n, train_len=train_len, test_len=test_len, stride=stride))

    # Collect all test indices
    test_indices = set()
    for fold in folds:
        test_indices.update(range(fold.test_lo, fold.test_hi + 1))

    # Expected test range: from train_len to n-1
    expected_test_range = set(range(train_len, n))

    # Check coverage (allow for partial final window)
    missing_indices = expected_test_range - test_indices
    assert len(missing_indices) <= test_len, (
        f"WF_COVERAGE: Too many missing indices: {missing_indices}"
    )


def test_no_overlap_between_train_test():
    """Test that train and test windows never overlap."""
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))

    for fold in folds:
        train_indices = set(range(fold.train_lo, fold.train_hi + 1))
        test_indices = set(range(fold.test_lo, fold.test_hi + 1))

        overlap = train_indices & test_indices
        assert len(overlap) == 0, f"WF_OVERLAP: Train and test overlap: {overlap}"


def test_anchored_mode_boundaries():
    """Test anchored mode (fixed train start)."""
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5, anchored=True))

    for fold in folds:
        # In anchored mode, train_lo should always be 0 (or warmup)
        assert fold.train_lo == 0, f"WF_ANCHORED: train_lo should be 0, got {fold.train_lo}"
        assert fold.train_hi < fold.test_lo, (
            f"WF_ANCHORED: train_hi ({fold.train_hi}) must be < test_lo ({fold.test_lo})"
        )


def test_warmup_handling():
    """Test warmup period handling."""
    warmup = 10
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5, warmup=warmup))

    for fold in folds:
        # Train should not start before warmup
        assert fold.train_lo >= warmup, (
            f"WF_WARMUP: train_lo ({fold.train_lo}) should be >= warmup ({warmup})"
        )


def test_edge_cases():
    """Test edge cases for fold generation."""
    # Very small dataset
    folds = list(gen_walkforward(n=10, train_len=5, test_len=3, stride=2))
    assert len(folds) > 0, "WF_EDGE: Should generate at least one fold for small dataset"

    # Equal train and test length
    folds = list(gen_walkforward(n=50, train_len=10, test_len=10, stride=10))
    for fold in folds:
        assert fold.train_hi < fold.test_lo, "WF_EDGE: No overlap with equal lengths"


def test_stride_effects():
    """Test different stride values."""
    # Non-overlapping test windows
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=10))
    assert len(folds) == 8, f"WF_STRIDE: Expected 8 folds with stride=10, got {len(folds)}"

    # Overlapping test windows
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))
    assert len(folds) == 16, f"WF_STRIDE: Expected 16 folds with stride=5, got {len(folds)}"


def test_fold_properties():
    """Test fold object properties."""
    fold = Fold(train_lo=0, train_hi=19, test_lo=20, test_hi=29, fold_id=0)

    # Test train and test lengths
    assert fold.train_hi - fold.train_lo + 1 == 20, "WF_PROP: Train length should be 20"
    assert fold.test_hi - fold.test_lo + 1 == 10, "WF_PROP: Test length should be 10"

    # Test boundary conditions
    assert fold.train_hi < fold.test_lo, "WF_PROP: train_hi should be < test_lo"


@pytest.mark.parametrize(
    "n,train_len,test_len,stride",
    [
        (40, 10, 5, 5),  # Standard case
        (41, 10, 5, 5),  # Off by one
        (39, 10, 5, 5),  # Off by one
        (50, 15, 8, 7),  # Different ratios
        (100, 30, 15, 10),  # Larger dataset
    ],
)
def test_off_by_one_edges(n, train_len, test_len, stride):
    """Test off-by-one edge cases with parametrized inputs."""
    folds = list(gen_walkforward(n=n, train_len=train_len, test_len=test_len, stride=stride))

    # Should generate at least one fold
    assert len(folds) > 0, (
        f"WF_OFFBYONE: No folds generated for n={n}, train={train_len}, test={test_len}, stride={stride}"
    )

    # All folds should have valid boundaries
    for fold in folds:
        assert fold.train_lo >= 0, f"WF_OFFBYONE: train_lo negative: {fold.train_lo}"
        assert fold.test_hi < n, f"WF_OFFBYONE: test_hi >= n: {fold.test_hi} >= {n}"
        assert fold.train_hi < fold.test_lo, (
            f"WF_OFFBYONE: train_hi >= test_lo: {fold.train_hi} >= {fold.test_lo}"
        )


def test_fold_sequence_consistency():
    """Test that fold sequence is consistent and monotonic."""
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))

    # Check fold IDs are sequential
    for i, fold in enumerate(folds):
        assert fold.fold_id == i, f"WF_SEQ: Expected fold_id {i}, got {fold.fold_id}"

    # Check that train and test windows advance monotonically
    for i in range(1, len(folds)):
        prev_fold = folds[i - 1]
        curr_fold = folds[i]

        # Train windows should advance
        assert curr_fold.train_lo >= prev_fold.train_lo, "WF_SEQ: train_lo should not decrease"
        assert curr_fold.train_hi >= prev_fold.train_hi, "WF_SEQ: train_hi should not decrease"

        # Test windows should advance
        assert curr_fold.test_lo >= prev_fold.test_lo, "WF_SEQ: test_lo should not decrease"
        assert curr_fold.test_hi >= prev_fold.test_hi, "WF_SEQ: test_hi should not decrease"
