#!/usr/bin/env python3
"""Test Crypto Golden Snapshot"""

import pandas as pd

from tests.golden.crypto_snapshot import (
    create_golden_crypto_data,
    create_golden_crypto_features
)


class TestCryptoGoldenSnapshot:
    """Test golden crypto snapshot for CI."""

    def test_golden_data_deterministic(self):
        """Test that golden data generation is deterministic."""
        data1 = create_golden_crypto_data()
        data2 = create_golden_crypto_data()
        
        pd.testing.assert_frame_equal(data1, data2, check_exact=True)
        assert len(data1) == 50
        assert 'BTCUSDT' in data1['symbol'].values

    def test_golden_features_deterministic(self):
        """Test that golden features are deterministic."""
        features1 = create_golden_crypto_features()
        features2 = create_golden_crypto_features()
        
        pd.testing.assert_frame_equal(features1['X'], features2['X'], check_exact=True)
        pd.testing.assert_series_equal(features1['y'], features2['y'], check_exact=True)
        
        assert features1['X'].shape == (50, 5)
        assert features1['y'].name == 'target_1h'
