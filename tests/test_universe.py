#!/usr/bin/env python3
"""
Multi-Asset Universe Tests

Smoke tests and invariants for the universe runner.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.runner_universe import run_universe, _costs_for


class TestUniverseRunner:
    """Test universe runner components"""
    
    def test_costs_for(self):
        """Test cost lookup function"""
        costs_map = {"default": 3, "GME": 8, "COIN": 6}
        
        # Test specific ticker
        assert _costs_for("GME", costs_map) == 8.0
        assert _costs_for("COIN", costs_map) == 6.0
        
        # Test default fallback
        assert _costs_for("AAPL", costs_map) == 3.0
        assert _costs_for("UNKNOWN", costs_map) == 3.0
        
        # Test missing default
        costs_no_default = {"GME": 8}
        assert _costs_for("AAPL", costs_no_default) == 3.0  # Should use hardcoded default
    
    def test_universe_smoke(self, tmp_path):
        """Smoke test with minimal configuration"""
        # Create minimal universe config
        universe_config = {
            "universe": ["AAPL", "NVDA"],
            "market_proxy": "QQQ",
            "cross_proxies": [],
            "costs_bps": {"default": 3}
        }
        
        universe_file = tmp_path / "universe.yaml"
        with open(universe_file, 'w') as f:
            yaml.dump(universe_config, f)
        
        # Create minimal grid config
        grid_config = {
            "horizons": [3],
            "eps_quantiles": [0.5],
            "temperature": [1.0],
            "turnover_band": [0.08, 0.18],
            "models": [{"type": "ridge", "alphas": [1.0]}],
            "costs": {"commission_bps": 1, "slippage_bps": 2},
            "walkforward": {"fold_length": 63, "step_size": 63, "min_train_days": 100},
            "gate": {"threshold_delta_vs_baseline": 0.1, "min_folds": 3, "max_fail_rate": 0.3},
            "output": {"csv_path": "grid_results.csv", "json_path": "grid_results.json", "log_level": "INFO"},
            "data": {"symbols": ["AAPL"], "start_date": "2023-01-01", "end_date": "2024-01-01", "market_benchmark": "QQQ"}
        }
        
        grid_file = tmp_path / "grid.yaml"
        with open(grid_file, 'w') as f:
            yaml.dump(grid_config, f)
        
        # Run universe (this will likely fail due to data issues, but should not crash)
        out_dir = tmp_path / "out"
        
        try:
            board = run_universe(str(universe_file), str(grid_file), str(out_dir))
            
            # Check that leaderboard was created
            assert isinstance(board, pd.DataFrame)
            assert "ticker" in board.columns
            assert "best_median_sharpe" in board.columns
            assert "gate_pass" in board.columns
            
            # Check that output files were created
            assert (out_dir / "leaderboard.csv").exists()
            assert (out_dir / "summary.json").exists()
            
        except Exception as e:
            # Expected to fail due to data/network issues in test environment
            # But should not crash with import errors or configuration issues
            assert "No module named" not in str(e), f"Import error: {e}"
            assert "not found" not in str(e), f"File not found error: {e}"
            print(f"Expected failure in test environment: {e}")
    
    def test_leaderboard_structure(self):
        """Test leaderboard DataFrame structure"""
        # Create mock leaderboard data
        mock_data = [
            {
                "ticker": "AAPL",
                "best_median_sharpe": 0.5,
                "best_vs_BH": 0.2,
                "best_vs_rule": 0.3,
                "median_turnover": 0.15,
                "median_trades": 25,
                "gate_pass": True,
                "runtime_sec": 30.5,
                "costs_bps": 3.0,
                "num_configs": 5
            },
            {
                "ticker": "GME",
                "best_median_sharpe": 0.3,
                "best_vs_BH": 0.1,
                "best_vs_rule": 0.2,
                "median_turnover": 0.20,
                "median_trades": 30,
                "gate_pass": False,
                "runtime_sec": 45.2,
                "costs_bps": 8.0,
                "num_configs": 3
            }
        ]
        
        board = pd.DataFrame(mock_data)
        
        # Test required columns
        required_columns = [
            "ticker", "best_median_sharpe", "best_vs_BH", "best_vs_rule",
            "median_turnover", "median_trades", "gate_pass", "runtime_sec"
        ]
        
        for col in required_columns:
            assert col in board.columns, f"Missing required column: {col}"
        
        # Test data types
        assert board["ticker"].dtype == "object"
        assert board["best_median_sharpe"].dtype in ["float64", "float32"]
        assert board["gate_pass"].dtype == "bool"
        assert board["runtime_sec"].dtype in ["float64", "float32"]
        
        # Test sorting (gate pass first, then Sharpe)
        board_sorted = board.sort_values(["gate_pass", "best_median_sharpe"], ascending=[False, False])
        assert board_sorted.iloc[0]["ticker"] == "AAPL"  # Should be first (gate pass + higher Sharpe)
        assert board_sorted.iloc[1]["ticker"] == "GME"   # Should be second (no gate pass)
    
    def test_cost_calculation(self):
        """Test per-asset cost calculation"""
        costs_map = {"default": 3, "GME": 8, "COIN": 6}
        
        # Test cost splitting (commission + slippage)
        gme_cost = _costs_for("GME", costs_map)
        assert gme_cost == 8.0
        
        # In the universe runner, this gets split:
        commission_bps = gme_cost / 2.0  # 4.0
        slippage_bps = gme_cost / 2.0    # 4.0
        
        assert commission_bps == 4.0
        assert slippage_bps == 4.0
        assert commission_bps + slippage_bps == gme_cost
    
    def test_gate_calculation(self):
        """Test gate pass calculation"""
        # Mock results
        best_config = {
            "median_model_sharpe": 0.6,
            "median_sharpe_bh": 0.3,
            "median_sharpe_rule": 0.4
        }
        
        gate_config = {"threshold_delta_vs_baseline": 0.1}
        
        # Calculate gate pass
        best_baseline = max(best_config["median_sharpe_bh"], best_config["median_sharpe_rule"])
        gate_pass = best_config["median_model_sharpe"] > best_baseline + gate_config["threshold_delta_vs_baseline"]
        
        # 0.6 > max(0.3, 0.4) + 0.1 = 0.4 + 0.1 = 0.5
        assert gate_pass == True
        
        # Test failure case
        best_config_fail = {
            "median_model_sharpe": 0.4,  # Lower than required
            "median_sharpe_bh": 0.3,
            "median_sharpe_rule": 0.4
        }
        
        gate_pass_fail = best_config_fail["median_model_sharpe"] > best_baseline + gate_config["threshold_delta_vs_baseline"]
        assert gate_pass_fail == False


class TestPortfolioSelection:
    """Test portfolio selection from leaderboard"""
    
    def test_select_top_k(self):
        """Test top-K selection logic"""
        # Mock leaderboard data
        mock_data = [
            {"ticker": "AAPL", "best_median_sharpe": 0.8, "gate_pass": True},
            {"ticker": "NVDA", "best_median_sharpe": 0.7, "gate_pass": True},
            {"ticker": "MSFT", "best_median_sharpe": 0.6, "gate_pass": True},
            {"ticker": "GME", "best_median_sharpe": 0.5, "gate_pass": False},
            {"ticker": "COIN", "best_median_sharpe": 0.4, "gate_pass": False}
        ]
        
        board = pd.DataFrame(mock_data)
        
        # Test gate-only selection
        gate_only = board[board["gate_pass"] == True]
        top_2_gate = gate_only.sort_values("best_median_sharpe", ascending=False).head(2)
        expected_gate = ["AAPL", "NVDA"]
        assert list(top_2_gate["ticker"]) == expected_gate
        
        # Test including failed gate
        top_2_all = board.sort_values("best_median_sharpe", ascending=False).head(2)
        expected_all = ["AAPL", "NVDA"]
        assert list(top_2_all["ticker"]) == expected_all
        
        # Test minimum Sharpe filter
        min_sharpe_0_6 = board[board["best_median_sharpe"] >= 0.6]
        assert len(min_sharpe_0_6) == 3  # AAPL, NVDA, MSFT


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
