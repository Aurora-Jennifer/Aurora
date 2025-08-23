"""
Performance regression gates - fail if key operations regress beyond threshold
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import your modules
try:
    from core.ml.build_features import build_matrix
    from scripts.paper_broker import HoldingsLedger, calculate_position_size
except ImportError:
    pytest.skip("Core modules not available", allow_module_level=True)


class PerformanceGate:
    """Track performance baselines and detect regressions"""

    def __init__(self, baseline_file: str = "reports/performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> dict[str, float]:
        """Load performance baselines from file"""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def save_baselines(self):
        """Save current baselines to file"""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)

    def check_regression(self, test_name: str, duration: float, max_regression: float = 0.20):
        """Check if performance regressed beyond threshold"""
        baseline = self.baselines.get(test_name)

        if baseline is None:
            # First run - set baseline
            self.baselines[test_name] = duration
            print(f"[PERF] Setting baseline for {test_name}: {duration:.3f}s")
            return True

        regression = (duration - baseline) / baseline
        if regression > max_regression:
            pytest.fail(
                f"Performance regression in {test_name}: "
                f"{duration:.3f}s vs baseline {baseline:.3f}s "
                f"({regression:.1%} > {max_regression:.1%} threshold)"
            )

        # Update baseline if we improved significantly
        if regression < -0.10:  # 10% improvement
            self.baselines[test_name] = duration
            print(f"[PERF] Updating baseline for {test_name}: {baseline:.3f}s → {duration:.3f}s")

        return True


# Global performance gate instance
perf_gate = PerformanceGate()


@pytest.fixture(scope="session", autouse=True)
def save_baselines_on_exit():
    """Save performance baselines when tests complete"""
    yield
    perf_gate.save_baselines()


@pytest.mark.benchmark
def test_feature_building_performance(benchmark):
    """Test feature building stays within performance bounds"""

    # Create synthetic OHLC data
    dates = pd.date_range("2023-01-01", periods=1000, freq="D")
    np.random.seed(42)  # Deterministic

    prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "SPY",
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": np.random.randint(1000000, 2000000, len(dates))
    })

    # Benchmark the feature building
    result = benchmark(build_matrix, df, include_returns=True)

    # Check regression
    duration = benchmark.stats.stats.mean
    perf_gate.check_regression("feature_building", duration, max_regression=0.25)

    # Sanity check result
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


@pytest.mark.benchmark
def test_paper_broker_performance(benchmark):
    """Test paper broker execution stays within performance bounds"""

    # Create large holdings ledger with many trades
    def run_many_trades():
        ledger = HoldingsLedger(initial_cash=1000000.0)

        # Execute 100 trades
        for i in range(100):
            symbol = f"STOCK_{i % 10}"  # 10 different stocks
            quantity = np.random.uniform(-100, 100)
            price = np.random.uniform(50, 200)

            ledger.execute_trade(symbol, quantity, price)

        return ledger

    # Benchmark the trading
    result = benchmark(run_many_trades)

    # Check regression
    duration = benchmark.stats.stats.mean
    perf_gate.check_regression("paper_broker_100_trades", duration, max_regression=0.30)

    # Sanity check result
    assert len(result.trades) == 100
    assert result.cash > 0  # Should still have some cash


@pytest.mark.benchmark
def test_position_sizing_performance(benchmark):
    """Test position sizing calculation performance"""

    def run_position_sizing():
        results = []
        for _i in range(1000):
            score = np.random.uniform(-1, 1)
            rank = np.random.uniform(0, 1)
            size = calculate_position_size(score, rank)
            results.append(size)
        return results

    # Benchmark position sizing
    result = benchmark(run_position_sizing)

    # Check regression
    duration = benchmark.stats.stats.mean
    perf_gate.check_regression("position_sizing_1000", duration, max_regression=0.20)

    # Sanity check
    assert len(result) == 1000
    assert all(0 <= size <= 150 for size in result)  # Max position should be capped


@pytest.mark.benchmark
def test_e2e_latency_slo(benchmark):
    """Test end-to-end pipeline meets latency SLO"""

    def simulate_e2e_decision():
        """Simulate the critical path: data → features → model → decision"""

        # 1. Data loading (simulated)
        time.sleep(0.01)  # 10ms data fetch

        # 2. Feature building
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "SPY",
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.randint(1000000, 2000000, len(dates))
        })

        build_matrix(df, include_returns=True)

        # 3. Model inference (simulated)
        time.sleep(0.005)  # 5ms inference
        score = np.random.uniform(-1, 1)

        # 4. Position sizing
        position_size = calculate_position_size(score, None)

        # 5. Risk check (simulated)
        time.sleep(0.002)  # 2ms risk check

        return {"score": score, "position_size": position_size}

    # Benchmark E2E pipeline
    result = benchmark(simulate_e2e_decision)

    # Check SLO: p95 ≤ 150ms
    duration = benchmark.stats.stats.mean
    p95_estimate = duration * 1.5  # Rough p95 estimate

    SLO_TARGET = 0.150  # 150ms
    if p95_estimate > SLO_TARGET:
        pytest.fail(
            f"E2E latency SLO violation: estimated p95 {p95_estimate*1000:.1f}ms > {SLO_TARGET*1000:.1f}ms target"
        )

    # Track performance
    perf_gate.check_regression("e2e_decision_latency", duration, max_regression=0.15)

    # Sanity check
    assert "score" in result
    assert "position_size" in result


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
