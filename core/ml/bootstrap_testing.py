"""
Bootstrap statistical testing for trading strategy validation
"""
import numpy as np
import pandas as pd


def circular_block_bootstrap_diff(model_returns: np.ndarray, 
                                 bench_returns: np.ndarray, 
                                 block: int = 5, 
                                 B: int = 5000, 
                                 seed: int = 42) -> tuple[float, float]:
    """
    Circular block bootstrap test for Sharpe ratio difference
    
    Args:
        model_returns: Strategy returns
        bench_returns: Benchmark returns (e.g., buy-and-hold)
        block: Block size for bootstrap (default 5 trading days)
        B: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Tuple of (observed_sharpe_diff, p_value)
    """
    rng = np.random.default_rng(seed)
    T = len(model_returns)
    
    def sample_one():
        """Generate one bootstrap sample using circular block bootstrap"""
        k = int(np.ceil(T / block))
        starts = rng.integers(0, T, size=k)
        sidx = np.concatenate([
            (np.arange(block) + st) % T 
            for st in starts
        ])[:T]
        return sidx
    
    def sharpe_ratio(returns):
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or np.std(returns, ddof=1) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    
    # Observed Sharpe ratio difference
    obs_sharpe_diff = sharpe_ratio(model_returns) - sharpe_ratio(bench_returns)
    
    # Bootstrap samples
    boot_sharpe_diffs = []
    for _ in range(B):
        sidx = sample_one()
        boot_model = model_returns[sidx]
        boot_bench = bench_returns[sidx]
        boot_diff = sharpe_ratio(boot_model) - sharpe_ratio(boot_bench)
        boot_sharpe_diffs.append(boot_diff)
    
    # One-sided p-value: P(bootstrap_diff <= 0)
    p_value = np.mean(np.array(boot_sharpe_diffs) <= 0.0)
    
    return obs_sharpe_diff, p_value


def cost_sensitivity_analysis(returns: np.ndarray, 
                            costs_bps: list[float] = [0, 2, 4, 8]) -> pd.DataFrame:
    """
    Analyze strategy performance across different cost levels
    
    Args:
        returns: Gross returns (before costs)
        costs_bps: List of cost levels in basis points
        
    Returns:
        DataFrame with metrics for each cost level
    """
    results = []
    
    for cost_bps in costs_bps:
        # Apply costs (assuming 1 trade per day on average)
        net_returns = returns - cost_bps * 1e-4
        
        # Calculate metrics
        sharpe = np.mean(net_returns) / np.std(net_returns, ddof=1) * np.sqrt(252) if np.std(net_returns, ddof=1) > 0 else 0
        total_return = np.prod(1 + net_returns) - 1
        max_dd = calculate_max_drawdown(net_returns)
        volatility = np.std(net_returns, ddof=1) * np.sqrt(252)
        
        results.append({
            'cost_bps': cost_bps,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'volatility': volatility,
            'net_returns_mean': np.mean(net_returns),
            'net_returns_std': np.std(net_returns, ddof=1)
        })
    
    return pd.DataFrame(results)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def psi(expected: np.ndarray, observed: np.ndarray, bins: int = 20) -> float:
    """
    Population Stability Index for feature drift detection
    
    Args:
        expected: Expected distribution (training data)
        observed: Observed distribution (test data)
        bins: Number of bins for histogram
        
    Returns:
        PSI value (0-0.1: stable, 0.1-0.2: moderate change, >0.2: significant change)
    """
    # Create bins based on expected data
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    q[0] = -np.inf  # Ensure all data is included
    q[-1] = np.inf
    
    # Calculate histograms
    expected_hist, _ = np.histogram(expected, bins=q)
    observed_hist, _ = np.histogram(observed, bins=q)
    
    # Normalize to probabilities
    expected_prob = expected_hist / len(expected)
    observed_prob = observed_hist / len(observed)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    expected_prob = np.clip(expected_prob, epsilon, None)
    observed_prob = np.clip(observed_prob, epsilon, None)
    
    # Calculate PSI
    psi_value = np.sum((observed_prob - expected_prob) * np.log(observed_prob / expected_prob))
    
    return float(psi_value)


def generate_aggregate_metrics(fold_results: list[dict]) -> dict:
    """
    Generate aggregate metrics across all folds
    
    Args:
        fold_results: List of fold result dictionaries
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not fold_results:
        return {}
    
    # Extract metrics from all folds
    sharpe_ratios = [fold['metrics']['sharpe_net'] for fold in fold_results]
    total_returns = [fold['metrics']['total_return'] for fold in fold_results]
    max_drawdowns = [fold['metrics']['mdd'] for fold in fold_results]
    turnovers = [fold['metrics']['turnover'] for fold in fold_results]
    
    # Calculate aggregate statistics
    aggregate_metrics = {
        'num_folds': len(fold_results),
        'median_sharpe': np.median(sharpe_ratios),
        'mean_sharpe': np.mean(sharpe_ratios),
        'std_sharpe': np.std(sharpe_ratios, ddof=1),
        'min_sharpe': np.min(sharpe_ratios),
        'max_sharpe': np.max(sharpe_ratios),
        'median_return': np.median(total_returns),
        'mean_return': np.mean(total_returns),
        'median_mdd': np.median(max_drawdowns),
        'mean_mdd': np.mean(max_drawdowns),
        'median_turnover': np.median(turnovers),
        'mean_turnover': np.mean(turnovers),
        'sharpe_ci_includes_zero': np.min(sharpe_ratios) <= 0 <= np.max(sharpe_ratios)
    }
    
    return aggregate_metrics
