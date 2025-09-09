#!/usr/bin/env python3
"""
Statistical validation and multiple testing control for trading systems.

Implements Deflated Sharpe Ratio, White's Reality Check, and other
statistical tests to prevent false discoveries in multiple testing scenarios.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.stats import norm
import warnings


def deflated_sharpe_ratio(sharpe_ratio: float, n_observations: int, 
                         n_trials: int, skewness: float = 0.0, 
                         kurtosis: float = 3.0) -> float:
    """
    Calculate the Deflated Sharpe Ratio to account for multiple testing.
    
    The Deflated Sharpe Ratio adjusts the Sharpe ratio for the number of
    trials performed, preventing false discoveries from multiple testing.
    
    Args:
        sharpe_ratio: The observed Sharpe ratio
        n_observations: Number of observations (trading days)
        n_trials: Number of trials/tests performed
        skewness: Skewness of returns (default: 0 for normal)
        kurtosis: Kurtosis of returns (default: 3 for normal)
    
    Returns:
        Deflated Sharpe ratio (p-value equivalent)
    
    References:
        Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
        Correcting for selection bias, backtest overfitting, and non-normality.
    """
    
    # Calculate the expected maximum Sharpe ratio under null hypothesis
    expected_max_sr = np.sqrt(2 * np.log(n_trials))
    
    # Calculate the variance of the Sharpe ratio
    sr_variance = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                   (kurtosis - 1) / 4 * sharpe_ratio**2) / n_observations
    
    # Calculate the deflated Sharpe ratio
    deflated_sr = (sharpe_ratio - expected_max_sr) / np.sqrt(sr_variance)
    
    # Convert to p-value (one-sided test)
    p_value = 1 - norm.cdf(deflated_sr)
    
    return p_value


def whites_reality_check(returns_list: List[pd.Series], 
                        benchmark_returns: pd.Series,
                        n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    White's Reality Check for multiple strategy comparison.
    
    Tests whether the best performing strategy significantly outperforms
    the benchmark after accounting for multiple testing.
    
    Args:
        returns_list: List of return series for different strategies
        benchmark_returns: Benchmark return series
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary with test statistics and p-values
    """
    
    # Calculate Sharpe ratios for all strategies
    sharpe_ratios = []
    for returns in returns_list:
        if len(returns) > 0 and returns.std() > 0:
            sr = returns.mean() / returns.std() * np.sqrt(252)
            sharpe_ratios.append(sr)
        else:
            sharpe_ratios.append(0.0)
    
    # Calculate benchmark Sharpe ratio
    if len(benchmark_returns) > 0 and benchmark_returns.std() > 0:
        benchmark_sr = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
    else:
        benchmark_sr = 0.0
    
    # Calculate excess Sharpe ratios
    excess_srs = [sr - benchmark_sr for sr in sharpe_ratios]
    max_excess_sr = max(excess_srs)
    
    # Bootstrap the null distribution
    bootstrap_max_srs = []
    
    for _ in range(n_bootstrap):
        # Resample returns with replacement
        bootstrap_returns = []
        for returns in returns_list:
            if len(returns) > 0:
                bootstrap_sample = returns.sample(n=len(returns), replace=True)
                if bootstrap_sample.std() > 0:
                    sr = bootstrap_sample.mean() / bootstrap_sample.std() * np.sqrt(252)
                    bootstrap_returns.append(sr)
                else:
                    bootstrap_returns.append(0.0)
            else:
                bootstrap_returns.append(0.0)
        
        # Calculate bootstrap benchmark
        if len(benchmark_returns) > 0:
            bootstrap_benchmark = benchmark_returns.sample(n=len(benchmark_returns), replace=True)
            if bootstrap_benchmark.std() > 0:
                bootstrap_benchmark_sr = (bootstrap_benchmark.mean() / 
                                        bootstrap_benchmark.std() * np.sqrt(252))
            else:
                bootstrap_benchmark_sr = 0.0
        else:
            bootstrap_benchmark_sr = 0.0
        
        # Calculate bootstrap excess Sharpe ratios
        bootstrap_excess_srs = [sr - bootstrap_benchmark_sr for sr in bootstrap_returns]
        bootstrap_max_sr = max(bootstrap_excess_srs)
        bootstrap_max_srs.append(bootstrap_max_sr)
    
    # Calculate p-value
    p_value = np.mean(np.array(bootstrap_max_srs) >= max_excess_sr)
    
    return {
        'max_excess_sharpe': max_excess_sr,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_strategies': len(returns_list),
        'n_bootstrap': n_bootstrap
    }


def stationary_bootstrap_ci(returns: pd.Series, confidence_level: float = 0.95,
                           n_bootstrap: int = 1000, block_size: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate confidence intervals for Sharpe ratio using stationary bootstrap.
    
    The stationary bootstrap accounts for the time series nature of returns
    and provides more accurate confidence intervals than standard bootstrap.
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap iterations
        block_size: Block size for bootstrap (auto-determined if None)
    
    Returns:
        Dictionary with confidence intervals and statistics
    """
    
    if len(returns) == 0 or returns.std() == 0:
        return {
            'sharpe_ratio': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
    
    # Auto-determine block size if not provided
    if block_size is None:
        # Use Politis & White (2004) rule of thumb
        n = len(returns)
        block_size = max(1, int(np.ceil(n**(1/3))))
    
    # Calculate original Sharpe ratio
    original_sr = returns.mean() / returns.std() * np.sqrt(252)
    
    # Stationary bootstrap
    bootstrap_srs = []
    n = len(returns)
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample using stationary bootstrap
        bootstrap_sample = []
        i = 0
        
        while len(bootstrap_sample) < n:
            # Random block length (geometric distribution)
            block_length = np.random.geometric(1.0 / block_size)
            
            # Random starting point
            start_idx = np.random.randint(0, n)
            
            # Add block to bootstrap sample
            for j in range(block_length):
                if len(bootstrap_sample) >= n:
                    break
                idx = (start_idx + j) % n
                bootstrap_sample.append(returns.iloc[idx])
        
        # Calculate bootstrap Sharpe ratio
        bootstrap_returns = pd.Series(bootstrap_sample[:n])
        if bootstrap_returns.std() > 0:
            sr = bootstrap_returns.mean() / bootstrap_returns.std() * np.sqrt(252)
            bootstrap_srs.append(sr)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_srs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_srs, 100 * (1 - alpha / 2))
    
    return {
        'sharpe_ratio': original_sr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'block_size': block_size
    }


def sign_test(returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Sign test for comparing strategy returns to benchmark.
    
    Tests whether the strategy significantly outperforms the benchmark
    using the non-parametric sign test.
    
    Args:
        returns: Strategy return series
        benchmark_returns: Benchmark return series
    
    Returns:
        Dictionary with test statistics and p-value
    """
    
    # Align returns
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        return {
            'n_observations': 0,
            'positive_differences': 0,
            'p_value': 1.0,
            'significant': False
        }
    
    strategy_returns = returns.loc[common_idx]
    benchmark_returns_aligned = benchmark_returns.loc[common_idx]
    
    # Calculate differences
    differences = strategy_returns - benchmark_returns_aligned
    positive_diffs = (differences > 0).sum()
    n_obs = len(differences)
    
    # Sign test (binomial test)
    from scipy.stats import binomtest
    p_value = binomtest(positive_diffs, n_obs, p=0.5, alternative='greater').pvalue
    
    return {
        'n_observations': n_obs,
        'positive_differences': positive_diffs,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def multiple_testing_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
    """
    Apply multiple testing correction to p-values.
    
    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'fdr_bh', 'holm')
    
    Returns:
        List of corrected p-values
    """
    
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        corrected = p_values * len(p_values)
        corrected = np.minimum(corrected, 1.0)
    
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        corrected = np.zeros_like(p_values)
        
        for i, p in enumerate(sorted_p):
            corrected[sorted_indices[i]] = min(1.0, p * (len(p_values) - i))
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR control
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        corrected = np.zeros_like(p_values)
        
        for i in range(len(sorted_p)):
            corrected[sorted_indices[i]] = min(1.0, sorted_p[i] * len(p_values) / (i + 1))
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected.tolist()


def validate_strategy_robustness(returns: pd.Series, benchmark_returns: pd.Series,
                               n_trials: int = 100) -> Dict[str, any]:
    """
    Comprehensive robustness validation for a trading strategy.
    
    Args:
        returns: Strategy return series
        benchmark_returns: Benchmark return series
        n_trials: Number of trials for deflated Sharpe calculation
    
    Returns:
        Dictionary with all validation results
    """
    
    results = {}
    
    # Basic statistics
    if len(returns) > 0 and returns.std() > 0:
        results['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        results['n_observations'] = len(returns)
    else:
        results['sharpe_ratio'] = 0.0
        results['n_observations'] = 0
    
    # Deflated Sharpe ratio
    if results['n_observations'] > 0:
        results['deflated_sharpe_pvalue'] = deflated_sharpe_ratio(
            results['sharpe_ratio'], 
            results['n_observations'], 
            n_trials
        )
        results['deflated_sharpe_significant'] = results['deflated_sharpe_pvalue'] < 0.05
    else:
        results['deflated_sharpe_pvalue'] = 1.0
        results['deflated_sharpe_significant'] = False
    
    # Stationary bootstrap confidence intervals
    results['bootstrap_ci'] = stationary_bootstrap_ci(returns)
    
    # Sign test
    results['sign_test'] = sign_test(returns, benchmark_returns)
    
    # Overall robustness score
    robustness_factors = [
        results['deflated_sharpe_significant'],
        results['sign_test']['significant'],
        results['bootstrap_ci']['ci_lower'] > 0
    ]
    results['robustness_score'] = sum(robustness_factors) / len(robustness_factors)
    
    return results


def daily_return_sign_test(returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    """
    Perform daily return sign test to detect non-random patterns.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns for relative sign test
        
    Returns:
        Dictionary with sign test results
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return {
            'positive_days': 0,
            'total_days': 0,
            'positive_ratio': np.nan,
            'expected_ratio': 0.5,
            'p_value': np.nan,
            'significant': False,
            'error': 'Insufficient data'
        }
    
    # Count positive and negative returns
    positive_days = (returns_clean > 0).sum()
    total_days = len(returns_clean)
    positive_ratio = positive_days / total_days
    
    # Binomial test for randomness
    from scipy.stats import binomtest
    p_value = binomtest(positive_days, total_days, p=0.5).pvalue
    
    # Relative sign test if benchmark provided
    relative_results = None
    if benchmark_returns is not None:
        benchmark_clean = benchmark_returns.dropna()
        if len(benchmark_clean) >= 30:
            # Align returns
            common_dates = returns_clean.index.intersection(benchmark_clean.index)
            if len(common_dates) >= 30:
                strategy_aligned = returns_clean.loc[common_dates]
                benchmark_aligned = benchmark_clean.loc[common_dates]
                
                # Count days where strategy outperforms benchmark
                outperformance_days = (strategy_aligned > benchmark_aligned).sum()
                relative_p_value = binomtest(outperformance_days, len(common_dates), p=0.5).pvalue
                
                relative_results = {
                    'outperformance_days': outperformance_days,
                    'total_comparison_days': len(common_dates),
                    'outperformance_ratio': outperformance_days / len(common_dates),
                    'relative_p_value': relative_p_value,
                    'relative_significant': relative_p_value < 0.05
                }
    
    return {
        'positive_days': positive_days,
        'total_days': total_days,
        'positive_ratio': positive_ratio,
        'expected_ratio': 0.5,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'relative_test': relative_results
    }


def compute_uncertainty_metrics(returns: pd.Series, benchmark_returns: pd.Series = None,
                               confidence_levels: list = [0.90, 0.95, 0.99]) -> dict:
    """
    Compute comprehensive uncertainty quantification metrics.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        confidence_levels: List of confidence levels for CIs
        
    Returns:
        Dictionary with all uncertainty metrics
    """
    results = {
        'sharpe_confidence_intervals': {},
        'sign_test': daily_return_sign_test(returns, benchmark_returns),
        'bootstrap_summary': {}
    }
    
    # Compute Sharpe CIs for different confidence levels
    for conf_level in confidence_levels:
        ci_result = stationary_bootstrap_ci(returns, conf_level)
        results['sharpe_confidence_intervals'][f'{int(conf_level*100)}%'] = ci_result
    
    # Bootstrap summary statistics
    if results['sharpe_confidence_intervals']:
        # Use 95% CI results for summary
        ci_95 = results['sharpe_confidence_intervals']['95%']
        results['bootstrap_summary'] = {
            'sharpe_original': ci_95.get('sharpe_ratio', np.nan),
            'sharpe_bootstrap_mean': np.nan,  # Would need to compute from bootstrap samples
            'sharpe_bootstrap_std': np.nan,   # Would need to compute from bootstrap samples
            'n_bootstrap_samples': ci_95.get('n_bootstrap', 0),
            'ci_width_95': ci_95.get('ci_upper', np.nan) - ci_95.get('ci_lower', np.nan) if not np.isnan(ci_95.get('ci_upper', np.nan)) else np.nan
        }
    
    # Uncertainty score (0-1, higher = more uncertain)
    uncertainty_factors = []
    
    # CI width factor
    if not np.isnan(results['bootstrap_summary'].get('ci_width_95', np.nan)):
        ci_width = results['bootstrap_summary']['ci_width_95']
        uncertainty_factors.append(min(ci_width / 2.0, 1.0))  # Normalize by 2.0 Sharpe units
    
    # Sign test factor
    sign_p = results['sign_test'].get('p_value', 1.0)
    if not np.isnan(sign_p):
        uncertainty_factors.append(1.0 - sign_p)  # Higher p-value = more uncertain
    
    results['uncertainty_score'] = np.mean(uncertainty_factors) if uncertainty_factors else 0.5
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_days = 252
    strategy_returns = pd.Series(np.random.normal(0.001, 0.02, n_days))
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, n_days))
    
    # Run validation
    results = validate_strategy_robustness(strategy_returns, benchmark_returns)
    
    print("Strategy Robustness Validation:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Test uncertainty quantification
    print("\nUncertainty Quantification:")
    uncertainty_results = compute_uncertainty_metrics(strategy_returns, benchmark_returns)
    for key, value in uncertainty_results.items():
        print(f"  {key}: {value}")
