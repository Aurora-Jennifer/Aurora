"""
IC (Information Coefficient) validation with statistical rigor.

This module implements professional-grade IC analysis with:
- Newey-West HAC standard errors for autocorrelation
- Block bootstrap for confidence intervals 
- Multiple testing awareness
- Regime-aware IC decomposition
"""

import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr


@dataclass
class ICResult:
    """Information Coefficient analysis results."""
    mean_ic: float
    ic_std: float
    t_stat: float
    p_value: float
    confidence_interval: tuple[float, float]
    hit_rate: float  # Proportion of positive IC periods
    num_observations: int
    method: str = "pearson"
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if IC is statistically significant."""
        return self.p_value < alpha
    
    def quality_score(self) -> str:
        """Classify IC quality."""
        if abs(self.mean_ic) < 0.02:
            return "weak"
        if abs(self.mean_ic) < 0.05:
            return "moderate"
        if abs(self.mean_ic) < 0.10:
            return "strong"
        return "exceptional"


class ICValidator:
    """Professional IC validation with statistical rigor."""
    
    def __init__(self, bootstrap_samples: int = 1000, block_length: int | None = None):
        """
        Initialize IC validator.
        
        Args:
            bootstrap_samples: Number of bootstrap resamples
            block_length: Block length for bootstrap (auto if None)
        """
        self.bootstrap_samples = bootstrap_samples
        self.block_length = block_length
    
    def compute_ic_stats(
        self, 
        predictions: pd.Series, 
        returns: pd.Series,
        method: str = "pearson",
        lags: int | None = None
    ) -> ICResult:
        """
        Compute IC statistics with HAC standard errors and bootstrap CI.
        
        Args:
            predictions: Model predictions (aligned with return dates)
            returns: Realized returns
            method: 'pearson' or 'spearman'
            lags: Newey-West lags (auto if None)
            
        Returns:
            ICResult with comprehensive statistics
        """
        # Align data and drop NaNs
        data = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
        
        if len(data) < 30:
            warnings.warn(f"IC computation with only {len(data)} observations may be unreliable")
        
        if len(data) == 0:
            return ICResult(0.0, 0.0, 0.0, 1.0, (0.0, 0.0), 0.5, 0, method)
        
        pred = data["pred"].values
        ret = data["ret"].values
        
        # Compute IC
        if method == "pearson":
            ic, _ = pearsonr(pred, ret)
        elif method == "spearman":
            ic, _ = spearmanr(pred, ret)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if np.isnan(ic):
            return ICResult(0.0, 0.0, 0.0, 1.0, (0.0, 0.0), 0.5, len(data), method)
        
        # Rolling IC for time series analysis
        window = max(21, len(data) // 10)  # At least 21 days or 10% of data
        rolling_ic = self._compute_rolling_ic(pred, ret, window, method)
        
        # HAC standard error (Newey-West)
        if lags is None:
            lags = int(np.floor(4 * (len(rolling_ic) / 100) ** (2/9)))  # Standard rule
        
        ic_se = self._newey_west_se(rolling_ic, lags)
        
        # T-statistic and p-value
        t_stat = np.mean(rolling_ic) / ic_se if ic_se > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(rolling_ic) - 1))
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(pred, ret, method)
        
        # Hit rate (proportion of positive IC periods)
        hit_rate = np.mean(rolling_ic > 0) if len(rolling_ic) > 0 else 0.5
        
        return ICResult(
            mean_ic=ic,
            ic_std=np.std(rolling_ic) if len(rolling_ic) > 1 else 0.0,
            t_stat=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            hit_rate=hit_rate,
            num_observations=len(data),
            method=method
        )
    
    def _compute_rolling_ic(
        self, 
        predictions: np.ndarray, 
        returns: np.ndarray, 
        window: int,
        method: str
    ) -> np.ndarray:
        """Compute rolling IC for time series analysis."""
        if len(predictions) < window:
            # Not enough data for rolling window
            if method == "pearson":
                ic, _ = pearsonr(predictions, returns)
            else:
                ic, _ = spearmanr(predictions, returns)
            return np.array([ic] if not np.isnan(ic) else [0.0])
        
        rolling_ics = []
        for i in range(window, len(predictions) + 1):
            pred_window = predictions[i-window:i]
            ret_window = returns[i-window:i]
            
            if method == "pearson":
                ic, _ = pearsonr(pred_window, ret_window)
            else:
                ic, _ = spearmanr(pred_window, ret_window)
            
            if not np.isnan(ic):
                rolling_ics.append(ic)
        
        return np.array(rolling_ics) if rolling_ics else np.array([0.0])
    
    def _newey_west_se(self, time_series: np.ndarray, lags: int) -> float:
        """
        Compute Newey-West HAC standard error.
        
        Accounts for heteroskedasticity and autocorrelation in IC time series.
        """
        if len(time_series) <= 1:
            return 0.0
        
        n = len(time_series)
        mean_ic = np.mean(time_series)
        
        # Deviations from mean
        deviations = time_series - mean_ic
        
        # Variance (lag 0)
        gamma_0 = np.mean(deviations ** 2)
        
        # Autocovariances
        gamma_sum = gamma_0
        for lag in range(1, min(lags + 1, n)):
            if n - lag <= 0:
                break
            
            gamma_lag = np.mean(deviations[:-lag] * deviations[lag:])
            
            # Bartlett weights
            weight = 1 - lag / (lags + 1)
            gamma_sum += 2 * weight * gamma_lag
        
        # HAC variance
        hac_variance = gamma_sum / n
        
        return np.sqrt(max(hac_variance, 1e-8))  # Avoid zero SE
    
    def _bootstrap_ci(
        self, 
        predictions: np.ndarray, 
        returns: np.ndarray, 
        method: str,
        alpha: float = 0.05
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval using block bootstrap.
        
        Block bootstrap preserves time series structure in IC estimates.
        """
        n = len(predictions)
        if n < 10:
            return (0.0, 0.0)
        
        # Auto block length if not specified
        if self.block_length is None:
            block_len = max(5, int(np.sqrt(n)))
        else:
            block_len = self.block_length
        
        bootstrap_ics = []
        
        for _ in range(self.bootstrap_samples):
            # Block bootstrap resample
            resampled_pred, resampled_ret = self._block_resample(
                predictions, returns, block_len
            )
            
            # Compute IC for bootstrap sample
            try:
                if method == "pearson":
                    ic, _ = pearsonr(resampled_pred, resampled_ret)
                else:
                    ic, _ = spearmanr(resampled_pred, resampled_ret)
                
                if not np.isnan(ic):
                    bootstrap_ics.append(ic)
            except:
                continue
        
        if not bootstrap_ics:
            return (0.0, 0.0)
        
        # Confidence interval
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_ics, lower_percentile)
        ci_upper = np.percentile(bootstrap_ics, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _block_resample(
        self, 
        predictions: np.ndarray, 
        returns: np.ndarray, 
        block_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform block bootstrap resampling.
        
        Maintains temporal structure within blocks while allowing 
        resampling across different time periods.
        """
        n = len(predictions)
        
        if block_length >= n:
            # Single block case
            return predictions.copy(), returns.copy()
        
        # Number of blocks needed
        num_blocks = int(np.ceil(n / block_length))
        
        resampled_pred = []
        resampled_ret = []
        
        for _ in range(num_blocks):
            # Random block start
            start_idx = np.random.randint(0, n - block_length + 1)
            end_idx = start_idx + block_length
            
            # Add block to resample
            resampled_pred.extend(predictions[start_idx:end_idx])
            resampled_ret.extend(returns[start_idx:end_idx])
        
        # Trim to original length
        resampled_pred = np.array(resampled_pred[:n])
        resampled_ret = np.array(resampled_ret[:n])
        
        return resampled_pred, resampled_ret
    
    def regime_analysis(
        self, 
        predictions: pd.Series, 
        returns: pd.Series,
        volatility: pd.Series,
        method: str = "pearson"
    ) -> dict:
        """
        Analyze IC performance across market regimes.
        
        Args:
            predictions: Model predictions
            returns: Realized returns
            volatility: Market volatility proxy (e.g., rolling std)
            method: Correlation method
            
        Returns:
            Dict with regime-specific IC statistics
        """
        # Align all series
        data = pd.DataFrame({
            "pred": predictions,
            "ret": returns, 
            "vol": volatility
        }).dropna()
        
        if len(data) < 30:
            return {"error": "Insufficient data for regime analysis"}
        
        # Define regimes based on volatility terciles
        vol_33 = data["vol"].quantile(0.33)
        vol_67 = data["vol"].quantile(0.67)
        
        regimes = {
            "low_vol": data[data["vol"] <= vol_33],
            "med_vol": data[(data["vol"] > vol_33) & (data["vol"] <= vol_67)],
            "high_vol": data[data["vol"] > vol_67]
        }
        
        regime_results = {}
        
        for regime_name, regime_data in regimes.items():
            if len(regime_data) >= 10:  # Minimum observations
                ic_result = self.compute_ic_stats(
                    regime_data["pred"], 
                    regime_data["ret"], 
                    method
                )
                regime_results[regime_name] = {
                    "ic": ic_result.mean_ic,
                    "t_stat": ic_result.t_stat,
                    "p_value": ic_result.p_value,
                    "hit_rate": ic_result.hit_rate,
                    "observations": len(regime_data),
                    "quality": ic_result.quality_score()
                }
            else:
                regime_results[regime_name] = {"error": "Insufficient observations"}
        
        return regime_results
    
    def multiple_testing_adjustment(
        self, 
        ic_results: list[ICResult], 
        method: str = "bonferroni"
    ) -> list[ICResult]:
        """
        Apply multiple testing correction to IC results.
        
        Args:
            ic_results: List of IC results from different configurations
            method: Correction method ('bonferroni', 'bh' for Benjamini-Hochberg)
            
        Returns:
            List of IC results with adjusted p-values
        """
        if not ic_results:
            return ic_results
        
        p_values = [result.p_value for result in ic_results]
        
        if method == "bonferroni":
            adjusted_p = [p * len(p_values) for p in p_values]
        elif method == "bh":  # Benjamini-Hochberg
            sorted_indices = np.argsort(p_values)
            adjusted_p = [0.0] * len(p_values)
            
            for i, idx in enumerate(sorted_indices):
                bh_factor = len(p_values) / (i + 1)
                adjusted_p[idx] = min(1.0, p_values[idx] * bh_factor)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Create new results with adjusted p-values
        adjusted_results = []
        for i, result in enumerate(ic_results):
            adjusted_result = ICResult(
                mean_ic=result.mean_ic,
                ic_std=result.ic_std,
                t_stat=result.t_stat,
                p_value=min(1.0, adjusted_p[i]),  # Cap at 1.0
                confidence_interval=result.confidence_interval,
                hit_rate=result.hit_rate,
                num_observations=result.num_observations,
                method=result.method
            )
            adjusted_results.append(adjusted_result)
        
        return adjusted_results
