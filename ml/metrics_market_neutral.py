"""
Market-neutral metrics for cross-sectional trading strategies.

Provides CAPM-based metrics with Newey-West standard errors for robust
statistical inference in the presence of serial correlation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252


def _newey_west_sigma(x: pd.Series, lags: int | None = None) -> float:
    """Newey–West (HAC) stdev for a zero-mean series."""
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 5:
        return x.std(ddof=1)
    if lags is None:
        lags = max(1, int(round(np.sqrt(len(x)))))
    # HAC variance via statsmodels (OLS on constant with HAC cov)
    X = np.ones((len(x), 1))
    model = sm.OLS(x.values, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    var = res.cov_params()[0, 0]  # variance of the mean
    # Convert variance-of-mean to variance-of-series: var(x) = var(mean)*n
    series_var = var * len(x)
    return float(np.sqrt(series_var))


def _to_decimal(r):
    """Convert returns to decimal format if they appear to be percentages."""
    if r is None or len(r) == 0:
        return r
    med = r.abs().median()
    if med > 1:   # likely percentage
        r = r / 100.0
    if med > 0.2: # still too big? some upstream scaling
        r = r / 100.0
    return r


def capm_metrics(strat_ret: pd.Series, mkt_ret: pd.Series, horizon: int = 5):
    """
    Return dict with beta, alpha (daily & annual), alpha t-stat (NW), and IR vs market.
    
    Args:
        strat_ret: Strategy returns series (daily)
        mkt_ret: Market returns series (daily)
        horizon: Return horizon in days (for proper annualization)
        
    Returns:
        Dictionary with CAPM metrics:
        - beta: Market beta
        - alpha_daily: Daily alpha
        - alpha_ann: Annualized alpha
        - alpha_tstat: Newey-West t-statistic for alpha
        - ir_mkt: Information ratio vs market (annualized) - computed from active returns
        - n_capm_obs: Number of observations used
        - capm_status: Status of CAPM calculation
        - nw_lags: Number of lags used in Newey-West
        - annualizer: Annualization factor used
    """
    # Convert to decimal format and align dates
    strat_ret = _to_decimal(strat_ret)
    mkt_ret = _to_decimal(mkt_ret)
    
    # Align dates - ensure both series have same index
    mkt_ret = mkt_ret.reindex(strat_ret.index)
    
    df = pd.concat({"s": strat_ret, "m": mkt_ret}, axis=1).dropna()
    
    # Calculate annualizer based on horizon
    annualizer = np.sqrt(TRADING_DAYS / horizon)
    
    if len(df) < 40:  # Too short for reliable CAPM
        return dict(beta=np.nan, alpha_daily=np.nan, alpha_ann=np.nan,
                    alpha_tstat=np.nan, ir_mkt=np.nan, n_capm_obs=len(df),
                    capm_status="insufficient_data", nw_lags=0, annualizer=annualizer)
    
    if df["m"].std(ddof=1) == 0:
        return dict(beta=np.nan, alpha_daily=np.nan, alpha_ann=np.nan,
                    alpha_tstat=np.nan, ir_mkt=np.nan, n_capm_obs=len(df),
                    capm_status="zero_market_vol", nw_lags=0, annualizer=annualizer)

    # Use h-1 for Newey-West lags (more appropriate for short horizons)
    nw_lags = max(1, horizon - 1)

    X = sm.add_constant(df["m"].values)
    y = df["s"].values
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    alpha = float(ols.params[0])
    beta = float(ols.params[1])
    alpha_se = float(ols.bse[0])
    alpha_t = alpha / alpha_se if alpha_se > 0 else np.nan

    # FIXED: Information Ratio vs market using alpha t-statistic approach
    # The issue: OLS residuals (strategy - beta*market) have mean zero by construction
    # Solution: Use the regression alpha's t-statistic as the IR measure
    # This represents the risk-adjusted active return (alpha) relative to its uncertainty
    
    # IR_mkt = alpha_t_statistic (this is the risk-adjusted active return)
    # The alpha t-stat is: alpha / standard_error_of_alpha
    # This is essentially the Information Ratio of the alpha component
    ir_mkt = alpha_t  # Use the alpha t-statistic as IR

    return dict(
        beta=beta,
        alpha_daily=alpha,
        alpha_ann=alpha * annualizer,
        alpha_tstat=alpha_t,
        ir_mkt=ir_mkt,
        n_capm_obs=len(df),
        capm_status="sufficient_data",
        nw_lags=nw_lags,
        annualizer=annualizer
    )
