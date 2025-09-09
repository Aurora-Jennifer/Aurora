"""
Risk neutralization module for cross-sectional strategies.

This module implements beta-neutralization and sector-neutralization
to remove unwanted market and sector exposures from cross-sectional signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings


def neutralize_scores(
    scores: pd.Series,
    market_beta: pd.Series,
    sector_dummies: Optional[pd.DataFrame] = None,
    size_factor: Optional[pd.Series] = None,
    momentum_factor: Optional[pd.Series] = None,
    method: str = "ols"
) -> pd.Series:
    """
    Neutralize cross-sectional scores against risk factors.
    
    Parameters:
    -----------
    scores : pd.Series
        Raw cross-sectional scores (index: symbols)
    market_beta : pd.Series
        Market beta for each symbol (index: symbols)
    sector_dummies : pd.DataFrame, optional
        Sector dummy variables (index: symbols, columns: sectors)
    size_factor : pd.Series, optional
        Size factor (e.g., log market cap) for each symbol
    momentum_factor : pd.Series, optional
        Momentum factor (e.g., 12-month return) for each symbol
    method : str
        Neutralization method: "ols" or "residual"
        
    Returns:
    --------
    pd.Series
        Neutralized scores (index: symbols)
    """
    # Align all series to common index
    common_idx = scores.index.intersection(market_beta.index)
    if len(common_idx) == 0:
        warnings.warn("No common symbols between scores and market_beta")
        return scores
    
    scores_aligned = scores.loc[common_idx]
    beta_aligned = market_beta.loc[common_idx]
    
    # Build feature matrix
    features = []
    feature_names = []
    
    # Market beta (always include)
    features.append(beta_aligned.values.reshape(-1, 1))
    feature_names.append("market_beta")
    
    # Sector dummies
    if sector_dummies is not None:
        sector_aligned = sector_dummies.loc[common_idx]
        # Drop one sector to avoid multicollinearity
        sector_cols = sector_aligned.columns[:-1]  # Keep all but last
        features.append(sector_aligned[sector_cols].values)
        feature_names.extend([f"sector_{col}" for col in sector_cols])
    
    # Size factor
    if size_factor is not None:
        size_aligned = size_factor.loc[common_idx]
        features.append(size_aligned.values.reshape(-1, 1))
        feature_names.append("size")
    
    # Momentum factor
    if momentum_factor is not None:
        mom_aligned = momentum_factor.loc[common_idx]
        features.append(mom_aligned.values.reshape(-1, 1))
        feature_names.append("momentum")
    
    # Combine features
    X = np.hstack(features) if len(features) > 1 else features[0]
    
    # Neutralize using OLS residuals
    if method == "ols":
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, scores_aligned.values)
            neutralized = scores_aligned.values - reg.predict(X)
        except ImportError:
            # Fallback to numpy
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            beta_coefs = np.linalg.lstsq(X_with_intercept, scores_aligned.values, rcond=None)[0]
            neutralized = scores_aligned.values - X_with_intercept @ beta_coefs
    
    elif method == "residual":
        # Simple residual method (remove mean of each factor)
        neutralized = scores_aligned.values.copy()
        for i, feature in enumerate(features):
            if feature.shape[1] == 1:  # Single factor
                factor_mean = np.mean(feature)
                neutralized -= factor_mean
            else:  # Multiple factors (e.g., sector dummies)
                for j in range(feature.shape[1]):
                    factor_mean = np.mean(feature[:, j])
                    neutralized -= factor_mean
    
    else:
        raise ValueError(f"Unknown neutralization method: {method}")
    
    # Return as Series with original index
    result = pd.Series(neutralized, index=common_idx, name="neutralized_scores")
    
    # Preserve original scores for symbols not in common index
    full_result = scores.copy()
    full_result.loc[common_idx] = result
    
    return full_result


def create_sector_dummies(
    sector_mapping: Dict[str, str],
    symbols: List[str]
) -> pd.DataFrame:
    """
    Create sector dummy variables from sector mapping.
    
    Parameters:
    -----------
    sector_mapping : Dict[str, str]
        Mapping from symbol to sector
    symbols : List[str]
        List of symbols to create dummies for
        
    Returns:
    --------
    pd.DataFrame
        Sector dummy variables (index: symbols, columns: sectors)
    """
    # Create sector series
    sector_series = pd.Series([sector_mapping.get(sym, "Unknown") for sym in symbols], 
                             index=symbols, name="sector")
    
    # Create dummy variables
    dummies = pd.get_dummies(sector_series, prefix="sector")
    
    return dummies


def estimate_market_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 252,
    min_obs: int = 60
) -> pd.Series:
    """
    Estimate market beta for each symbol using rolling regression.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Stock returns (index: dates, columns: symbols)
    market_returns : pd.Series
        Market returns (index: dates)
    lookback : int
        Lookback window for beta estimation
    min_obs : int
        Minimum observations required for beta estimation
        
    Returns:
    --------
    pd.Series
        Market beta for each symbol (index: symbols)
    """
    betas = {}
    
    # Align returns and market returns
    common_dates = returns.index.intersection(market_returns.index)
    returns_aligned = returns.loc[common_dates]
    market_aligned = market_returns.loc[common_dates]
    
    for symbol in returns_aligned.columns:
        stock_returns = returns_aligned[symbol].dropna()
        
        if len(stock_returns) < min_obs:
            betas[symbol] = np.nan
            continue
        
        # Use most recent data for beta estimation
        if len(stock_returns) > lookback:
            stock_returns = stock_returns.tail(lookback)
            market_returns_subset = market_aligned.loc[stock_returns.index]
        else:
            market_returns_subset = market_aligned.loc[stock_returns.index]
        
        # Align market returns
        common_idx = stock_returns.index.intersection(market_returns_subset.index)
        if len(common_idx) < min_obs:
            betas[symbol] = np.nan
            continue
        
        stock_aligned = stock_returns.loc[common_idx]
        market_aligned_subset = market_returns_subset.loc[common_idx]
        
        # Estimate beta using simple regression
        try:
            # Beta = Cov(stock, market) / Var(market)
            covariance = np.cov(stock_aligned, market_aligned_subset)[0, 1]
            market_variance = np.var(market_aligned_subset, ddof=1)
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = np.nan
                
            betas[symbol] = beta
            
        except Exception:
            betas[symbol] = np.nan
    
    return pd.Series(betas, name="market_beta")


def create_size_factor(
    market_caps: pd.DataFrame,
    method: str = "log"
) -> pd.Series:
    """
    Create size factor from market capitalization data.
    
    Parameters:
    -----------
    market_caps : pd.DataFrame
        Market capitalization data (index: dates, columns: symbols)
    method : str
        Size factor method: "log", "rank", or "zscore"
        
    Returns:
    --------
    pd.Series
        Size factor for each symbol (index: symbols)
    """
    # Use most recent market cap data
    latest_caps = market_caps.iloc[-1].dropna()
    
    if method == "log":
        size_factor = np.log(latest_caps)
    elif method == "rank":
        size_factor = latest_caps.rank(pct=True)
    elif method == "zscore":
        size_factor = (latest_caps - latest_caps.mean()) / latest_caps.std()
    else:
        raise ValueError(f"Unknown size factor method: {method}")
    
    return size_factor




def create_sector_dummies(symbols: List[str], n_sectors: int = 10) -> pd.DataFrame:
    """
    Create sector dummy variables for a list of symbols using algorithmic classification.
    
    Uses a simple, data-driven approach that doesn't rely on hardcoded mappings.
    This creates balanced sector exposure without manual maintenance.
    
    Parameters:
    -----------
    symbols : List[str]
        List of stock symbols
    n_sectors : int
        Number of sectors to create (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with symbols as index and sector dummy columns
    """
    # Simple algorithmic approach: divide symbols into balanced groups
    # This avoids hardcoding and works for any universe
    n_symbols = len(symbols)
    
    # Create balanced sector assignments
    symbols_per_sector = n_symbols // n_sectors
    remainder = n_symbols % n_sectors
    
    sector_assignments = []
    current_sector = 0
    symbols_in_current_sector = 0
    
    for i, symbol in enumerate(symbols):
        # Add extra symbol to first 'remainder' sectors to balance
        sector_size = symbols_per_sector + (1 if current_sector < remainder else 0)
        
        if symbols_in_current_sector >= sector_size:
            current_sector += 1
            symbols_in_current_sector = 0
        
        sector_assignments.append(f'Sector_{current_sector + 1}')
        symbols_in_current_sector += 1
    
    # Create DataFrame with sector dummies
    sector_df = pd.DataFrame({'symbol': symbols, 'sector': sector_assignments})
    sector_dummies = pd.get_dummies(sector_df['sector'], prefix='sector')
    sector_dummies.index = symbols
    
    # Drop one sector to avoid collinearity
    if len(sector_dummies.columns) > 1:
        sector_dummies = sector_dummies.drop(sector_dummies.columns[-1], axis=1)
    
    return sector_dummies


def _ridge_residual(y, X, ridge=1e-6):
    """
    Compute ridge regression residuals for numerical stability.
    
    Parameters:
    -----------
    y : np.ndarray
        Target values (n,)
    X : np.ndarray
        Feature matrix (n, k)
    ridge : float
        Ridge regularization parameter
        
    Returns:
    --------
    resid : np.ndarray
        Residuals (n,)
    beta : np.ndarray
        Ridge coefficients (k,)
    """
    XtX = X.T @ X
    k = XtX.shape[0]
    beta = np.linalg.solve(XtX + ridge * np.eye(k), X.T @ y)
    resid = y - X @ beta
    return resid, beta


def partial_neutralize_series(y: pd.Series,
                              exposures: pd.DataFrame,
                              gamma: float = 0.6,
                              ridge: float = 1e-6) -> pd.Series:
    """
    Apply partial neutralization to a series using ridge regression.
    
    Formula: y_partial = (1-γ) * y + γ * resid(y | X)
    
    Parameters:
    -----------
    y : pd.Series
        Target series to neutralize
    exposures : pd.DataFrame
        Exposure factors (same index as y)
    gamma : float
        Neutralization strength (0=no neutralization, 1=full neutralization)
    ridge : float
        Ridge regularization parameter for numerical stability
        
    Returns:
    --------
    pd.Series
        Partially neutralized series
    """
    # Align data
    X = exposures.loc[y.index].astype(float).values
    yy = y.values.astype(float)
    
    # Handle edge cases
    if X.shape[1] == 0 or np.all(np.isnan(yy)) or np.all(np.isnan(X)):
        return y
    
    # Remove rows with any NaN values
    valid_mask = np.isfinite(yy) & np.isfinite(X).all(axis=1)
    if valid_mask.sum() < 10:  # Need minimum observations
        return y
    
    y_clean = yy[valid_mask]
    X_clean = X[valid_mask]
    
    # Compute ridge residuals
    resid, beta = _ridge_residual(y_clean, X_clean, ridge=ridge)
    
    # Apply partial neutralization with proper dtype casting
    y_partial_clean = ((1 - gamma) * y_clean + gamma * resid).astype(np.float32)
    
    # Reconstruct full series
    y_partial = y.copy()
    y_partial.iloc[valid_mask] = y_partial_clean
    
    return y_partial


def winsorize_by_date(df: pd.DataFrame, cols, lo=0.025, hi=0.975):
    """
    Winsorize specified columns by date (cross-sectional clipping).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data with 'date' column
    cols : list
        Column names to winsorize
    lo : float
        Lower quantile threshold
    hi : float
        Upper quantile threshold
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with winsorized columns
    """
    # Use transform to avoid FutureWarning and improve performance
    for col in cols:
        g = df.groupby('date')[col]
        ql = g.transform(lambda x: x.quantile(lo))
        qh = g.transform(lambda x: x.quantile(hi))
        df[col] = df[col].clip(lower=ql, upper=qh)
    return df


def cs_zscore_features(df: pd.DataFrame, feature_cols):
    """
    Apply cross-sectional z-scoring to features by date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data with 'date' column
    feature_cols : list
        Feature column names to z-score
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with z-scored features
    """
    # Use transform to avoid FutureWarning and improve performance
    g = df.groupby('date')[feature_cols]
    mu = g.transform('mean')
    sd = g.transform('std', ddof=0).replace(0.0, np.nan)
    df[feature_cols] = ((df[feature_cols] - mu) / sd).fillna(0.0)
    return df


def create_momentum_factor(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip_days: int = 21
) -> pd.Series:
    """
    Create momentum factor from historical returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Stock returns (index: dates, columns: symbols)
    lookback : int
        Lookback window for momentum calculation
    skip_days : int
        Days to skip at the end (to avoid look-ahead bias)
        
    Returns:
    --------
    pd.Series
        Momentum factor for each symbol (index: symbols)
    """
    # Calculate momentum as cumulative return over lookback period
    momentum = {}
    
    for symbol in returns.columns:
        stock_returns = returns[symbol].dropna()
        
        if len(stock_returns) < lookback + skip_days:
            momentum[symbol] = np.nan
            continue
        
        # Use returns from lookback period, skipping recent days
        start_idx = len(stock_returns) - lookback - skip_days
        end_idx = len(stock_returns) - skip_days
        
        if start_idx < 0:
            momentum[symbol] = np.nan
            continue
        
        momentum_returns = stock_returns.iloc[start_idx:end_idx]
        momentum[symbol] = (1 + momentum_returns).prod() - 1
    
    return pd.Series(momentum, name="momentum")


def apply_risk_neutralization(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    market_returns: pd.Series,
    market_caps: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,
    volumes: Optional[pd.DataFrame] = None,
    sector_mapping: Optional[Dict[str, str]] = None,
    neutralization_config: Optional[Dict] = None,
    rolling_beta_panel: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply risk neutralization to cross-sectional scores.
    
    Parameters:
    -----------
    scores : pd.DataFrame
        Cross-sectional scores (index: dates, columns: symbols)
    returns : pd.DataFrame
        Stock returns (index: dates, columns: symbols)
    market_returns : pd.Series
        Market returns (index: dates)
    market_caps : pd.DataFrame, optional
        Market capitalization data (index: dates, columns: symbols)
    sector_mapping : Dict[str, str], optional
        Mapping from symbol to sector
    neutralization_config : Dict, optional
        Configuration for neutralization
        
    Returns:
    --------
    pd.DataFrame
        Risk-neutralized scores (index: dates, columns: symbols)
    """
    if neutralization_config is None:
        neutralization_config = {
            "beta_lookback": 252,
            "beta_min_obs": 60,
            "momentum_lookback": 252,
            "momentum_skip_days": 21,
            "size_method": "log",
            "neutralization_method": "ols"
        }
    
    # Initialize result
    neutralized_scores = scores.copy()
    
    # Use rolling beta panel if provided, otherwise estimate market beta
    if rolling_beta_panel is not None:
        print("📊 Using pre-computed rolling beta panel for risk neutralization...")
        # For each date, use the beta values from the rolling panel
        market_beta = rolling_beta_panel.loc[scores.index].mean(axis=0)  # Average across dates
    else:
        print("📊 Estimating market beta for risk neutralization...")
        market_beta = estimate_market_beta(
            returns, market_returns,
            lookback=neutralization_config["beta_lookback"],
            min_obs=neutralization_config["beta_min_obs"]
        )
    
    # Log beta coverage
    valid_beta_pct = (market_beta.notna().sum() / len(market_beta)) * 100
    print(f"🔍 Debug: Beta coverage: {valid_beta_pct:.1f}% ({market_beta.notna().sum()}/{len(market_beta)} symbols)")
    print(f"🔍 Debug: Beta stats: mean={market_beta.mean():.3f}, std={market_beta.std():.3f}")
    
    # Create sector dummies (once for all dates)
    sector_dummies = None
    print("🏢 Creating sector dummy variables...")
    try:
        sector_dummies = create_sector_dummies(scores.columns.tolist())
        print(f"🏢 Sector dummies created: {sector_dummies.shape[1]} sectors for {sector_dummies.shape[0]} symbols")
    except Exception as e:
        print(f"⚠️ Could not create sector dummies: {e}")
        sector_dummies = None
    
    # Create size factor (once for all dates)
    size_factor = None
    if market_caps is not None:
        print("📏 Creating size factor from market cap...")
        size_factor = create_size_factor(
            market_caps, 
            method=neutralization_config["size_method"]
        )
    elif prices is not None and volumes is not None:
        print("📏 Creating size factor from ADV (close * volume)...")
        try:
            size_factor = create_adv_size_factor(
                prices, volumes,
                lookback=neutralization_config.get("size_lookback", 60),
                method=neutralization_config["size_method"]
            )
        except Exception as e:
            print(f"⚠️ Could not create ADV size factor: {e}")
            size_factor = None
    else:
        print("⚠️ No market cap or price/volume data available - skipping size factor")
    
    # Create momentum factor (once for all dates)
    print("🚀 Creating momentum factor...")
    momentum_factor = create_momentum_factor(
        returns,
        lookback=neutralization_config["momentum_lookback"],
        skip_days=neutralization_config["momentum_skip_days"]
    )
    
    # Log exposure factors summary
    exposures = []
    if market_beta is not None and market_beta.notna().any():
        exposures.append(f"market_beta({market_beta.notna().sum()})")
    if sector_dummies is not None:
        exposures.append(f"sector_dummies({len(sector_dummies.columns)})")
    if size_factor is not None and size_factor.notna().any():
        exposures.append(f"size({size_factor.notna().sum()})")
    if momentum_factor is not None and momentum_factor.notna().any():
        exposures.append(f"momentum({momentum_factor.notna().sum()})")
    
    print(f"🔍 Debug: Neutralizing on exposures: {', '.join(exposures)}")
    if not exposures:
        print("⚠️  WARNING: No exposure factors available for neutralization!")
    
    # Apply neutralization for each date
    print("🛡️  Applying risk neutralization to cross-sectional scores...")
    for date in scores.index:
        date_scores = scores.loc[date]
        
        # Skip if all scores are NaN
        if date_scores.isna().all():
            continue
        
        # Apply neutralization
        neutralized = neutralize_scores(
            date_scores,
            market_beta,
            sector_dummies,
            size_factor,
            momentum_factor,
            method=neutralization_config["neutralization_method"]
        )
        
        neutralized_scores.loc[date] = neutralized
    
    print(f"✅ Risk neutralization complete. Processed {len(scores)} dates.")
    
    return neutralized_scores
