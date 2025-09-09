"""
Panel Dataset Builder for Global Cross-Sectional Models

Builds a proper panel dataset (date × symbol) with real market features
for training global cross-sectional ranking models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PanelBuilder:
    """Builds panel datasets with market features for global models."""
    
    def __init__(self, 
                 universe: List[str],
                 market_proxy: str = 'SPY',
                 cross_proxies: List[str] = None):
        """Initialize panel builder."""
        self.universe = universe
        self.market_proxy = market_proxy
        self.cross_proxies = cross_proxies or ['QQQ', 'TLT', 'UUP', 'VIXY']
        
    def download_data(self, 
                     start_date: str = '2020-01-01',
                     end_date: str = '2024-01-01') -> Dict[str, pd.DataFrame]:
        """Download price data for all symbols."""
        logger.info(f"Downloading data for {len(self.universe)} symbols")
        
        all_data = {}
        
        # Download universe data
        for symbol in self.universe:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
                if not df.empty:
                    # Handle multi-level columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [f"{symbol}_{col[0].lower()}" for col in df.columns]
                    else:
                        df.columns = [f"{symbol}_{col.lower()}" for col in df.columns]
                    all_data[symbol] = df
                    logger.info(f"Downloaded {len(df)} days for {symbol}")
                else:
                    logger.warning(f"No data for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
        
        # Download market proxy
        try:
            market_df = yf.download(self.market_proxy, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not market_df.empty:
                # Handle multi-level columns
                if isinstance(market_df.columns, pd.MultiIndex):
                    market_df.columns = [f"market_{col[0].lower()}" for col in market_df.columns]
                else:
                    market_df.columns = [f"market_{col.lower()}" for col in market_df.columns]
                all_data['market'] = market_df
                logger.info(f"Downloaded {len(market_df)} days for market proxy")
        except Exception as e:
            logger.error(f"Error downloading market proxy: {e}")
        
        # Download cross-asset proxies
        for proxy in self.cross_proxies:
            try:
                proxy_df = yf.download(proxy, start=start_date, end=end_date, progress=False, auto_adjust=False)
                if not proxy_df.empty:
                    # Handle multi-level columns
                    if isinstance(proxy_df.columns, pd.MultiIndex):
                        proxy_df.columns = [f"{proxy.lower()}_{col[0].lower()}" for col in proxy_df.columns]
                    else:
                        proxy_df.columns = [f"{proxy.lower()}_{col.lower()}" for col in proxy_df.columns]
                    all_data[proxy] = proxy_df
                    logger.info(f"Downloaded {len(proxy_df)} days for {proxy}")
            except Exception as e:
                logger.error(f"Error downloading {proxy}: {e}")
        
        return all_data
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate features for each symbol."""
        logger.info("Calculating features for all symbols")
        
        all_features = []
        
        for symbol in self.universe:
            if symbol not in data:
                continue
                
            df = data[symbol].copy()
            
            # Store raw values for relative feature construction
            df['close'] = df[f'{symbol}_close']
            df['volume'] = df[f'{symbol}_volume']
            
            # Returns
            df['ret1'] = df['close'].pct_change(1)
            df['ret5'] = df['close'].pct_change(5)
            df['ret10'] = df['close'].pct_change(10)
            df['ret20'] = df['close'].pct_change(20)
            
            # 🔧 CRITICAL FIX: Volatility must be per-symbol rolling over time
            # We're already in a per-symbol loop, so just use rolling directly
            df['vol_5'] = df['ret1'].rolling(5, min_periods=5).std(ddof=0)
            df['vol_20'] = df['ret1'].rolling(20, min_periods=10).std(ddof=0)
            
            # Technical indicators
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Moving averages
            df['ma_fast'] = df['close'].rolling(10).mean()
            df['ma_slow'] = df['close'].rolling(20).mean()
            df['ma_ratio'] = df['ma_fast'] / df['ma_slow']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price features
            df['price_ma_ratio'] = df['close'] / df['ma_slow']
            df['vol_ratio'] = df['vol_5'] / df['vol_20']
            df['ret_vol_ratio'] = df['ret5'] / df['vol_5']
            
            # Momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Regime features (simplified)
            df['regime_trend'] = (df['ma_fast'] > df['ma_slow']).astype(int)
            df['regime_vol'] = (df['vol_5'] > df['vol_20']).astype(int)
            df['regime_chop'] = (df['bb_width'] > df['bb_width'].rolling(20).mean()).astype(int)
            
            # ROBUST WORKHORSE FEATURES - commonly used in cross-sectional alpha generation
            
            # 1. Short-term reversal (1-5 day mean reversion)
            df['reversal_1_5'] = -df['ret1'].rolling(5).mean()  # Negative for mean reversion
            
            # 2. Medium-term momentum (5-20 day trend following)
            df['momentum_5_20'] = df['close'] / df['close'].shift(5) - df['close'].shift(5) / df['close'].shift(20)
            
            # 3. Volatility-adjusted returns (Sharpe-like ratios)
            df['sharpe_5'] = df['ret5'] / (df['vol_5'] + 1e-8)  # Add small epsilon to avoid division by zero
            df['sharpe_20'] = df['ret20'] / (df['vol_20'] + 1e-8)
            
            # 4. Volume-price trend (VPT) - cumulative volume-weighted price change
            df['vpt'] = (df['volume'] * df['ret1']).rolling(20).sum()
            
            # 5. Price acceleration (second derivative of price)
            df['price_accel'] = df['ret1'].diff()  # Change in returns
            
            # 6. Volume acceleration
            df['volume_accel'] = df['volume_ratio'].diff()
            
            # 7. Relative strength vs. market (will be calculated later with market data)
            df['relative_strength_5'] = df['ret5']  # Placeholder, will be adjusted vs market
            df['relative_strength_20'] = df['ret20']  # Placeholder, will be adjusted vs market
            
            # 8. Volatility regime features
            df['vol_regime_high'] = (df['vol_5'] > df['vol_20'].rolling(20).quantile(0.8)).astype(int)
            df['vol_regime_low'] = (df['vol_5'] < df['vol_20'].rolling(20).quantile(0.2)).astype(int)
            
            # 9. Price position within recent range
            df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-8)
            
            # 10. Volume position within recent range
            df['volume_position_20'] = (df['volume'] - df['volume'].rolling(20).min()) / (df['volume'].rolling(20).max() - df['volume'].rolling(20).min() + 1e-8)
            
            # 11. Quality/Profitability proxy (using price stability as proxy)
            df['quality_proxy'] = 1 / (df['vol_20'] + 1e-8)  # Inverse volatility as quality proxy
            
            # 12. Liquidity proxy (using volume stability)
            df['liquidity_proxy'] = df['volume'].rolling(20).mean() / (df['volume'].rolling(20).std() + 1e-8)
            
            # 13. Trend strength (using multiple timeframes)
            df['trend_strength'] = (df['ma_fast'] / df['ma_slow'] - 1) * (df['close'] / df['ma_slow'] - 1)
            
            # 14. Mean reversion signal (using RSI and Bollinger position)
            df['mean_reversion'] = (df['rsi_14'] - 50) * (df['bb_position'] - 0.5)
            
            # 15. Breakout signal (price vs recent highs/lows)
            df['breakout_signal'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(int) - (df['close'] < df['close'].rolling(20).min().shift(1)).astype(int)
            
            # Add symbol identifier and date
            df['symbol'] = symbol
            df['date'] = df.index  # Use the index as date
            
            # Select feature columns (including new robust workhorse features)
            feature_cols = [
                'symbol', 'date', 'close', 'volume',
                'ret1', 'ret5', 'ret10', 'ret20',
                'vol_5', 'vol_20',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'ma_fast', 'ma_slow', 'ma_ratio',
                'rsi_14',
                'volume_ratio', 'price_ma_ratio', 'vol_ratio', 'ret_vol_ratio',
                'momentum_5', 'momentum_10', 'momentum_20',
                'regime_trend', 'regime_vol', 'regime_chop',
                # New robust workhorse features
                'reversal_1_5', 'momentum_5_20',
                'sharpe_5', 'sharpe_20',
                'vpt', 'price_accel', 'volume_accel',
                'relative_strength_5', 'relative_strength_20',
                'vol_regime_high', 'vol_regime_low',
                'price_position_20', 'volume_position_20',
                'quality_proxy', 'liquidity_proxy',
                'trend_strength', 'mean_reversion', 'breakout_signal'
            ]
            
            # Only keep columns that exist
            available_cols = [col for col in feature_cols if col in df.columns]
            df_features = df[available_cols].copy()
            
            all_features.append(df_features)
            logger.info(f"Calculated {len(available_cols)} features for {symbol}")
        
        # Combine all features
        if not all_features:
            raise ValueError("No features calculated")
        
        panel_df = pd.concat(all_features, ignore_index=True)
        
        # Sort by symbol and date (if available)
        if 'date' in panel_df.columns:
            panel_df = panel_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        else:
            panel_df = panel_df.sort_values('symbol').reset_index(drop=True)
        
        logger.info(f"Panel dataset created: {len(panel_df)} rows, {len(panel_df.columns)} columns")
        
        return panel_df
    
    def add_market_features(self, panel_df: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add market and cross-asset features."""
        if 'market' not in data:
            logger.warning("No market data available")
            return panel_df
        
        market_df = data['market'].copy()
        
        # Market returns
        market_df['market_ret1'] = market_df['market_close'].pct_change(1)
        market_df['market_ret5'] = market_df['market_close'].pct_change(5)
        market_df['market_vol'] = market_df['market_ret1'].rolling(20).std()
        
        # Add cross-asset features
        for proxy in self.cross_proxies:
            if proxy in data:
                proxy_df = data[proxy].copy()
                proxy_df[f'{proxy.lower()}_ret1'] = proxy_df[f'{proxy.lower()}_close'].pct_change(1)
                proxy_df[f'{proxy.lower()}_ret5'] = proxy_df[f'{proxy.lower()}_close'].pct_change(5)
                
                # Merge with market data
                market_df = market_df.merge(
                    proxy_df[[f'{proxy.lower()}_ret1', f'{proxy.lower()}_ret5']], 
                    left_index=True, right_index=True, how='left'
                )
        
        # Merge market features with panel data
        # For now, we'll add market features as constants (you'd want to properly align by date)
        for col in market_df.columns:
            if col.startswith('market_') or any(proxy.lower() in col for proxy in self.cross_proxies):
                panel_df[col] = market_df[col].mean()  # Simplified - use mean for now
        
        return panel_df
    
    def create_forward_returns(self, panel_df: pd.DataFrame, horizons: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Create forward returns for different horizons."""
        logger.info(f"Creating forward returns for horizons: {horizons}")
        
        df = panel_df.copy()
        
        # CRITICAL FIX: Sort by symbol and date to ensure proper time series order
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        for horizon in horizons:
            # Forward returns
            df[f'ret_fwd_{horizon}'] = df.groupby('symbol')['close'].pct_change(horizon).shift(-horizon)
            
            # Excess returns (vs market)
            if 'market_ret1' in df.columns:
                market_forward = df.groupby('symbol')['market_ret1'].shift(-horizon)
                df[f'excess_ret_fwd_{horizon}'] = df[f'ret_fwd_{horizon}'] - market_forward
        
        return df
    
    def build_panel(self, 
                   start_date: str = '2020-01-01',
                   end_date: str = '2024-01-01',
                   horizons: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Build complete panel dataset."""
        logger.info("Building complete panel dataset")
        
        # Download data
        data = self.download_data(start_date, end_date)
        
        # Calculate features
        panel_df = self.calculate_features(data)
        
        # Add market features
        panel_df = self.add_market_features(panel_df, data)
        
        # 🔧 CRITICAL FIX: Apply CS transformations BEFORE creating forward returns
        # This prevents forward-looking data from being cross-sectionally transformed
        print("🔧 PIPELINE ORDER FIX: CS transforms BEFORE forward returns to prevent leakage")
        
        # 🔧 PRESERVE ESSENTIAL COLUMNS for forward returns calculation
        essential_cols = ['close', 'market_ret1', 'date', 'symbol']
        preserved_data = panel_df[essential_cols].copy()
        
        # Transform raw features to cross-sectional relative features (SAFE - no forward data yet)
        panel_df = self.create_cross_sectional_features(panel_df)
        
        # 🔧 RESTORE ESSENTIAL COLUMNS for forward returns calculation
        for col in essential_cols:
            if col in preserved_data.columns and col not in panel_df.columns:
                panel_df[col] = preserved_data[col]
        
        # Create forward returns AFTER CS transformations (so they can't be CS-transformed)
        panel_df = self.create_forward_returns(panel_df, horizons)
        
        # Clean data
        panel_df = self.clean_data(panel_df)
        
        logger.info(f"Final panel dataset: {len(panel_df)} rows, {len(panel_df.columns)} columns")
        
        return panel_df
    
    def create_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using robust cross-sectional transforms with fallbacks."""
        logger.info("Creating robust cross-sectional features")
        
        import numpy as np
        RNG = np.random.default_rng(7)
        
        def cs_rank(s: pd.Series):
            """Stable rank 0..1; if constant, return 0.5"""
            if s.nunique(dropna=False) <= 1:
                return pd.Series(0.5, index=s.index)
            return s.rank(method='average', pct=True)

        def cs_zscore(s: pd.Series):
            """Z-score with fallback for zero variance"""
            mu = s.mean()
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(0.0, index=s.index)  # fallback: zero-mean constant
            return (s - mu) / (sd + 1e-12)
        
        # Get all feature columns (excluding meta/target columns)
        excluded_cols = ['date', 'symbol', 'ret_fwd_3', 'ret_fwd_5', 'ret_fwd_10', 
                        'excess_ret_fwd_3', 'excess_ret_fwd_5', 'excess_ret_fwd_10']
        feat_cols = [c for c in df.columns if c not in excluded_cols]
        
        # Add a basic sector column for residualization (simplified)
        df['sector'] = df['symbol'].str[0]  # Simple first-letter grouping
        
        df = df.sort_values(['date', 'symbol']).copy()
        
        # 1) Identify time-only features (no cross-sectional variation on any date)
        time_only = []
        for c in feat_cols:
            if c in df.columns:
                max_nunique = df.groupby('date')[c].nunique(dropna=False).max()
                if max_nunique <= 1:
                    time_only.append(c)
        
        keep = [c for c in feat_cols if c not in time_only and c in df.columns]
        logger.info(f"Dropping {len(time_only)} time-only features: {time_only[:5]}...")
        
        # 🔍 DEBUG: Check raw feature dispersion BEFORE transformation
        if keep:
            raw_sample_date = df['date'].unique()[0]
            sample_data = df[df['date'] == raw_sample_date]
            sample_stds = sample_data[keep].std()
            n_zero_std = (sample_stds == 0).sum()
            print(f"🔍 RAW feature check (date {raw_sample_date}): {n_zero_std}/{len(keep)} features have zero std")
            print(f"🔍 Sample size: {len(sample_data)} symbols on this date")
            
            if n_zero_std > len(keep) * 0.5:
                print(f"🚨 WARNING: {n_zero_std}/{len(keep)} raw features already have zero variance!")
                print(f"Sample zero-std features: {sample_stds[sample_stds == 0].index[:5].tolist()}")
                
                # Detailed debugging - check actual values
                zero_std_features = sample_stds[sample_stds == 0].index
                if len(zero_std_features) > 0:
                    first_feature = zero_std_features[0]
                    feature_values = sample_data[first_feature]
                    print(f"🔍 Detailed debug for {first_feature}:")
                    print(f"   Values: {feature_values.head().tolist()} (showing first 5)")
                    print(f"   All identical? {feature_values.nunique() <= 1}")
                    print(f"   Has NaN? {feature_values.isna().any()}")
                    print(f"   All NaN? {feature_values.isna().all()}")
                    
            print(f"Sample feature variance: {sample_stds.describe()}")
        
        # 2) Classify features for appropriate transforms
        rank_like = [c for c in keep if any(k in c.lower() for k in
                     ['vol_', 'volume', 'liquidity', 'turnover', 'beta', 'sharpe', 'momentum', 'skew', 'kurt', 'close'])]
        ratio_like = [c for c in keep if any(k in c.lower() for k in
                     ['ratio', '_pct', 'percentile', 'position', '_z', 'cs', 'rel', 'bb_', 'ma_', 'rsi', 'stoch']) and c not in rank_like]
        level_like = [c for c in keep if c not in set(rank_like) | set(ratio_like)]
        
        # 3) Per-date transformation
        def transform_per_date(g):
            out = {}
            debug_sample_date = df['date'].unique()[0]
            
            # a) RANK for scale/level-driven features (robust)
            for c in rank_like + level_like:
                if c in g.columns:
                    raw_values = g[c]
                    ranked_values = cs_rank(raw_values)
                    out[c + '_csr'] = ranked_values
                    
                    # Debug only for first date and first few features
                    if g['date'].iloc[0] == debug_sample_date and c in (rank_like + level_like)[:2]:
                        print(f"🔍 RANK DEBUG {c}: raw_nunique={raw_values.nunique()}, raw_std={raw_values.std():.6f}")
                        print(f"   Raw sample: {raw_values.iloc[:3].values}")
                        print(f"   Ranked sample: {ranked_values.iloc[:3].values}")
            
            # b) Z-SCORE for already bounded/ratio features (keeps sign info)  
            for c in ratio_like:
                if c in g.columns:
                    raw_values = g[c]
                    z = cs_zscore(raw_values)
                    # fallback: if z all zeros, mix in small noise to break ties
                    if (z.abs().sum() == 0):
                        z = pd.Series(RNG.normal(0, 1e-6, size=len(g)), index=g.index)
                    out[c + '_csz'] = z
                    
                    # Debug only for first date and first few features
                    if g['date'].iloc[0] == debug_sample_date and c in ratio_like[:2]:
                        print(f"🔍 ZSCORE DEBUG {c}: raw_nunique={raw_values.nunique()}, raw_std={raw_values.std():.6f}")
                        print(f"   Z-score sample: {z.iloc[:3].values}")
            
            # Convert to DataFrame first so we can reference the new columns
            temp_df = pd.DataFrame(out, index=g.index)
            
            # c) Sector residualization for core rank features
            if 'sector' in g.columns and len(temp_df.columns) > 0:
                # Only residualize rank features, limit to first 16 for performance
                rank_cols = [c for c in temp_df.columns if c.endswith('_csr')][:16]
                for c in rank_cols:
                    if c in temp_df.columns:
                        # Merge sector info for residualization
                        temp_with_sector = temp_df.join(g[['sector']])
                        if len(temp_with_sector.groupby('sector')) > 1:  # Only if multiple sectors
                            mu_sec = temp_with_sector.groupby('sector')[c].transform('mean')
                            temp_df[c + '_res'] = temp_df[c] - mu_sec
            
            return temp_df
        
        # 🔧 CRITICAL FIX: Use transform() instead of apply() to avoid index alignment issues
        print("🔧 Applying cross-sectional transforms using groupby.transform for proper alignment...")
        
        # Get feature columns only (exclude metadata)
        feats_to_transform = [c for c in rank_like + ratio_like + level_like if c in df.columns]
        print(f"🔧 Processing {len(feats_to_transform)} features...")
        
        # Apply cross-sectional ranking for rank_like + level_like features
        rank_features = [c for c in rank_like + level_like if c in df.columns]
        if rank_features:
            print(f"🔧 Applying CS ranks to {len(rank_features)} features...")
            for c in rank_features:
                df[c + '_csr'] = df.groupby('date', group_keys=False)[c].transform(cs_rank)
        
        # Apply cross-sectional z-scoring for ratio_like features  
        zscore_features = [c for c in ratio_like if c in df.columns]
        if zscore_features:
            print(f"🔧 Applying CS z-scores to {len(zscore_features)} features...")
            for c in zscore_features:
                df[c + '_csz'] = df.groupby('date', group_keys=False)[c].transform(cs_zscore)
        
        # 🔧 RE-ENABLED: Safe residualization with proper index alignment
        print("🔧 Applying safe residualization with vectorized operations...")
        
        from ml.safe_residualization import build_residualized_features
        
        # Select features for residualization (limit for performance)
        rank_cols_for_resid = [c + '_csr' for c in rank_features if c + '_csr' in df.columns][:16]
        
        if rank_cols_for_resid and len(rank_cols_for_resid) > 0:
            # Build residualized features safely
            market_col = 'market_ret' if 'market_ret' in df.columns else None
            sector_col = 'sector' if 'sector' in df.columns else None
            
            residual_config = {
                'market_residualization': market_col is not None,
                'sector_residualization': sector_col is not None,
                'normalization_method': 'zscore',  # Will be applied exactly once
                'max_residual_features': 16
            }
            
            df, new_residual_cols = build_residualized_features(
                df, rank_cols_for_resid, market_col, sector_col, residual_config
            )
            
            print(f"✅ Safe residualization complete: {len(new_residual_cols)} new features")
        else:
            print("🔧 No features available for residualization")
        
        # Debug check on first date
        debug_sample_date = df['date'].unique()[0] 
        sample_data = df[df['date'] == debug_sample_date]
        
        # Check a few transformed features
        debug_features = [c for c in df.columns if c.endswith(('_csr', '_csz', '_res'))][:3]
        for feat in debug_features:
            if feat in sample_data.columns:
                feat_std = sample_data[feat].std()
                feat_nunique = sample_data[feat].nunique()
                print(f"🔍 Transform check {feat}: std={feat_std:.6f}, nunique={feat_nunique}")
                if feat_std == 0:
                    print(f"⚠️ WARNING: {feat} has zero std - transformation failed!")
        
        # 4) Final feature selection: prefer residuals -> ranks -> z-scores
        new_feats = ([c for c in df.columns if c.endswith('_res')] +
                    [c for c in df.columns if c.endswith('_csr')] +
                    [c for c in df.columns if c.endswith('_csz')])
        
        # 5) Drop features that are flat on >20% of dates
        if new_feats:
            wstd = df.groupby('date')[new_feats].std()
            flat_frac = (wstd.fillna(0) == 0).sum() / wstd.shape[0]
            keep_final = [c for c in new_feats if flat_frac[c] < 0.2]
            drop_count = len(new_feats) - len(keep_final)
            if drop_count > 0:
                logger.info(f"Dropping {drop_count} features flat on >20% dates")
        else:
            keep_final = []
        
        # Create final dataframe with metadata + forward returns + transformed features
        metadata_cols = ['date', 'symbol']
        forward_ret_cols = [c for c in df.columns if c.startswith(('ret_fwd_', 'excess_ret_fwd_'))]
        
        # Preserve any existing target columns (cs_target will be created later)
        final_cols = metadata_cols + forward_ret_cols + keep_final
        df_final = df[final_cols].copy()
        
        logger.info(f"Cross-sectional transformation complete: {len(keep_final)} features retained")
        logger.info(f"Feature breakdown: {len([c for c in keep_final if c.endswith('_res')])} residuals, "
                   f"{len([c for c in keep_final if c.endswith('_csr')])} ranks, "
                   f"{len([c for c in keep_final if c.endswith('_csz')])} z-scores")
        
        # 🔍 FEATURE DISPERSION VERIFICATION: Ensure features vary within dates
        logger.info("🔍 Verifying per-date feature dispersion...")
        feature_cols = [c for c in keep_final if c.startswith('f_')]
        if feature_cols and 'date' in df_final.columns:
            per_date_stds = df_final.groupby('date')[feature_cols].std()
            flat_features_by_date = (per_date_stds <= 1e-6).sum(axis=1)
            avg_flat_features = flat_features_by_date.mean()
            pct_dates_with_flat_features = (flat_features_by_date > 0).mean()
            
            logger.info(f"📊 Feature dispersion check: avg {avg_flat_features:.1f} flat features per date (out of {len(feature_cols)})")
            logger.info(f"📊 Feature dispersion check: {pct_dates_with_flat_features:.1%} dates have some flat features")
            
            # Find worst offending features
            total_flat_days = (per_date_stds <= 1e-6).sum(axis=0)
            worst_features = total_flat_days.nlargest(5)
            if worst_features.max() > 0:
                logger.warning(f"⚠️ Features with most flat days: {dict(worst_features)}")
            
            # Critical assertion
            if avg_flat_features > len(feature_cols) * 0.5:
                logger.error(f"❌ CRITICAL: Too many flat features per date: {avg_flat_features:.1f}/{len(feature_cols)}")
                raise ValueError(f"Cross-sectional transformation failed: {avg_flat_features:.1f}/{len(feature_cols)} features flat per date")
            else:
                logger.info("✅ Feature dispersion verification passed")
                
        # 🛡️ POST-FIX GUARDRAILS: Fail fast if transformation produced constants
        if feature_cols:
            g = df_final.groupby('date', sort=False)
            flat_counts = (g[feature_cols].nunique() <= 1).sum()  # per-feature count of flat dates
            completely_flat = flat_counts[flat_counts == g.ngroups]  # features flat on ALL dates
            
            if len(completely_flat) > 0:
                logger.error(f"❌ FATAL: {len(completely_flat)} features are flat across ALL dates!")
                logger.error(f"❌ Completely flat features: {list(completely_flat.index)[:5]}...")
                raise ValueError(f"Cross-sectional transformation failed: {len(completely_flat)} features are constants")
            
            # Quick snapshot on first date for manual inspection
            if g.ngroups > 0:
                first_date = df_final['date'].iloc[0]
                snap = df_final.loc[df_final['date'] == first_date, feature_cols]
                snap_stats = snap.std(ddof=0).describe()
                logger.info(f"📊 First date feature std distribution: min={snap_stats['min']:.6f}, mean={snap_stats['mean']:.6f}")
                if snap_stats['min'] == 0:
                    zero_std_features = snap.std(ddof=0) == 0
                    logger.warning(f"⚠️ {zero_std_features.sum()} features have zero std on first date: {zero_std_features[zero_std_features].index[:3].tolist()}")
        
        return df_final
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the panel dataset."""
        logger.info("Cleaning panel dataset")
        
        initial_rows = len(df)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with too many NaN values (more than 50% missing)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        # Fill remaining NaN values with forward fill, then backward fill
        df = df.ffill().bfill()
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        final_rows = len(df)
        logger.info(f"Data cleaning: {initial_rows} -> {final_rows} rows")
        
        return df


def build_panel_from_universe(universe_config_path: str, 
                             output_path: str,
                             start_date: str = '2020-01-01',
                             end_date: str = '2024-01-01') -> pd.DataFrame:
    """Build panel dataset from universe configuration."""
    import yaml
    
    # Load universe configuration
    with open(universe_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    universe = config['universe']
    market_proxy = config.get('market_proxy', 'SPY')
    cross_proxies = config.get('cross_proxies', ['QQQ', 'TLT', 'UUP', 'VIXY'])
    
    # Build panel
    builder = PanelBuilder(universe, market_proxy, cross_proxies)
    panel_df = builder.build_panel(start_date, end_date)
    
    # Save panel dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False)
    
    logger.info(f"Panel dataset saved to {output_path}")
    
    return panel_df
