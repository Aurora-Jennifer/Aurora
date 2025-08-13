"""
Feature Engineering Module
Comprehensive feature generation for trading strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    # Trend/Momentum features
    sma_periods: List[Tuple[int, int]] = None  # [(fast, slow), ...]
    ema_periods: List[Tuple[int, int]] = None
    macd_params: Tuple[int, int, int] = (12, 26, 9)  # (fast, slow, signal)
    rsi_period: int = 14
    roc_period: int = 10
    
    # Volatility features
    atr_period: int = 14
    vol_period: int = 20
    adx_period: int = 14
    
    # Mean reversion features
    bb_period: int = 20
    bb_std: float = 2.0
    zscore_period: int = 20
    
    # Volume features
    volume_period: int = 20
    mfi_period: int = 14
    
    # Regime filters
    choppiness_period: int = 14
    regime_threshold: float = 0.5


class FeatureEngine:
    """
    Comprehensive feature engineering for trading strategies.
    
    Supports:
    - Trend/Momentum features
    - Volatility & regime filters
    - Mean reversion features
    - Volume/flow features
    - Market context features
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize FeatureEngine.
        
        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.features: Dict[str, pd.Series] = {}
        
        # Set default SMA periods if not provided
        if self.config.sma_periods is None:
            self.config.sma_periods = [(5, 50), (10, 200), (20, 180)]
        
        if self.config.ema_periods is None:
            self.config.ema_periods = [(5, 50), (10, 200), (20, 180)]
        
        logger.info("Initialized FeatureEngine")
    
    def generate_all_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate all features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dict of feature names to feature series
        """
        logger.info("Generating all features...")
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten column names
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
            logger.info(f"Flattened MultiIndex columns: {data.columns.tolist()}")
        
        # Clear existing features
        self.features = {}
        
        # Generate trend/momentum features
        self._generate_trend_features(data)
        
        # Generate volatility features
        self._generate_volatility_features(data)
        
        # Generate mean reversion features
        self._generate_mean_reversion_features(data)
        
        # Generate volume features
        self._generate_volume_features(data)
        
        # Generate regime filters
        self._generate_regime_features(data)
        
        logger.info(f"Generated {len(self.features)} features")
        return self.features.copy()
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature generation."""
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten column names
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
            logger.info(f"Flattened MultiIndex columns: {data.columns.tolist()}")
        
        return data
    
    def _generate_trend_features(self, data: pd.DataFrame) -> None:
        """Generate trend and momentum features."""
        data = self._prepare_data(data)
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # SMA Crossovers
        for fast, slow in self.config.sma_periods:
            sma_fast = close.rolling(fast).mean()
            sma_slow = close.rolling(slow).mean()
            
            # Crossover signal
            signal = ((sma_fast > sma_slow) * 2 - 1).fillna(0)
            self.features[f'SMA_Crossover_{fast}_{slow}'] = signal
            
            # Slope of SMA
            slope = sma_fast.diff(5) / sma_fast.shift(5)
            self.features[f'SMA_Slope_{fast}'] = slope.fillna(0)
        
        # EMA Crossovers
        for fast, slow in self.config.ema_periods:
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            
            signal = ((ema_fast > ema_slow) * 2 - 1).fillna(0)
            self.features[f'EMA_Crossover_{fast}_{slow}'] = signal
            
            # Slope of EMA
            slope = ema_fast.diff(5) / ema_fast.shift(5)
            self.features[f'EMA_Slope_{fast}'] = slope.fillna(0)
        
        # Donchian Breakout
        for period in [20, 50, 100]:
            high_n = high.rolling(period).max()
            low_n = low.rolling(period).min()
            
            breakout_signal = ((close > high_n.shift(1)) * 1 + 
                              (close < low_n.shift(1)) * -1).fillna(0)
            self.features[f'Donchian_Breakout_{period}'] = breakout_signal
        
        # MACD
        fast, slow, signal_period = self.config.macd_params
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        self.features['MACD_Line'] = macd_line
        self.features['MACD_Signal'] = signal_line
        self.features['MACD_Histogram'] = histogram
        self.features['MACD_Crossover'] = ((macd_line > signal_line) * 2 - 1).fillna(0)
        
        # RSI Momentum
        rsi = self._calculate_rsi(close, self.config.rsi_period)
        self.features['RSI'] = rsi
        self.features['RSI_Momentum'] = ((rsi > 50) * 2 - 1).fillna(0)
        self.features['RSI_Overbought'] = (rsi > 70).astype(int)
        self.features['RSI_Oversold'] = (rsi < 30).astype(int)
        
        # Rate of Change
        roc = (close / close.shift(self.config.roc_period) - 1) * 100
        self.features[f'ROC_{self.config.roc_period}'] = roc.fillna(0)
    
    def _generate_volatility_features(self, data: pd.DataFrame) -> None:
        """Generate volatility and regime filter features."""
        data = self._prepare_data(data)
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # ATR and ATR%
        atr = self._calculate_atr(data, self.config.atr_period)
        atr_pct = atr / close
        self.features['ATR'] = atr
        self.features['ATR_Pct'] = atr_pct
        
        # Realized Volatility
        from core.utils import calculate_returns
        returns = calculate_returns(close, shift=0)
        for period in [10, 20, 50]:
            vol = returns.rolling(period).std() * np.sqrt(252)
            self.features[f'Realized_Vol_{period}'] = vol.fillna(0)
        
        # ADX (simplified)
        adx = self._calculate_adx(data, self.config.adx_period)
        self.features['ADX'] = adx
        self.features['ADX_Trend'] = (adx > 25).astype(int)
        
        # Volatility Regime
        vol_20 = returns.rolling(20).std()
        vol_high = vol_20 > vol_20.rolling(100).quantile(0.8)
        vol_low = vol_20 < vol_20.rolling(100).quantile(0.2)
        
        self.features['Vol_High'] = vol_high.astype(int)
        self.features['Vol_Low'] = vol_low.astype(int)
    
    def _generate_mean_reversion_features(self, data: pd.DataFrame) -> None:
        """Generate mean reversion features."""
        data = self._prepare_data(data)
        close = data['Close']
        
        # RSI Extremes
        rsi = self._calculate_rsi(close, 14)
        rsi_2 = self._calculate_rsi(close, 2)
        rsi_3 = self._calculate_rsi(close, 3)
        
        self.features['RSI_2_Extreme'] = ((rsi_2 < 10) * -1 + (rsi_2 > 90) * 1).fillna(0)
        self.features['RSI_3_Extreme'] = ((rsi_3 < 10) * -1 + (rsi_3 > 90) * 1).fillna(0)
        
        # Z-score of price vs MA
        for period in [20, 50, 100]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            zscore = (close - ma) / (std + 1e-6)
            self.features[f'ZScore_{period}'] = zscore.fillna(0)
        
        # Bollinger Bands
        bb_period = self.config.bb_period
        bb_std = self.config.bb_std
        
        bb_ma = close.rolling(bb_period).mean()
        bb_std_series = close.rolling(bb_period).std()
        bb_upper = bb_ma + bb_std * bb_std_series
        bb_lower = bb_ma - bb_std * bb_std_series
        
        # BB touches and penetrations
        bb_touch_upper = (close >= bb_upper * 0.99).astype(int)
        bb_touch_lower = (close <= bb_lower * 1.01).astype(int)
        bb_penetrate_upper = (close > bb_upper).astype(int)
        bb_penetrate_lower = (close < bb_lower).astype(int)
        
        self.features['BB_Touch_Upper'] = bb_touch_upper
        self.features['BB_Touch_Lower'] = bb_touch_lower
        self.features['BB_Penetrate_Upper'] = bb_penetrate_upper
        self.features['BB_Penetrate_Lower'] = bb_penetrate_lower
        
        # BB position
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-6)
        self.features['BB_Position'] = bb_position.fillna(0.5)
        
        # VWAP deviation (simplified)
        if 'Volume' in data.columns:
            vwap = (data['Close'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
            vwap_dev = (close - vwap) / vwap
            self.features['VWAP_Deviation'] = vwap_dev.fillna(0)
    
    def _generate_volume_features(self, data: pd.DataFrame) -> None:
        """Generate volume and flow features."""
        data = self._prepare_data(data)
        if 'Volume' not in data.columns:
            logger.warning("Volume data not available, skipping volume features")
            return
        
        close = data['Close']
        volume = data['Volume']
        
        # On-Balance Volume slope
        obv = self._calculate_obv(close, volume)
        obv_slope = obv.diff(5) / obv.shift(5)
        self.features['OBV_Slope'] = obv_slope.fillna(0)
        
        # Volume Z-score
        vol_ma = volume.rolling(self.config.volume_period).mean()
        vol_std = volume.rolling(self.config.volume_period).std()
        vol_zscore = (volume - vol_ma) / (vol_std + 1e-6)
        self.features['Volume_ZScore'] = vol_zscore.fillna(0)
        
        # Accumulation/Distribution line slope
        ad_line = self._calculate_ad_line(data)
        ad_slope = ad_line.diff(5) / ad_line.shift(5)
        self.features['AD_Slope'] = ad_slope.fillna(0)
        
        # Money Flow Index
        mfi = self._calculate_mfi(data, self.config.mfi_period)
        self.features['MFI'] = mfi
        self.features['MFI_Overbought'] = (mfi > 80).astype(int)
        self.features['MFI_Oversold'] = (mfi < 20).astype(int)
    
    def _generate_regime_features(self, data: pd.DataFrame) -> None:
        """Generate regime detection features."""
        data = self._prepare_data(data)
        close = data['Close']
        
        # Choppiness Index
        choppiness = self._calculate_choppiness(data, self.config.choppiness_period)
        self.features['Choppiness'] = choppiness
        self.features['Choppy_Market'] = (choppiness > 61.8).astype(int)
        
        # Trend vs Mean Reversion Regime
        from core.utils import calculate_returns
        returns = calculate_returns(close, shift=0)
        vol = returns.rolling(20).std()
        trend_strength = abs(returns.rolling(20).mean()) / vol
        
        self.features['Trend_Strength'] = trend_strength.fillna(0)
        self.features['Trend_Regime'] = (trend_strength > self.config.regime_threshold).astype(int)
        self.features['Mean_Reversion_Regime'] = (trend_strength < 0.1).astype(int)
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close_prev = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX (simplified version)."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate smoothed values
        tr = self._calculate_atr(data, period)
        plus_di = 100 * plus_dm.rolling(period).mean() / tr
        minus_di = 100 * minus_dm.rolling(period).mean() / tr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        close_diff = close.diff()
        obv = pd.Series(0, index=close.index)
        
        for i in range(1, len(close)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution line."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        mfm = ((close - low) - (high - close)) / (high - low + 1e-6)
        mfm = mfm.clip(-1, 1)
        mfv = mfm * volume
        
        ad_line = mfv.cumsum()
        return ad_line
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-6)))
        return mfi
    
    def _calculate_choppiness(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Choppiness Index."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr = self._calculate_atr(data, 1) * period
        range_sum = (high - low).rolling(period).sum()
        
        choppiness = 100 * np.log10(range_sum / tr) / np.log10(period)
        return choppiness
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics for all features."""
        if not self.features:
            return pd.DataFrame()
        
        summary_data = []
        for name, feature in self.features.items():
            summary_data.append({
                'Feature': name,
                'Mean': feature.mean(),
                'Std': feature.std(),
                'Min': feature.min(),
                'Max': feature.max(),
                'Non_Zero_Pct': (feature != 0).mean(),
                'Missing_Pct': feature.isna().mean()
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_features(self, save_path: Optional[str] = None) -> None:
        """Plot all features."""
        try:
            import matplotlib.pyplot as plt
            
            n_features = len(self.features)
            if n_features == 0:
                logger.warning("No features to plot")
                return
            
            # Calculate grid dimensions
            cols = 3
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (name, feature) in enumerate(self.features.items()):
                row = i // cols
                col = i % cols
                
                feature.plot(ax=axes[row, col], alpha=0.7)
                axes[row, col].set_title(name)
                axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_features, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping feature plots")


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Fetch sample data
    data = yf.download("SPY", start="2020-01-01", end="2025-01-01")
    
    # Create feature engine
    config = FeatureConfig()
    engine = FeatureEngine(config)
    
    # Generate all features
    features = engine.generate_all_features(data)
    
    # Print feature summary
    print("Feature Summary:")
    print(engine.get_feature_summary())
    
    # Print feature names
    print(f"\nGenerated {len(features)} features:")
    for name in features.keys():
        print(f"  - {name}")
