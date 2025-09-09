"""
Comprehensive Market Analysis System

Builds advanced indicators, market regimes, and context for reward-based training.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: talib not available, using simplified indicators")


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float  # 0-1
    trend_strength: float  # 0-1
    volatility_regime: str  # 'low', 'normal', 'high'
    sector_rotation: dict[str, float]  # sector performance


@dataclass
class MarketContext:
    """Comprehensive market context for trading decisions"""
    regime: MarketRegime
    technical_signals: dict[str, float]
    sentiment_indicators: dict[str, float]
    risk_metrics: dict[str, float]
    time_of_day: str
    day_of_week: str
    market_session: str


class ComprehensiveMarketAnalyzer:
    """
    Builds comprehensive market analysis for reward-based training
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.lookback_periods = config.get('lookback_periods', {
            'short': 5, 'medium': 20, 'long': 50, 'very_long': 200
        })
        
    def build_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set including:
        - Multi-timeframe technical indicators
        - Market regime analysis
        - Sentiment indicators
        - Risk metrics
        - Time-based features
        """
        features = pd.DataFrame(index=data.index)
        
        # 1. Multi-timeframe technical indicators
        features = self._build_technical_indicators(data, features)
        
        # 2. Market regime features
        features = self._build_regime_features(data, features)
        
        # 3. Sentiment and momentum features
        features = self._build_sentiment_features(data, features)
        
        # 4. Risk and volatility features
        features = self._build_risk_features(data, features)
        
        # 5. Time-based features
        features = self._build_time_features(data, features)
        
        # 6. Cross-asset features (if available)
        features = self._build_cross_asset_features(data, features)
        
        # Fill NaN values with forward fill, then backward fill, then fill remaining with 0
        features = features.ffill().bfill().fillna(0)
        
        # Only return rows that have at least some non-zero values
        non_zero_rows = (features != 0).any(axis=1)
        return features[non_zero_rows]
    
    def _build_technical_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build multi-timeframe technical indicators"""
        
        # Convert data to float64 for TA-Lib compatibility
        if TALIB_AVAILABLE:
            close_values = data['Close'].values.astype(np.float64)
            volume_values = data['Volume'].values.astype(np.float64)
            high_values = data['High'].values.astype(np.float64)
            low_values = data['Low'].values.astype(np.float64)
            open_values = data['Open'].values.astype(np.float64)
        
        # Price-based indicators
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'std_{period}'] = data['Close'].rolling(period).std()
            features[f'bb_upper_{period}'] = features[f'sma_{period}'] + (2 * features[f'std_{period}'])
            features[f'bb_lower_{period}'] = features[f'sma_{period}'] - (2 * features[f'std_{period}'])
            features[f'bb_position_{period}'] = (data['Close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Momentum indicators
        for period in [3, 5, 10, 20, 50]:
            features[f'momentum_{period}'] = data['Close'].pct_change(period)
            features[f'roc_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            
            if TALIB_AVAILABLE:
                features[f'rsi_{period}'] = talib.RSI(close_values, timeperiod=period)
                features[f'stoch_k_{period}'] = talib.STOCH(high_values, low_values, close_values, fastk_period=period)[0]
                features[f'stoch_d_{period}'] = talib.STOCH(high_values, low_values, close_values, fastk_period=period)[1]
            else:
                # Simplified RSI calculation
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                # Simplified stochastic
                low_min = data['Low'].rolling(window=period).min()
                high_max = data['High'].rolling(window=period).max()
                features[f'stoch_k_{period}'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
                features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Volume indicators
        features['volume_sma_20'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
        
        if TALIB_AVAILABLE:
            features['obv'] = talib.OBV(close_values, volume_values)
            features['ad'] = talib.AD(high_values, low_values, close_values, volume_values)
        else:
            # Simplified OBV
            features['obv'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
            # Simplified AD
            features['ad'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
        
        # Volatility indicators
        if TALIB_AVAILABLE:
            features['atr_14'] = talib.ATR(high_values, low_values, close_values, timeperiod=14)
            features['natr_14'] = talib.NATR(high_values, low_values, close_values, timeperiod=14)
            features['trange'] = talib.TRANGE(high_values, low_values, close_values)
        else:
            # Simplified ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features['atr_14'] = true_range.rolling(window=14).mean()
            features['natr_14'] = features['atr_14'] / data['Close'] * 100
            features['trange'] = true_range
        
        # Trend indicators
        if TALIB_AVAILABLE:
            features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(close_values)
            features['adx'] = talib.ADX(high_values, low_values, close_values, timeperiod=14)
            features['di_plus'] = talib.PLUS_DI(high_values, low_values, close_values, timeperiod=14)
            features['di_minus'] = talib.MINUS_DI(high_values, low_values, close_values, timeperiod=14)
        else:
            # Simplified MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # Simplified ADX (placeholder)
            features['adx'] = 50.0  # Neutral value
            features['di_plus'] = 50.0
            features['di_minus'] = 50.0
        
        # Pattern recognition (simplified)
        if TALIB_AVAILABLE:
            features['doji'] = talib.CDLDOJI(open_values, high_values, low_values, close_values)
            features['hammer'] = talib.CDLHAMMER(open_values, high_values, low_values, close_values)
            features['engulfing'] = talib.CDLENGULFING(open_values, high_values, low_values, close_values)
        else:
            # Simplified pattern recognition
            body_size = np.abs(data['Close'] - data['Open'])
            total_range = data['High'] - data['Low']
            features['doji'] = (body_size / total_range < 0.1).astype(int)
            features['hammer'] = 0  # Placeholder
            features['engulfing'] = 0  # Placeholder
        
        return features
    
    def _build_regime_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build market regime classification features"""
        
        # Trend strength
        short_trend = data['Close'].rolling(20).mean()
        long_trend = data['Close'].rolling(50).mean()
        features['trend_strength'] = (short_trend - long_trend) / long_trend
        features['trend_direction'] = np.where(features['trend_strength'] > 0, 1, -1)
        
        # Volatility regime (convert to numeric)
        volatility = data['Close'].rolling(20).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        features['volatility_regime'] = np.where(vol_percentile > 0.8, 2,  # high
                                               np.where(vol_percentile < 0.2, 0, 1))  # low, normal
        
        # Market regime classification
        features['regime_bull'] = ((features['trend_strength'] > 0.02) & (vol_percentile < 0.6)).astype(int)
        features['regime_bear'] = ((features['trend_strength'] < -0.02) & (vol_percentile < 0.6)).astype(int)
        features['regime_sideways'] = ((abs(features['trend_strength']) < 0.01) & (vol_percentile < 0.6)).astype(int)
        features['regime_volatile'] = (vol_percentile > 0.8).astype(int)
        
        return features
    
    def _build_sentiment_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build sentiment and momentum features"""
        
        # Price momentum
        features['momentum_1d'] = data['Close'].pct_change(1)
        features['momentum_3d'] = data['Close'].pct_change(3)
        features['momentum_5d'] = data['Close'].pct_change(5)
        features['momentum_10d'] = data['Close'].pct_change(10)
        features['momentum_20d'] = data['Close'].pct_change(20)
        
        # Momentum strength
        features['momentum_strength'] = features['momentum_1d'] * features['momentum_5d'] * features['momentum_20d']
        
        # Gap analysis
        features['gap_up'] = (data['Open'] > data['Close'].shift(1)).astype(int)
        features['gap_down'] = (data['Open'] < data['Close'].shift(1)).astype(int)
        features['gap_size'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Intraday patterns
        features['intraday_range'] = (data['High'] - data['Low']) / data['Open']
        features['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        return features
    
    def _build_risk_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build risk and volatility features"""
        
        # Value at Risk (VaR)
        returns = data['Close'].pct_change()
        features['var_95'] = returns.rolling(20).quantile(0.05)
        features['var_99'] = returns.rolling(20).quantile(0.01)
        
        # Expected Shortfall (CVaR)
        features['cvar_95'] = returns.rolling(20).apply(lambda x: x[x <= x.quantile(0.05)].mean())
        features['cvar_99'] = returns.rolling(20).apply(lambda x: x[x <= x.quantile(0.01)].mean())
        
        # Volatility clustering
        features['vol_cluster'] = returns.rolling(20).std().rolling(20).std()
        
        # Skewness and Kurtosis
        features['skewness'] = returns.rolling(20).skew()
        features['kurtosis'] = returns.rolling(20).kurt()
        
        # Drawdown analysis
        rolling_max = data['Close'].rolling(20).max()
        features['drawdown'] = (data['Close'] - rolling_max) / rolling_max
        features['max_drawdown'] = features['drawdown'].rolling(20).min()
        
        return features
    
    def _build_time_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build time-based features"""
        
        # Time of day (for intraday data)
        if 'Time' in data.columns:
            features['hour'] = pd.to_datetime(data['Time']).dt.hour
            features['minute'] = pd.to_datetime(data['Time']).dt.minute
            features['time_of_day'] = features['hour'] + features['minute'] / 60
        else:
            # For daily data, use day of week
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
        
        # Market session indicators
        features['is_monday'] = (data.index.dayofweek == 0).astype(int)
        features['is_friday'] = (data.index.dayofweek == 4).astype(int)
        features['is_month_end'] = (data.index.day >= 28).astype(int)
        features['is_quarter_end'] = (data.index.month % 3 == 0).astype(int)
        
        return features
    
    def _build_cross_asset_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Build cross-asset features (if multiple assets available)"""
        
        # This would be expanded with actual cross-asset data
        # For now, just add placeholders
        features['cross_asset_momentum'] = 0.0
        features['cross_asset_volatility'] = 0.0
        features['cross_asset_correlation'] = 0.0
        
        return features
    
    def analyze_market_context(self, data: pd.DataFrame, current_time: datetime) -> MarketContext:
        """
        Analyze current market context for trading decisions
        """
        
        # Get latest data
        latest_data = data.iloc[-1]
        
        # Classify market regime
        regime = self._classify_market_regime(data)
        
        # Build technical signals
        technical_signals = self._build_technical_signals(data)
        
        # Build sentiment indicators
        sentiment_indicators = self._build_sentiment_indicators(data)
        
        # Build risk metrics
        risk_metrics = self._build_risk_metrics(data)
        
        # Determine time context
        time_of_day = self._get_time_context(current_time)
        day_of_week = current_time.strftime('%A')
        market_session = self._get_market_session(current_time)
        
        return MarketContext(
            regime=regime,
            technical_signals=technical_signals,
            sentiment_indicators=sentiment_indicators,
            risk_metrics=risk_metrics,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            market_session=market_session
        )
    
    def _classify_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Classify current market regime"""
        
        # Simple regime classification based on trend and volatility
        short_trend = data['Close'].rolling(20).mean().iloc[-1]
        long_trend = data['Close'].rolling(50).mean().iloc[-1]
        trend_strength = (short_trend - long_trend) / long_trend
        
        volatility = data['Close'].rolling(20).std().iloc[-1]
        vol_percentile = data['Close'].rolling(100).std().rank(pct=True).iloc[-1]
        
        # Classify regime
        if trend_strength > 0.02 and vol_percentile < 0.6:
            regime = 'bull'
            confidence = min(abs(trend_strength) * 10, 1.0)
        elif trend_strength < -0.02 and vol_percentile < 0.6:
            regime = 'bear'
            confidence = min(abs(trend_strength) * 10, 1.0)
        elif vol_percentile > 0.8:
            regime = 'volatile'
            confidence = min(vol_percentile, 1.0)
        else:
            regime = 'sideways'
            confidence = 0.5
        
        # Volatility regime
        if vol_percentile > 0.8:
            vol_regime = 'high'
        elif vol_percentile < 0.2:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return MarketRegime(
            regime=regime,
            confidence=confidence,
            trend_strength=abs(trend_strength),
            volatility_regime=vol_regime,
            sector_rotation={}  # Would be populated with actual sector data
        )
    
    def _build_technical_signals(self, data: pd.DataFrame) -> dict[str, float]:
        """Build technical trading signals"""
        
        latest = data.iloc[-1]
        
        # Use actual column names from features
        rsi_20 = latest.get('rsi_20', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        bb_position_20 = latest.get('bb_position_20', 0.5)
        sma_20 = latest.get('sma_20', 0)
        sma_50 = latest.get('sma_50', 0)
        volume_ratio = latest.get('volume_ratio', 1)
        
        signals = {
            'rsi_signal': 1 if rsi_20 < 30 else (-1 if rsi_20 > 70 else 0),
            'macd_signal': 1 if macd > macd_signal else -1,
            'bb_signal': 1 if bb_position_20 < 0.2 else (-1 if bb_position_20 > 0.8 else 0),
            'trend_signal': 1 if sma_20 > sma_50 else -1,
            'volume_signal': 1 if volume_ratio > 1.2 else (-1 if volume_ratio < 0.8 else 0)
        }
        
        return signals
    
    def _build_sentiment_indicators(self, data: pd.DataFrame) -> dict[str, float]:
        """Build sentiment indicators"""
        
        latest = data.iloc[-1]
        
        # Use actual column names from features
        momentum_5d = latest.get('momentum_5d', 0)
        gap_size = latest.get('gap_size', 0)
        close_position = latest.get('close_position', 0.5)
        vol_cluster = latest.get('vol_cluster', 0)
        
        sentiment = {
            'momentum_sentiment': momentum_5d,
            'gap_sentiment': gap_size,
            'intraday_sentiment': close_position - 0.5,
            'volatility_sentiment': vol_cluster
        }
        
        return sentiment
    
    def _build_risk_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        """Build risk metrics"""
        
        latest = data.iloc[-1]
        
        # Use actual column names from features
        var_95 = latest.get('var_95', 0)
        cvar_95 = latest.get('cvar_95', 0)
        max_drawdown = latest.get('max_drawdown', 0)
        std_20 = latest.get('std_20', 0)
        skewness = latest.get('skewness', 0)
        kurtosis = latest.get('kurtosis', 0)
        
        risk = {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'volatility': std_20,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        return risk
    
    def _get_time_context(self, current_time: datetime) -> str:
        """Get time of day context"""
        
        hour = current_time.hour
        
        if 6 <= hour < 12:
            return 'morning'
        if 12 <= hour < 18:
            return 'afternoon'
        if 18 <= hour < 22:
            return 'evening'
        return 'night'
    
    def _get_market_session(self, current_time: datetime) -> str:
        """Get market session context"""
        
        hour = current_time.hour
        
        if 9 <= hour < 16:
            return 'regular'
        if 4 <= hour < 9 or 16 <= hour < 20:
            return 'extended'
        return 'closed'
