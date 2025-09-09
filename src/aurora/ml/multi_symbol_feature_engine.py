"""
Multi-Symbol Feature Engine

Generates features across multiple symbols for portfolio-level trading decisions.
Handles cross-asset features, correlation features, and portfolio-level indicators.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from ..data.multi_symbol_manager import MultiSymbolDataManager
from .market_analyzer import ComprehensiveMarketAnalyzer


@dataclass
class MultiSymbolFeatureConfig:
    """Configuration for multi-symbol feature generation"""
    symbols: list[str]
    individual_features: bool = True
    cross_asset_features: bool = True
    correlation_features: bool = True
    portfolio_features: bool = True
    market_regime_features: bool = True
    feature_lookback: int = 20
    correlation_window: int = 30


class MultiSymbolFeatureEngine:
    """
    Generates features for multi-symbol trading
    
    Features include:
    - Individual symbol features (technical indicators)
    - Cross-asset features (relative strength, momentum)
    - Correlation features (rolling correlations)
    - Portfolio features (diversification, concentration)
    - Market regime features (overall market state)
    """
    
    def __init__(self, config: MultiSymbolFeatureConfig):
        self.config = config
        self.market_analyzer = ComprehensiveMarketAnalyzer({})
        self.feature_cache: dict[str, pd.DataFrame] = {}
        
    def build_all_features(self, data_manager: MultiSymbolDataManager) -> pd.DataFrame:
        """Build comprehensive features for all symbols"""
        
        print("🔧 Building multi-symbol features...")
        
        # Get aligned data
        aligned_data = data_manager.get_aligned_data()
        if aligned_data is None:
            raise ValueError("No aligned data available")
        
        all_features = []
        
        # 1. Individual symbol features
        if self.config.individual_features:
            print("   Building individual symbol features...")
            individual_features = self._build_individual_features(data_manager)
            all_features.append(individual_features)
        
        # 2. Cross-asset features
        if self.config.cross_asset_features:
            print("   Building cross-asset features...")
            cross_asset_features = self._build_cross_asset_features(aligned_data)
            all_features.append(cross_asset_features)
        
        # 3. Correlation features
        if self.config.correlation_features:
            print("   Building correlation features...")
            correlation_features = self._build_correlation_features(aligned_data)
            all_features.append(correlation_features)
        
        # 4. Portfolio features
        if self.config.portfolio_features:
            print("   Building portfolio features...")
            portfolio_features = self._build_portfolio_features(aligned_data)
            all_features.append(portfolio_features)
        
        # 5. Market regime features
        if self.config.market_regime_features:
            print("   Building market regime features...")
            regime_features = self._build_market_regime_features(aligned_data)
            all_features.append(regime_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            # Remove duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        else:
            combined_features = pd.DataFrame(index=aligned_data.index)
        
        print(f"✅ Multi-symbol features built: {combined_features.shape}")
        return combined_features
    
    def _build_individual_features(self, data_manager: MultiSymbolDataManager) -> pd.DataFrame:
        """Build individual features for each symbol"""
        
        individual_features = []
        
        for symbol in data_manager.get_available_symbols():
            symbol_data = data_manager.get_symbol_data(symbol)
            if symbol_data is None:
                continue
            
            # Generate features for this symbol
            features = self.market_analyzer.build_comprehensive_features(symbol_data)
            
            # Add symbol prefix to column names
            features.columns = [f'{symbol}_{col}' for col in features.columns]
            
            individual_features.append(features)
        
        if individual_features:
            return pd.concat(individual_features, axis=1)
        return pd.DataFrame()
    
    def _build_cross_asset_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Build cross-asset features (relative strength, momentum)"""
        
        cross_features = []
        symbols = aligned_data.columns.get_level_values('Symbol').unique()
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Calculate returns for all symbols
        returns_data = {}
        for symbol in symbols:
            close_prices = aligned_data[symbol]['Close']
            returns_data[symbol] = close_prices.pct_change()
        
        returns_df = pd.DataFrame(returns_data)
        
        # 1. Relative strength features
        for symbol in symbols:
            symbol_returns = returns_df[symbol]
            
            # Relative strength vs market (average of all symbols)
            market_returns = returns_df.mean(axis=1)
            relative_strength = symbol_returns - market_returns
            cross_features.append(pd.DataFrame({
                f'{symbol}_relative_strength': relative_strength,
                f'{symbol}_relative_strength_ma': relative_strength.rolling(self.config.feature_lookback).mean(),
                f'{symbol}_relative_strength_std': relative_strength.rolling(self.config.feature_lookback).std()
            }))
        
        # 2. Momentum ranking features
        momentum_ranking = returns_df.rolling(self.config.feature_lookback).mean().rank(axis=1, ascending=False)
        for symbol in symbols:
            cross_features.append(pd.DataFrame({
                f'{symbol}_momentum_rank': momentum_ranking[symbol],
                f'{symbol}_momentum_percentile': momentum_ranking[symbol] / len(symbols)
            }))
        
        # 3. Volatility ranking features
        volatility_ranking = returns_df.rolling(self.config.feature_lookback).std().rank(axis=1, ascending=False)
        for symbol in symbols:
            cross_features.append(pd.DataFrame({
                f'{symbol}_volatility_rank': volatility_ranking[symbol],
                f'{symbol}_volatility_percentile': volatility_ranking[symbol] / len(symbols)
            }))
        
        if cross_features:
            return pd.concat(cross_features, axis=1)
        return pd.DataFrame()
    
    def _build_correlation_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Build correlation-based features"""
        
        correlation_features = []
        symbols = aligned_data.columns.get_level_values('Symbol').unique()
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Calculate returns for all symbols
        returns_data = {}
        for symbol in symbols:
            close_prices = aligned_data[symbol]['Close']
            returns_data[symbol] = close_prices.pct_change()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Rolling correlation features
        window = self.config.correlation_window
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Avoid duplicates
                    # Rolling correlation
                    rolling_corr = returns_df[symbol1].rolling(window).corr(returns_df[symbol2])
                    correlation_features.append(pd.DataFrame({
                        f'{symbol1}_{symbol2}_correlation': rolling_corr,
                        f'{symbol1}_{symbol2}_correlation_ma': rolling_corr.rolling(10).mean(),
                        f'{symbol1}_{symbol2}_correlation_std': rolling_corr.rolling(10).std()
                    }))
        
        # Average correlation for each symbol
        for symbol in symbols:
            symbol_correlations = []
            for other_symbol in symbols:
                if symbol != other_symbol:
                    rolling_corr = returns_df[symbol].rolling(window).corr(returns_df[other_symbol])
                    symbol_correlations.append(rolling_corr)
            
            if symbol_correlations:
                avg_correlation = pd.concat(symbol_correlations, axis=1).mean(axis=1)
                correlation_features.append(pd.DataFrame({
                    f'{symbol}_avg_correlation': avg_correlation,
                    f'{symbol}_avg_correlation_ma': avg_correlation.rolling(10).mean()
                }))
        
        if correlation_features:
            return pd.concat(correlation_features, axis=1)
        return pd.DataFrame()
    
    def _build_portfolio_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Build portfolio-level features"""
        
        portfolio_features = []
        symbols = aligned_data.columns.get_level_values('Symbol').unique()
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Calculate returns for all symbols
        returns_data = {}
        for symbol in symbols:
            close_prices = aligned_data[symbol]['Close']
            returns_data[symbol] = close_prices.pct_change()
        
        returns_df = pd.DataFrame(returns_data)
        
        # 1. Portfolio diversification metrics
        window = self.config.feature_lookback
        
        # Rolling correlation matrix
        rolling_correlations = []
        for i in range(len(returns_df)):
            if i >= window:
                corr_matrix = returns_df.iloc[i-window:i].corr()
                # Average correlation (excluding diagonal)
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                rolling_correlations.append(avg_corr)
            else:
                rolling_correlations.append(np.nan)
        
        portfolio_features.append(pd.DataFrame({
            'portfolio_avg_correlation': rolling_correlations,
            'portfolio_diversification': 1 - np.array(rolling_correlations)  # Higher = more diversified
        }))
        
        # 2. Portfolio concentration metrics
        # Rolling volatility of each symbol
        symbol_volatilities = returns_df.rolling(window).std()
        
        # Portfolio volatility (equal weight)
        portfolio_returns = returns_df.mean(axis=1)
        portfolio_volatility = portfolio_returns.rolling(window).std()
        
        # Concentration ratio (max weight / sum of weights)
        max_volatility = symbol_volatilities.max(axis=1)
        sum_volatility = symbol_volatilities.sum(axis=1)
        concentration_ratio = max_volatility / sum_volatility
        
        portfolio_features.append(pd.DataFrame({
            'portfolio_volatility': portfolio_volatility,
            'portfolio_concentration': concentration_ratio,
            'portfolio_max_volatility': max_volatility
        }))
        
        # 3. Market breadth indicators
        # Percentage of symbols with positive returns
        positive_returns = (returns_df > 0).sum(axis=1)
        total_symbols = len(symbols)
        market_breadth = positive_returns / total_symbols
        
        portfolio_features.append(pd.DataFrame({
            'market_breadth': market_breadth,
            'market_breadth_ma': market_breadth.rolling(10).mean(),
            'market_breadth_std': market_breadth.rolling(10).std()
        }))
        
        if portfolio_features:
            return pd.concat(portfolio_features, axis=1)
        return pd.DataFrame()
    
    def _build_market_regime_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Build market regime features"""
        
        regime_features = []
        symbols = aligned_data.columns.get_level_values('Symbol').unique()
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Calculate returns for all symbols
        returns_data = {}
        for symbol in symbols:
            close_prices = aligned_data[symbol]['Close']
            returns_data[symbol] = close_prices.pct_change()
        
        returns_df = pd.DataFrame(returns_data)
        
        # 1. Market regime classification
        window = self.config.feature_lookback
        
        # Market return (equal weight portfolio)
        market_returns = returns_df.mean(axis=1)
        market_volatility = market_returns.rolling(window).std()
        market_trend = market_returns.rolling(window).mean()
        
        # Regime classification
        high_vol = market_volatility > market_volatility.rolling(50).quantile(0.8)
        positive_trend = market_trend > 0
        negative_trend = market_trend < -market_volatility.rolling(50).quantile(0.2)
        
        # Regime labels
        regime = pd.Series('normal', index=market_returns.index)
        regime[high_vol & positive_trend] = 'high_vol_bull'
        regime[high_vol & negative_trend] = 'high_vol_bear'
        regime[~high_vol & positive_trend] = 'low_vol_bull'
        regime[~high_vol & negative_trend] = 'low_vol_bear'
        
        # Convert to numeric
        regime_map = {'normal': 0, 'low_vol_bull': 1, 'low_vol_bear': 2, 'high_vol_bull': 3, 'high_vol_bear': 4}
        regime_numeric = regime.map(regime_map)
        
        regime_features.append(pd.DataFrame({
            'market_regime': regime_numeric,
            'market_trend': market_trend,
            'market_volatility': market_volatility,
            'market_volatility_percentile': market_volatility.rolling(50).rank(pct=True)
        }))
        
        # 2. Sector rotation indicators (if we have enough symbols)
        if len(symbols) >= 4:
            # Calculate relative performance of different quartiles
            # For each time point, calculate quartiles across symbols
            sector_rotation = []
            for i in range(len(returns_df)):
                if i >= window:
                    period_returns = returns_df.iloc[i-window:i]
                    # Calculate quartiles for each symbol over the window
                    symbol_quartiles = period_returns.quantile([0.25, 0.5, 0.75])
                    # Use the median (50th percentile) as the sector rotation indicator
                    sector_rotation.append(symbol_quartiles.loc[0.5].mean())
                else:
                    sector_rotation.append(0.0)
            
            regime_features.append(pd.DataFrame({
                'sector_rotation': sector_rotation,
                'sector_rotation_ma': pd.Series(sector_rotation).rolling(10).mean()
            }))
        
        if regime_features:
            return pd.concat(regime_features, axis=1)
        return pd.DataFrame()
    
    def get_feature_summary(self, features: pd.DataFrame) -> dict[str, Any]:
        """Get summary of generated features"""
        
        summary = {
            'total_features': len(features.columns),
            'feature_categories': {},
            'missing_data_pct': features.isnull().sum().sum() / (len(features) * len(features.columns)),
            'feature_types': {}
        }
        
        # Categorize features
        for col in features.columns:
            if '_relative_strength' in col:
                category = 'cross_asset'
            elif '_correlation' in col:
                category = 'correlation'
            elif col.startswith('portfolio_') or col.startswith('market_'):
                category = 'portfolio'
            else:
                category = 'individual'
            
            if category not in summary['feature_categories']:
                summary['feature_categories'][category] = 0
            summary['feature_categories'][category] += 1
        
        # Feature types
        for col in features.columns:
            dtype = str(features[col].dtype)
            if dtype not in summary['feature_types']:
                summary['feature_types'][dtype] = 0
            summary['feature_types'][dtype] += 1
        
        return summary
