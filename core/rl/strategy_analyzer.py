"""
Strategy Analyzer

Analyzes trading patterns to identify what contributes to profitable trades
and suggests strategy improvements.
"""

import pandas as pd
from typing import Any
from dataclasses import dataclass
import json


@dataclass
class StrategyPattern:
    """A profitable trading pattern"""
    name: str
    description: str
    conditions: dict[str, Any]
    success_rate: float
    avg_reward: float
    frequency: int
    confidence: float


@dataclass
class StrategyRecommendation:
    """Strategy improvement recommendation"""
    pattern: StrategyPattern
    action: str  # 'increase_position', 'add_filter', 'adjust_timing'
    reasoning: str
    expected_improvement: float


class StrategyAnalyzer:
    """
    Analyzes trading patterns to identify profitable strategies
    """
    
    def __init__(self, config: dict[str, Any]):
        self.min_pattern_frequency = config.get('min_pattern_frequency', 5)
        self.min_success_rate = config.get('min_success_rate', 0.6)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.patterns = []
        self.recommendations = []
        
    def analyze_trades(self, trade_results: list[Any]) -> dict[str, Any]:
        """
        Analyze trade results to identify patterns
        
        Args:
            trade_results: List of TradeResult objects
            
        Returns:
            Analysis results with patterns and recommendations
        """
        if not trade_results:
            return {"error": "No trades to analyze"}
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'action': t.action,
            'position_size': t.position_size,
            'price_change': t.price_change,
            'reward': t.reward,
            'success': t.success,
            **t.features,
            **{f"market_{k}": v for k, v in t.market_context.items()}
        } for t in trade_results])
        
        # Find patterns
        patterns = self._find_patterns(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, df)
        
        # Calculate overall statistics
        stats = self._calculate_statistics(df)
        
        return {
            "patterns": [p.__dict__ for p in patterns],
            "recommendations": [r.__dict__ for r in recommendations],
            "statistics": stats,
            "top_patterns": self._get_top_patterns(patterns)
        }
    
    def _find_patterns(self, df: pd.DataFrame) -> list[StrategyPattern]:
        """Find profitable trading patterns"""
        patterns = []
        
        # Pattern 1: Feature-based patterns
        feature_patterns = self._find_feature_patterns(df)
        patterns.extend(feature_patterns)
        
        # Pattern 2: Market condition patterns
        market_patterns = self._find_market_patterns(df)
        patterns.extend(market_patterns)
        
        # Pattern 3: Action-based patterns
        action_patterns = self._find_action_patterns(df)
        patterns.extend(action_patterns)
        
        # Pattern 4: Timing patterns
        timing_patterns = self._find_timing_patterns(df)
        patterns.extend(timing_patterns)
        
        return patterns
    
    def _find_feature_patterns(self, df: pd.DataFrame) -> list[StrategyPattern]:
        """Find patterns based on technical features"""
        patterns = []
        
        # Get feature columns (exclude non-feature columns)
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'action', 'position_size', 'price_change', 
                        'reward', 'success'] and not col.startswith('market_')]
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
                
            # Find optimal thresholds for this feature
            thresholds = self._find_optimal_thresholds(df, feature)
            
            for threshold, direction in thresholds:
                # Filter trades that meet this condition
                if direction == 'above':
                    mask = df[feature] > threshold
                else:
                    mask = df[feature] < threshold
                    
                if mask.sum() < self.min_pattern_frequency:
                    continue
                    
                subset = df[mask]
                success_rate = subset['success'].mean()
                
                if success_rate >= self.min_success_rate:
                    pattern = StrategyPattern(
                        name=f"{feature}_{direction}_{threshold:.3f}",
                        description=f"Trades when {feature} is {direction} {threshold:.3f}",
                        conditions={feature: {"threshold": threshold, "direction": direction}},
                        success_rate=success_rate,
                        avg_reward=subset['reward'].mean(),
                        frequency=len(subset),
                        confidence=self._calculate_confidence(subset)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_market_patterns(self, df: pd.DataFrame) -> list[StrategyPattern]:
        """Find patterns based on market conditions"""
        patterns = []
        
        market_cols = [col for col in df.columns if col.startswith('market_')]
        
        for col in market_cols:
            if col not in df.columns:
                continue
                
            # Market regime patterns
            if 'market_regime' in col:
                for regime in df[col].unique():
                    if pd.isna(regime):
                        continue
                        
                    mask = df[col] == regime
                    if mask.sum() < self.min_pattern_frequency:
                        continue
                        
                    subset = df[mask]
                    success_rate = subset['success'].mean()
                    
                    if success_rate >= self.min_success_rate:
                        pattern = StrategyPattern(
                            name=f"market_regime_{regime}",
                            description=f"Trades during {regime} market regime",
                            conditions={"market_regime": regime},
                            success_rate=success_rate,
                            avg_reward=subset['reward'].mean(),
                            frequency=len(subset),
                            confidence=self._calculate_confidence(subset)
                        )
                        patterns.append(pattern)
            
            # Volatility patterns
            elif 'volatility' in col:
                thresholds = self._find_optimal_thresholds(df, col)
                
                for threshold, direction in thresholds:
                    if direction == 'above':
                        mask = df[col] > threshold
                    else:
                        mask = df[col] < threshold
                        
                    if mask.sum() < self.min_pattern_frequency:
                        continue
                        
                    subset = df[mask]
                    success_rate = subset['success'].mean()
                    
                    if success_rate >= self.min_success_rate:
                        pattern = StrategyPattern(
                            name=f"volatility_{direction}_{threshold:.3f}",
                            description=f"Trades when volatility is {direction} {threshold:.3f}",
                            conditions={"volatility": {"threshold": threshold, "direction": direction}},
                            success_rate=success_rate,
                            avg_reward=subset['reward'].mean(),
                            frequency=len(subset),
                            confidence=self._calculate_confidence(subset)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _find_action_patterns(self, df: pd.DataFrame) -> list[StrategyPattern]:
        """Find patterns based on trading actions"""
        patterns = []
        
        for action in df['action'].unique():
            mask = df['action'] == action
            if mask.sum() < self.min_pattern_frequency:
                continue
                
            subset = df[mask]
            success_rate = subset['success'].mean()
            
            if success_rate >= self.min_success_rate:
                pattern = StrategyPattern(
                    name=f"action_{action}",
                    description=f"Trades using {action} action",
                    conditions={"action": action},
                    success_rate=success_rate,
                    avg_reward=subset['reward'].mean(),
                    frequency=len(subset),
                    confidence=self._calculate_confidence(subset)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_timing_patterns(self, df: pd.DataFrame) -> list[StrategyPattern]:
        """Find patterns based on timing"""
        patterns = []
        
        # Day of week patterns
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        for day in df['day_of_week'].unique():
            mask = df['day_of_week'] == day
            if mask.sum() < self.min_pattern_frequency:
                continue
                
            subset = df[mask]
            success_rate = subset['success'].mean()
            
            if success_rate >= self.min_success_rate:
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day]
                pattern = StrategyPattern(
                    name=f"timing_{day_name.lower()}",
                    description=f"Trades on {day_name}",
                    conditions={"day_of_week": day},
                    success_rate=success_rate,
                    avg_reward=subset['reward'].mean(),
                    frequency=len(subset),
                    confidence=self._calculate_confidence(subset)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_optimal_thresholds(self, df: pd.DataFrame, feature: str) -> list[tuple]:
        """Find optimal thresholds for a feature"""
        thresholds = []
        
        # Use percentiles to find thresholds
        percentiles = [10, 25, 50, 75, 90]
        
        for p in percentiles:
            threshold = df[feature].quantile(p / 100)
            
            # Test above threshold
            above_mask = df[feature] > threshold
            if above_mask.sum() >= self.min_pattern_frequency:
                above_success = df[above_mask]['success'].mean()
                if above_success >= self.min_success_rate:
                    thresholds.append((threshold, 'above'))
            
            # Test below threshold
            below_mask = df[feature] < threshold
            if below_mask.sum() >= self.min_pattern_frequency:
                below_success = df[below_mask]['success'].mean()
                if below_success >= self.min_success_rate:
                    thresholds.append((threshold, 'below'))
        
        return thresholds
    
    def _calculate_confidence(self, subset: pd.DataFrame) -> float:
        """Calculate confidence in a pattern"""
        n = len(subset)
        success_rate = subset['success'].mean()
        
        # Simple confidence based on sample size and success rate
        confidence = min(1.0, (n / 100) * success_rate)
        return confidence
    
    def _generate_recommendations(self, patterns: list[StrategyPattern], 
                                df: pd.DataFrame) -> list[StrategyRecommendation]:
        """Generate strategy improvement recommendations"""
        recommendations = []
        
        # Sort patterns by success rate and frequency
        sorted_patterns = sorted(patterns, 
                               key=lambda p: p.success_rate * p.frequency, 
                               reverse=True)
        
        for pattern in sorted_patterns[:5]:  # Top 5 patterns
            if pattern.confidence >= self.min_confidence:
                # Recommendation 1: Increase position size for high-confidence patterns
                if pattern.success_rate > 0.8 and pattern.frequency > 10:
                    rec = StrategyRecommendation(
                        pattern=pattern,
                        action="increase_position",
                        reasoning=f"High success rate ({pattern.success_rate:.2%}) with good frequency ({pattern.frequency})",
                        expected_improvement=pattern.avg_reward * 0.2
                    )
                    recommendations.append(rec)
                
                # Recommendation 2: Add filters for medium-confidence patterns
                elif pattern.success_rate > 0.7:
                    rec = StrategyRecommendation(
                        pattern=pattern,
                        action="add_filter",
                        reasoning=f"Good success rate ({pattern.success_rate:.2%}) but could benefit from additional filters",
                        expected_improvement=pattern.avg_reward * 0.1
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _calculate_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate overall trading statistics"""
        return {
            "total_trades": len(df),
            "success_rate": df['success'].mean(),
            "avg_reward": df['reward'].mean(),
            "total_reward": df['reward'].sum(),
            "best_trade": df['reward'].max(),
            "worst_trade": df['reward'].min(),
            "reward_std": df['reward'].std(),
            "sharpe_ratio": df['reward'].mean() / (df['reward'].std() + 1e-8)
        }
    
    def _get_top_patterns(self, patterns: list[StrategyPattern]) -> list[dict[str, Any]]:
        """Get top patterns by success rate and frequency"""
        if not patterns:
            return []
            
        # Score patterns by success rate * frequency * confidence
        scored_patterns = []
        for pattern in patterns:
            score = pattern.success_rate * pattern.frequency * pattern.confidence
            scored_patterns.append({
                "pattern": pattern.__dict__,
                "score": score
            })
        
        # Sort by score and return top 5
        scored_patterns.sort(key=lambda x: x["score"], reverse=True)
        return scored_patterns[:5]
    
    def save_analysis(self, analysis: dict[str, Any], filepath: str):
        """Save analysis results to file"""
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    def load_analysis(self, filepath: str) -> dict[str, Any]:
        """Load analysis results from file"""
        with open(filepath) as f:
            return json.load(f)

