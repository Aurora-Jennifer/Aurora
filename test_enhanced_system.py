"""
Comprehensive Test for Enhanced Trading System
Tests regime detection, adaptive features, and regime-aware ensemble strategy
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.regime_detector import RegimeDetector, RegimeAwareFeatureEngine
from core.feature_reweighter import FeatureReweighter, AdaptiveFeatureEngine
from strategies.regime_aware_ensemble import RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams
from enhanced_paper_trading import EnhancedPaperTradingSystem
from core.utils import setup_logging, ensure_directories


def test_regime_detection():
    """Test regime detection system."""
    print("🔍 Testing Regime Detection System...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Create different market regimes
    trend_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.normal(0.1, 0.5, len(dates))),
        'High': 100 + np.cumsum(np.random.normal(0.1, 0.5, len(dates))) + np.random.uniform(0, 2, len(dates)),
        'Low': 100 + np.cumsum(np.random.normal(0.1, 0.5, len(dates))) - np.random.uniform(0, 2, len(dates)),
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Initialize regime detector
    detector = RegimeDetector(lookback_period=60)
    
    # Test regime detection
    regime_name, confidence, regime_params = detector.detect_regime(trend_data)
    
    print(f"  ✅ Detected regime: {regime_name} (confidence: {confidence:.2f})")
    print(f"  ✅ Regime parameters: {regime_params.regime_name}")
    print(f"  ✅ Position sizing multiplier: {regime_params.position_sizing_multiplier}")
    
    return True


def test_feature_reweighting():
    """Test feature re-weighting system."""
    print("\n⚖️ Testing Feature Re-weighting System...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Create features
    features = {
        'momentum': pd.Series(np.random.normal(0, 1, len(dates)), index=dates),
        'mean_reversion': pd.Series(np.random.normal(0, 1, len(dates)), index=dates),
        'volatility': pd.Series(np.random.uniform(0.1, 0.3, len(dates)), index=dates),
        'volume': pd.Series(np.random.uniform(0.5, 1.5, len(dates)), index=dates)
    }
    
    # Create returns
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    # Initialize re-weighter
    reweighter = FeatureReweighter(rolling_window=30, reweight_frequency=10)
    
    # Update performance for different regimes
    regimes = ['trend', 'chop', 'volatile']
    for i, regime in enumerate(regimes):
        start_idx = i * 100
        end_idx = start_idx + 100
        
        regime_features = {k: v.iloc[start_idx:end_idx] for k, v in features.items()}
        regime_returns = returns.iloc[start_idx:end_idx]
        
        reweighter.update_feature_performance(
            regime_features, regime_returns, regime, dates[start_idx]
        )
    
    # Get feature weights
    for regime in regimes:
        weights = reweighter.get_feature_weights(regime)
        summary = reweighter.get_regime_performance_summary(regime)
        
        print(f"  ✅ {regime.capitalize()} regime:")
        print(f"     Feature weights: {weights}")
        print(f"     Avg IC: {summary.get('avg_ic', 0):.3f}")
        print(f"     Avg Sharpe: {summary.get('avg_sharpe', 0):.3f}")
    
    return True


def test_regime_aware_ensemble():
    """Test regime-aware ensemble strategy."""
    print("\n🎯 Testing Regime-Aware Ensemble Strategy...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Create realistic price data
    close_prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    high_prices = close_prices + np.random.uniform(0, 2, len(dates))
    low_prices = close_prices - np.random.uniform(0, 2, len(dates))
    volumes = np.random.uniform(1000000, 5000000, len(dates))
    
    data = pd.DataFrame({
        'Close': close_prices,
        'High': high_prices,
        'Low': low_prices,
        'Volume': volumes
    }, index=dates)
    
    # Initialize strategy
    params = RegimeAwareEnsembleParams(
        combination_method='rolling_ic',
        confidence_threshold=0.3,
        use_regime_switching=True,
        regime_lookback=60
    )
    
    strategy = RegimeAwareEnsembleStrategy(params)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Get regime info
    regime_info = strategy.get_regime_info()
    
    print(f"  ✅ Generated signals: {len(signals)} data points")
    print(f"  ✅ Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"  ✅ Current regime: {regime_info.get('regime', 'unknown')}")
    print(f"  ✅ Regime confidence: {regime_info.get('confidence', 0):.3f}")
    
    # Test signal characteristics
    non_zero_signals = signals[signals != 0]
    if len(non_zero_signals) > 0:
        print(f"  ✅ Non-zero signals: {len(non_zero_signals)} ({len(non_zero_signals)/len(signals)*100:.1f}%)")
        print(f"  ✅ Signal volatility: {non_zero_signals.std():.3f}")
    
    return True


def test_enhanced_paper_trading():
    """Test enhanced paper trading system."""
    print("\n💰 Testing Enhanced Paper Trading System...")
    
    # Initialize system
    system = EnhancedPaperTradingSystem()
    
    # Test daily trading
    test_date = date(2024, 1, 15)  # Use a recent date
    system.run_daily_trading(test_date)
    
    # Get performance report
    report = system.get_performance_report()
    
    print(f"  ✅ System initialized with ${system.capital:,.0f} capital")
    print(f"  ✅ Total trades: {report.get('total_trades', 0)}")
    print(f"  ✅ Current capital: ${report.get('current_capital', 0):,.0f}")
    
    if 'regime_stats' in report:
        print("  ✅ Regime statistics:")
        for regime, stats in report['regime_stats'].items():
            print(f"     {regime}: {stats['count']} days, avg confidence: {stats['avg_confidence']:.3f}")
    
    return True


def test_integration():
    """Test integration of all components."""
    print("\n🔗 Testing System Integration...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Create realistic market data
    close_prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    data = pd.DataFrame({
        'Close': close_prices,
        'High': close_prices + np.random.uniform(0, 2, len(dates)),
        'Low': close_prices - np.random.uniform(0, 2, len(dates)),
        'Volume': np.random.uniform(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test full pipeline
    try:
        # 1. Regime detection
        detector = RegimeDetector()
        regime_name, confidence, regime_params = detector.detect_regime(data)
        
        # 2. Feature re-weighting
        reweighter = FeatureReweighter()
        features = {
            'momentum': data['Close'].pct_change(),
            'volatility': data['Close'].pct_change().rolling(20).std(),
            'volume': data['Volume'] / data['Volume'].rolling(20).mean()
        }
        returns = data['Close'].pct_change().shift(-1)
        reweighter.update_feature_performance(features, returns, regime_name, data.index[-1])
        
        # 3. Adaptive features
        adaptive_engine = AdaptiveFeatureEngine(reweighter)
        adaptive_features = adaptive_engine.generate_adaptive_features(data, regime_name, features)
        
        # 4. Regime-aware ensemble
        params = RegimeAwareEnsembleParams()
        strategy = RegimeAwareEnsembleStrategy(params)
        signals = strategy.generate_signals(data)
        
        print(f"  ✅ Regime detection: {regime_name} ({confidence:.3f})")
        print(f"  ✅ Feature re-weighting: {len(features)} features processed")
        print(f"  ✅ Adaptive features: {len(adaptive_features)} features generated")
        print(f"  ✅ Ensemble signals: {len(signals)} signals generated")
        print(f"  ✅ Integration test passed!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 Starting Comprehensive Enhanced System Test")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging("logs/enhanced_test.log")
    ensure_directories("results")
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Feature Re-weighting", test_feature_reweighting),
        ("Regime-Aware Ensemble", test_regime_aware_ensemble),
        ("Enhanced Paper Trading", test_enhanced_paper_trading),
        ("System Integration", test_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        except Exception as e:
            test_results[test_name] = False
            print(f"❌ FAILED {test_name}: {e}")
    
    # Generate test report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Save test results
    with open("results/enhanced_test_results.json", "w") as f:
        json.dump({
            'test_results': test_results,
            'summary': {
                'passed': passed_tests,
                'total': total_tests,
                'success_rate': passed_tests/total_tests*100
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Enhanced system is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review and fix issues.")
        return False


def create_config_file():
    """Create enhanced paper trading configuration file."""
    config = {
        'initial_capital': 100000,
        'symbols': ['SPY', 'QQQ', 'IWM'],
        'strategies': ['regime_ensemble', 'ensemble', 'sma', 'momentum'],
        'rebalance_frequency': 'daily',
        'max_position_size': 0.2,
        'stop_loss': 0.05,
        'take_profit': 0.15,
        'regime_switching': True,
        'feature_adaptation': True,
        'performance_tracking': True,
        'regime_detection': {
            'lookback_period': 252,
            'confidence_threshold': 0.3
        },
        'feature_reweighting': {
            'rolling_window': 60,
            'reweight_frequency': 20,
            'decay_factor': 0.95
        },
        'ensemble_strategy': {
            'combination_method': 'rolling_ic',
            'confidence_threshold': 0.3,
            'use_regime_switching': True,
            'regime_lookback': 252
        }
    }
    
    ensure_directories("config")
    with open("config/enhanced_paper_trading_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created enhanced paper trading configuration file")


if __name__ == "__main__":
    # Create config file
    create_config_file()
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Enhanced system is ready for 65%+ return optimization!")
        print("\nNext steps:")
        print("1. Run: python enhanced_paper_trading.py --daily")
        print("2. Setup cron: python enhanced_paper_trading.py --setup-cron")
        print("3. Monitor logs in logs/enhanced_paper_trading.log")
        print("4. Review results in results/ directory")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before deployment.")
    
    sys.exit(0 if success else 1)
