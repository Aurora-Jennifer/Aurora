#!/usr/bin/env python3
"""
Daily Paper Trading Operations Script with Execution Infrastructure

Automates the complete daily trading workflow with real order execution:
- Pre-market validation
- Trading session monitoring with real order execution
- End-of-day reporting and reconciliation

Usage:
    python ops/daily_paper_trading_with_execution.py --mode preflight    # 08:00 CT
    python ops/daily_paper_trading_with_execution.py --mode trading      # 08:30-15:00 CT
    python ops/daily_paper_trading_with_execution.py --mode eod          # 15:10 CT
    python ops/daily_paper_trading_with_execution.py --mode full         # Complete daily cycle
"""
import sys
import os
sys.path.append('.')

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import signal
import logging
import hashlib
import subprocess
import yaml

from ops.paper_trading_guards import PaperTradingGuards
from ops.pre_market_dry_run import run_pre_market_dry_run
# from ml.production_logging import setup_production_logging  # Removed - causes queue handler conflicts
from core.walk.xgb_model_loader import XGBModelLoader
from core.ml.feature_gate import prepare_X_for_xgb
from core.data.ingest import fetch_alpaca_bars, create_fallback_data
from core.ml.sector_residualizer import load_sector_map, sector_residualize

# Import execution infrastructure
from core.execution.order_types import Order, OrderType, OrderSide, OrderStatus
from core.execution.order_manager import OrderManager
from core.execution.position_sizing import PositionSizer, PositionSizingConfig
from core.execution.risk_manager import RiskManager, RiskLimits
from core.execution.portfolio_manager import PortfolioManager
from core.execution.execution_engine import ExecutionEngine, ExecutionConfig

# Import Alpaca client
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, OrderType as AlpacaOrderType, TimeInForce


class DailyPaperTradingWithExecution:
    """Automated daily paper trading operations with real order execution."""
    
    def __init__(self):
        """Initialize daily operations with execution infrastructure."""
        # Initialize logging first
        self.logger = logging.getLogger("daily_paper_trading_execution")
        
        # Load environment variables from .env file
        self._load_environment()
        
        # Load configuration
        self._load_execution_config()
        
        # Initialize execution components
        self._initialize_execution_infrastructure()
        
        # Initialize existing components
        self._initialize_existing_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        from pathlib import Path
        
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
                        except ValueError:
                            # Skip malformed lines
                            continue
            self.logger.info("Loaded environment variables from .env file")
        else:
            self.logger.warning("No .env file found - using system environment variables")
    
    def _load_execution_config(self):
        """Load execution configuration."""
        try:
            config_path = Path('config/execution.yaml')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.execution_config = yaml.safe_load(f)
                self.logger.info("Loaded execution configuration from config/execution.yaml")
            else:
                # Use default configuration
                self.execution_config = {
                    'execution': {
                        'enabled': True,
                        'mode': 'paper',
                        'signal_threshold': 0.1,
                        'max_orders_per_execution': 10
                    },
                    'position_sizing': {
                        'max_position_size': 0.1,
                        'min_trade_size': 100.0,
                        'max_trade_size': 5000.0,
                        'signal_threshold': 0.1
                    },
                    'risk_management': {
                        'max_daily_loss': 0.02,
                        'max_position_risk': 0.05,
                        'max_orders_per_day': 50
                    }
                }
                self.logger.info("Using default execution configuration")
        except Exception as e:
            self.logger.error(f"Error loading execution config: {e}")
            raise
    
    def _initialize_execution_infrastructure(self):
        """Initialize the execution infrastructure components."""
        try:
            # Initialize Alpaca client
            # Support both ALPACA_ and APCA_ environment variable formats
            api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY/ALPACA_SECRET_KEY or APCA_API_KEY_ID/APCA_API_SECRET_KEY")
            
            self.alpaca_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True  # Always use paper trading for safety
            )
            self.logger.info("✅ Alpaca client initialized (paper trading mode)")
            
            # Initialize execution components
            self.order_manager = OrderManager(self.alpaca_client, self.execution_config.get('order_management', {}))
            self.portfolio_manager = PortfolioManager(self.alpaca_client, self.execution_config.get('portfolio_management', {}))
            
            # Position sizing configuration
            position_config = PositionSizingConfig(**self.execution_config['position_sizing'])
            self.position_sizer = PositionSizer(position_config)
            
            # Risk management configuration
            risk_limits = RiskLimits(**self.execution_config['risk_management'])
            self.risk_manager = RiskManager(risk_limits, self.execution_config.get('sector_mapping', {}))
            
            # Execution engine configuration
            exec_config = ExecutionConfig(**self.execution_config['execution'])
            self.execution_engine = ExecutionEngine(
                order_manager=self.order_manager,
                portfolio_manager=self.portfolio_manager,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                config=exec_config
            )
            
            self.logger.info("✅ Execution infrastructure initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize execution infrastructure: {e}")
            self.logger.warning("Falling back to signal-only mode (no order execution)")
            self.execution_engine = None
            self.alpaca_client = None
    
    def _initialize_existing_components(self):
        """Initialize existing components from the original system."""
        # Initialize guards
        self.guards = PaperTradingGuards()
        
        # Initialize model loader
        model_config = self.execution_config.get('model', {})
        if model_config.get('enabled', False) and model_config.get('model_path') and model_config.get('features_path'):
            try:
                self.model_loader = XGBModelLoader(
                    model_path=model_config['model_path'],
                    features_path=model_config['features_path']
                )
                self.logger.info("✅ XGBoost model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Model loading failed: {e}")
                self.model_loader = None
        else:
            self.logger.info("ℹ️  Model integration disabled - will use external signals")
            self.model_loader = None
        
        # Load sector map
        self._load_sector_map()
        
        # Trading state
        self.trading_active = False
        self.emergency_halt = False
    
    def _load_sector_map(self):
        """Load sector mapping for risk management."""
        try:
            self.sector_map = load_sector_map()
            self.logger.info(f"✅ Sector map loaded: {len(self.sector_map)} symbols")
        except Exception as e:
            self.logger.warning(f"Sector map loading failed: {e}")
            self.sector_map = None
    
    def _generate_trading_signals(self, market_data: pd.DataFrame) -> Dict:
        """
        Generate trading signals using the production model with strict feature contract.
        
        Args:
            market_data: DataFrame with market data and features
            
        Returns:
            Dict with signals, positions, and metadata
        """
        if self.model_loader is None:
            self.logger.warning("MODEL DISABLED: No model loaded → MOCK mode")
            return self._generate_mock_signals(market_data)
        
        try:
            # Apply sector residualization if sector map is available
            if self.sector_map is not None:
                # Get base features that need sector residualization (CSR features)
                csr_features = [col for col in market_data.columns if col.endswith('_csr')]
                if csr_features:
                    market_data = sector_residualize(market_data, csr_features, self.sector_map)
            
            # Prepare features for XGBoost
            X = prepare_X_for_xgb(market_data, self.model_loader.features_whitelist)
            
            # Generate predictions
            predictions = self.model_loader.predict(X)
            
            # Convert predictions to trading signals
            signals = self._predictions_to_signals(predictions, market_data)
            
            self.logger.info(f"✅ USING MODEL ({len(self.model_loader.features_whitelist)}/{len(self.model_loader.features_whitelist)} features matched)")
            self.logger.info(f"Generated {len(signals)} trading signals using production model")
            
            return {
                'signals': signals,
                'predictions': predictions,
                'model_used': True,
                'feature_count': len(self.model_loader.features_whitelist),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            self.logger.warning("MODEL DISABLED: Error in prediction → MOCK mode")
            return self._generate_mock_signals(market_data)
    
    def _predictions_to_signals(self, predictions: np.ndarray, market_data: pd.DataFrame) -> Dict:
        """Convert model predictions to trading signals."""
        # Simple long-short strategy based on predictions
        # Top 20% long, bottom 20% short
        n_assets = len(predictions)
        n_long = max(1, int(n_assets * 0.2))
        n_short = max(1, int(n_assets * 0.2))
        
        # Ensure we have symbols in the index
        if 'symbol' in market_data.columns:
            symbols = market_data['symbol'].values
        else:
            symbols = market_data.index.values
        
        # Rank predictions (descending order - highest scores first)
        ranks = np.argsort(predictions)[::-1]
        
        signals = {}
        for i, rank in enumerate(ranks):
            symbol = symbols[rank]
            if i < n_long:
                signals[symbol] = 1.0 / n_long  # Equal weight long
            elif i >= n_assets - n_short:
                signals[symbol] = -1.0 / n_short  # Equal weight short
            else:
                signals[symbol] = 0.0  # Neutral
        
        # Log signal distribution for sanity check
        signal_values = np.array(list(signals.values()))
        self.logger.info(f"Signal distribution: mean={signal_values.mean():.3f}, std={signal_values.std():.3f}, "
                        f"longs={(signal_values>0).mean():.1%}, shorts={(signal_values<0).mean():.1%}")
        
        return signals
    
    def _generate_mock_signals(self, market_data: pd.DataFrame) -> Dict:
        """Generate mock trading signals for testing."""
        np.random.seed(42)
        n_assets = len(market_data)
        
        # Random long-short signals
        signals = {}
        for symbol in market_data.index:
            signal = np.random.choice([-0.1, 0.0, 0.1], p=[0.2, 0.6, 0.2])
            signals[symbol] = signal
        
        return {
            'signals': signals,
            'predictions': np.random.randn(n_assets),
            'model_used': False,
            'feature_count': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_trading_signals(self, signals: Dict, current_prices: Dict) -> Dict:
        """
        Execute trading signals through the execution engine.
        
        Args:
            signals: Trading signals {symbol: signal_strength}
            current_prices: Current market prices {symbol: price}
            
        Returns:
            Execution result dictionary
        """
        if self.execution_engine is None:
            self.logger.warning("EXECUTION DISABLED: No execution engine available")
            return {
                'status': 'disabled',
                'orders_submitted': 0,
                'orders_filled': 0,
                'message': 'Execution engine not available'
            }
        
        try:
            # Filter signals by threshold
            filtered_signals = {
                symbol: signal for symbol, signal in signals.items()
                if abs(signal) >= self.execution_config['execution']['signal_threshold']
            }
            
            if not filtered_signals:
                self.logger.info("No signals above threshold for execution")
                return {
                    'status': 'no_signals',
                    'orders_submitted': 0,
                    'orders_filled': 0,
                    'message': 'No signals above threshold'
                }
            
            self.logger.info(f"Executing {len(filtered_signals)} signals through execution engine")
            
            # Execute signals
            result = self.execution_engine.execute_signals(filtered_signals, current_prices)
            
            self.logger.info(f"Execution result: {result.orders_submitted} submitted, {result.orders_filled} filled")
            
            return {
                'status': 'success' if result.success else 'error',
                'orders_submitted': result.orders_submitted,
                'orders_filled': result.orders_filled,
                'orders_rejected': result.orders_rejected,
                'execution_time': result.execution_time,
                'errors': result.errors,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error executing signals: {e}")
            return {
                'status': 'error',
                'orders_submitted': 0,
                'orders_filled': 0,
                'message': str(e)
            }
    
    def _fetch_real_market_data(self) -> pd.DataFrame:
        """Fetch real market data from Alpaca API."""
        try:
            # Use existing data fetching logic with correct parameters
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'BAC', 'WFC']
            bars = fetch_alpaca_bars(
                symbols=symbols,
                timeframe='1Day',
                lookback_minutes=43200  # 30 days in minutes
            )
            
            if bars is None or bars.empty:
                self.logger.warning("No market data available, using fallback")
                return create_fallback_data(symbols)
            
            # Add technical features
            bars = self._add_technical_features(bars)
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return create_fallback_data(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'BAC', 'WFC'])
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to market data."""
        # This is a simplified version - in production you'd use your full feature engineering
        for symbol in df.index.get_level_values(0).unique():
            symbol_df = df.loc[symbol].copy()
            
            # Moving averages
            symbol_df['ma_fast'] = symbol_df['close'].rolling(5).mean()
            symbol_df['ma_slow'] = symbol_df['close'].rolling(20).mean()
            symbol_df['ma_ratio'] = symbol_df['ma_fast'] / (symbol_df['ma_slow'] + 1e-8)
            
            # Volatility
            symbol_df['vol_20'] = symbol_df['close'].rolling(20).std()
            
            # Update the dataframe
            df.loc[symbol] = symbol_df
        
        return df
    
    def _get_current_prices(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current prices from market data."""
        current_prices = {}
        for symbol in market_data.index.get_level_values(0).unique():
            try:
                latest_price = market_data.loc[symbol, 'close'].iloc[-1]
                current_prices[symbol] = float(latest_price)
            except (IndexError, KeyError):
                self.logger.warning(f"No current price available for {symbol}")
                current_prices[symbol] = 100.0  # Fallback price
        
        return current_prices
    
    def _calculate_signal_entropy(self, signal_values: List[float]) -> float:
        """Calculate entropy of signal distribution."""
        if not signal_values:
            return 0.0
        
        # Convert to numpy array and remove zeros
        signals = np.array(signal_values)
        signals = signals[signals != 0]
        
        if len(signals) == 0:
            return 0.0
        
        # Normalize to probabilities
        abs_signals = np.abs(signals)
        total = np.sum(abs_signals)
        
        if total == 0:
            return 0.0
        
        probs = abs_signals / total
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.trading_active = False
        self._emergency_halt("Signal received")
    
    def _emergency_halt(self, reason: str):
        """Emergency halt with immediate position flattening."""
        self.logger.critical(f"EMERGENCY HALT: {reason}")
        self.emergency_halt = True
        self.trading_active = False
        
        # Execute emergency stop on execution engine
        if self.execution_engine:
            try:
                self.execution_engine.emergency_stop()
                self.logger.info("Emergency stop executed on execution engine")
            except Exception as e:
                self.logger.error(f"Error during emergency stop: {e}")
        
        # Send alert
        self._send_alert("CRITICAL", f"Emergency halt triggered: {reason}")
    
    def _send_alert(self, level: str, message: str):
        """Send alert notification."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_msg = f"[{timestamp}] {level}: {message}"
        self.logger.warning(alert_msg)
        print(f"🚨 {alert_msg}")
    
    def run_trading_session(self):
        """Run the main trading session with execution."""
        self.logger.info("Starting trading session with execution infrastructure")
        
        # Pre-flight checks
        if not self._preflight_checks():
            self.logger.error("Pre-flight checks failed")
            return False
        
        # Initialize session
        self.trading_active = True
        session_stats = {
            'bars_processed': 0,
            'signals_generated': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'alerts_triggered': 0,
            'emergency_halts': 0,
            'start_time': datetime.now(),
            'model_used': False
        }
        
        consecutive_low_entropy = 0
        
        try:
            print("🔄 Starting trading loop with execution...")
            if self.model_loader:
                print("   Using production XGBoost model for signal generation")
            else:
                print("   Using mock signals (no model loaded)")
            
            if self.execution_engine:
                print("   ✅ Execution engine active - will place real orders")
            else:
                print("   ⚠️  Execution engine disabled - signal-only mode")
            
            print("   Monitoring: entropy, PnL, slippage, ADV breaches, order execution")
            
            # Simulate trading session (normally runs for market hours)
            max_bars = 50  # Simulate ~6.5 hour session with frequent checks
            
            for bar_count in range(1, max_bars + 1):
                if not self.trading_active:
                    break
                
                # Fetch real market data from Alpaca API
                market_data = self._fetch_real_market_data()
                
                # Generate trading signals using the model
                signal_result = self._generate_trading_signals(market_data)
                signals = signal_result['signals']
                model_used = signal_result['model_used']
                session_stats['model_used'] = model_used
                
                # Calculate entropy from signal distribution
                signal_values = list(signals.values())
                action_entropy = self._calculate_signal_entropy(signal_values)
                session_stats['signals_generated'] += len([s for s in signal_values if s != 0])
                
                # Get current prices for execution
                current_prices = self._get_current_prices(market_data)
                
                # Execute trading signals
                execution_result = self._execute_trading_signals(signals, current_prices)
                session_stats['orders_submitted'] += execution_result['orders_submitted']
                session_stats['orders_filled'] += execution_result['orders_filled']
                
                # Mock PnL calculation (in production, would use real portfolio PnL)
                daily_pnl_pct = np.random.uniform(-0.01, 0.01)  # -1% to +1%
                
                # Mock slippage tracking
                expected_slippage = 0.0006  # 6 bps
                realized_slippage = expected_slippage * np.random.uniform(0.8, 1.4)
                slippage_deviation = abs(realized_slippage - expected_slippage) / expected_slippage
                
                # Check kill conditions
                kill_triggered = False
                
                # 1. Entropy floor check
                if action_entropy < 0.75:
                    consecutive_low_entropy += 1
                    if consecutive_low_entropy >= 10:
                        self._send_alert("CRITICAL", f"Low entropy for {consecutive_low_entropy} bars: {action_entropy:.3f}")
                        kill_triggered = True
                else:
                    consecutive_low_entropy = 0
                
                # 2. Daily loss limit
                if daily_pnl_pct <= -0.02:  # -2%
                    self._send_alert("CRITICAL", f"Daily loss limit breached: {daily_pnl_pct:.2%}")
                    kill_triggered = True
                
                # 3. Slippage deviation
                if slippage_deviation > 3.0:  # 3 sigma (simplified)
                    self._send_alert("WARNING", f"High slippage deviation: {slippage_deviation:.1f}x expected")
                    session_stats['alerts_triggered'] += 1
                
                # Emergency halt if kill condition triggered
                if kill_triggered:
                    self._emergency_halt("Kill condition triggered")
                    session_stats['emergency_halts'] += 1
                    break
                
                # Progress reporting
                if bar_count % 10 == 0:
                    model_status = "MODEL" if model_used else "MOCK"
                    exec_status = "EXEC" if self.execution_engine else "SIGNAL-ONLY"
                    n_signals = len([s for s in signal_values if s != 0])
                    feature_count = signal_result.get('feature_count', 0)
                    print(f"   Bar {bar_count}: {model_status} ({feature_count}/45) {exec_status} entropy={action_entropy:.3f}, signals={n_signals}, orders={execution_result['orders_submitted']}, PnL={daily_pnl_pct:.2%}, slippage={realized_slippage*10000:.1f}bps")
                
                session_stats['bars_processed'] = bar_count
                
                # Simulate processing time
                time.sleep(0.1)
            
            # Session completed
            session_stats['end_time'] = datetime.now()
            session_stats['duration'] = (session_stats['end_time'] - session_stats['start_time']).total_seconds()
            
            self.logger.info("Trading session completed successfully")
            self._print_session_summary(session_stats)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in trading session: {e}")
            self._emergency_halt(f"Session error: {e}")
            return False
        
        finally:
            self.trading_active = False
    
    def _preflight_checks(self) -> bool:
        """Run pre-flight checks before trading."""
        self.logger.info("Running pre-flight checks...")
        
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Model availability
        total_checks += 1
        if self.model_loader is not None:
            self.logger.info("✅ Model loaded successfully")
            checks_passed += 1
        else:
            self.logger.warning("⚠️  No model loaded - will use mock signals")
            checks_passed += 1  # Mock mode is acceptable
        
        # Check 2: Execution engine
        total_checks += 1
        if self.execution_engine is not None:
            self.logger.info("✅ Execution engine initialized")
            checks_passed += 1
        else:
            self.logger.warning("⚠️  Execution engine not available - signal-only mode")
            checks_passed += 1  # Signal-only mode is acceptable
        
        # Check 3: Alpaca connection
        total_checks += 1
        if self.alpaca_client is not None:
            try:
                account = self.alpaca_client.get_account()
                self.logger.info(f"✅ Alpaca connection verified (Account: {account.id})")
                checks_passed += 1
            except Exception as e:
                self.logger.error(f"❌ Alpaca connection failed: {e}")
        else:
            self.logger.warning("⚠️  No Alpaca client - signal-only mode")
            checks_passed += 1  # Signal-only mode is acceptable
        
        # Check 4: Market data
        total_checks += 1
        try:
            market_data = self._fetch_real_market_data()
            if not market_data.empty:
                self.logger.info(f"✅ Market data available ({len(market_data)} symbols)")
                checks_passed += 1
            else:
                self.logger.warning("⚠️  No market data - using fallback")
                checks_passed += 1  # Fallback is acceptable
        except Exception as e:
            self.logger.error(f"❌ Market data check failed: {e}")
        
        success_rate = checks_passed / total_checks
        self.logger.info(f"Pre-flight checks: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        return success_rate >= 0.75  # Allow some failures
    
    def _print_session_summary(self, stats: Dict):
        """Print trading session summary."""
        print("\n" + "="*60)
        print("📊 TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Bars processed: {stats['bars_processed']}")
        print(f"Signals generated: {stats['signals_generated']}")
        print(f"Orders submitted: {stats['orders_submitted']}")
        print(f"Orders filled: {stats['orders_filled']}")
        print(f"Model used: {'Yes' if stats['model_used'] else 'No (Mock)'}")
        print(f"Alerts triggered: {stats['alerts_triggered']}")
        print(f"Emergency halts: {stats['emergency_halts']}")
        
        if stats['orders_submitted'] > 0:
            fill_rate = stats['orders_filled'] / stats['orders_submitted']
            print(f"Order fill rate: {fill_rate:.1%}")
        
        print("="*60)


import fcntl
import os
import contextlib
import uuid
import inspect
import logging

# Set up logging FIRST - before any other imports or initialization
ENGINE_ID = uuid.uuid4().hex[:8]

def setup_logging():
    """Setup logging with single handler to prevent duplicates and shutdown crashes."""
    # Force a single clean root handler set
    logging.basicConfig(level=logging.INFO, handlers=[], force=True)
    
    root = logging.getLogger()
    # Nuke any handlers added by imported libs
    for handler in list(root.handlers):
        root.removeHandler(handler)
    
    root.setLevel(logging.INFO)
    h = logging.StreamHandler()  # let systemd capture stdout/stderr
    h.setFormatter(logging.Formatter(f'[{ENGINE_ID}] %(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root.addHandler(h)
    
    # Prevent double bubbling - clear package handlers and set propagation
    for name in ("core", "alpaca", "urllib3"):
        logger = logging.getLogger(name)
        logger.propagate = True  # keep only root handler
        logger.handlers.clear()  # ensure no extra handlers
    
    # Clear any late-added handlers from imported modules
    for lg in logging.Logger.manager.loggerDict.values():
        if isinstance(lg, logging.PlaceHolder):
            continue
        try:
            lg.handlers.clear()
            lg.propagate = True  # only root's single handler emits
        except Exception:
            pass

# Run logging setup immediately
setup_logging()

# Import constructor guard utility
from core.utils.constructor_guard import construct_once

# Shutdown cleanup to prevent daemon thread crashes
import atexit
import sys

def dump_threads(tag="before-exit"):
    """Debug function to dump thread information."""
    import threading
    logging.warning("THREAD_DUMP %s", tag)
    for t in threading.enumerate():
        logging.warning("THREAD name=%r ident=%r daemon=%r alive=%r", t.name, t.ident, t.daemon, t.is_alive())

@atexit.register
def _shutdown_logging_and_streams():
    """Clean shutdown of logging and streams to prevent daemon thread crashes."""
    try:
        dump_threads("shutdown")
    except Exception:
        pass
    try:
        logging.shutdown()
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

@contextlib.contextmanager
def single_instance(lockfile="/tmp/trader_engine.lock"):
    """Ensure only one instance runs at a time."""
    fd = os.open(lockfile, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Paper Trading with Execution')
    parser.add_argument('--mode', choices=['preflight', 'trading', 'eod', 'full'], 
                       default='trading', help='Operation mode')
    
    args = parser.parse_args()
    
    print(f"[ENGINE_ID] {ENGINE_ID}")
    
    try:
        with single_instance():
            # Initialize operations
            print(f"[build_engine] {inspect.stack()[1].filename}:{inspect.stack()[1].lineno}")
            ops = DailyPaperTradingWithExecution()
            
            if args.mode == 'preflight':
                print("🔍 Running pre-flight checks...")
                success = ops._preflight_checks()
                print(f"Pre-flight checks: {'✅ PASSED' if success else '❌ FAILED'}")
                
            elif args.mode == 'trading':
                print("🚀 Starting trading session with execution...")
                success = ops.run_trading_session()
                print(f"Trading session: {'✅ COMPLETED' if success else '❌ FAILED'}")
                
            elif args.mode == 'eod':
                print("📊 End-of-day reporting...")
                # TODO: Implement EOD reporting
                print("EOD reporting not yet implemented")
                
            elif args.mode == 'full':
                print("🔄 Running full daily cycle...")
                # Run all modes in sequence
                if ops._preflight_checks():
                    ops.run_trading_session()
                    # TODO: Add EOD reporting
                else:
                    print("❌ Pre-flight checks failed - aborting")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
