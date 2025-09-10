"""
Test integration of execution infrastructure with daily paper trading.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.daily_paper_trading_with_execution import DailyPaperTradingWithExecution


class TestExecutionIntegrationDailyTrading:
    """Test integration between execution infrastructure and daily trading."""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_api_key',
            'ALPACA_SECRET_KEY': 'test_secret_key'
        }):
            yield
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create mock Alpaca client."""
        client = Mock()
        client.get_account.return_value = Mock(id="test_account", cash=100000.0)
        client.get_all_positions.return_value = []
        client.get_all_orders.return_value = []
        client.submit_order.return_value = Mock(id="order_123")
        return client
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': 100.0 + np.random.randn() * 5,
                    'high': 105.0 + np.random.randn() * 5,
                    'low': 95.0 + np.random.randn() * 5,
                    'close': 100.0 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randint(0, 500000)
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(['symbol', 'timestamp'])
        return df
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_initialization_with_execution(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client):
        """Test initialization with execution infrastructure."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Verify execution components are initialized
        assert ops.alpaca_client is not None
        assert ops.order_manager is not None
        assert ops.portfolio_manager is not None
        assert ops.position_sizer is not None
        assert ops.risk_manager is not None
        assert ops.execution_engine is not None
        
        # Verify configuration is loaded
        assert ops.execution_config is not None
        assert 'execution' in ops.execution_config
        assert 'position_sizing' in ops.execution_config
        assert 'risk_management' in ops.execution_config
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_initialization_without_credentials(self, mock_model_loader, mock_trading_client):
        """Test initialization without Alpaca credentials (fallback mode)."""
        # Setup mocks to simulate missing credentials
        mock_trading_client.side_effect = Exception("No credentials")
        mock_model_loader.return_value = Mock()
        
        # Initialize (should not crash)
        ops = DailyPaperTradingWithExecution()
        
        # Verify fallback behavior
        assert ops.alpaca_client is None
        assert ops.execution_engine is None
        assert ops.model_loader is not None  # Model should still work
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_signal_generation(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client, mock_market_data):
        """Test signal generation with execution infrastructure."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, -0.6, 0.2])
        # Use realistic feature names that exist in the mock data
        mock_model.features_whitelist = ['close', 'volume']
        mock_model_loader.return_value = mock_model
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test signal generation
        with patch.object(ops, '_fetch_real_market_data', return_value=mock_market_data):
            signal_result = ops._generate_trading_signals(mock_market_data)
        
        # Verify signal generation
        assert 'signals' in signal_result
        assert 'model_used' in signal_result
        assert signal_result['model_used'] is True
        assert len(signal_result['signals']) > 0
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_signal_execution(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client):
        """Test signal execution through execution engine."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test signal execution
        signals = {'AAPL': 0.8, 'MSFT': -0.6, 'GOOGL': 0.05}
        current_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2800.0}
        
        result = ops._execute_trading_signals(signals, current_prices)
        
        # Verify execution result
        assert 'status' in result
        assert 'orders_submitted' in result
        assert 'orders_filled' in result
        assert result['orders_submitted'] >= 0
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_execution_without_engine(self, mock_model_loader, mock_trading_client):
        """Test signal execution when execution engine is not available."""
        # Setup mocks to simulate no execution engine
        mock_trading_client.side_effect = Exception("No credentials")
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test signal execution (should fallback gracefully)
        signals = {'AAPL': 0.8, 'MSFT': -0.6}
        current_prices = {'AAPL': 150.0, 'MSFT': 300.0}
        
        result = ops._execute_trading_signals(signals, current_prices)
        
        # Verify fallback behavior
        assert result['status'] == 'disabled'
        assert result['orders_submitted'] == 0
        assert 'message' in result
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_preflight_checks(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client):
        """Test pre-flight checks."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test pre-flight checks
        with patch.object(ops, '_fetch_real_market_data', return_value=pd.DataFrame({'close': [100.0]})):
            success = ops._preflight_checks()
        
        # Verify checks pass
        assert success is True
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_emergency_halt(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client):
        """Test emergency halt functionality."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test emergency halt
        ops._emergency_halt("Test emergency")
        
        # Verify emergency state
        assert ops.emergency_halt is True
        assert ops.trading_active is False
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_current_prices_extraction(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client, mock_market_data):
        """Test current prices extraction from market data."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test price extraction
        current_prices = ops._get_current_prices(mock_market_data)
        
        # Verify prices are extracted
        assert len(current_prices) > 0
        assert all(isinstance(price, float) for price in current_prices.values())
        assert all(price > 0 for price in current_prices.values())
    
    @patch('ops.daily_paper_trading_with_execution.TradingClient')
    @patch('ops.daily_paper_trading_with_execution.XGBModelLoader')
    def test_signal_entropy_calculation(self, mock_model_loader, mock_trading_client, mock_env_vars, mock_alpaca_client):
        """Test signal entropy calculation."""
        # Setup mocks
        mock_trading_client.return_value = mock_alpaca_client
        mock_model_loader.return_value = Mock()
        
        # Initialize
        ops = DailyPaperTradingWithExecution()
        
        # Test entropy calculation
        signal_values = [0.8, -0.6, 0.2, 0.0, -0.1]
        entropy = ops._calculate_signal_entropy(signal_values)
        
        # Verify entropy calculation
        assert isinstance(entropy, float)
        assert entropy >= 0.0
        assert entropy <= 2.0  # Reasonable upper bound for entropy
    
    def test_configuration_loading(self):
        """Test execution configuration loading."""
        with patch('ops.daily_paper_trading_with_execution.TradingClient'), \
             patch('ops.daily_paper_trading_with_execution.XGBModelLoader'):
            
            # Test with existing config file
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open_config()):
                
                ops = DailyPaperTradingWithExecution()
                assert ops.execution_config is not None
                assert 'execution' in ops.execution_config


def mock_open_config():
    """Mock configuration file content."""
    import yaml
    config_content = {
        'execution': {
            'enabled': True,
            'mode': 'paper',
            'signal_threshold': 0.1
        },
        'position_sizing': {
            'max_position_size': 0.1,
            'min_trade_size': 100.0
        },
        'risk_management': {
            'max_daily_loss': 0.02,
            'max_position_risk': 0.05
        }
    }
    
    from unittest.mock import mock_open
    return mock_open(read_data=yaml.dump(config_content))
