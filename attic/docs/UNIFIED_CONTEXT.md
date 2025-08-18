# 🎯 **UNIFIED CODEBASE CONTEXT**

## **📋 Overview**
This document provides a unified reference for naming conventions, patterns, and structures across the entire trading system codebase to ensure consistency and reduce code rewriting.

---

## **🏗️ NAMING CONVENTIONS**

### **1. Classes (PascalCase)**
```python
# Core Engine Classes
PaperTradingEngine
BacktestEngine
PortfolioState
TradeBook
TradeRecord

# Strategy Classes
BaseStrategy
StrategyParams
EnsembleStrategy
RegimeAwareEnsembleStrategy
MeanReversionStrategy
MomentumStrategy
SMACrossoverStrategy

# Feature Classes
FeatureEngine
FeatureConfig
FeatureReweighter
AdaptiveFeatureEngine

# Utility Classes
TradingLogger
DiscordNotifier
RegimeDetector
StrategyFactory
```

### **2. Functions/Methods (snake_case)**
```python
# Core Engine Methods
run_trading_cycle()
run_backtest()
load_config()
initialize_components()
get_market_data()
detect_regime()
generate_signals()
execute_trades()
update_portfolio()
calculate_performance_metrics()

# Strategy Methods
generate_signals()
get_default_params()
get_param_ranges()
validate_params()
get_description()
backtest()

# Portfolio Methods
value_at()
mark_to_market()
execute_order()
close_all_positions()
get_position()
get_summary()

# Trade Logger Methods
on_buy()
on_sell()
reset()
mark_drawdown()
get_open_positions()
get_closed_trades()
get_ledger()
get_trades()
export_trades_csv()

# Utility Methods
setup_logging()
normalize_prices()
apply_slippage()
calculate_drawdown()
validate_trade()
calculate_returns()
ensure_directories()
safe_divide()
clean_dataframe()
format_percentage()
format_currency()
```

### **3. Variables (snake_case)**
```python
# Core State Variables
initial_capital
current_capital
portfolio_value
cash_balance
positions
trade_history
daily_returns
equity_curve
performance_metrics

# Trading Variables
symbol
quantity
price
side  # "BUY" or "SELL"
order_type
order_id
trade_id
entry_date
exit_date
realized_pnl
unrealized_pnl
fees_paid

# Strategy Variables
signal_strength
confidence_threshold
position_sizing_multiplier
stop_loss_multiplier
take_profit_multiplier
regime_name
regime_confidence

# Feature Variables
feature_weights
feature_importance
feature_performance
rolling_window
reweight_frequency
decay_factor

# Configuration Variables
config_file
profile_file
log_file
data_dir
results_dir
```

### **4. Constants (ALL_CAPS)**
```python
# Trading Constants
DEFAULT_COMMISSION_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0
MIN_HISTORY_DAYS = 60
MAX_POSITION_SIZE_PCT = 0.1
MIN_TRADE_SIZE = 100.0
MAX_DRAWDOWN_LIMIT = 0.2

# Time Constants
TRADING_DAYS_PER_YEAR = 252
WARMUP_PERIOD_DAYS = 200
LOOKBACK_PERIOD_DAYS = 252

# Feature Constants
DEFAULT_RSI_PERIOD = 14
DEFAULT_ATR_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
```

---

## **📊 DATA STRUCTURES**

### **1. DataFrame Columns**
```python
# Price Data Columns
"Open", "High", "Low", "Close", "Volume"
"Adj_Close", "Dividends", "Stock_Splits"

# Feature Columns
"ret1", "ma20", "vol20", "zscore20"
"rsi_14", "atr_14", "bb_position"
"trend_strength", "volatility_regime"
"regime_confidence", "signal_strength"

# Trade Columns
"date", "symbol", "side", "quantity", "price"
"value", "fees", "realized_pnl", "regime"
"entry_date", "exit_date", "entry_price", "exit_price"
```

### **2. Dictionary Keys**
```python
# Configuration Keys
"symbols", "initial_capital", "commission_bps"
"max_position_size", "stop_loss_pct", "take_profit_pct"
"regime_detection", "feature_engineering", "risk_management"

# Performance Keys
"total_return", "annualized_return", "volatility"
"sharpe_ratio", "max_drawdown", "win_rate"
"profit_factor", "total_trades", "avg_trade_size"

# Trade Keys
"symbol", "side", "quantity", "price", "value"
"fees", "realized_pnl", "entry_date", "exit_date"
"regime", "confidence", "signal_strength"
```

### **3. Type Hints**
```python
# Common Type Hints
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Function Signatures
def generate_signals(df: pd.DataFrame) -> pd.Series:
def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
def execute_order(order: Dict[str, Any]) -> str:
def validate_params(params: Dict[str, Any]) -> bool:
def setup_logging(log_file: str, level: int = logging.INFO) -> logging.Logger:
```

---

## **🔧 FUNCTION PATTERNS**

### **1. Strategy Interface**
```python
class BaseStrategy(ABC):
    def __init__(self, params: StrategyParams):
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from price data."""
        pass

    @abstractmethod
    def get_default_params(self) -> StrategyParams:
        """Return default parameters."""
        pass

    @abstractmethod
    def get_param_ranges(self) -> Dict[str, Any]:
        """Return parameter ranges for optimization."""
        pass

    def validate_params(self, params: StrategyParams) -> bool:
        """Validate strategy parameters."""
        pass

    def get_description(self) -> str:
        """Return strategy description."""
        pass
```

### **2. Engine Interface**
```python
class TradingEngine:
    def __init__(self, config_file: str, profile_file: str = None):
        self.config = self.load_config(config_file)
        self.initialize_components()

    def run_trading_cycle(self, date: str) -> Dict[str, Any]:
        """Run a single trading cycle."""
        pass

    def get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get market data for symbols."""
        pass

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals."""
        pass

    def execute_trades(self, signals: Dict[str, float]) -> List[Dict]:
        """Execute trades based on signals."""
        pass

    def update_portfolio(self, trades: List[Dict]):
        """Update portfolio with executed trades."""
        pass
```

### **3. Utility Function Patterns**
```python
# Data Processing
def normalize_prices(prices: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
def calculate_returns(close: pd.Series, shift: int = -1) -> pd.Series:

# Performance Calculation
def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
def calculate_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> float:
def calculate_sharpe_ratio(returns: pd.Series) -> float:

# Trade Processing
def apply_slippage(price: float, quantity: float, slippage_bps: float) -> float:
def validate_trade(symbol: str, quantity: float, price: float, cash: float) -> Tuple[bool, str]:
def calculate_commission(value: float, commission_bps: float) -> float:

# Formatting
def format_percentage(value: float, decimals: int = 2) -> str:
def format_currency(value: float, decimals: int = 2) -> str:
def format_timestamp(timestamp: Union[str, datetime]) -> str:
```

---

## **📁 MODULE STRUCTURE**

### **1. Core Module (`core/`)**
```
core/
├── engine/
│   ├── paper.py          # Paper trading engine
│   └── backtest.py       # Backtesting engine
├── portfolio.py          # Portfolio management
├── trade_logger.py       # Trade logging
├── utils.py             # Utility functions
├── factory.py           # Component factory
├── strategy.py          # Strategy base
├── regime_detector.py   # Regime detection
├── feature_reweighter.py # Feature engineering
├── enhanced_logging.py  # Enhanced logging
├── notifications.py     # Notifications
├── performance.py       # Performance metrics
└── walk/                # Walk-forward analysis
    ├── pipeline.py
    ├── run.py
    └── folds.py
```

### **2. Strategies Module (`strategies/`)**
```
strategies/
├── base.py                    # Strategy base class
├── factory.py                 # Strategy factory
├── ensemble_strategy.py       # Ensemble strategy
├── regime_aware_ensemble.py   # Regime-aware ensemble
├── mean_reversion.py          # Mean reversion strategy
├── momentum.py                # Momentum strategy
└── sma_crossover.py           # SMA crossover strategy
```

### **3. Features Module (`features/`)**
```
features/
├── feature_engine.py          # Feature engineering
└── ensemble.py                # Feature ensemble
```

### **4. Brokers Module (`brokers/`)**
```
brokers/
├── data_provider.py           # Data provider
└── ibkr_broker.py             # IBKR broker
```

---

## **🎨 CODE PATTERNS**

### **1. Error Handling**
```python
try:
    # Operation
    result = perform_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return fallback_value
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### **2. Logging**
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Operation started")
logger.debug(f"Processing data: {data.shape}")
logger.warning("Potential issue detected")
logger.error("Operation failed")
```

### **3. Configuration Loading**
```python
def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        return {}
```

### **4. Data Validation**
```python
def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return False
    return True

def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> bool:
    """Validate numeric value is within range."""
    if not (min_val <= value <= max_val):
        logger.error(f"{name} must be between {min_val} and {max_val}, got {value}")
        return False
    return True
```

---

## **🔗 COMMON IMPORTS**

### **1. Standard Library**
```python
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
```

### **2. Third Party**
```python
import numpy as np
import pandas as pd
import yfinance as yf
```

### **3. Local Imports**
```python
from core.utils import setup_logging, calculate_returns
from core.portfolio import PortfolioState
from core.trade_logger import TradeBook
from strategies.base import BaseStrategy, StrategyParams
from features.feature_engine import FeatureEngine
```

---

## **📝 DOCUMENTATION PATTERNS**

### **1. Function Docstrings**
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: default_value)

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised

    Example:
        >>> function_name("example", 42)
        "expected_output"
    """
    pass
```

### **2. Class Docstrings**
```python
class ClassName:
    """
    Brief description of the class.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Methods:
        method1: Brief description of method1
        method2: Brief description of method2

    Example:
        >>> instance = ClassName()
        >>> instance.method1()
        "expected_output"
    """
    pass
```

---

## **🎯 USAGE GUIDELINES**

### **1. When Adding New Code**
1. **Follow naming conventions** - Use snake_case for functions/variables, PascalCase for classes
2. **Add type hints** - Include parameter and return type annotations
3. **Write docstrings** - Document purpose, parameters, returns, and examples
4. **Add logging** - Use appropriate log levels for debugging and monitoring
5. **Handle errors** - Use try/except blocks with specific exception types
6. **Validate inputs** - Check data types, ranges, and required fields

### **2. When Modifying Existing Code**
1. **Preserve interfaces** - Don't change public method signatures without deprecation
2. **Maintain compatibility** - Ensure existing functionality continues to work
3. **Update documentation** - Keep docstrings and comments current
4. **Add tests** - Include tests for new functionality or bug fixes

### **3. When Creating New Modules**
1. **Follow module structure** - Place code in appropriate directories
2. **Use consistent imports** - Follow the import patterns shown above
3. **Include __init__.py** - Make modules importable packages
4. **Add module docstring** - Document the module's purpose and contents

---

## **✅ VERIFICATION CHECKLIST**

Before submitting code, ensure:
- [ ] All functions use snake_case naming
- [ ] All classes use PascalCase naming
- [ ] All constants use ALL_CAPS naming
- [ ] Type hints are included for all functions
- [ ] Docstrings are present and complete
- [ ] Error handling is appropriate
- [ ] Logging is used appropriately
- [ ] Code follows the patterns shown above
- [ ] No hardcoded magic numbers (use constants)
- [ ] Consistent import organization

---

**📚 This document serves as the single source of truth for codebase conventions and patterns. Refer to it when writing or modifying code to ensure consistency across the entire trading system.**
