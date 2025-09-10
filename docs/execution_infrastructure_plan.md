# Execution Infrastructure Implementation Plan

## Overview
Transform the current signal generation system into a complete execution infrastructure that places real orders on Alpaca and manages a paper trading portfolio.

## Current State Analysis
- ✅ **Signal Generation**: Production XGBoost model generating 53 signals per bar
- ✅ **Data Pipeline**: Real Alpaca data → Feature Engineering → Model → Signals
- ✅ **Safety Systems**: Feature gates, entropy monitoring, emergency halts
- ❌ **Order Execution**: Signals generated but not converted to orders
- ❌ **Portfolio Management**: No position tracking or risk management
- ❌ **Trade Reconciliation**: No order status monitoring or reporting

## Architecture Design

### 1. Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Signal Gen    │───▶│  Order Manager   │───▶│  Alpaca API     │
│   (Existing)    │    │  (New)           │    │  (Existing)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Portfolio Manager│
                       │ (New)            │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Risk Manager     │
                       │ (New)            │
                       └──────────────────┘
```

### 2. Data Flow

```
Signals → Position Sizing → Risk Checks → Order Generation → Execution → Monitoring
```

## Implementation Plan

### Phase 1: Order Management System (Week 1)

#### 1.1 Order Types and States
```python
# core/execution/order_types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    alpaca_order_id: Optional[str] = None
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
```

#### 1.2 Order Manager
```python
# core/execution/order_manager.py
class OrderManager:
    def __init__(self, alpaca_api: REST):
        self.alpaca_api = alpaca_api
        self.pending_orders = {}
        self.filled_orders = {}
        self.logger = logging.getLogger(__name__)
    
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca and track it."""
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status from Alpaca."""
        
    def reconcile_orders(self) -> None:
        """Sync local order state with Alpaca."""
```

### Phase 2: Position Sizing and Risk Management (Week 1-2)

#### 2.1 Position Sizing Engine
```python
# core/execution/position_sizing.py
class PositionSizer:
    def __init__(self, config: dict):
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_total_exposure = config.get('max_total_exposure', 0.8)  # 80% of portfolio
        self.min_trade_size = config.get('min_trade_size', 100)  # $100 minimum
        
    def calculate_position_size(self, signal: float, symbol: str, 
                              portfolio_value: float, current_positions: dict) -> int:
        """Calculate position size based on signal strength and risk limits."""
        
    def validate_position(self, symbol: str, quantity: int, 
                         portfolio_value: float, current_positions: dict) -> bool:
        """Validate position against risk limits."""
```

#### 2.2 Risk Manager
```python
# core/execution/risk_manager.py
class RiskManager:
    def __init__(self, config: dict):
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2%
        self.max_position_risk = config.get('max_position_risk', 0.05)  # 5%
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30%
        
    def check_risk_limits(self, proposed_order: Order, 
                         current_positions: dict, daily_pnl: float) -> bool:
        """Check if order violates risk limits."""
        
    def calculate_portfolio_risk(self, positions: dict) -> dict:
        """Calculate current portfolio risk metrics."""
```

### Phase 3: Portfolio Management (Week 2)

#### 3.1 Portfolio Manager
```python
# core/execution/portfolio_manager.py
class PortfolioManager:
    def __init__(self, alpaca_api: REST):
        self.alpaca_api = alpaca_api
        self.positions = {}
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.logger = logging.getLogger(__name__)
    
    def update_positions(self) -> None:
        """Sync positions with Alpaca."""
        
    def get_position(self, symbol: str) -> dict:
        """Get current position for symbol."""
        
    def calculate_portfolio_metrics(self) -> dict:
        """Calculate portfolio performance metrics."""
        
    def rebalance_portfolio(self, target_weights: dict) -> list:
        """Generate rebalancing orders."""
```

### Phase 4: Signal-to-Order Conversion (Week 2-3)

#### 4.1 Signal Processor
```python
# core/execution/signal_processor.py
class SignalProcessor:
    def __init__(self, position_sizer: PositionSizer, risk_manager: RiskManager):
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
    
    def process_signals(self, signals: dict, current_positions: dict, 
                       portfolio_value: float) -> list:
        """Convert signals to orders with risk checks."""
        
    def generate_rebalancing_orders(self, current_positions: dict, 
                                   target_positions: dict) -> list:
        """Generate orders to rebalance portfolio."""
```

### Phase 5: Execution Engine Integration (Week 3)

#### 5.1 Execution Engine
```python
# core/execution/execution_engine.py
class ExecutionEngine:
    def __init__(self, order_manager: OrderManager, portfolio_manager: PortfolioManager):
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.logger = logging.getLogger(__name__)
    
    def execute_signals(self, signals: dict) -> dict:
        """Main execution loop: signals → orders → execution."""
        
    def monitor_execution(self) -> None:
        """Monitor order execution and update positions."""
        
    def handle_order_updates(self, order_updates: list) -> None:
        """Process order status updates from Alpaca."""
```

### Phase 6: Integration with Daily Paper Trading (Week 3-4)

#### 6.1 Modified Daily Paper Trading
```python
# ops/daily_paper_trading.py (modifications)
class DailyPaperTradingOperations:
    def __init__(self):
        # ... existing initialization ...
        
        # Add execution components
        self.execution_engine = ExecutionEngine(
            order_manager=OrderManager(self.alpaca_api),
            portfolio_manager=PortfolioManager(self.alpaca_api)
        )
        
    def _execute_trading_signals(self, signals: dict) -> dict:
        """Execute signals through the execution engine."""
        try:
            execution_result = self.execution_engine.execute_signals(signals)
            return execution_result
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return {"status": "error", "message": str(e)}
```

## Configuration

### 6.1 Execution Configuration
```yaml
# config/execution.yaml
execution:
  enabled: true
  mode: "paper"  # paper, live
  
position_sizing:
  max_position_size: 0.1  # 10% of portfolio per position
  max_total_exposure: 0.8  # 80% of portfolio total
  min_trade_size: 100  # $100 minimum trade
  signal_threshold: 0.1  # Minimum signal strength to trade
  
risk_management:
  max_daily_loss: 0.02  # 2% daily loss limit
  max_position_risk: 0.05  # 5% per position risk
  max_sector_exposure: 0.3  # 30% sector exposure limit
  stop_loss_pct: 0.05  # 5% stop loss
  
order_management:
  default_order_type: "limit"
  limit_price_offset: 0.001  # 0.1% offset from market price
  time_in_force: "day"
  max_orders_per_bar: 10
  
monitoring:
  order_reconciliation_interval: 30  # seconds
  position_update_interval: 60  # seconds
  risk_check_interval: 10  # seconds
```

## Testing Strategy

### 7.1 Unit Tests
- Order creation and validation
- Position sizing calculations
- Risk limit checks
- Portfolio metrics calculations

### 7.2 Integration Tests
- End-to-end signal → order → execution
- Order status reconciliation
- Portfolio synchronization
- Risk limit enforcement

### 7.3 Paper Trading Tests
- Small position sizes ($100-500)
- Limited number of symbols (5-10)
- Short time periods (1-2 hours)
- Manual monitoring and validation

## Implementation Timeline

### Week 1: Foundation
- [ ] Order types and data structures
- [ ] Order manager with Alpaca integration
- [ ] Basic position sizing logic
- [ ] Unit tests for core components

### Week 2: Risk and Portfolio
- [ ] Risk management system
- [ ] Portfolio manager
- [ ] Position tracking and reconciliation
- [ ] Integration tests

### Week 3: Execution Engine
- [ ] Signal-to-order conversion
- [ ] Execution engine
- [ ] Order monitoring and updates
- [ ] End-to-end testing

### Week 4: Integration and Testing
- [ ] Integration with daily paper trading
- [ ] Configuration management
- [ ] Paper trading validation
- [ ] Documentation and monitoring

## Success Criteria

### Functional Requirements
- [ ] Convert signals to orders with proper position sizing
- [ ] Execute orders on Alpaca paper trading
- [ ] Track positions and portfolio value
- [ ] Enforce risk limits and stop losses
- [ ] Reconcile order status with Alpaca
- [ ] Generate execution reports

### Performance Requirements
- [ ] Order execution within 5 seconds of signal generation
- [ ] Position updates within 1 minute
- [ ] Risk checks every 10 seconds
- [ ] Handle up to 100 concurrent orders

### Safety Requirements
- [ ] Never exceed position size limits
- [ ] Never exceed daily loss limits
- [ ] Emergency stop on system errors
- [ ] Complete audit trail of all orders
- [ ] Manual override capabilities

## Risk Mitigation

### 1. Gradual Rollout
- Start with small position sizes
- Limited number of symbols
- Manual monitoring required
- Easy rollback to signal-only mode

### 2. Safety Mechanisms
- Multiple risk checks before order submission
- Position size validation
- Daily loss limits
- Emergency stop functionality

### 3. Monitoring and Alerts
- Real-time order status monitoring
- Portfolio risk metrics
- Execution performance tracking
- Error rate monitoring

## Next Steps

1. **Review and approve this plan**
2. **Set up development environment**
3. **Implement Phase 1: Order Management System**
4. **Create unit tests for each component**
5. **Begin integration testing with small positions**

This execution infrastructure will transform the current signal generation system into a complete automated trading system while maintaining all existing safety mechanisms.
