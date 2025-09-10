# Technical Implementation Guide: Alpha Source Architecture

## 🏗️ **System Architecture Overview**

This guide provides the technical implementation details for transforming Aurora from a differentiator-level system to an alpha-source territory platform.

---

## 📊 **Current Architecture vs. Target Architecture**

### **Current Architecture (Differentiator Level)**
```
Data Pipeline → Feature Engineering → XGBoost → Position Sizing → Execution
```

### **Target Architecture (Alpha Source Level)**
```
Data Pipeline → Multi-View Features → Model Zoo → Ensemble → Calibration → Policy Layer → Execution
```

---

## 🔧 **Phase 1: Feature Expansion Implementation**

### **1.1 Sector Neutralization Engine**

#### **Core Implementation**
```python
# core/features/sector_neutralizer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class SectorNeutralizer:
    def __init__(self, sector_mapping: Dict[str, str]):
        self.sector_mapping = sector_mapping
        self.sector_returns = None
        
    def fit(self, returns: pd.DataFrame) -> 'SectorNeutralizer':
        """Calculate sector returns from stock returns"""
        returns_with_sector = returns.copy()
        returns_with_sector['sector'] = returns_with_sector.index.map(self.sector_mapping)
        
        # Calculate sector returns (equal-weighted)
        self.sector_returns = returns_with_sector.groupby('sector').mean()
        return self
    
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Residualize stock returns against sector returns"""
        if self.sector_returns is None:
            raise ValueError("Must fit before transform")
            
        residual_returns = returns.copy()
        
        for symbol in returns.columns:
            sector = self.sector_mapping.get(symbol)
            if sector and sector in self.sector_returns.index:
                sector_return = self.sector_returns.loc[sector]
                residual_returns[symbol] = returns[symbol] - sector_return
        
        return residual_returns
    
    def create_sector_neutral_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create sector-neutral features"""
        residual_returns = self.transform(returns)
        
        features = pd.DataFrame(index=returns.index)
        
        # Sector-neutral momentum
        features['momentum_5_20_sector_neutral'] = (
            residual_returns.rolling(5).mean() / 
            residual_returns.rolling(20).std()
        )
        
        # Sector-neutral volatility
        features['vol_20_sector_neutral'] = residual_returns.rolling(20).std()
        
        # Sector-neutral mean reversion
        features['mean_reversion_5_sector_neutral'] = (
            residual_returns.rolling(5).mean() / 
            residual_returns.rolling(20).mean()
        )
        
        return features
```

#### **Integration with Existing Pipeline**
```python
# ml/panel_builder.py (modifications)
from core.features.sector_neutralizer import SectorNeutralizer

class PanelBuilder:
    def __init__(self, config):
        self.config = config
        self.sector_neutralizer = SectorNeutralizer(config.sector_mapping)
        
    def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Existing features
        features = self._build_existing_features(data)
        
        # Add sector-neutral features
        if hasattr(self.config, 'sector_neutralization') and self.config.sector_neutralization:
            sector_features = self.sector_neutralizer.create_sector_neutral_features(data['returns'])
            features = pd.concat([features, sector_features], axis=1)
        
        return features
```

### **1.2 Fundamental Features Engine**

#### **Core Implementation**
```python
# core/features/fundamental_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf

class FundamentalEngine:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.fundamental_data = {}
        
    def fetch_fundamental_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch fundamental data for all symbols"""
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract key fundamental metrics
                self.fundamental_data[symbol] = {
                    'eps_surprise': info.get('earningsQuarterlyGrowth', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0)
                }
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                self.fundamental_data[symbol] = {}
        
        return self.fundamental_data
    
    def create_fundamental_features(self, date: pd.Timestamp) -> pd.DataFrame:
        """Create fundamental features for a given date"""
        features = pd.DataFrame(index=self.symbols)
        
        for symbol in self.symbols:
            if symbol in self.fundamental_data:
                data = self.fundamental_data[symbol]
                
                # EPS surprise (normalized)
                features.loc[symbol, 'eps_surprise'] = data.get('eps_surprise', 0)
                
                # Revenue growth (normalized)
                features.loc[symbol, 'revenue_growth'] = data.get('revenue_growth', 0)
                
                # Profitability metrics
                features.loc[symbol, 'profit_margin'] = data.get('profit_margin', 0)
                features.loc[symbol, 'roe'] = data.get('roe', 0)
                
                # Valuation metrics
                features.loc[symbol, 'pe_ratio'] = data.get('pe_ratio', 0)
                features.loc[symbol, 'pb_ratio'] = data.get('pb_ratio', 0)
                
                # Financial health
                features.loc[symbol, 'debt_to_equity'] = data.get('debt_to_equity', 0)
        
        # Cross-sectional z-scores
        for col in features.columns:
            features[f'{col}_zscore'] = (features[col] - features[col].mean()) / features[col].std()
        
        return features
```

---

## 🤖 **Phase 2: Model Zoo Implementation**

### **2.1 Model Factory**

#### **Core Implementation**
```python
# core/models/model_factory.py
from typing import Dict, Any, List
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

class ModelFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_models(self) -> Dict[str, Any]:
        """Create all models in the zoo"""
        models = {}
        
        # Tree-based models
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 6),
            random_state=42
        )
        
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 6),
            random_state=42,
            verbose=-1
        )
        
        models['catboost'] = CatBoostRegressor(
            iterations=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            depth=self.config.get('max_depth', 6),
            random_seed=42,
            verbose=False
        )
        
        # Linear models
        models['lasso'] = Lasso(
            alpha=self.config.get('lasso_alpha', 0.01),
            random_state=42
        )
        
        models['ridge'] = Ridge(
            alpha=self.config.get('ridge_alpha', 1.0),
            random_state=42
        )
        
        models['elastic_net'] = ElasticNet(
            alpha=self.config.get('elastic_alpha', 0.01),
            l1_ratio=self.config.get('elastic_l1_ratio', 0.5),
            random_state=42
        )
        
        # Ensemble models
        models['random_forest'] = RandomForestRegressor(
            n_estimators=self.config.get('rf_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            random_state=42
        )
        
        return models
```

### **2.2 Model Training Pipeline**

#### **Core Implementation**
```python
# core/models/training_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

class ModelTrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_performance = {}
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models with walk-forward validation"""
        # Create models
        model_factory = ModelFactory(self.config)
        self.models = model_factory.create_models()
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'mse': [],
                'mae': [],
                'ic': [],
                'sharpe': []
            }
            
            # Walk-forward training
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                ic = np.corrcoef(y_val, y_pred)[0, 1]
                
                # Store metrics
                self.model_performance[model_name]['mse'].append(mse)
                self.model_performance[model_name]['mae'].append(mae)
                self.model_performance[model_name]['ic'].append(ic)
            
            # Retrain on full dataset
            model.fit(X, y)
            
            # Save model
            self._save_model(model, model_name)
        
        return self.models
    
    def _save_model(self, model: Any, model_name: str) -> None:
        """Save trained model"""
        model_dir = f"models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/model.pkl")
        
        # Save performance metrics
        performance_df = pd.DataFrame(self.model_performance[model_name])
        performance_df.to_csv(f"{model_dir}/performance.csv")
    
    def get_model_rankings(self) -> pd.DataFrame:
        """Get model performance rankings"""
        rankings = []
        
        for model_name, metrics in self.model_performance.items():
            rankings.append({
                'model': model_name,
                'avg_ic': np.mean(metrics['ic']),
                'avg_mse': np.mean(metrics['mse']),
                'avg_mae': np.mean(metrics['mae']),
                'ic_std': np.std(metrics['ic'])
            })
        
        return pd.DataFrame(rankings).sort_values('avg_ic', ascending=False)
```

---

## 🎯 **Phase 3: Ensemble Implementation**

### **3.1 IC-Weighted Ensemble**

#### **Core Implementation**
```python
# core/ensemble/ic_weighted_ensemble.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib

class ICWeightedEnsemble:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.ic_weights = {}
        self.rolling_ic_window = config.get('rolling_ic_window', 20)
        
    def load_models(self, model_dir: str) -> None:
        """Load all trained models"""
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name, 'model.pkl')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
    
    def calculate_rolling_ic(self, predictions: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Calculate rolling Information Coefficient for each model"""
        rolling_ic = pd.DataFrame(index=predictions.index)
        
        for model_name in predictions.columns:
            rolling_ic[model_name] = predictions[model_name].rolling(
                self.rolling_ic_window
            ).corr(returns)
        
        return rolling_ic
    
    def update_weights(self, rolling_ic: pd.DataFrame) -> None:
        """Update model weights based on recent IC performance"""
        # Use most recent IC values
        recent_ic = rolling_ic.iloc[-1]
        
        # Convert IC to weights (higher IC = higher weight)
        # Add small constant to avoid zero weights
        ic_positive = recent_ic + 0.01
        
        # Normalize weights
        self.ic_weights = (ic_positive / ic_positive.sum()).to_dict()
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate ensemble predictions"""
        predictions = pd.DataFrame(index=X.index)
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        # Calculate rolling IC (if returns available)
        if hasattr(self, 'returns'):
            rolling_ic = self.calculate_rolling_ic(predictions, self.returns)
            self.update_weights(rolling_ic)
        
        # Weighted ensemble prediction
        ensemble_pred = pd.Series(0, index=X.index)
        for model_name, weight in self.ic_weights.items():
            ensemble_pred += weight * predictions[model_name]
        
        return ensemble_pred
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get contribution of each model to ensemble prediction"""
        predictions = pd.DataFrame(index=X.index)
        
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        # Weighted contributions
        contributions = pd.DataFrame(index=X.index)
        for model_name, weight in self.ic_weights.items():
            contributions[f'{model_name}_contribution'] = weight * predictions[model_name]
        
        return contributions
```

### **3.2 Signal Calibration**

#### **Core Implementation**
```python
# core/ensemble/signal_calibrator.py
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import Tuple

class SignalCalibrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, predictions: pd.Series, actual_returns: pd.Series) -> 'SignalCalibrator':
        """Fit calibration model"""
        # Remove NaN values
        valid_idx = ~(predictions.isna() | actual_returns.isna())
        pred_clean = predictions[valid_idx]
        actual_clean = actual_returns[valid_idx]
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(pred_clean, actual_clean)
        self.is_fitted = True
        
        return self
    
    def transform(self, predictions: pd.Series) -> pd.Series:
        """Transform predictions to calibrated expected returns"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        return pd.Series(
            self.calibrator.transform(predictions),
            index=predictions.index
        )
    
    def get_confidence_intervals(self, predictions: pd.Series, confidence: float = 0.95) -> Tuple[pd.Series, pd.Series]:
        """Get confidence intervals for predictions"""
        if not self.is_fitted:
            raise ValueError("Must fit before getting confidence intervals")
        
        # Simple approach: use prediction uncertainty
        calibrated = self.transform(predictions)
        
        # Estimate uncertainty from prediction variance
        uncertainty = np.std(calibrated) * (1 - confidence)
        
        lower_bound = calibrated - uncertainty
        upper_bound = calibrated + uncertainty
        
        return lower_bound, upper_bound
```

---

## 💰 **Phase 4: Policy Layer Implementation**

### **4.1 Cost-Aware Position Sizing**

#### **Core Implementation**
```python
# core/policy/cost_aware_sizer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CostModel:
    spread_bps: float = 5.0  # 5 basis points
    commission_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 5.0  # 5 basis points
    market_impact_bps: float = 2.0  # 2 basis points per $10k

class CostAwarePositionSizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_model = CostModel(**config.get('cost_model', {}))
        self.min_expected_return_bps = config.get('min_expected_return_bps', 20)
        
    def estimate_costs(self, symbol: str, quantity: int, price: float) -> float:
        """Estimate total transaction costs"""
        notional = abs(quantity) * price
        
        # Spread cost
        spread_cost = notional * (self.cost_model.spread_bps / 10000)
        
        # Commission cost
        commission_cost = notional * (self.cost_model.commission_bps / 10000)
        
        # Slippage cost
        slippage_cost = notional * (self.cost_model.slippage_bps / 10000)
        
        # Market impact cost
        market_impact_cost = notional * (self.cost_model.market_impact_bps / 10000) * (notional / 10000)
        
        total_cost = spread_cost + commission_cost + slippage_cost + market_impact_cost
        
        return total_cost
    
    def should_trade(self, expected_return: float, quantity: int, price: float) -> bool:
        """Determine if trade should be executed based on costs"""
        if quantity == 0:
            return False
        
        costs = self.estimate_costs('', quantity, price)
        expected_profit = abs(expected_return) * abs(quantity) * price
        
        # Only trade if expected profit > costs + minimum threshold
        return expected_profit > costs * (1 + self.min_expected_return_bps / 10000)
    
    def size_position(self, expected_return: float, price: float, portfolio_value: float) -> int:
        """Size position considering costs"""
        # Base position size (from existing logic)
        base_size = int((expected_return * portfolio_value) / price)
        
        # Check if trade is profitable
        if not self.should_trade(expected_return, base_size, price):
            return 0
        
        # Adjust size to account for costs
        # Reduce size if costs are high relative to expected return
        cost_ratio = self.estimate_costs('', base_size, price) / (abs(expected_return) * abs(base_size) * price)
        
        if cost_ratio > 0.1:  # If costs > 10% of expected return
            # Reduce position size
            adjusted_size = int(base_size * (1 - cost_ratio))
            return adjusted_size
        
        return base_size
```

### **4.2 Turnover Budget Manager**

#### **Core Implementation**
```python
# core/policy/turnover_manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TurnoverBudget:
    max_daily_turnover: float = 0.05  # 5% max daily turnover
    max_weekly_turnover: float = 0.15  # 15% max weekly turnover
    max_monthly_turnover: float = 0.50  # 50% max monthly turnover

class TurnoverManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.budget = TurnoverBudget(**config.get('turnover_budget', {}))
        self.turnover_history = pd.DataFrame()
        
    def calculate_turnover(self, current_positions: pd.Series, target_positions: pd.Series, prices: pd.Series) -> float:
        """Calculate portfolio turnover"""
        # Calculate notional value of position changes
        position_changes = (target_positions - current_positions).abs()
        turnover_notional = (position_changes * prices).sum()
        
        # Calculate total portfolio value
        portfolio_value = (current_positions.abs() * prices).sum()
        
        # Calculate turnover as percentage
        turnover = turnover_notional / portfolio_value if portfolio_value > 0 else 0
        
        return turnover
    
    def check_turnover_budget(self, current_positions: pd.Series, target_positions: pd.Series, prices: pd.Series) -> bool:
        """Check if proposed trades exceed turnover budget"""
        turnover = self.calculate_turnover(current_positions, target_positions, prices)
        
        # Check daily budget
        if turnover > self.budget.max_daily_turnover:
            return False
        
        # Check weekly budget (if history available)
        if len(self.turnover_history) >= 5:  # 5 trading days
            weekly_turnover = self.turnover_history['turnover'].tail(5).sum()
            if weekly_turnover + turnover > self.budget.max_weekly_turnover:
                return False
        
        # Check monthly budget (if history available)
        if len(self.turnover_history) >= 20:  # 20 trading days
            monthly_turnover = self.turnover_history['turnover'].tail(20).sum()
            if monthly_turnover + turnover > self.budget.max_monthly_turnover:
                return False
        
        return True
    
    def adjust_positions(self, current_positions: pd.Series, target_positions: pd.Series, prices: pd.Series) -> pd.Series:
        """Adjust target positions to respect turnover budget"""
        if self.check_turnover_budget(current_positions, target_positions, prices):
            return target_positions
        
        # Reduce position changes proportionally
        position_changes = target_positions - current_positions
        max_turnover_notional = self.budget.max_daily_turnover * (current_positions.abs() * prices).sum()
        
        # Calculate scaling factor
        current_turnover_notional = (position_changes.abs() * prices).sum()
        scaling_factor = max_turnover_notional / current_turnover_notional
        
        # Apply scaling
        adjusted_changes = position_changes * scaling_factor
        adjusted_positions = current_positions + adjusted_changes
        
        return adjusted_positions
    
    def record_turnover(self, turnover: float, date: pd.Timestamp) -> None:
        """Record turnover for budget tracking"""
        self.turnover_history = pd.concat([
            self.turnover_history,
            pd.DataFrame({'turnover': [turnover], 'date': [date]})
        ], ignore_index=True)
```

---

## 📊 **Integration with Existing System**

### **Modified Execution Engine**

```python
# core/execution/execution_engine.py (modifications)
from core.ensemble.ic_weighted_ensemble import ICWeightedEnsemble
from core.ensemble.signal_calibrator import SignalCalibrator
from core.policy.cost_aware_sizer import CostAwarePositionSizer
from core.policy.turnover_manager import TurnoverManager

class ExecutionEngine:
    def __init__(self, config):
        self.config = config
        self.ensemble = ICWeightedEnsemble(config.ensemble)
        self.calibrator = SignalCalibrator(config.calibration)
        self.cost_sizer = CostAwarePositionSizer(config.cost_aware)
        self.turnover_manager = TurnoverManager(config.turnover)
        
    def execute_signals(self, signals: Dict[str, float], current_positions: Dict[str, int]) -> ExecutionResult:
        """Execute signals with alpha-source pipeline"""
        
        # 1. Generate ensemble predictions
        ensemble_predictions = self.ensemble.predict(signals)
        
        # 2. Calibrate to expected returns
        expected_returns = self.calibrator.transform(ensemble_predictions)
        
        # 3. Calculate target positions
        target_positions = {}
        for symbol, expected_return in expected_returns.items():
            target_positions[symbol] = self.cost_sizer.size_position(
                expected_return, 
                self.get_price(symbol), 
                self.portfolio_value
            )
        
        # 4. Apply turnover budget
        adjusted_positions = self.turnover_manager.adjust_positions(
            current_positions, target_positions, self.get_prices()
        )
        
        # 5. Execute trades
        return self._execute_trades(adjusted_positions, current_positions)
```

---

## 🚀 **Deployment Strategy**

### **Phase 1 Deployment (Weeks 1-4)**
1. **Sector Neutralization**: Deploy to paper trading
2. **Model Diversification**: Add LightGBM/CatBoost
3. **Validation**: Backtest and paper trade
4. **Monitoring**: Track performance improvements

### **Phase 2 Deployment (Weeks 5-8)**
1. **Cost-Aware Sizing**: Deploy to paper trading
2. **Fundamental Features**: Add to feature pipeline
3. **Validation**: Compare with Phase 1 results
4. **Monitoring**: Track cost savings and alpha improvement

### **Phase 3 Deployment (Weeks 9-12)**
1. **Ensemble System**: Deploy IC-weighted ensemble
2. **Signal Calibration**: Deploy calibrated expected returns
3. **Validation**: Full system backtesting
4. **Monitoring**: Track ensemble performance

### **Phase 4 Deployment (Months 4-6)**
1. **Alternative Data**: Integrate new data sources
2. **Advanced Models**: Deploy neural networks
3. **Full System**: Complete alpha-source pipeline
4. **Production**: Deploy to live trading

---

## 🎯 **Success Metrics**

### **Technical Metrics**
- **Feature Count**: 45 → 80+ features
- **Model Count**: 1 → 7+ models
- **Ensemble Stability**: IC standard deviation < 0.02
- **Calibration Quality**: < 5% calibration error

### **Performance Metrics**
- **Information Coefficient**: 0.02 → 0.10+
- **Sharpe Ratio**: 1.0 → 2.5+
- **Maximum Drawdown**: 10% → 8%
- **Cost Efficiency**: > 80% of gross alpha retained

---

**This technical implementation guide provides the complete architecture for transforming Aurora into an alpha-source territory system.** 🚀
