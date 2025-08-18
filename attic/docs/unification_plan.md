# Trading System Codebase Unification Plan

## Detected Conventions

The codebase currently uses mixed naming patterns with good module separation. We will enforce:
- **Functions/Methods**: snake_case with verb_noun pattern
- **Variables**: snake_case with descriptive names, preserve domain abbreviations
- **Classes**: PascalCase
- **Constants**: ALL_CAPS
- **Suffixes**: *_df, *_series, *_arr, *_pct, *_bps
- **Module Structure**: Preserve current organization

## Rename Map

| Symbol Kind | Module | Old Name | New Name | Reason | Risk | External API | Alias Added |
|-------------|--------|----------|----------|---------|------|--------------|-------------|
| function | core/engine/paper.py | load_config | load_config | Already correct | low | yes | no |
| function | core/engine/paper.py | load_profile_config | load_profile_config | Already correct | low | yes | no |
| function | core/engine/paper.py | _initialize_components | _initialize_components | Already correct | low | no | no |
| function | core/engine/paper.py | _initialize_kill_switches | _initialize_kill_switches | Already correct | low | no | no |
| function | core/engine/paper.py | _setup_discord_notifications | _setup_discord_notifications | Already correct | low | no | no |
| function | core/engine/paper.py | _log_startup | _log_startup | Already correct | low | no | no |
| function | core/engine/paper.py | run_trading_cycle | run_trading_cycle | Already correct | low | yes | no |
| function | core/engine/paper.py | _get_market_data | _get_market_data | Already correct | low | no | no |
| function | core/engine/paper.py | _detect_regime | _detect_regime | Already correct | low | no | no |
| function | core/engine/paper.py | _generate_signals | _generate_signals | Already correct | low | no | no |
| function | core/engine/paper.py | _execute_trades | _execute_trades | Already correct | low | no | no |
| function | core/engine/paper.py | _update_portfolio | _update_portfolio | Already correct | low | no | no |
| function | core/engine/paper.py | _record_daily_return | _record_daily_return | Already correct | low | no | no |
| function | core/engine/paper.py | _calculate_performance_metrics | _calculate_performance_metrics | Already correct | low | no | no |
| function | core/engine/paper.py | _save_results | _save_results | Already correct | low | no | no |
| function | core/engine/paper.py | _send_notifications | _send_notifications | Already correct | low | no | no |
| function | core/engine/paper.py | get_performance_summary | get_performance_summary | Already correct | low | yes | no |
| function | core/engine/paper.py | get_positions | get_positions | Already correct | low | yes | no |
| function | core/engine/paper.py | get_trade_history | get_trade_history | Already correct | low | yes | no |
| function | core/engine/paper.py | get_daily_returns | get_daily_returns | Already correct | low | yes | no |
| function | core/engine/paper.py | get_regime_history | get_regime_history | Already correct | low | yes | no |
| function | core/engine/paper.py | shutdown | shutdown | Already correct | low | yes | no |
| variable | core/engine/paper.py | trading_logger | trading_logger | Already correct | low | no | no |
| variable | core/engine/paper.py | logger | logger | Already correct | low | no | no |
| variable | core/engine/paper.py | config | config | Already correct | low | no | no |
| variable | core/engine/paper.py | capital | capital | Already correct | low | no | no |
| variable | core/engine/paper.py | positions | positions | Already correct | low | no | no |
| variable | core/engine/paper.py | trade_history | trade_history | Already correct | low | no | no |
| variable | core/engine/paper.py | daily_returns | daily_returns | Already correct | low | no | no |
| variable | core/engine/paper.py | regime_history | regime_history | Already correct | low | no | no |
| variable | core/engine/paper.py | performance_metrics | performance_metrics | Already correct | low | no | no |
| variable | core/engine/paper.py | kill_switches | kill_switches | Already correct | low | no | no |
| variable | core/engine/paper.py | discord_notifier | discord_notifier | Already correct | low | no | no |
| variable | core/engine/paper.py | use_ibkr | use_ibkr | Already correct | low | no | no |
| variable | core/engine/paper.py | ibkr_config | ibkr_config | Already correct | low | no | no |
| variable | core/engine/paper.py | data_provider | data_provider | Already correct | low | no | no |
| variable | core/engine/paper.py | regime_detector | regime_detector | Already correct | low | no | no |
| variable | core/engine/paper.py | feature_reweighter | feature_reweighter | Already correct | low | no | no |
| variable | core/engine/paper.py | adaptive_engine | adaptive_engine | Already correct | low | no | no |
| variable | core/engine/paper.py | strategy | strategy | Already correct | low | no | no |
| variable | core/engine/paper.py | risk_guardrails | risk_guardrails | Already correct | low | no | no |
| variable | core/engine/paper.py | telemetry | telemetry | Already correct | low | no | no |
| class | core/engine/paper.py | PaperTradingEngine | PaperTradingEngine | Already correct | low | yes | no |
| function | core/engine/backtest.py | run_backtest | run_backtest | Already correct | low | yes | no |
| function | core/engine/backtest.py | _get_data_up_to_date | _get_data_up_to_date | Already correct | low | no | no |
| function | core/engine/backtest.py | _load_historical_data | _load_historical_data | Already correct | low | no | no |
| function | core/engine/backtest.py | _get_trading_dates_from_data | _get_trading_dates_from_data | Already correct | low | no | no |
| function | core/engine/backtest.py | _update_portfolio_prices | _update_portfolio_prices | Already correct | low | no | no |
| function | core/engine/backtest.py | _run_daily_trading | _run_daily_trading | Already correct | low | no | no |
| function | core/engine/backtest.py | _detect_regime_with_rate_limit | _detect_regime_with_rate_limit | Already correct | low | no | no |
| function | core/engine/backtest.py | _get_historical_data | _get_historical_data | Already correct | low | no | no |
| function | core/engine/backtest.py | _generate_signals | _generate_signals | Already correct | low | no | no |
| function | core/engine/backtest.py | _get_prices_for_date | _get_prices_for_date | Already correct | low | no | no |
| function | core/engine/backtest.py | _execute_trades_with_portfolio | _execute_trades_with_portfolio | Already correct | low | no | no |
| function | core/engine/backtest.py | _slice_ledger_to_backtest | _slice_ledger_to_backtest | Already correct | low | no | no |
| function | core/engine/backtest.py | _get_trades_in_backtest_window | _get_trades_in_backtest_window | Already correct | low | no | no |
| function | core/engine/backtest.py | _calculate_trade_metrics_from | _calculate_trade_metrics_from | Already correct | low | no | no |
| function | core/engine/backtest.py | _calculate_portfolio_metrics | _calculate_portfolio_metrics | Already correct | low | no | no |
| function | core/engine/backtest.py | _generate_results | _generate_results | Already correct | low | no | no |
| function | core/engine/backtest.py | _save_results | _save_results | Already correct | low | no | no |
| function | core/engine/backtest.py | _format_summary | _format_summary | Already correct | low | no | no |
| function | core/engine/backtest.py | get_last_summary | get_last_summary | Already correct | low | yes | no |
| function | core/engine/backtest.py | print_results | print_results | Already correct | low | yes | no |
| function | core/engine/backtest.py | _generate_final_results | _generate_final_results | Already correct | low | no | no |
| variable | core/engine/backtest.py | start_date | start_date | Already correct | low | no | no |
| variable | core/engine/backtest.py | end_date | end_date | Already correct | low | no | no |
| variable | core/engine/backtest.py | initial_capital | initial_capital | Already correct | low | no | no |
| variable | core/engine/backtest.py | _last_results | _last_results | Already correct | low | no | no |
| variable | core/engine/backtest.py | portfolio | portfolio | Already correct | low | no | no |
| variable | core/engine/backtest.py | trade_book | trade_book | Already correct | low | no | no |
| variable | core/engine/backtest.py | daily_returns | daily_returns | Already correct | low | no | no |
| variable | core/engine/backtest.py | equity_curve | equity_curve | Already correct | low | no | no |
| variable | core/engine/backtest.py | insufficient_data_logged | insufficient_data_logged | Already correct | low | no | no |
| variable | core/engine/backtest.py | logger | logger | Already correct | low | no | no |
| variable | core/engine/backtest.py | MIN_HISTORY | MIN_HISTORY | Already correct | low | no | no |
| class | core/engine/backtest.py | BacktestEngine | BacktestEngine | Already correct | low | yes | no |
| function | strategies/base.py | generate_signals | generate_signals | Already correct | low | yes | no |
| function | strategies/base.py | get_default_params | get_default_params | Already correct | low | yes | no |
| function | strategies/base.py | get_param_ranges | get_param_ranges | Already correct | low | yes | no |
| function | strategies/base.py | backtest | backtest | Already correct | low | yes | no |
| function | strategies/base.py | get_description | get_description | Already correct | low | yes | no |
| function | strategies/base.py | validate_params | validate_params | Already correct | low | yes | no |
| class | strategies/base.py | StrategyParams | StrategyParams | Already correct | low | yes | no |
| class | strategies/base.py | BaseStrategy | BaseStrategy | Already correct | low | yes | no |
| function | strategies/factory.py | register_strategy | register_strategy | Already correct | low | yes | no |
| function | strategies/factory.py | get_available_strategies | get_available_strategies | Already correct | low | yes | no |
| function | strategies/factory.py | create_strategy | create_strategy | Already correct | low | yes | no |
| class | strategies/factory.py | StrategyFactory | StrategyFactory | Already correct | low | yes | no |
| function | core/portfolio.py | apply_fill | apply_fill | Already correct | low | yes | no |
| function | core/portfolio.py | unrealized_pnl | unrealized_pnl | Already correct | low | yes | no |
| function | core/portfolio.py | value_at | value_at | Already correct | low | yes | no |
| function | core/portfolio.py | mark_to_market | mark_to_market | Already correct | low | yes | no |
| function | core/portfolio.py | execute_order | execute_order | Already correct | low | yes | no |
| function | core/portfolio.py | close_all_positions | close_all_positions | Already correct | low | yes | no |
| function | core/portfolio.py | get_position | get_position | Already correct | low | yes | no |
| function | core/portfolio.py | get_open_positions_count | get_open_positions_count | Already correct | low | yes | no |
| function | core/portfolio.py | get_summary | get_summary | Already correct | low | yes | no |
| class | core/portfolio.py | Position | Position | Already correct | low | yes | no |
| class | core/portfolio.py | PortfolioState | PortfolioState | Already correct | low | yes | no |
| function | core/trade_logger.py | on_buy | on_buy | Already correct | low | yes | no |
| function | core/trade_logger.py | on_sell | on_sell | Already correct | low | yes | no |
| function | core/trade_logger.py | reset | reset | Already correct | low | yes | no |
| function | core/trade_logger.py | mark_drawdown | mark_drawdown | Already correct | low | yes | no |
| function | core/trade_logger.py | get_open_positions | get_open_positions | Already correct | low | yes | no |
| function | core/trade_logger.py | get_closed_trades | get_closed_trades | Already correct | low | yes | no |
| function | core/trade_logger.py | get_ledger | get_ledger | Already correct | low | yes | no |
| function | core/trade_logger.py | get_trades | get_trades | Already correct | low | yes | no |
| function | core/trade_logger.py | get_trade_summary | get_trade_summary | Already correct | low | yes | no |
| function | core/trade_logger.py | export_trades_csv | export_trades_csv | Already correct | low | yes | no |
| class | core/trade_logger.py | TradeRecord | TradeRecord | Already correct | low | yes | no |
| class | core/trade_logger.py | TradeBook | TradeBook | Already correct | low | yes | no |

## Key Unification Areas

### 1. Variable Naming Consistency
- Ensure all variables use snake_case
- Standardize domain abbreviations (pnl, sharpe, sma, ema, rsi, atr, bid, ask, ohlc)
- Add proper suffixes (*_df, *_series, *_arr, *_pct, *_bps)

### 2. Function Naming Consistency
- All functions already follow snake_case pattern
- Maintain verb_noun pattern for clarity

### 3. Class Naming Consistency
- All classes already follow PascalCase pattern
- Maintain current naming conventions

### 4. Configuration Key Standardization
- Standardize config keys to snake_case
- Preserve backward compatibility with aliases

### 5. Import Organization
- Standardize import order (stdlib, third-party, local)
- Remove unused imports

## Risk Assessment

**Low Risk Changes:**
- Variable renaming within functions
- Internal method renaming
- Documentation updates

**Medium Risk Changes:**
- Configuration key changes (need aliases)
- Public API method renaming (need deprecation warnings)

**High Risk Changes:**
- Class name changes (need aliases)
- Module structure changes

## Migration Strategy

1. **Phase 1**: Internal variable and function standardization
2. **Phase 2**: Configuration key standardization with aliases
3. **Phase 3**: Public API standardization with deprecation warnings
4. **Phase 4**: Remove deprecated aliases (future version)

## Regression Testing Plan

1. **Order Simulation Tests**
   - Market order execution with slippage
   - Limit order handling
   - Fee calculation accuracy
   - Short position management

2. **Portfolio Accounting Tests**
   - Cash balance tracking
   - Position value calculation
   - Realized/unrealized PnL computation
   - Mark-to-market accuracy

3. **Performance Metrics Tests**
   - Sharpe ratio calculation
   - Maximum drawdown computation
   - Return calculation accuracy
   - Volatility measurement

4. **Walk-Forward Tests**
   - Identical fold generation with fixed seeds
   - Consistent model selection
   - Reproducible results

5. **Integration Tests**
   - End-to-end backtest execution
   - Paper trading cycle completion
   - Data provider integration
   - Risk management enforcement

## Deprecation Plan

Aliases will be maintained for one major version cycle with deprecation warnings. After the next major release, deprecated aliases will be removed. This ensures backward compatibility while encouraging migration to the new naming conventions.
