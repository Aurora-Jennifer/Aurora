# Configuration Guide

This document describes all configuration options for the trading system.

## Environment Variables

### IBKR Configuration
- `IBKR_PAPER_TRADING`: Set to `true` for paper trading, `false` for live trading
- `IBKR_HOST`: IBKR Gateway host (default: `127.0.0.1`)
- `IBKR_PORT`: IBKR Gateway port (default: `7497` for paper, `7496` for live)
- `IBKR_CLIENT_ID`: Unique client ID for IBKR connection
- `IBKR_TIMEOUT`: Connection timeout in seconds (default: `20`)
- `IBKR_MAX_RETRIES`: Maximum connection retries (default: `3`)

### Data Configuration
- `DATA_CACHE_DIR`: Directory for caching market data (default: `data/ibkr`)
- `DATA_USE_CACHE`: Enable data caching (default: `true`)
- `DATA_FALLBACK_TO_YFINANCE`: Enable yfinance fallback (default: `true`)

### Logging Configuration
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `LOG_FILE_ROTATION`: Enable log file rotation (default: `true`)
- `LOG_MAX_FILE_SIZE`: Maximum log file size (default: `10MB`)
- `LOG_BACKUP_COUNT`: Number of backup log files (default: `5`)

### Discord Notifications
- `DISCORD_ENABLED`: Enable Discord notifications (default: `false`)
- `DISCORD_WEBHOOK_URL`: Discord webhook URL
- `DISCORD_BOT_NAME`: Bot name for notifications (default: `Trading Bot`)

## Configuration Files

### Main Configuration
- `config/enhanced_paper_trading_config.json`: Main trading system configuration
- `config/config.yaml`: Legacy configuration (deprecated)

### Strategy Configuration
- `config/strategies_config.json`: Strategy-specific parameters
- `config/paper_trading_config.json`: Paper trading settings
- `config/live_config.json`: Live trading settings

### IBKR Configuration
- `config/ibkr_config.json`: IBKR connection settings
- `config/live_config_ibkr.json`: Live IBKR settings

### Notifications
- `config/notifications/discord_config.json`: Discord notification settings

## Configuration Structure

### Enhanced Paper Trading Config
```json
{
  "symbols": ["SPY", "AAPL", "NVDA", "GOOGL", "MSFT"],
  "initial_capital": 100000,
  "use_ibkr": true,
  "ibkr_config": {
    "paper_trading": true,
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 12399
  },
  "risk_params": {
    "max_weight_per_symbol": 0.25,
    "max_drawdown": 0.15,
    "max_daily_loss": 0.02
  },
  "strategy_params": {
    "regime_aware_ensemble": {
      "confidence_threshold": 0.3,
      "regime_lookback": 252
    }
  }
}
```

### Strategy Parameters
- `confidence_threshold`: Minimum confidence for trade execution
- `regime_lookback`: Lookback period for regime detection
- `trend_following_weight`: Weight for trend-following signals
- `mean_reversion_weight`: Weight for mean-reversion signals
- `rolling_window`: Rolling window for performance calculation

### Risk Parameters
- `max_weight_per_symbol`: Maximum position size per symbol
- `max_drawdown`: Maximum allowed drawdown
- `max_daily_loss`: Maximum daily loss limit
- `position_sizing`: Regime-specific position sizing multipliers

## Configuration Validation

The system validates configuration files on startup:

1. **Required fields**: Ensures all required configuration fields are present
2. **Type validation**: Validates data types for configuration values
3. **Range validation**: Ensures values are within acceptable ranges
4. **Path validation**: Validates file paths and ensures they exist

## Environment-Specific Configuration

### Development
- Use paper trading mode
- Enable verbose logging
- Use local IBKR Gateway

### Production
- Use live trading mode (when ready)
- Enable Discord notifications
- Use production IBKR Gateway

### Testing
- Use minimal symbol list
- Enable test mode flags
- Use mock data providers

## Configuration Best Practices

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Version control configs**: Keep configuration files in version control
3. **Environment separation**: Use different configs for different environments
4. **Documentation**: Document all configuration options
5. **Validation**: Always validate configuration on startup

## Troubleshooting

### Common Issues
1. **IBKR Connection Failed**: Check host, port, and client ID
2. **Configuration Not Found**: Ensure config files exist and are readable
3. **Invalid Parameters**: Check configuration validation errors
4. **Permission Denied**: Ensure proper file permissions

### Debug Mode
Enable debug logging by setting `LOG_LEVEL=DEBUG` to see detailed configuration loading information.
