# Data Pipeline Architecture

## Overview

The Aurora trading system's data pipeline is designed for reliability, accuracy, and real-time market data integration. This document outlines the complete data flow from raw market data to feature engineering.

## Data Flow Architecture

```
Alpaca API → fetch_bars_alpaca.py → prices.parquet → panel_builder.py → features.parquet → run_universe.py → predictions.json
```

## Component Details

### 1. Data Source: Alpaca API

#### API Configuration
- **Endpoint**: `https://data-api.alpaca.markets/v2/stocks/bars`
- **Authentication**: API Key + Secret Key
- **Rate Limits**: 200 requests/minute
- **Data Types**: OHLCV bars, corporate actions, fundamentals

#### Data Quality
- **Real-time**: Live market data during trading hours
- **Historical**: Up to 2 years of historical data
- **Coverage**: 130 symbols (S&P 500 subset)
- **Frequency**: Daily bars with intraday precision

### 2. Data Fetcher: `tools/fetch_bars_alpaca.py`

#### Core Features
- **Symbol Normalization**: Handles Alpaca-specific symbol formats
- **Batching**: Processes large symbol sets efficiently (50 symbols/request)
- **Error Handling**: Graceful fallback and retry logic
- **Data Validation**: Ensures complete data coverage

#### Symbol Normalization
```python
NORMALIZE_MAP = {
    "BRK-B": "BRK.B",    # Berkshire Hathaway
    "BF-B": "BF.B",      # Brown-Forman
    "ANTM": "ELV",       # Anthem → Elevance Health
    "HCP": "PEAK",       # HCP → Healthpeak Properties
    "FB": "META",        # Facebook → Meta
}
```

#### Batching Strategy
- **Chunk Size**: 50 symbols per API request
- **Parallel Processing**: Sequential requests to respect rate limits
- **Error Recovery**: Individual chunk failures don't stop entire process
- **Data Aggregation**: All chunks combined into single output file

#### Data Validation
- **Coverage Check**: Ensures 130/130 symbols present
- **Date Validation**: Confirms data for expected trading days
- **Schema Validation**: Verifies required columns present
- **Quality Checks**: Identifies missing or invalid data

### 3. Data Storage: `data/latest/prices.parquet`

#### File Structure
```
data/
├── latest/
│   ├── prices.parquet          # Main price data
│   ├── fundamentals.parquet    # Company fundamentals
│   └── sector_map.parquet      # Sector classifications
├── universe/
│   └── top300.txt             # Symbol universe
└── archive/                   # Historical data backups
```

#### Data Schema
```python
prices.parquet:
- date: datetime64[ns]         # Trading date
- symbol: string               # Stock symbol
- open: float64                # Opening price
- high: float64                # High price
- low: float64                 # Low price
- close: float64               # Closing price
- volume: int64                # Trading volume
```

#### Data Quality Metrics
- **Completeness**: 100% symbol coverage required
- **Freshness**: Data updated daily by 16:00 CT
- **Accuracy**: Price data validated against market standards
- **Consistency**: Uniform schema across all symbols

### 4. Feature Engineering: `ml/panel_builder.py`

#### Feature Categories

##### Price-Based Features
- **Returns**: Daily, weekly, monthly returns
- **Volatility**: Rolling volatility measures
- **Momentum**: Price momentum indicators
- **Reversal**: Short-term reversal signals

##### Volume-Based Features
- **Volume Surprise**: Unexpected volume changes
- **Volume-Price Trend**: Volume-price relationships
- **Liquidity**: Amihud illiquidity measure
- **Turnover**: Trading activity metrics

##### Cross-Sectional Features
- **Ranking**: Cross-sectional rank features
- **Z-Scoring**: Cross-sectional standardization
- **Residualization**: Market/sector neutralization
- **Relative Strength**: Peer-relative performance

#### Feature Engineering Pipeline

##### Step 1: Raw Feature Construction
```python
# Price features
df['ret_1d'] = df.groupby('symbol')['close'].pct_change()
df['vol_5d'] = df.groupby('symbol')['ret_1d'].rolling(5).std()

# Volume features
df['volume_surprise'] = df.groupby('symbol')['volume'].pct_change()
df['amihud_illiquidity'] = abs(df['ret_1d']) / df['volume']
```

##### Step 2: Cross-Sectional Transformations
```python
# Cross-sectional ranking (using transform to preserve index)
df['cs_rank_ret'] = df.groupby('date')['ret_1d'].transform(lambda x: x.rank(pct=True))

# Cross-sectional z-scoring
df['cs_zscore_vol'] = df.groupby('date')['vol_5d'].transform(lambda x: (x - x.mean()) / x.std())
```

##### Step 3: Residualization
```python
# Market residualization
df['ret_market_neutral'] = residualize_against_factor(
    df, 'ret_1d', 'market_return', 'date'
)

# Sector residualization
df['vol_sector_neutral'] = residualize_against_factor(
    df, 'vol_5d', 'sector_mean', 'date'
)
```

#### Feature Validation

##### Dispersion Checks
- **Cross-Sectional Variance**: Features must have cross-sectional variation
- **Temporal Stability**: Features should be stable over time
- **Outlier Detection**: Extreme values identified and handled
- **Missing Data**: Proper handling of NaN values

##### Quality Gates
- **Feature Count**: Exactly 45 features required
- **Symbol Coverage**: All 130 symbols must have features
- **Date Coverage**: Features for all required dates
- **Data Types**: Consistent data types across features

### 5. Feature Storage: `features.parquet`

#### File Structure
```
features.parquet:
- date: datetime64[ns]         # Feature date
- symbol: string               # Stock symbol
- f_ret_1d: float64            # 1-day return
- f_vol_5d: float64            # 5-day volatility
- f_cs_rank_ret: float64       # Cross-sectional rank
- ... (45 total features)
```

#### Feature Metadata
```python
feature_metadata = {
    'feature_count': 45,
    'symbol_count': 130,
    'date_range': ('2020-01-01', '2025-09-08'),
    'feature_types': {
        'price': 15,
        'volume': 10,
        'cross_sectional': 12,
        'residualized': 8
    }
}
```

## Data Quality Assurance

### Validation Pipeline

#### 1. Data Coverage Validation
```python
def validate_data_coverage():
    """Ensure 100% symbol coverage"""
    universe_symbols = load_universe()  # 130 symbols
    data_symbols = df['symbol'].nunique()
    
    if data_symbols != universe_symbols:
        raise ValueError(f"Coverage mismatch: {data_symbols}/{universe_symbols}")
    
    return True
```

#### 2. Feature Quality Validation
```python
def validate_feature_quality(df):
    """Ensure feature quality standards"""
    # Check for flat features
    for col in feature_cols:
        if df.groupby('date')[col].std().mean() < 0.01:
            raise ValueError(f"Flat feature detected: {col}")
    
    # Check for excessive missing data
    missing_pct = df[feature_cols].isnull().mean().max()
    if missing_pct > 0.05:
        raise ValueError(f"Excessive missing data: {missing_pct:.1%}")
    
    return True
```

#### 3. Temporal Validation
```python
def validate_temporal_consistency(df):
    """Ensure temporal consistency"""
    # Check for lookahead bias
    feature_date = df['date'].max()
    decision_date = get_decision_date()
    
    if feature_date >= decision_date:
        raise ValueError("Lookahead bias detected")
    
    return True
```

### Error Handling

#### Data Fetch Errors
- **API Failures**: Retry with exponential backoff
- **Rate Limiting**: Respect API rate limits
- **Network Issues**: Graceful degradation
- **Authentication**: Clear error messages

#### Feature Engineering Errors
- **Missing Data**: Proper imputation or exclusion
- **Calculation Errors**: Safe mathematical operations
- **Memory Issues**: Efficient data processing
- **Index Alignment**: Preserve DataFrame structure

## Performance Optimization

### Data Processing

#### Efficient Operations
- **Vectorized Operations**: Use pandas vectorized functions
- **Memory Management**: Process data in chunks if needed
- **Caching**: Cache intermediate results
- **Parallel Processing**: Use multiprocessing for independent operations

#### Storage Optimization
- **Parquet Format**: Efficient columnar storage
- **Compression**: Reduce file sizes
- **Partitioning**: Organize data by date/symbol
- **Indexing**: Optimize query performance

### Monitoring

#### Data Pipeline Metrics
- **Processing Time**: Track data processing duration
- **Memory Usage**: Monitor memory consumption
- **Error Rates**: Track failure rates
- **Data Quality**: Monitor data quality metrics

#### Alerting
- **Coverage Alerts**: Notify if coverage < 100%
- **Quality Alerts**: Notify if quality metrics degrade
- **Performance Alerts**: Notify if processing time increases
- **Error Alerts**: Notify of critical errors

## Backup and Recovery

### Data Backup Strategy
- **Daily Backups**: Backup data files daily
- **Version Control**: Track data file versions
- **Archive Strategy**: Move old data to archive
- **Recovery Testing**: Regular recovery drills

### Disaster Recovery
- **Data Redundancy**: Multiple data sources
- **Backup Locations**: Offsite backup storage
- **Recovery Procedures**: Documented recovery steps
- **Testing**: Regular disaster recovery testing

## Security and Compliance

### Data Security
- **Access Control**: Restrict data access
- **Encryption**: Encrypt sensitive data
- **Audit Logging**: Log data access
- **Data Retention**: Follow retention policies

### Compliance
- **Data Privacy**: Protect personal information
- **Regulatory Compliance**: Follow financial regulations
- **Data Governance**: Implement data governance policies
- **Risk Management**: Manage data-related risks

## Future Enhancements

### Data Sources
- **Additional Exchanges**: Expand to other exchanges
- **Alternative Data**: Incorporate alternative data sources
- **Real-time Data**: WebSocket feeds for real-time updates
- **International Markets**: Global market coverage

### Processing Improvements
- **Streaming Processing**: Real-time data processing
- **Machine Learning**: ML-based data quality
- **Automated Anomaly Detection**: Detect data anomalies
- **Predictive Maintenance**: Predict system failures

### Scalability
- **Distributed Processing**: Scale across multiple machines
- **Cloud Integration**: Cloud-based data processing
- **Microservices**: Decompose into microservices
- **API Gateway**: Centralized API management
