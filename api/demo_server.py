"""
FastAPI Demo Server
Exposes trading system functionality via API without revealing source code.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from pydantic import BaseModel

# Import compiled modules (will be compiled with Nuitka)
try:
    from core_compiled import composer_integration, data_sanity, config_loader
except ImportError:
    # Fallback to source for development
    from core.engine import composer_integration
    from core import data_sanity, config_loader

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
API_TOKEN = os.getenv("TRADING_API_TOKEN", "demo_token_2025")

app = FastAPI(
    title="Advanced Trading System API",
    description="Professional algorithmic trading system API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
)

# Pydantic models
class MarketData(BaseModel):
    symbol: str
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    timestamps: List[str]

class PredictionRequest(BaseModel):
    market_data: MarketData
    lookback: int = 50

class PredictionResponse(BaseModel):
    signal: float
    confidence: float
    regime: str
    strategy_weights: Dict[str, float]
    timestamp: str
    processing_time_ms: float

class ValidationRequest(BaseModel):
    market_data: MarketData

class ValidationResponse(BaseModel):
    is_valid: bool
    issues: List[str]
    repairs: List[str]
    validation_time_ms: float

# Dependency for API token validation
async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid API token")
    return authorization

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """
    Generate trading signal prediction.
    
    This endpoint processes market data and returns a trading signal
    using the advanced composer system without exposing internal logic.
    """
    start_time = datetime.now()
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame({
            'Open': request.market_data.open,
            'High': request.market_data.high,
            'Low': request.market_data.low,
            'Close': request.market_data.close,
            'Volume': request.market_data.volume
        }, index=pd.to_datetime(request.market_data.timestamps))
        
        # Load configuration
        config = config_loader.load_config(['config/base.yaml'])
        
        # Initialize composer integration
        composer = composer_integration.ComposerIntegration(config)
        
        # Get prediction
        current_idx = len(df) - 1
        signal, metadata = composer.get_composer_decision(
            df, request.market_data.symbol, current_idx
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            signal=float(signal),
            confidence=metadata.get('confidence', 0.5),
            regime=metadata.get('regime', 'unknown'),
            strategy_weights=metadata.get('strategy_weights', {}),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/validate", response_model=ValidationResponse)
async def validate_data(
    request: ValidationRequest,
    token: str = Depends(verify_token)
):
    """
    Validate market data for integrity and quality.
    
    This endpoint performs comprehensive data validation without
    exposing the validation logic details.
    """
    start_time = datetime.now()
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame({
            'Open': request.market_data.open,
            'High': request.market_data.high,
            'Low': request.market_data.low,
            'Close': request.market_data.close,
            'Volume': request.market_data.volume
        }, index=pd.to_datetime(request.market_data.timestamps))
        
        # Perform validation
        validator = data_sanity.DataSanityValidator()
        clean_data, result = validator.validate_and_repair(df, request.market_data.symbol)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ValidationResponse(
            is_valid=len(result.repairs) == 0 and len(result.flags) == 0,
            issues=result.flags,
            repairs=result.repairs,
            validation_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/capabilities")
async def get_capabilities(token: str = Depends(verify_token)):
    """Get system capabilities without exposing implementation details."""
    return {
        "features": [
            "Advanced ML-based signal generation",
            "Multi-strategy composer system",
            "Regime-aware trading",
            "Comprehensive data validation",
            "Real-time risk management"
        ],
        "supported_assets": ["equities", "etfs", "crypto"],
        "max_lookback": 252,
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Only enable docs in development
    enable_docs = os.getenv("ENABLE_DOCS", "false").lower() == "true"
    
    uvicorn.run(
        "demo_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for security
        access_log=True
    )
