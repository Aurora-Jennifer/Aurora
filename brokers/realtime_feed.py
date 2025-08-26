"""
Realtime WebSocket feed driver with heartbeat monitoring and kill-switch.

Provides incremental OHLCV data with execution trust guarantees:
- Heartbeat monitoring (< 5s staleness)
- Kill switch on stale feeds
- Latency telemetry
- Duplicate timestamp detection
"""

import asyncio
import json
import logging
import os
import time
import websockets
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
import pandas as pd

logger = logging.getLogger(__name__)


class RealtimeFeed:
    """Real-time WebSocket feed with execution trust features."""
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 interval: str = "1m",
                 testnet: bool = True,
                 heartbeat_timeout: float = 5.0):
        self.symbol = symbol.upper()
        self.interval = interval
        self.testnet = testnet
        self.heartbeat_timeout = heartbeat_timeout
        
        # State tracking
        self.last_msg_ts: Optional[float] = None
        self.last_bar_ts: Optional[pd.Timestamp] = None
        self.trading_halted: bool = False
        self.latency_stats = []
        
        # Callbacks
        self.on_bar: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_heartbeat: Optional[Callable[[float], None]] = None
        self.on_halt: Optional[Callable[[str], None]] = None
        
        # WebSocket URL (Binance testnet)
        if testnet:
            self.ws_url = "wss://testnet.binance.vision/ws/" + f"{symbol.lower()}@kline_{interval}"
        else:
            self.ws_url = "wss://stream.binance.com:9443/ws/" + f"{symbol.lower()}@kline_{interval}"
    
    def is_stale(self) -> bool:
        """Check if feed is stale (> heartbeat_timeout seconds)."""
        if self.last_msg_ts is None:
            return True
        return (time.time() - self.last_msg_ts) > self.heartbeat_timeout
    
    def halt_trading(self, reason: str) -> None:
        """Activate kill switch and halt trading."""
        self.trading_halted = True
        os.environ["FLAG_TRADING_HALTED"] = "1"
        logger.error(f"TRADING HALTED: {reason}")
        if self.on_halt:
            self.on_halt(reason)
    
    def resume_trading(self) -> None:
        """Resume trading (clear halt flag)."""
        self.trading_halted = False
        os.environ.pop("FLAG_TRADING_HALTED", None)
        logger.info("Trading resumed")
    
    def _parse_kline(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Binance kline message into OHLCV bar."""
        try:
            kline = msg.get('k', {})
            if not kline.get('x', False):  # Only process closed bars
                return None
                
            bar_ts = pd.Timestamp(kline['t'], unit='ms', tz=timezone.utc)
            
            # Skip duplicate timestamps
            if self.last_bar_ts and bar_ts <= self.last_bar_ts:
                logger.debug(f"Skipping duplicate/old bar: {bar_ts}")
                return None
                
            self.last_bar_ts = bar_ts
            
            return {
                'symbol': self.symbol,
                'timestamp': bar_ts,
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'feed_ts': time.time()  # For latency tracking
            }
        except Exception as e:
            logger.error(f"Failed to parse kline: {e}")
            return None
    
    async def _heartbeat_monitor(self):
        """Monitor feed health and trigger kill switch if stale."""
        while True:
            await asyncio.sleep(1.0)
            
            if self.is_stale():
                if not self.trading_halted:
                    self.halt_trading(f"Feed stale: {self.heartbeat_timeout}s timeout")
            else:
                if self.trading_halted:
                    self.resume_trading()
                    
            # Heartbeat callback
            if self.on_heartbeat:
                self.on_heartbeat(time.time() - (self.last_msg_ts or 0))
    
    async def start(self) -> None:
        """Start the real-time feed."""
        logger.info(f"Starting realtime feed: {self.symbol} {self.interval}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        
        # Start heartbeat monitor
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info("WebSocket connected")
                
                async for message in websocket:
                    self.last_msg_ts = time.time()
                    
                    try:
                        msg = json.loads(message)
                        bar = self._parse_kline(msg)
                        
                        if bar and self.on_bar:
                            self.on_bar(bar)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.halt_trading(f"WebSocket connection failed: {e}")
        finally:
            heartbeat_task.cancel()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feed statistics for telemetry."""
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'last_msg_ts': self.last_msg_ts,
            'last_bar_ts': self.last_bar_ts.isoformat() if self.last_bar_ts else None,
            'trading_halted': self.trading_halted,
            'is_stale': self.is_stale(),
            'staleness_sec': time.time() - (self.last_msg_ts or 0) if self.last_msg_ts else None
        }


# Feature flag check
def is_realtime_enabled() -> bool:
    """Check if realtime mode is enabled via feature flag."""
    return os.getenv("FLAG_REALTIME", "0") == "1"


def check_trading_halted() -> bool:
    """Check if trading is halted via kill switch."""
    return os.getenv("FLAG_TRADING_HALTED", "0") == "1"


# Example usage
if __name__ == "__main__":
    import asyncio
    
    from core.utils import setup_logging
    setup_logging("logs/realtime_feed.log", logging.INFO)
    
    feed = RealtimeFeed("BTCUSDT", "1m", testnet=True)
    
    def on_bar(bar):
        print(f"📊 New bar: {bar['symbol']} {bar['timestamp']} C={bar['close']:.2f}")
    
    def on_heartbeat(staleness):
        if staleness > 2:
            print(f"💓 Heartbeat: {staleness:.1f}s since last message")
    
    def on_halt(reason):
        print(f"🛑 HALT: {reason}")
    
    feed.on_bar = on_bar
    feed.on_heartbeat = on_heartbeat
    feed.on_halt = on_halt
    
    try:
        asyncio.run(feed.start())
    except KeyboardInterrupt:
        print("Feed stopped")
