from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
import time
from typing import Any, Protocol


class QuoteProvider(Protocol):
    def quote(self, symbol: str) -> dict[str, Any]:
        """Return a quote dict with keys: bid, ask, mid, ts (ISO UTC)."""
        ...


class DummyMidProvider:
    def __init__(self, last_px_fn: Callable[[str], float]) -> None:
        self._last = last_px_fn

    def quote(self, symbol: str) -> dict[str, Any]:
        mid = float(self._last(symbol))
        return {
            "bid": mid,
            "ask": mid,
            "mid": mid,
            "ts": datetime.now(UTC).isoformat(),
        }


class RateLimitedProvider:
    """Simple sliding-window rate limiter wrapper for a QuoteProvider."""

    def __init__(self, inner: QuoteProvider, max_requests_per_minute: int) -> None:
        self._inner = inner
        self._max = int(max_requests_per_minute)
        self._hits = deque()  # store monotonic timestamps (seconds)

    def _prune(self, now_s: float) -> None:
        one_minute_ago = now_s - 60.0
        while self._hits and self._hits[0] < one_minute_ago:
            self._hits.popleft()

    def quote(self, symbol: str) -> dict[str, Any]:
        now_s = time.monotonic()
        self._prune(now_s)
        if len(self._hits) >= self._max:
            # Fail closed to trigger higher-level abort/lockout paths
            raise RuntimeError("RATE_LIMIT_EXCEEDED: max_requests_per_minute reached")
        self._hits.append(now_s)
        return self._inner.quote(symbol)

    # Delegate optional lifecycle methods if present
    def __getattr__(self, item):
        # Allow closing, etc., on inner providers
        return getattr(self._inner, item)


def get_quote_provider(provider_type: str = "dummy") -> QuoteProvider:
    """Factory function to get quote provider.

    Args:
        provider_type: Provider type ("dummy", "ibkr", etc.)

    Returns:
        QuoteProvider instance
    """
    if provider_type == "dummy":
        provider: QuoteProvider = DummyMidProvider(lambda s: 100.0)
    elif provider_type == "ibkr":
        from pathlib import Path

        import yaml

        from .quotes_provider_ibkr import IBKRQuoteConfig, IBKRQuoteProvider

        # Load IBKR config from overlay
        ibkr_cfg = yaml.safe_load(Path("config/brokers/ibkr.yaml").read_text())

        # Create config
        config = IBKRQuoteConfig(
            host=ibkr_cfg.get("ibkr", {}).get("host", "127.0.0.1"),
            port=ibkr_cfg.get("ibkr", {}).get("port", 7497),
            client_id=ibkr_cfg.get("ibkr", {}).get("client_id", 321),
            route=ibkr_cfg.get("ibkr", {}).get("route", "SMART"),
            currency=ibkr_cfg.get("ibkr", {}).get("currency", "USD"),
            snap_ms=ibkr_cfg.get("ibkr", {}).get("snap_ms", 150),
        )
        provider = IBKRQuoteProvider(config)
    else:
        raise ValueError(f"Unknown quote provider: {provider_type}")

    # Optional rate limiting from config/base.yaml
    try:
        from pathlib import Path
        import yaml

        base_cfg = yaml.safe_load(Path("config/base.yaml").read_text()) or {}
        rate_limit_cfg = ((base_cfg.get("risk") or {}).get("rate_limit") or {})
        max_rpm = int(rate_limit_cfg.get("max_requests_per_minute") or 0)
        if max_rpm > 0:
            provider = RateLimitedProvider(provider, max_rpm)
    except Exception:
        # On any error, proceed without rate limiting (fail-open at construction time).
        pass

    return provider
