"""
Timezone utilities for consistent datetime handling.
"""

from datetime import datetime, timezone
from dateutil import parser as dtparse


def ensure_aware_utc(dt) -> datetime:
    """
    Ensure datetime is timezone-aware in UTC.
    
    Args:
        dt: datetime object, string, or None
        
    Returns:
        timezone-aware datetime in UTC, or None if input is None
    """
    if dt is None:
        return None
        
    if isinstance(dt, str):
        dt = dtparse.isoparse(dt)  # respects embedded TZ if present
        
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
        
    return dt


def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(timezone.utc)
