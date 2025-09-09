#!/usr/bin/env python3
"""
Date alignment helpers for trading decisions vs feature dates.

Key insight: We trade on decision_date using features from feature_date (previous session).
This prevents lookahead bias and aligns with how features are timestamped.
"""
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta


# Initialize NYSE calendar
XNYS = mcal.get_calendar("XNYS")


def prev_session(d):
    """
    Get the previous trading session before date d.
    
    Args:
        d: Date (string, pd.Timestamp, or date object)
        
    Returns:
        date: Previous trading session
    """
    d = pd.Timestamp(d)
    
    # Look back 10 days to find recent sessions
    start_date = d - pd.Timedelta(days=10)
    end_date = d
    
    schedule = XNYS.schedule(start_date=start_date, end_date=end_date)
    
    if schedule.empty:
        raise ValueError(f"No trading sessions found before {d}")
    
    # Find sessions strictly before d
    before_d = schedule.index[schedule.index.date < d.date()]
    
    if before_d.empty:
        raise ValueError(f"No trading sessions found before {d}")
    
    return before_d[-1].date()


def next_session(d):
    """
    Get the next trading session after date d.
    
    Args:
        d: Date (string, pd.Timestamp, or date object)
        
    Returns:
        date: Next trading session
    """
    d = pd.Timestamp(d)
    
    # Look ahead 10 days to find upcoming sessions
    start_date = d
    end_date = d + pd.Timedelta(days=10)
    
    schedule = XNYS.schedule(start_date=start_date, end_date=end_date)
    
    if schedule.empty:
        raise ValueError(f"No trading sessions found after {d}")
    
    # Find sessions strictly after d
    after_d = schedule.index[schedule.index.date > d.date()]
    
    if after_d.empty:
        # If d is a trading day and we want the next one
        same_or_after = schedule.index[schedule.index.date >= d.date()]
        if len(same_or_after) > 1:
            return same_or_after[1].date()
        else:
            raise ValueError(f"No trading sessions found after {d}")
    
    return after_d[0].date()


def get_decision_and_feature_dates(now=None):
    """
    Get aligned decision_date and feature_date for trading.
    
    Logic:
    - Pre-market (before 9:30 AM ET): decision_date = today, feature_date = prev_session(today)
    - After close (after 4:00 PM ET): decision_date = next_session(today), feature_date = today
    
    Args:
        now: Current timestamp (defaults to now)
        
    Returns:
        tuple: (decision_date, feature_date) as date objects
    """
    if now is None:
        now = pd.Timestamp.now(tz='America/New_York')
    else:
        now = pd.Timestamp(now, tz='America/New_York')
    
    today = now.date()
    
    # Check if today is a trading day
    schedule = XNYS.schedule(start_date=today, end_date=today)
    is_trading_day = not schedule.empty
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = pd.Timestamp("09:30", tz="America/New_York").time()
    market_close = pd.Timestamp("16:00", tz="America/New_York").time()
    current_time = now.time()
    
    if is_trading_day and current_time < market_open:
        # Pre-market: trade today using yesterday's features
        decision_date = today
        feature_date = prev_session(today)
    elif is_trading_day and current_time >= market_close:
        # After close: trade tomorrow using today's features
        decision_date = next_session(today)
        feature_date = today
    elif is_trading_day:
        # During market hours: trade today using yesterday's features
        decision_date = today
        feature_date = prev_session(today)
    else:
        # Non-trading day: trade next session using prev session features
        decision_date = next_session(today)
        feature_date = prev_session(decision_date)
    
    return decision_date, feature_date


def validate_date_alignment(decision_date, feature_date):
    """
    Validate that feature_date is before decision_date to prevent lookahead.
    
    Args:
        decision_date: Date we're making trading decisions for
        feature_date: Date features are calculated from
        
    Returns:
        bool: True if alignment is valid
    """
    decision_date = pd.Timestamp(decision_date).date()
    feature_date = pd.Timestamp(feature_date).date()
    
    if feature_date >= decision_date:
        raise ValueError(f"Lookahead detected: feature_date {feature_date} >= decision_date {decision_date}")
    
    # Check both are trading days
    for date, name in [(decision_date, 'decision'), (feature_date, 'feature')]:
        schedule = XNYS.schedule(start_date=date, end_date=date)
        if schedule.empty:
            raise ValueError(f"{name}_date {date} is not a trading day")
    
    return True


def last_trading_day():
    """
    Get the most recent trading day (today if trading day, otherwise previous trading day).
    
    Returns:
        str: Last trading day in YYYY-MM-DD format
    """
    today = pd.Timestamp.now(tz='America/New_York').date()
    
    # Check if today is a trading day
    schedule = XNYS.schedule(start_date=today, end_date=today)
    if not schedule.empty:
        return today.strftime('%Y-%m-%d')
    
    # Today is not a trading day, get previous session
    return prev_session(today).strftime('%Y-%m-%d')


def get_feature_date_range(feature_date, lookback_days=60):
    """
    Get the date range needed for feature calculation with sufficient lookback.
    
    Args:
        feature_date: Target date for features
        lookback_days: Number of days of history needed
        
    Returns:
        tuple: (start_date, end_date) for data fetching
    """
    feature_date = pd.Timestamp(feature_date)
    
    # Add buffer for weekends and holidays
    start_date = feature_date - pd.Timedelta(days=lookback_days + 7)
    end_date = feature_date
    
    return start_date.date(), end_date.date()


def main():
    """Test the date helpers."""
    print("📅 TESTING DATE ALIGNMENT HELPERS")
    print("=" * 40)
    
    # Test current date alignment
    try:
        decision_date, feature_date = get_decision_and_feature_dates()
        print(f"✅ Current alignment:")
        print(f"   Decision date: {decision_date}")
        print(f"   Feature date: {feature_date}")
        
        # Validate alignment
        validate_date_alignment(decision_date, feature_date)
        print(f"✅ Date alignment valid (no lookahead)")
        
        # Test date range for features
        start_date, end_date = get_feature_date_range(feature_date)
        print(f"✅ Feature data range:")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}")
        print(f"   Days: {(pd.Timestamp(end_date) - pd.Timestamp(start_date)).days}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Test edge cases
    print(f"\n🧪 Testing edge cases:")
    
    test_dates = ['2025-01-01', '2025-07-04', '2025-12-25']  # Holidays
    for test_date in test_dates:
        try:
            prev = prev_session(test_date)
            next_sess = next_session(test_date)
            print(f"   {test_date}: prev={prev}, next={next_sess}")
        except Exception as e:
            print(f"   {test_date}: ⚠️ {e}")
    
    print(f"\n✅ Date helpers validated!")
    return 0


if __name__ == "__main__":
    exit(main())
