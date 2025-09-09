#!/usr/bin/env python3
"""
Get the last NYSE trading day relative to a given timestamp.
Handles weekends, holidays, and market hours properly.
"""
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, date, timedelta
import argparse


def last_trading_day(ts=None, tz="America/Chicago"):
    """
    Get the last NYSE trading day relative to timestamp.
    
    Args:
        ts: Timestamp to use as reference (default: now)
        tz: Timezone for interpretation (default: America/Chicago)
        
    Returns:
        date: Last trading day as date object
    """
    try:
        # Handle timezone-aware timestamp
        if ts is None:
            now = pd.Timestamp.now(tz=tz)
        else:
            now = pd.Timestamp(ts, tz=tz)
        
        # Get NYSE calendar
        nyse = mcal.get_calendar("XNYS")
        
        # Look back 10 business days to find recent sessions
        end_date = now.date()
        start_date = end_date - timedelta(days=15)  # Buffer for holidays
        
        # Get trading schedule
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        
        if schedule.empty:
            raise ValueError(f"No trading sessions found between {start_date} and {end_date}")
        
        # Convert now to NY time for market hours comparison
        ny_now = now.tz_convert("America/New_York")
        
        # Find the most recent trading session
        # If today is a trading day and after market close (4 PM ET), use today
        # Otherwise, use the previous trading session
        
        trading_sessions = schedule.index
        
        # Check if today is a trading day
        today_date = ny_now.date()
        if pd.Timestamp(today_date) in trading_sessions:
            # Today is a trading day - check if market is closed
            market_close = pd.Timestamp("16:00", tz="America/New_York").time()
            if ny_now.time() >= market_close:
                # After close, use today
                last_session = pd.Timestamp(today_date)
            else:
                # Before close, use previous session
                previous_sessions = trading_sessions[trading_sessions < pd.Timestamp(today_date)]
                if not previous_sessions.empty:
                    last_session = previous_sessions.max()
                else:
                    last_session = pd.Timestamp(today_date)
        else:
            # Today is not a trading day, use most recent session
            last_session = trading_sessions.max()
        
        return last_session.date()
        
    except Exception as e:
        # Fallback: use simple business day logic
        print(f"⚠️  Calendar error: {e}, using business day fallback")
        
        if ts is None:
            ref_date = datetime.now().date()
        else:
            ref_date = pd.Timestamp(ts).date()
            
        # Simple fallback - go back to most recent weekday
        while ref_date.weekday() >= 5:  # Saturday=5, Sunday=6
            ref_date -= timedelta(days=1)
            
        return ref_date


def validate_trading_day(target_date):
    """Validate that a date is a trading day."""
    try:
        nyse = mcal.get_calendar("XNYS")
        schedule = nyse.schedule(start_date=target_date, end_date=target_date)
        return not schedule.empty
    except Exception:
        # Fallback - check if it's a weekday
        if isinstance(target_date, str):
            target_date = pd.Timestamp(target_date).date()
        return target_date.weekday() < 5


def main():
    parser = argparse.ArgumentParser(description='Get last trading day')
    parser.add_argument('--date', help='Reference date (default: now)')
    parser.add_argument('--tz', default='America/Chicago', help='Timezone')
    parser.add_argument('--validate', help='Validate if given date is trading day')
    
    args = parser.parse_args()
    
    if args.validate:
        is_trading = validate_trading_day(args.validate)
        print(f"📅 {args.validate} is {'✅ trading day' if is_trading else '❌ not trading day'}")
        return
    
    try:
        last_day = last_trading_day(args.date, args.tz)
        print(f"📅 Last trading day: {last_day}")
        
        # Validation
        is_trading = validate_trading_day(last_day)
        print(f"✅ Validation: {last_day} is {'trading day' if is_trading else 'NOT trading day'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
