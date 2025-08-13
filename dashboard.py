#!/usr/bin/env python3
"""
Trading Bot Dashboard
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px

app = Flask(__name__)

def load_performance_data():
    """Load performance data."""
    try:
        with open('results/performance_report.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def load_trade_history():
    """Load trade history."""
    try:
        return pd.read_csv('results/trade_history.csv')
    except:
        return pd.DataFrame()

def load_daily_returns():
    """Load daily returns."""
    try:
        return pd.read_csv('results/daily_returns.csv')
    except:
        return pd.DataFrame()

def create_performance_chart():
    """Create performance chart."""
    returns_df = load_daily_returns()
    if returns_df.empty:
        return None
    
    # Create cumulative returns
    returns_df['cumulative_return'] = (1 + returns_df['return']).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_df['date'],
        y=returns_df['cumulative_return'],
        mode='lines',
        name='Cumulative Return',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_trade_chart():
    """Create trade chart."""
    trades_df = load_trade_history()
    if trades_df.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['pnl'],
        mode='markers',
        name='Trade PnL',
        marker=dict(
            color=trades_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red'),
            size=8
        )
    ))
    
    fig.update_layout(
        title='Trade PnL',
        xaxis_title='Date',
        yaxis_title='PnL',
        template='plotly_white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def dashboard():
    """Main dashboard page."""
    perf_data = load_performance_data()
    trades_df = load_trade_history()
    returns_df = load_daily_returns()
    
    # Calculate metrics
    total_trades = len(trades_df) if not trades_df.empty else 0
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    
    return render_template('dashboard.html',
                         performance=perf_data,
                         total_trades=total_trades,
                         win_rate=win_rate,
                         total_pnl=total_pnl)

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance data."""
    return jsonify(load_performance_data())

@app.route('/api/trades')
def api_trades():
    """API endpoint for trade data."""
    trades_df = load_trade_history()
    if not trades_df.empty:
        return jsonify(trades_df.to_dict('records'))
    return jsonify([])

@app.route('/api/charts/performance')
def api_performance_chart():
    """API endpoint for performance chart."""
    chart_json = create_performance_chart()
    return chart_json if chart_json else jsonify({})

@app.route('/api/charts/trades')
def api_trade_chart():
    """API endpoint for trade chart."""
    chart_json = create_trade_chart()
    return chart_json if chart_json else jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
