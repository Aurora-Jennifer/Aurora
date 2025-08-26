#!/bin/bash

# Performance Analysis Starter Script
# This script sets up the environment and starts profiling the ML walkforward system

set -e

echo "🚀 Starting Performance Analysis for ML Walkforward System"
echo "=========================================================="

# Create performance directories
mkdir -p .perf
mkdir -p results/perf_baseline

# Install profiling tools if not already installed
echo "📦 Installing profiling tools..."
pip install line_profiler memory_profiler psutil pyarrow

# Set environment variables
export PYTHONPATH=/home/Jennifer/projects/trader
export PYTHONUNBUFFERED=1

echo "🔍 Running baseline performance measurements..."

# Small workload profile (2 years, 4 folds)
echo "Running small workload profile..."
python -m cProfile -o .perf/small_profile.pstats scripts/ml_walkforward.py \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --fold-length 252 \
  --step-size 126 \
  --warm-start \
  --config config/ml_backtest_unified.json \
  --output-dir results/perf_baseline/small

echo "✅ Small workload profile completed"

# Medium workload profile (5 years, 10 folds)
echo "Running medium workload profile..."
python -m cProfile -o .perf/medium_profile.pstats scripts/ml_walkforward.py \
  --start-date 2018-01-01 \
  --end-date 2022-12-31 \
  --fold-length 252 \
  --step-size 126 \
  --warm-start \
  --config config/ml_backtest_unified.json \
  --output-dir results/perf_baseline/medium

echo "✅ Medium workload profile completed"

# Analyze profiles
echo "📊 Analyzing performance profiles..."
echo ""
echo "=== SMALL WORKLOAD HOTSPOTS ==="
python -c "
import pstats
p = pstats.Stats('.perf/small_profile.pstats')
p.sort_stats('cumulative').print_stats(10)
"

echo ""
echo "=== MEDIUM WORKLOAD HOTSPOTS ==="
python -c "
import pstats
p = pstats.Stats('.perf/medium_profile.pstats')
p.sort_stats('cumulative').print_stats(10)
"

echo ""
echo "🎯 Performance analysis completed!"
echo "📁 Results saved to:"
echo "   - .perf/small_profile.pstats"
echo "   - .perf/medium_profile.pstats"
echo "   - results/perf_baseline/"
echo ""
echo "📋 Next steps:"
echo "   1. Review the hotspots above"
echo "   2. Implement parallelization (see PERF_PLAN.md)"
echo "   3. Run line profiling on top functions"
echo "   4. Begin optimization implementation"
