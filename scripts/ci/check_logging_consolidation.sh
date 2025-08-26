#!/usr/bin/env bash
set -euo pipefail

# Logging Consolidation CI Gate
# This script enforces that all logging setup uses the centralized core.utils.logging module

echo "🔍 Checking logging consolidation contract..."

# Find violations of ad-hoc logging setup
violations=$(git ls-files '*.py' | xargs grep -nE 'basicConfig\(|StreamHandler\(|FileHandler\(|Formatter\(' || true)

if [[ -n "$violations" ]]; then
    echo "❌ Logging consolidation violation(s) found:"
    echo "$violations"
    echo ""
    echo "💡 Fix by using centralized logging:"
    echo "   from core.utils import setup_logging"
    echo "   logger = setup_logging('logs/script_name.log', logging.INFO)"
    echo ""
    echo "📁 Allowed exceptions:"
    echo "   - core/utils/logging.py (the centralized module itself)"
    echo "   - tests/ (test files may have custom logging)"
    echo "   - attic/ (archived code)"
    exit 1
fi

echo "✅ Logging consolidation contract honored."
echo "   All scripts use centralized logging setup."
