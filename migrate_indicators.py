#!/usr/bin/env python3
"""
Technical Indicators Migration Script

This script helps migrate from inline indicator calculations to centralized functions.
Run this script to automatically update your code files.
"""

import os
import re
import shutil


def backup_file(file_path: str) -> str:
    """Create a backup of the file."""
    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    return backup_path


def migrate_rsi_calculations(content: str) -> str:
    """Migrate RSI calculations to use centralized function."""
    # Pattern for RSI calculation
    pattern = (
        r"delta\s*=\s*.*\.diff\(\)\s*\n.*gain\s*=\s*.*where\(delta\s*>\s*0.*\n.*"
        r"loss\s*=\s*.*where\(delta\s*<\s*0.*\n.*rs\s*=\s*gain\s*/\s*loss\s*\n.*"
        r"rsi\s*=\s*100\s*-\s*\(100\s*/\s*\(1\s*\+\s*rs\)\)"
    )

    def replace_rsi(match):
        return "from utils.indicators import rsi\nrsi_val = rsi(close, window=14)"

    return re.sub(pattern, replace_rsi, content, flags=re.MULTILINE | re.DOTALL)


def migrate_macd_calculations(content: str) -> str:
    """Migrate MACD calculations to use centralized function."""
    # Pattern for MACD calculation
    pattern = r"ema.*\s*=\s*.*\.ewm\(span=\d+\)\.mean\(\)\s*\n.*macd.*\s*=\s*ema.*\s*-\s*ema.*\s*\n.*signal.*\s*=\s*macd.*\.ewm\(span=\d+\)\.mean\(\)"

    def replace_macd(match):
        return "from utils.indicators import macd\nmacd_data = macd(close)\nmacd_line = macd_data['macd']\nsignal_line = macd_data['signal']"

    return re.sub(pattern, replace_macd, content, flags=re.MULTILINE | re.DOTALL)


def migrate_bollinger_bands(content: str) -> str:
    """Migrate Bollinger Bands calculations to use centralized function."""
    # Pattern for Bollinger Bands calculation
    pattern = (
        r"bb_.*\s*=\s*.*\.rolling\(window=\d+\)\.mean\(\)\s*\n.*"
        r"bb_.*\s*=\s*.*\.rolling\(window=\d+\)\.std\(\)\s*\n.*"
        r"bb_upper\s*=\s*bb_.*\s*\+\s*\(bb_.*\s*\*\s*\d+\)\s*\n.*"
        r"bb_lower\s*=\s*bb_.*\s*-\s*\(bb_.*\s*\*\s*\d+\)"
    )

    def replace_bb(match):
        return "from utils.indicators import bollinger_bands\nbb_data = bollinger_bands(close)\nbb_upper = bb_data['upper']\nbb_lower = bb_data['lower']"

    return re.sub(pattern, replace_bb, content, flags=re.MULTILINE | re.DOTALL)


def migrate_file(file_path: str) -> bool:
    """Migrate a single file."""
    try:
        # Read file
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Create backup
        backup_path = backup_file(file_path)
        print(f"Backed up {file_path} to {backup_path}")

        # Apply migrations
        original_content = content
        content = migrate_rsi_calculations(content)
        content = migrate_macd_calculations(content)
        content = migrate_bollinger_bands(content)

        # Check if content changed
        if content != original_content:
            # Write updated content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Migrated {file_path}")
            return True
        else:
            print(f"No changes needed for {file_path}")
            return False

    except Exception as e:
        print(f"Error migrating {file_path}: {e}")
        return False


def main():
    """Main migration function."""
    print("Technical Indicators Migration Script")
    print("=" * 40)

    # Get list of Python files to migrate
    python_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ["__pycache__", "node_modules", "venv", "env"]
        ]
        for file in files:
            if file.endswith(".py") and file != "migrate_indicators.py":
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files to analyze")

    # Migrate files
    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            migrated_count += 1

    print(f"\nMigration complete! {migrated_count} files updated.")
    print("Please review the changes and test your code thoroughly.")


if __name__ == "__main__":
    main()
