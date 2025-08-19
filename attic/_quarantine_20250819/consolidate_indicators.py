#!/usr/bin/env python3
"""
Technical Indicators Consolidation Script

This script helps migrate the codebase from inline technical indicator calculations
to using the centralized utils/indicators.py functions. It identifies duplicate
calculations and provides migration suggestions.
"""

import logging
import os
import re
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorConsolidator:
    """Analyzes and consolidates technical indicator calculations."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.duplicate_patterns = self._get_duplicate_patterns()
        self.files_analyzed = []
        self.duplicates_found = []

    def _get_duplicate_patterns(self) -> dict[str, dict]:
        """Define patterns for duplicate indicator calculations."""
        return {
            "rsi": {
                "pattern": r"delta\s*=\s*.*\.diff\(\)\s*\n.*gain\s*=\s*.*where\(delta\s*>\s*0.*\n.*loss\s*=\s*.*where\(delta\s*<\s*0.*\n.*rs\s*=\s*gain\s*/\s*loss\s*\n.*rsi\s*=\s*100\s*-\s*\(100\s*/\s*\(1\s*\+\s*rs\)\)",
                "replacement": "from utils.indicators import rsi\nrsi_val = rsi(close, window=14)",
                "description": "RSI calculation",
            },
            "macd": {
                "pattern": r"ema.*\s*=\s*.*\.ewm\(span=\d+\)\.mean\(\)\s*\n.*macd.*\s*=\s*ema.*\s*-\s*ema.*\s*\n.*signal.*\s*=\s*macd.*\.ewm\(span=\d+\)\.mean\(\)",
                "replacement": 'from utils.indicators import macd\nmacd_data = macd(close)\nmacd_line = macd_data["macd"]\nsignal_line = macd_data["signal"]',
                "description": "MACD calculation",
            },
            "bollinger_bands": {
                "pattern": r"bb_.*\s*=\s*.*\.rolling\(window=\d+\)\.mean\(\)\s*\n.*bb_.*\s*=\s*.*\.rolling\(window=\d+\)\.std\(\)\s*\n.*bb_upper\s*=\s*bb_.*\s*\+\s*\(bb_.*\s*\*\s*\d+\)\s*\n.*bb_lower\s*=\s*bb_.*\s*-\s*\(bb_.*\s*\*\s*\d+\)",
                "replacement": 'from utils.indicators import bollinger_bands\nbb_data = bollinger_bands(close)\nbb_upper = bb_data["upper"]\nbb_lower = bb_data["lower"]',
                "description": "Bollinger Bands calculation",
            },
            "atr": {
                "pattern": r"tr1\s*=\s*high\s*-\s*low\s*\n.*tr2\s*=\s*abs\(high\s*-\s*close\.shift\(1\)\)\s*\n.*tr3\s*=\s*abs\(low\s*-\s*close\.shift\(1\)\)\s*\n.*true_range\s*=\s*pd\.concat\(\[tr1,\s*tr2,\s*tr3\],\s*axis=1\)\.max\(axis=1\)\s*\n.*atr\s*=\s*true_range\.rolling\(window=\d+\)\.mean\(\)",
                "replacement": "from utils.indicators import atr\natr_val = atr(high, low, close, window=14)",
                "description": "ATR calculation",
            },
            "rolling_mean": {
                "pattern": r"\.rolling\(window=\d+\)\.mean\(\)",
                "replacement": "from utils.indicators import rolling_mean\nrolling_mean(data, window=WINDOW)",
                "description": "Rolling mean calculation",
            },
            "rolling_std": {
                "pattern": r"\.rolling\(window=\d+\)\.std\(\)",
                "replacement": "from utils.indicators import rolling_std\nrolling_std(data, window=WINDOW)",
                "description": "Rolling standard deviation calculation",
            },
        }

    def scan_project(self) -> dict[str, list[str]]:
        """Scan the project for Python files with technical indicators."""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in ["__pycache__", "node_modules", "venv", "env"]
            ]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        logger.info(f"Found {len(python_files)} Python files to analyze")
        return python_files

    def analyze_file(self, file_path: str) -> dict[str, list[tuple[int, str]]]:
        """Analyze a single file for duplicate indicator patterns."""
        results = {}

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                content.split("\n")

            for indicator, pattern_info in self.duplicate_patterns.items():
                matches = []
                pattern = pattern_info["pattern"]

                # Find all matches
                for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                    line_num = content[: match.start()].count("\n") + 1
                    matches.append((line_num, match.group()))

                if matches:
                    results[indicator] = matches

            self.files_analyzed.append(file_path)

        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")

        return results

    def generate_migration_report(self) -> str:
        """Generate a comprehensive migration report."""
        report = []
        report.append("# Technical Indicators Consolidation Report")
        report.append("")
        report.append(f"**Generated**: {pd.Timestamp.now()}")
        report.append(f"**Files Analyzed**: {len(self.files_analyzed)}")
        report.append("")

        # Summary statistics
        total_duplicates = sum(len(duplicates) for duplicates in self.duplicates_found)
        report.append(f"**Total Duplicate Patterns Found**: {total_duplicates}")
        report.append("")

        # Detailed findings
        for file_path, duplicates in self.duplicates_found:
            if duplicates:
                report.append(f"## {file_path}")
                report.append("")

                for indicator, matches in duplicates.items():
                    pattern_info = self.duplicate_patterns[indicator]
                    report.append(f"### {pattern_info['description']} ({len(matches)} instances)")
                    report.append("")

                    for line_num, match in matches:
                        report.append(f"**Line {line_num}**:")
                        report.append("```python")
                        report.append(match[:200] + "..." if len(match) > 200 else match)
                        report.append("```")
                        report.append("")
                        report.append("**Suggested Replacement**:")
                        report.append("```python")
                        report.append(pattern_info["replacement"])
                        report.append("```")
                        report.append("")

        # Migration recommendations
        report.append("## Migration Recommendations")
        report.append("")
        report.append(
            "1. **Start with high-impact files**: Focus on files with the most duplicates"
        )
        report.append("2. **Use the centralized functions**: Import from `utils.indicators`")
        report.append("3. **Test thoroughly**: Ensure calculations remain identical")
        report.append("4. **Update imports**: Add `from utils.indicators import ...` statements")
        report.append("")

        # Performance benefits
        report.append("## Expected Benefits")
        report.append("")
        report.append("- **Reduced code duplication**: Centralized indicator calculations")
        report.append("- **Improved maintainability**: Single source of truth for indicators")
        report.append("- **Better performance**: Optimized vectorized operations")
        report.append("- **Enhanced testing**: Easier to test indicator accuracy")
        report.append("- **Consistent behavior**: Standardized calculation methods")

        return "\n".join(report)

    def create_migration_script(self, output_file: str = "migrate_indicators.py"):
        """Create a script to help with the migration."""
        script_content = '''#!/usr/bin/env python3
"""
Technical Indicators Migration Script

This script helps migrate from inline indicator calculations to centralized functions.
Run this script to automatically update your code files.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

def backup_file(file_path: str) -> str:
    """Create a backup of the file."""
    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    return backup_path

def migrate_rsi_calculations(content: str) -> str:
    """Migrate RSI calculations to use centralized function."""
    # Pattern for RSI calculation
    pattern = r"delta\\s*=\\s*.*\\.diff\\(\\)\\s*\\n.*gain\\s*=\\s*.*where\\(delta\\s*>\\s*0.*\\n.*loss\\s*=\\s*.*where\\(delta\\s*<\\s*0.*\\n.*rs\\s*=\\s*gain\\s*/\\s*loss\\s*\\n.*rsi\\s*=\\s*100\\s*-\\s*\\(100\\s*/\\s*\\(1\\s*\\+\\s*rs\\)\\)"

    def replace_rsi(match):
        return "from utils.indicators import rsi\\nrsi_val = rsi(close, window=14)"

    return re.sub(pattern, replace_rsi, content, flags=re.MULTILINE | re.DOTALL)

def migrate_macd_calculations(content: str) -> str:
    """Migrate MACD calculations to use centralized function."""
    # Pattern for MACD calculation
    pattern = r"ema.*\\s*=\\s*.*\\.ewm\\(span=\\d+\\)\\.mean\\(\\)\\s*\\n.*macd.*\\s*=\\s*ema.*\\s*-\\s*ema.*\\s*\\n.*signal.*\\s*=\\s*macd.*\\.ewm\\(span=\\d+\\)\\.mean\\(\\)"

    def replace_macd(match):
        return "from utils.indicators import macd\\nmacd_data = macd(close)\\nmacd_line = macd_data['macd']\\nsignal_line = macd_data['signal']"

    return re.sub(pattern, replace_macd, content, flags=re.MULTILINE | re.DOTALL)

def migrate_bollinger_bands(content: str) -> str:
    """Migrate Bollinger Bands calculations to use centralized function."""
    # Pattern for Bollinger Bands calculation
    pattern = r"bb_.*\\s*=\\s*.*\\.rolling\\(window=\\d+\\)\\.mean\\(\\)\\s*\\n.*bb_.*\\s*=\\s*.*\\.rolling\\(window=\\d+\\)\\.std\\(\\)\\s*\\n.*bb_upper\\s*=\\s*bb_.*\\s*\\+\\s*\\(bb_.*\\s*\\*\\s*\\d+\\)\\s*\\n.*bb_lower\\s*=\\s*bb_.*\\s*-\\s*\\(bb_.*\\s*\\*\\s*\\d+\\)"

    def replace_bb(match):
        return "from utils.indicators import bollinger_bands\\nbb_data = bollinger_bands(close)\\nbb_upper = bb_data['upper']\\nbb_lower = bb_data['lower']"

    return re.sub(pattern, replace_bb, content, flags=re.MULTILINE | re.DOTALL)

def migrate_file(file_path: str) -> bool:
    """Migrate a single file."""
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
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
            with open(file_path, 'w', encoding='utf-8') as f:
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
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        for file in files:
            if file.endswith('.py') and file != 'migrate_indicators.py':
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files to analyze")

    # Migrate files
    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            migrated_count += 1

    print(f"\\nMigration complete! {migrated_count} files updated.")
    print("Please review the changes and test your code thoroughly.")

if __name__ == "__main__":
    main()
'''

        with open(output_file, "w") as f:
            f.write(script_content)

        logger.info(f"Migration script created: {output_file}")

    def run_analysis(self) -> str:
        """Run the complete analysis."""
        logger.info("Starting technical indicators consolidation analysis...")

        # Scan for Python files
        python_files = self.scan_project()

        # Analyze each file
        for file_path in python_files:
            duplicates = self.analyze_file(file_path)
            if duplicates:
                self.duplicates_found.append((file_path, duplicates))

        # Generate report
        report = self.generate_migration_report()

        # Create migration script
        self.create_migration_script()

        logger.info(f"Analysis complete! Found duplicates in {len(self.duplicates_found)} files.")

        return report


def main():
    """Main function to run the consolidation analysis."""
    consolidator = IndicatorConsolidator()
    report = consolidator.run_analysis()

    # Save report
    with open("INDICATOR_CONSOLIDATION_REPORT.md", "w") as f:
        f.write(report)

    print("Analysis complete! Check INDICATOR_CONSOLIDATION_REPORT.md for details.")
    print("Run migrate_indicators.py to automatically update your code files.")


if __name__ == "__main__":
    main()
