#!/usr/bin/env python3
"""
Audit Decision Paths - Find and flag all decision logic in the codebase

This script audits the entire codebase to identify all decision paths
and ensure only the unified decision core is used in production.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))



def find_decision_patterns(root_dir: str) -> dict[str, list[tuple[str, int, str]]]:
    """
    Find all decision-related patterns in the codebase.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        Dictionary mapping pattern names to list of (file, line, content) tuples
    """
    patterns = {
        'action_assignments': [],
        'argmax_calls': [],
        'softmax_calls': [],
        'buy_sell_hold': [],
        'position_updates': [],
        'cost_calculations': [],
        'tau_thresholds': [],
        'decision_logic': []
    }
    
    # Pattern definitions
    action_patterns = [
        r'action\s*=\s*[012]',
        r'action\s*=\s*argmax',
        r'action\s*=\s*softmax',
        r'action\s*=\s*BUY|SELL|HOLD',
        r'action\s*=\s*0|1|2'
    ]
    
    argmax_patterns = [
        r'\.argmax\(\)',
        r'torch\.argmax',
        r'np\.argmax'
    ]
    
    softmax_patterns = [
        r'\.softmax\(\)',
        r'torch\.softmax',
        r'F\.softmax'
    ]
    
    buy_sell_hold_patterns = [
        r'BUY|SELL|HOLD',
        r'buy|sell|hold'
    ]
    
    position_patterns = [
        r'position\s*[+\-]?=',
        r'pos\s*[+\-]?=',
        r'new_pos\s*=',
        r'prev_pos\s*='
    ]
    
    cost_patterns = [
        r'cost\s*[+\-]?=',
        r'commission',
        r'slippage',
        r'bps'
    ]
    
    tau_patterns = [
        r'tau',
        r'threshold',
        r'gate'
    ]
    
    decision_patterns = [
        r'if.*action',
        r'if.*position',
        r'if.*tau',
        r'if.*threshold'
    ]
    
    # Search patterns
    all_patterns = [
        ('action_assignments', action_patterns),
        ('argmax_calls', argmax_patterns),
        ('softmax_calls', softmax_patterns),
        ('buy_sell_hold', buy_sell_hold_patterns),
        ('position_updates', position_patterns),
        ('cost_calculations', cost_patterns),
        ('tau_thresholds', tau_patterns),
        ('decision_logic', decision_patterns)
    ]
    
    # Search files
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.pytest_cache']]
        
        for file in files:
            if file.endswith(('.py', '.yaml', '.yml', '.json')):
                file_path = os.path.join(root, file)
                try:
                    if os.path.exists(file_path):
                        with open(file_path, encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                for pattern_name, pattern_list in all_patterns:
                                    for pattern in pattern_list:
                                        if re.search(pattern, line, re.IGNORECASE):
                                            patterns[pattern_name].append((file_path, line_num, line.strip()))
                except (UnicodeDecodeError, PermissionError, FileNotFoundError):
                    continue
    
    return patterns


def analyze_decision_paths(patterns: dict[str, list[tuple[str, int, str]]]) -> dict[str, Any]:
    """
    Analyze decision paths and categorize them.
    
    Args:
        patterns: Dictionary of found patterns
        
    Returns:
        Analysis results
    """
    analysis = {
        'legacy_paths': [],
        'unified_core_usage': [],
        'suspicious_patterns': [],
        'safe_patterns': [],
        'total_matches': 0
    }
    
    # Files that should use unified decision core
    expected_unified_files = [
        'scripts/walkforward.py',
        'scripts/falsification_harness.py',
        'core/decision_core.py'
    ]
    
    # Files that are allowed to have decision logic (legacy)
    allowed_legacy_files = [
        'scripts/test_decision_core.py',
        'core/decision_core.py'
    ]
    
    for pattern_name, matches in patterns.items():
        analysis['total_matches'] += len(matches)
        
        for file_path, line_num, content in matches:
            # Normalize file path
            rel_path = os.path.relpath(file_path)
            
            # Check if this is expected unified core usage
            if any(expected in rel_path for expected in expected_unified_files):
                analysis['unified_core_usage'].append((rel_path, line_num, content))
            # Check if this is allowed legacy usage
            elif any(allowed in rel_path for allowed in allowed_legacy_files):
                analysis['safe_patterns'].append((rel_path, line_num, content))
            # Check for suspicious patterns
            elif pattern_name in ['action_assignments', 'argmax_calls', 'decision_logic']:
                analysis['suspicious_patterns'].append((rel_path, line_num, content))
            # Everything else is potentially legacy
            else:
                analysis['legacy_paths'].append((rel_path, line_num, content))
    
    return analysis


def generate_audit_report(analysis: dict[str, Any]) -> str:
    """
    Generate a comprehensive audit report.
    
    Args:
        analysis: Analysis results
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("🔍 Decision Paths Audit Report")
    report.append("=" * 50)
    
    # Summary
    report.append("\n📊 Summary:")
    report.append(f"   Total matches: {analysis['total_matches']}")
    report.append(f"   Unified core usage: {len(analysis['unified_core_usage'])}")
    report.append(f"   Safe patterns: {len(analysis['safe_patterns'])}")
    report.append(f"   Suspicious patterns: {len(analysis['suspicious_patterns'])}")
    report.append(f"   Legacy paths: {len(analysis['legacy_paths'])}")
    
    # Unified core usage
    if analysis['unified_core_usage']:
        report.append(f"\n✅ Unified Core Usage ({len(analysis['unified_core_usage'])} matches):")
        for file_path, line_num, content in analysis['unified_core_usage'][:10]:  # Show first 10
            report.append(f"   {file_path}:{line_num} - {content}")
        if len(analysis['unified_core_usage']) > 10:
            report.append(f"   ... and {len(analysis['unified_core_usage']) - 10} more")
    
    # Safe patterns
    if analysis['safe_patterns']:
        report.append(f"\n✅ Safe Patterns ({len(analysis['safe_patterns'])} matches):")
        for file_path, line_num, content in analysis['safe_patterns'][:5]:  # Show first 5
            report.append(f"   {file_path}:{line_num} - {content}")
        if len(analysis['safe_patterns']) > 5:
            report.append(f"   ... and {len(analysis['safe_patterns']) - 5} more")
    
    # Suspicious patterns
    if analysis['suspicious_patterns']:
        report.append(f"\n⚠️  Suspicious Patterns ({len(analysis['suspicious_patterns'])} matches):")
        for file_path, line_num, content in analysis['suspicious_patterns']:
            report.append(f"   {file_path}:{line_num} - {content}")
    
    # Legacy paths
    if analysis['legacy_paths']:
        report.append(f"\n❌ Legacy Paths ({len(analysis['legacy_paths'])} matches):")
        for file_path, line_num, content in analysis['legacy_paths']:
            report.append(f"   {file_path}:{line_num} - {content}")
    
    # Recommendations
    report.append("\n💡 Recommendations:")
    if analysis['suspicious_patterns']:
        report.append("   1. Review suspicious patterns for potential legacy decision logic")
    if analysis['legacy_paths']:
        report.append("   2. Refactor legacy paths to use unified decision core")
    report.append("   3. Add feature flags to disable legacy paths in production")
    report.append("   4. Run this audit regularly to catch new decision paths")
    
    return "\n".join(report)


def main():
    """Main audit function"""
    parser = argparse.ArgumentParser(description="Audit decision paths in codebase")
    parser.add_argument("--root", default=".", help="Root directory to search")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🔍 Starting Decision Paths Audit...")
    
    # Find patterns
    patterns = find_decision_patterns(args.root)
    
    # Analyze patterns
    analysis = analyze_decision_paths(patterns)
    
    # Generate report
    report = generate_audit_report(analysis)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {args.output}")
    else:
        print(report)
    
    # Return exit code based on findings
    if analysis['suspicious_patterns'] or analysis['legacy_paths']:
        print("\n❌ Audit found issues that need attention")
        return 1
    print("\n✅ Audit passed - no issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
