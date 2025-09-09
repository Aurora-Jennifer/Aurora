#!/usr/bin/env python3
"""
Focused Decision Audit - Find actual decision logic, not false positives

This script looks for actual decision-making code that could conflict
with the unified decision core.
"""

import os
import re
import sys
from typing import Any

def find_actual_decision_logic(root_dir: str) -> dict[str, list[tuple[str, int, str]]]:
    """Find actual decision logic patterns"""
    patterns = {
        'action_decisions': [],
        'position_logic': [],
        'cost_logic': [],
        'tau_logic': []
    }
    
    # More specific patterns for actual decision logic
    action_patterns = [
        r'action\s*=\s*(?:0|1|2|BUY|SELL|HOLD)',
        r'action\s*=\s*argmax\(.*\)',
        r'action\s*=\s*softmax\(.*\)\.argmax\(\)',
        r'if.*action.*==.*(?:0|1|2|BUY|SELL|HOLD)',
        r'action.*=.*torch\.argmax',
        r'action.*=.*np\.argmax'
    ]
    
    position_patterns = [
        r'position\s*=\s*(?:1|-1|0)',
        r'new_pos\s*=\s*(?:1|-1|0)',
        r'pos\s*=\s*(?:1|-1|0)',
        r'if.*position.*==.*(?:1|-1|0)',
        r'position\s*[+\-]=\s*1'
    ]
    
    cost_patterns = [
        r'cost\s*=\s*.*bps',
        r'commission\s*=\s*',
        r'slippage\s*=\s*',
        r'cost.*\*.*(?:0\.0001|0\.0004)',
        r'cost.*=.*position.*\*'
    ]
    
    tau_patterns = [
        r'tau\s*[<>=]',
        r'threshold\s*[<>=]',
        r'if.*tau',
        r'if.*threshold',
        r'abs\(.*\)\s*[<>=]\s*tau'
    ]
    
    # Search files
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.pytest_cache', 'logs', 'reports', 'artifacts']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    if os.path.exists(file_path):
                        with open(file_path, encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                # Action decisions
                                for pattern in action_patterns:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        patterns['action_decisions'].append((file_path, line_num, line.strip()))
                                
                                # Position logic
                                for pattern in position_patterns:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        patterns['position_logic'].append((file_path, line_num, line.strip()))
                                
                                # Cost logic
                                for pattern in cost_patterns:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        patterns['cost_logic'].append((file_path, line_num, line.strip()))
                                
                                # Tau logic
                                for pattern in tau_patterns:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        patterns['tau_logic'].append((file_path, line_num, line.strip()))
                                
                except (UnicodeDecodeError, PermissionError, FileNotFoundError):
                    continue
    
    return patterns

def analyze_focused_patterns(patterns: dict[str, list[tuple[str, int, str]]]) -> dict[str, Any]:
    """Analyze focused patterns"""
    analysis = {
        'legacy_decision_logic': [],
        'unified_core_usage': [],
        'safe_patterns': [],
        'total_issues': 0
    }
    
    # Files that should use unified decision core
    expected_unified_files = [
        'scripts/walkforward.py',
        'scripts/falsification_harness.py',
        'core/decision_core.py'
    ]
    
    # Files that are allowed to have decision logic
    allowed_files = [
        'scripts/test_decision_core.py',
        'core/decision_core.py'
    ]
    
    for pattern_name, matches in patterns.items():
        for file_path, line_num, content in matches:
            rel_path = os.path.relpath(file_path)
            
            # Check if this is expected unified core usage
            if any(expected in rel_path for expected in expected_unified_files):
                analysis['unified_core_usage'].append((rel_path, line_num, content))
            # Check if this is allowed
            elif any(allowed in rel_path for allowed in allowed_files):
                analysis['safe_patterns'].append((rel_path, line_num, content))
            # Everything else is potentially problematic
            else:
                analysis['legacy_decision_logic'].append((rel_path, line_num, content))
                analysis['total_issues'] += 1
    
    return analysis

def generate_focused_report(analysis: dict[str, Any]) -> str:
    """Generate focused audit report"""
    report = []
    report.append("🎯 Focused Decision Logic Audit")
    report.append("=" * 40)
    
    # Summary
    report.append("\n📊 Summary:")
    report.append(f"   Total issues: {analysis['total_issues']}")
    report.append(f"   Unified core usage: {len(analysis['unified_core_usage'])}")
    report.append(f"   Safe patterns: {len(analysis['safe_patterns'])}")
    report.append(f"   Legacy decision logic: {len(analysis['legacy_decision_logic'])}")
    
    # Legacy decision logic
    if analysis['legacy_decision_logic']:
        report.append(f"\n❌ Legacy Decision Logic ({len(analysis['legacy_decision_logic'])} issues):")
        for file_path, line_num, content in analysis['legacy_decision_logic']:
            report.append(f"   {file_path}:{line_num} - {content}")
    
    # Unified core usage
    if analysis['unified_core_usage']:
        report.append(f"\n✅ Unified Core Usage ({len(analysis['unified_core_usage'])} matches):")
        for file_path, line_num, content in analysis['unified_core_usage'][:5]:
            report.append(f"   {file_path}:{line_num} - {content}")
        if len(analysis['unified_core_usage']) > 5:
            report.append(f"   ... and {len(analysis['unified_core_usage']) - 5} more")
    
    # Safe patterns
    if analysis['safe_patterns']:
        report.append(f"\n✅ Safe Patterns ({len(analysis['safe_patterns'])} matches):")
        for file_path, line_num, content in analysis['safe_patterns'][:3]:
            report.append(f"   {file_path}:{line_num} - {content}")
        if len(analysis['safe_patterns']) > 3:
            report.append(f"   ... and {len(analysis['safe_patterns']) - 3} more")
    
    return "\n".join(report)

def main():
    """Main function"""
    print("🎯 Starting Focused Decision Logic Audit...")
    
    # Find patterns
    patterns = find_actual_decision_logic(".")
    
    # Analyze patterns
    analysis = analyze_focused_patterns(patterns)
    
    # Generate report
    report = generate_focused_report(analysis)
    
    print(report)
    
    # Return exit code
    if analysis['total_issues'] > 0:
        print(f"\n❌ Found {analysis['total_issues']} potential issues")
        return 1
    print("\n✅ No issues found")
    return 0

if __name__ == "__main__":
    sys.exit(main())
