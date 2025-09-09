#!/usr/bin/env python3
"""
Consolidation Report Script

Generates a comprehensive report on code consolidation status.
Run this before and after each consolidation phase to measure impact.
"""

import subprocess
from datetime import datetime
from pathlib import Path


def run_command(cmd: str, capture_output: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out: {cmd}"
    except Exception as e:
        return -1, "", f"Command failed: {e}"


def get_code_size() -> dict:
    """Get code size metrics using cloc."""
    print("== Code size ==")
    returncode, stdout, stderr = run_command("cloc . --json")
    
    if returncode == 0:
        try:
            import json
            data = json.loads(stdout)
            python_data = data.get("Python", {})
            return {
                "total_files": python_data.get("nFiles", 0),
                "total_lines": python_data.get("code", 0),
                "blank_lines": python_data.get("blank", 0),
                "comment_lines": python_data.get("comment", 0),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fallback to parsing cloc output
    returncode, stdout, stderr = run_command("cloc .")
    print(stdout)
    return {"raw_output": stdout}


def get_duplicate_code() -> dict:
    """Get duplicate code analysis using pylint."""
    print("\n== Duplicate code (pylint) ==")
    returncode, stdout, stderr = run_command(
        "pylint . --disable=all --enable=duplicate-code"
    )
    
    if returncode == 0:
        print(stdout[:1000])  # First 1000 chars
        return {"duplicate_blocks": stdout.count("duplicate-code")}
    print(f"Pylint failed: {stderr}")
    return {"error": stderr}


def get_logging_hits() -> dict:
    """Count logging setup hits."""
    print("\n== Logging hits ==")
    returncode, stdout, stderr = run_command(
        "git ls-files '*.py' | xargs grep -nE '(basicConfig\\(|StreamHandler\\(|FileHandler\\(|Formatter\\()' || true"
    )
    
    hits = stdout.strip().split('\n') if stdout.strip() else []
    hit_count = len([h for h in hits if h.strip()])
    print(f"Found {hit_count} logging setup hits")
    
    return {
        "hit_count": hit_count,
        "hits": hits[:10]  # First 10 hits for reference
    }


def get_validator_hits() -> dict:
    """Count data validation hits."""
    print("\n== Validator hits ==")
    returncode, stdout, stderr = run_command(
        "git ls-files '*.py' | xargs grep -nE '(validate_ohlcv|check_monotonic|check_duplicates|check_tz|DataSanity)' || true"
    )
    
    hits = stdout.strip().split('\n') if stdout.strip() else []
    hit_count = len([h for h in hits if h.strip()])
    print(f"Found {hit_count} validator hits")
    
    return {
        "hit_count": hit_count,
        "hits": hits[:10]  # First 10 hits for reference
    }


def get_complexity() -> dict:
    """Get code complexity using radon."""
    print("\n== Complexity (radon) ==")
    returncode, stdout, stderr = run_command("radon cc -s -a .")
    
    if returncode == 0:
        print(stdout[:1000])  # First 1000 chars
        return {"complexity_data": stdout}
    print(f"Radon failed: {stderr}")
    return {"error": stderr}


def get_test_status() -> dict:
    """Get test status."""
    print("\n== Tests ==")
    returncode, stdout, stderr = run_command("pytest -q --tb=no")
    
    if returncode == 0:
        print("All tests passed")
        return {"status": "passed", "output": stdout}
    print(f"Tests failed: {stdout}")
    return {"status": "failed", "output": stdout, "error": stderr}


def main():
    """Generate consolidation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"artifacts/consolidation_report_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating consolidation report: {output_dir}")
    
    # Collect metrics
    metrics = {
        "timestamp": timestamp,
        "code_size": get_code_size(),
        "duplicate_code": get_duplicate_code(),
        "logging_hits": get_logging_hits(),
        "validator_hits": get_validator_hits(),
        "complexity": get_complexity(),
        "test_status": get_test_status(),
    }
    
    # Save detailed report
    report_file = output_dir / "detailed_report.json"
    try:
        import json
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nDetailed report saved: {report_file}")
    except Exception as e:
        print(f"Failed to save detailed report: {e}")
    
    # Generate summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Consolidation Report - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Code Size:\n")
        if "total_lines" in metrics["code_size"]:
            f.write(f"  Python files: {metrics['code_size']['total_files']}\n")
            f.write(f"  Lines of code: {metrics['code_size']['total_lines']:,}\n")
        
        f.write(f"\nLogging Hits: {metrics['logging_hits']['hit_count']}\n")
        f.write(f"Validator Hits: {metrics['validator_hits']['hit_count']}\n")
        f.write(f"Test Status: {metrics['test_status']['status']}\n")
        
        if "duplicate_blocks" in metrics["duplicate_code"]:
            f.write(f"Duplicate Code Blocks: {metrics['duplicate_code']['duplicate_blocks']}\n")
    
    print(f"Summary saved: {summary_file}")
    
    # Print summary to console
    print(f"\n{'='*50}")
    print(f"CONSOLIDATION SUMMARY - {timestamp}")
    print(f"{'='*50}")
    print(f"Python files: {metrics['code_size'].get('total_files', 'N/A')}")
    print(f"Lines of code: {metrics['code_size'].get('total_lines', 'N/A'):,}")
    print(f"Logging hits: {metrics['logging_hits']['hit_count']}")
    print(f"Validator hits: {metrics['validator_hits']['hit_count']}")
    print(f"Test status: {metrics['test_status']['status']}")
    print(f"Report location: {output_dir}")


if __name__ == "__main__":
    main()
