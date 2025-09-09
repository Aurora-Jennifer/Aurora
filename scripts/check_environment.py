#!/usr/bin/env python3
"""
Environment validation script for production runs.

Checks for required dependencies and provides clear error messages
for missing components without failing the entire run.
"""

import sys
import importlib
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_python_version() -> str:
    """Get Python version string"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_dependency(module_name: str, package_name: Optional[str] = None) -> Dict[str, any]:
    """Check if a dependency is available and get version info"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return {
            'available': True,
            'version': version,
            'package': package_name
        }
    except ImportError:
        return {
            'available': False,
            'version': None,
            'package': package_name
        }


def check_lightgbm() -> Dict[str, any]:
    """Special check for LightGBM with detailed error info"""
    result = check_dependency('lightgbm')
    if not result['available']:
        result['error'] = (
            "LightGBM not available. This is expected on Python 3.13. "
            "LGBM grid items will be skipped. Use conda environment with Python 3.11/3.12 "
            "for full LightGBM support."
        )
    return result


def get_feature_schema_hash() -> str:
    """Generate hash of feature schema for reproducibility"""
    try:
        # Hash the features module to detect schema changes
        features_path = Path("ml/features.py")
        if features_path.exists():
            content = features_path.read_text()
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        return "unknown"
    except Exception:
        return "unknown"


def validate_environment() -> Dict[str, any]:
    """Comprehensive environment validation"""
    
    # Core dependencies
    core_deps = ['numpy', 'pandas', 'scipy', 'sklearn', 'yfinance']
    optional_deps = ['lightgbm', 'xgboost', 'torch']
    
    results = {
        'python_version': get_python_version(),
        'git_commit': get_git_commit(),
        'feature_schema_hash': get_feature_schema_hash(),
        'dependencies': {},
        'missing_models': [],
        'warnings': [],
        'errors': []
    }
    
    # Check core dependencies
    for dep in core_deps:
        results['dependencies'][dep] = check_dependency(dep)
        if not results['dependencies'][dep]['available']:
            results['errors'].append(f"Core dependency {dep} is missing")
    
    # Check optional dependencies
    for dep in optional_deps:
        if dep == 'lightgbm':
            results['dependencies'][dep] = check_lightgbm()
        else:
            results['dependencies'][dep] = check_dependency(dep)
        
        if not results['dependencies'][dep]['available']:
            results['missing_models'].append(dep)
            if dep == 'lightgbm':
                results['warnings'].append(results['dependencies'][dep].get('error', ''))
            else:
                results['warnings'].append(f"Optional dependency {dep} is missing")
    
    # Python version warnings
    if sys.version_info >= (3, 13):
        results['warnings'].append(
            "Python 3.13+ detected. LightGBM may not be available. "
            "Consider using Python 3.11/3.12 for full compatibility."
        )
    
    return results


def main():
    """Main validation function"""
    print("🔍 Validating environment...")
    
    env_info = validate_environment()
    
    # Print summary
    print(f"\n📊 Environment Summary:")
    print(f"  Python: {env_info['python_version']}")
    print(f"  Git commit: {env_info['git_commit'][:8]}")
    print(f"  Feature schema: {env_info['feature_schema_hash']}")
    
    # Print dependency status
    print(f"\n📦 Dependencies:")
    for name, info in env_info['dependencies'].items():
        status = "✅" if info['available'] else "❌"
        version = f" (v{info['version']})" if info['available'] and info['version'] != 'unknown' else ""
        print(f"  {status} {name}{version}")
    
    # Print warnings
    if env_info['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in env_info['warnings']:
            print(f"  • {warning}")
    
    # Print errors
    if env_info['errors']:
        print(f"\n❌ Errors:")
        for error in env_info['errors']:
            print(f"  • {error}")
        print(f"\n💥 Environment validation failed!")
        sys.exit(1)
    
    # Save environment info
    output_file = Path("results/environment_info.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"\n✅ Environment validation passed!")
    print(f"📄 Environment info saved to: {output_file}")
    
    if env_info['missing_models']:
        print(f"⚠️  Missing models will be skipped: {env_info['missing_models']}")


if __name__ == "__main__":
    main()
