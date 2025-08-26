#!/usr/bin/env python3
"""
Check if Phase 2 files are imported by current core/organized files.

This script checks if any files in cleanup+remove/phase2_conservative/ are imported
by files in the current organized structure (core/, training/, walkforward/, etc.).
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Set, Dict, List

def get_phase2_files() -> Set[str]:
    """Get all files in phase2_conservative"""
    phase2_dir = Path("scripts/cleanup+remove/phase2_conservative")
    if not phase2_dir.exists():
        return set()
    
    files = set()
    for file in phase2_dir.iterdir():
        if file.is_file() and file.suffix == '.py':
            # Remove .py extension for import matching
            files.add(file.stem)
    return files

def get_core_files() -> Set[str]:
    """Get all Python files in organized directories"""
    core_dirs = [
        "scripts/core",
        "scripts/training", 
        "scripts/walkforward",
        "scripts/validation",
        "scripts/data",
        "scripts/analysis",
        "scripts/experiments",
        "scripts/tools"
    ]
    
    files = set()
    for dir_path in core_dirs:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            for file in dir_obj.glob("*.py"):
                files.add(str(file))
    return files

def check_imports_in_file(file_path: str, phase2_modules: Set[str]) -> Set[str]:
    """Check if a file imports any Phase 2 modules"""
    imports = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for various import patterns
        patterns = [
            r'from\s+scripts\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # from scripts.module import
            r'import\s+scripts\.([a-zA-Z_][a-zA-Z0-9_]*)',       # import scripts.module
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',         # from module import (relative)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match in phase2_modules:
                    imports.add(match)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def check_dynamic_imports(file_path: str, phase2_modules: Set[str]) -> Set[str]:
    """Check for dynamic imports (importlib, __import__, etc.)"""
    imports = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for dynamic import patterns
        patterns = [
            r'importlib\.import_module\([\'"]([^\'"]+)[\'"]',  # importlib.import_module("module")
            r'__import__\([\'"]([^\'"]+)[\'"]',               # __import__("module")
            r'exec\([\'"]([^\'"]*import[^\'"]*)[\'"]',        # exec("import module")
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Extract module name from dynamic import
                if 'scripts.' in match:
                    module = match.split('scripts.')[-1].split('.')[0]
                    if module in phase2_modules:
                        imports.add(module)
                        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def main():
    """Main analysis function"""
    print("🔍 Checking Phase 2 import dependencies...")
    print()
    
    phase2_modules = get_phase2_files()
    core_files = get_core_files()
    
    print(f"📊 SUMMARY:")
    print(f"   Phase 2 modules: {len(phase2_modules)}")
    print(f"   Core files to check: {len(core_files)}")
    print()
    
    if not phase2_modules:
        print("❌ No Phase 2 modules found")
        return
    
    if not core_files:
        print("❌ No core files found")
        return
    
    # Check each core file for imports
    dependencies = {}
    total_imports = set()
    
    for file_path in core_files:
        static_imports = check_imports_in_file(file_path, phase2_modules)
        dynamic_imports = check_dynamic_imports(file_path, phase2_modules)
        all_imports = static_imports | dynamic_imports
        
        if all_imports:
            dependencies[file_path] = all_imports
            total_imports.update(all_imports)
    
    # Report results
    if dependencies:
        print("⚠️  DEPENDENCIES FOUND (Phase 2 files are imported):")
        print()
        for file_path, imports in dependencies.items():
            print(f"   📁 {file_path}")
            for imp in imports:
                print(f"      └── imports: {imp}")
        print()
        
        print("🚨 RECOMMENDATION:")
        print(f"   {len(total_imports)} Phase 2 modules are imported by core files.")
        print("   These files should NOT be removed until imports are updated.")
        print()
        
        print("📋 AFFECTED PHASE 2 MODULES:")
        for module in sorted(total_imports):
            print(f"   ❌ {module}")
        print()
        
    else:
        print("✅ NO DEPENDENCIES FOUND")
        print("   Phase 2 files are not imported by any core files.")
        print("   Safe to proceed with Phase 2 removal.")
        print()
    
    # Save detailed report
    report = {
        "phase2_modules": sorted(list(phase2_modules)),
        "core_files_checked": sorted(list(core_files)),
        "dependencies": {str(k): list(v) for k, v in dependencies.items()},
        "total_imports": sorted(list(total_imports))
    }
    
    report_file = "artifacts/phase2_import_analysis.json"
    os.makedirs("artifacts", exist_ok=True)
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
