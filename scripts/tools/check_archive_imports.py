#!/usr/bin/env python3
"""
Check if remaining archive files are imported by current core/organized files.

This script checks if any files in scripts/archive/ are imported
by files in the current organized structure (core/, training/, walkforward/, etc.).
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Set, Dict, List

def get_archive_files() -> Set[str]:
    """Get all Python files in scripts/archive/"""
    archive_dir = Path("scripts/archive")
    if not archive_dir.exists():
        return set()
    
    files = set()
    for file in archive_dir.iterdir():
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

def check_imports_in_file(file_path: str, archive_modules: Set[str]) -> Set[str]:
    """Check if a file imports any archive modules"""
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
                if match in archive_modules:
                    imports.add(match)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def check_dynamic_imports(file_path: str, archive_modules: Set[str]) -> Set[str]:
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
                    if module in archive_modules:
                        imports.add(module)
                        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def categorize_archive_files(archive_modules: Set[str]) -> Dict[str, List[str]]:
    """Categorize archive files by usage type"""
    categories = {
        "makefile_used": [],
        "ci_used": [],
        "imported": [],
        "potentially_unused": []
    }
    
    # Check Makefile usage
    try:
        with open("Makefile", "r") as f:
            makefile_content = f.read()
        
        for module in archive_modules:
            if f"scripts/{module}.py" in makefile_content:
                categories["makefile_used"].append(module)
            elif module in categories["imported"]:
                continue  # Already marked as imported
            else:
                categories["potentially_unused"].append(module)
                
    except FileNotFoundError:
        # If no Makefile, mark all as potentially unused
        categories["potentially_unused"] = list(archive_modules)
    
    # Check CI usage
    ci_dir = Path(".github/workflows")
    if ci_dir.exists():
        for yml_file in ci_dir.glob("*.yml"):
            try:
                with open(yml_file, "r") as f:
                    content = f.read()
                
                for module in archive_modules:
                    if f"scripts/{module}.py" in content and module not in categories["ci_used"]:
                        categories["ci_used"].append(module)
                        
            except Exception:
                continue
    
    return categories

def main():
    """Main analysis function"""
    print("🔍 Checking archive import dependencies...")
    print()
    
    archive_modules = get_archive_files()
    core_files = get_core_files()
    
    print(f"📊 SUMMARY:")
    print(f"   Archive modules: {len(archive_modules)}")
    print(f"   Core files to check: {len(core_files)}")
    print()
    
    if not archive_modules:
        print("❌ No archive modules found")
        return
    
    if not core_files:
        print("❌ No core files found")
        return
    
    # Check each core file for imports
    dependencies = {}
    total_imports = set()
    
    for file_path in core_files:
        static_imports = check_imports_in_file(file_path, archive_modules)
        dynamic_imports = check_dynamic_imports(file_path, archive_modules)
        all_imports = static_imports | dynamic_imports
        
        if all_imports:
            dependencies[file_path] = all_imports
            total_imports.update(all_imports)
    
    # Categorize archive files
    categories = categorize_archive_files(archive_modules)
    
    # Report results
    if dependencies:
        print("⚠️  DEPENDENCIES FOUND (Archive files are imported):")
        print()
        for file_path, imports in dependencies.items():
            print(f"   📁 {file_path}")
            for imp in imports:
                print(f"      └── imports: {imp}")
        print()
        
        print("🚨 RECOMMENDATION:")
        print(f"   {len(total_imports)} archive modules are imported by core files.")
        print("   These files should NOT be removed until imports are updated.")
        print()
        
        print("📋 AFFECTED ARCHIVE MODULES:")
        for module in sorted(total_imports):
            print(f"   ❌ {module}")
        print()
        
    else:
        print("✅ NO IMPORT DEPENDENCIES FOUND")
        print("   Archive files are not imported by any core files.")
        print()
    
    # Show categorization
    print("📋 ARCHIVE FILE CATEGORIZATION:")
    print()
    
    if categories["makefile_used"]:
        print("🔧 MAKEFILE USED (Keep these):")
        for module in sorted(categories["makefile_used"]):
            print(f"   ✅ {module}")
        print()
    
    if categories["ci_used"]:
        print("🚀 CI USED (Keep these):")
        for module in sorted(categories["ci_used"]):
            print(f"   ✅ {module}")
        print()
    
    if categories["imported"]:
        print("📥 IMPORTED (Keep these):")
        for module in sorted(categories["imported"]):
            print(f"   ✅ {module}")
        print()
    
    if categories["potentially_unused"]:
        print("🤔 POTENTIALLY UNUSED (Phase 3 candidates):")
        for module in sorted(categories["potentially_unused"]):
            print(f"   ❓ {module}")
        print()
        
        print("💡 PHASE 3 RECOMMENDATION:")
        print(f"   {len(categories['potentially_unused'])} files may be safe for Phase 3 removal.")
        print("   Review these files manually before removal.")
    
    # Save detailed report
    report = {
        "archive_modules": sorted(list(archive_modules)),
        "core_files_checked": sorted(list(core_files)),
        "dependencies": {str(k): list(v) for k, v in dependencies.items()},
        "total_imports": sorted(list(total_imports)),
        "categories": categories
    }
    
    report_file = "artifacts/archive_import_analysis.json"
    os.makedirs("artifacts", exist_ok=True)
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
