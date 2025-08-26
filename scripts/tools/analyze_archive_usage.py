#!/usr/bin/env python3
"""
Analyze which files in scripts/archive/ are actually used.

This script checks:
1. Makefile references
2. CI workflow references  
3. Python import references
4. Direct file references
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Set, Dict, List

def get_archive_files() -> Set[str]:
    """Get all files in scripts/archive/"""
    archive_dir = Path("scripts/archive")
    if not archive_dir.exists():
        return set()
    
    files = set()
    for file in archive_dir.iterdir():
        if file.is_file():
            files.add(file.name)
    return files

def check_makefile_references() -> Set[str]:
    """Check which archive files are referenced in Makefile"""
    used = set()
    
    try:
        with open("Makefile", "r") as f:
            content = f.read()
        
        # Find all script references
        script_refs = re.findall(r'scripts/([^/\s]+\.py)', content)
        for ref in script_refs:
            if ref in get_archive_files():
                used.add(ref)
                
    except FileNotFoundError:
        pass
    
    return used

def check_ci_references() -> Set[str]:
    """Check which archive files are referenced in CI workflows"""
    used = set()
    
    ci_dir = Path(".github/workflows")
    if not ci_dir.exists():
        return used
    
    for yml_file in ci_dir.glob("*.yml"):
        try:
            with open(yml_file, "r") as f:
                content = f.read()
            
            # Find all script references
            script_refs = re.findall(r'scripts/([^/\s]+\.py)', content)
            for ref in script_refs:
                if ref in get_archive_files():
                    used.add(ref)
                    
        except Exception:
            continue
    
    return used

def check_python_imports() -> Set[str]:
    """Check which archive files are imported in Python code"""
    used = set()
    
    # Search for import statements
    try:
        result = subprocess.run(
            ["grep", "-r", "from scripts\\."], 
            capture_output=True, text=True, cwd="."
        )
        
        if result.stdout:
            # Extract script names from import statements
            import_refs = re.findall(r'from scripts\.([^\.]+)', result.stdout)
            for ref in import_refs:
                if ref in get_archive_files():
                    used.add(ref)
                    
    except Exception:
        pass
    
    return used

def check_direct_references() -> Set[str]:
    """Check for direct file references (not imports)"""
    used = set()
    
    try:
        result = subprocess.run(
            ["grep", "-r", "scripts/"], 
            capture_output=True, text=True, cwd="."
        )
        
        if result.stdout:
            # Extract script names from references
            script_refs = re.findall(r'scripts/([^/\s]+\.py)', result.stdout)
            for ref in script_refs:
                if ref in get_archive_files():
                    used.add(ref)
                    
    except Exception:
        pass
    
    return used

def analyze_usage() -> Dict[str, List[str]]:
    """Analyze usage of archive files"""
    archive_files = get_archive_files()
    
    makefile_used = check_makefile_references()
    ci_used = check_ci_references()
    imports_used = check_python_imports()
    direct_used = check_direct_references()
    
    # Combine all usage
    all_used = makefile_used | ci_used | imports_used | direct_used
    
    # Categorize files
    unused = archive_files - all_used
    used = archive_files & all_used
    
    return {
        "used": sorted(list(used)),
        "unused": sorted(list(unused)),
        "makefile_used": sorted(list(makefile_used)),
        "ci_used": sorted(list(ci_used)),
        "imports_used": sorted(list(imports_used)),
        "direct_used": sorted(list(direct_used))
    }

def main():
    """Main analysis function"""
    print("🔍 Analyzing scripts/archive/ usage...")
    print()
    
    results = analyze_usage()
    
    print(f"📊 SUMMARY:")
    print(f"   Total archive files: {len(results['used']) + len(results['unused'])}")
    print(f"   Used files: {len(results['used'])}")
    print(f"   Unused files: {len(results['unused'])}")
    print(f"   Safe to remove: {len(results['unused'])}")
    print()
    
    if results['used']:
        print("📋 USED FILES (keep these):")
        for file in results['used']:
            sources = []
            if file in results['makefile_used']:
                sources.append("Makefile")
            if file in results['ci_used']:
                sources.append("CI")
            if file in results['imports_used']:
                sources.append("imports")
            if file in results['direct_used']:
                sources.append("direct")
            print(f"   ✅ {file} (used by: {', '.join(sources)})")
        print()
    
    if results['unused']:
        print("🗑️  UNUSED FILES (safe to remove):")
        for file in results['unused']:
            print(f"   ❌ {file}")
        print()
        
        print("💡 RECOMMENDATION:")
        print(f"   You can safely remove {len(results['unused'])} files from scripts/archive/")
        print("   These files are not referenced anywhere in the codebase.")
    
    # Save detailed report
    report_file = "artifacts/archive_usage_analysis.json"
    os.makedirs("artifacts", exist_ok=True)
    
    import json
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
