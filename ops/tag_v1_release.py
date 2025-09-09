#!/usr/bin/env python3
"""
Tag v1.0.0 Release Script

Creates a comprehensive release manifest and tags the v1.0.0 release
for paper trading launch.
"""
import subprocess
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys


def get_git_info() -> Dict[str, str]:
    """Get current git commit information."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        
        # Check if working directory is clean
        try:
            subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])
            is_clean = True
        except subprocess.CalledProcessError:
            is_clean = False
        
        return {
            'commit_hash': commit_hash,
            'short_hash': short_hash,
            'branch': branch,
            'is_clean': is_clean
        }
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting git info: {e}")
        return {}


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    if not Path(file_path).exists():
        return "FILE_NOT_FOUND"
    
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def create_release_manifest() -> Dict:
    """Create comprehensive release manifest."""
    print("📋 Creating v1.0.0 release manifest...")
    
    git_info = get_git_info()
    
    # Core system files to track
    critical_files = {
        'feature_whitelist': 'results/production/features_whitelist.json',
        'feature_whitelist_hash': 'results/production/features_whitelist.json.hash',
        'base_config': 'config/base.yaml',
        'risk_limits': 'config/risk_limits.json',
        'trading_mode': 'config/trading_mode.json',
        'env_requirements': 'requirements.txt'
    }
    
    # Compute hashes for all critical files
    file_hashes = {}
    for name, path in critical_files.items():
        file_hashes[name] = {
            'path': path,
            'hash': compute_file_hash(path),
            'exists': Path(path).exists()
        }
    
    # Load feature whitelist for verification
    whitelist_info = {}
    try:
        with open('results/production/features_whitelist.json', 'r') as f:
            whitelist = json.load(f)
            whitelist_info = {
                'feature_count': len(whitelist),
                'features': whitelist
            }
    except Exception as e:
        whitelist_info = {'error': str(e)}
    
    # System configuration
    system_config = {
        'paper_trading_mode': True,
        'trading_timezone': 'America/Chicago',
        'python_hash_seed': '42',
        'environment_type': 'paper'
    }
    
    # Create manifest
    manifest = {
        'release': {
            'version': 'v1.0.0',
            'name': 'Paper Trading Launch',
            'description': 'Production-ready quantitative trading system for paper trading validation',
            'created_at': datetime.now().isoformat(),
            'created_by': 'automated_release_script'
        },
        'git': git_info,
        'file_integrity': file_hashes,
        'feature_whitelist': whitelist_info,
        'system_config': system_config,
        'validation': {
            'all_critical_files_present': all(info['exists'] for info in file_hashes.values()),
            'git_working_directory_clean': git_info.get('is_clean', False),
            'feature_whitelist_validated': 'feature_count' in whitelist_info
        },
        'paper_trading': {
            'duration_days': 20,
            'success_criteria': {
                'ic_minimum': 0.015,
                'sharpe_minimum': 0.30,
                'turnover_maximum': 2.0,
                'max_guard_breaches_per_week': 1,
                'max_cost_deviation_pct': 25
            },
            'risk_limits': {
                'daily_loss_kill_pct': 2.0,
                'max_position_pct': 2.0,
                'max_gross_exposure_pct': 30.0,
                'max_adv_participation_pct': 2.0
            }
        }
    }
    
    return manifest


def validate_release_readiness(manifest: Dict) -> bool:
    """Validate that the system is ready for release."""
    print("🔍 Validating release readiness...")
    
    validation_passed = True
    issues = []
    
    # Check git status
    if not manifest['git'].get('is_clean', False):
        issues.append("Working directory is not clean")
        validation_passed = False
    
    # Check critical files
    if not manifest['validation']['all_critical_files_present']:
        issues.append("Not all critical files are present")
        validation_passed = False
    
    # Check feature whitelist
    if not manifest['validation']['feature_whitelist_validated']:
        issues.append("Feature whitelist validation failed")
        validation_passed = False
    
    # Check if feature whitelist hash exists and matches
    whitelist_hash_file = 'results/production/features_whitelist.json.hash'
    if Path(whitelist_hash_file).exists():
        try:
            with open('results/production/features_whitelist.json', 'r') as f:
                whitelist = json.load(f)
            
            content = json.dumps(sorted(whitelist), sort_keys=True)
            computed_hash = hashlib.sha256(content.encode()).hexdigest()
            
            with open(whitelist_hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if computed_hash != stored_hash:
                issues.append("Feature whitelist hash mismatch")
                validation_passed = False
        except Exception as e:
            issues.append(f"Feature whitelist hash validation error: {e}")
            validation_passed = False
    else:
        issues.append("Feature whitelist hash file missing")
        validation_passed = False
    
    # Print validation results
    if validation_passed:
        print("✅ Release validation passed")
    else:
        print("❌ Release validation failed:")
        for issue in issues:
            print(f"   - {issue}")
    
    return validation_passed


def tag_release(manifest: Dict) -> bool:
    """Tag the release in git."""
    print("🏷️ Tagging v1.0.0 release...")
    
    try:
        # Create annotated tag
        tag_message = f"""v1.0.0 - Paper Trading Launch

Production-ready quantitative trading system
- Features: {manifest['feature_whitelist'].get('feature_count', 'unknown')} validated features
- Performance: IC ~0.017, Sharpe ~0.32 (leak-safe)
- Risk controls: ADV limits, position caps, kill-switches
- Monitoring: Automated daily reports and alerts
- Infrastructure: Production-grade CI/CD with rollback

Commit: {manifest['git'].get('short_hash', 'unknown')}
Created: {manifest['release']['created_at']}
"""
        
        # Create the tag
        subprocess.check_call([
            'git', 'tag', '-a', 'v1.0.0', '-m', tag_message
        ])
        
        print("✅ Tag v1.0.0 created successfully")
        
        # Ask about pushing
        push_tag = input("Push tag to remote? (y/N): ").lower().strip()
        if push_tag in ['y', 'yes']:
            subprocess.check_call(['git', 'push', 'origin', 'v1.0.0'])
            print("✅ Tag pushed to remote")
        else:
            print("ℹ️ Tag created locally (not pushed)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create tag: {e}")
        return False


def main():
    """Main release tagging process."""
    print("🚀 PAPER TRADING v1.0.0 RELEASE TAGGING")
    print("="*50)
    
    # Step 1: Create manifest
    manifest = create_release_manifest()
    
    # Step 2: Save manifest
    manifest_dir = Path("results/releases")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_file = manifest_dir / "v1.0.0_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📄 Release manifest saved: {manifest_file}")
    
    # Step 3: Display manifest summary
    print(f"\n📋 RELEASE MANIFEST SUMMARY:")
    print(f"   Version: {manifest['release']['version']}")
    print(f"   Git commit: {manifest['git'].get('short_hash', 'unknown')}")
    print(f"   Features: {manifest['feature_whitelist'].get('feature_count', 'unknown')}")
    print(f"   Critical files: {len([f for f in manifest['file_integrity'].values() if f['exists']])}/{len(manifest['file_integrity'])}")
    
    # Step 4: Validate readiness
    if not validate_release_readiness(manifest):
        print("\n❌ RELEASE VALIDATION FAILED")
        print("Please fix the issues above before tagging the release.")
        sys.exit(1)
    
    # Step 5: Tag release
    print(f"\n🏷️ READY TO TAG v1.0.0 RELEASE")
    print(f"   This will create an annotated git tag for paper trading launch")
    
    proceed = input("Proceed with tagging? (y/N): ").lower().strip()
    if proceed not in ['y', 'yes']:
        print("❌ Release tagging cancelled")
        sys.exit(0)
    
    if tag_release(manifest):
        print(f"\n🎉 RELEASE v1.0.0 TAGGED SUCCESSFULLY!")
        print(f"="*50)
        print(f"✅ Release manifest: {manifest_file}")
        print(f"✅ Git tag: v1.0.0")
        print(f"✅ System ready for paper trading launch")
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run: ./start_paper_trading.sh")
        print(f"2. Monitor: ./status_paper_trading.sh")
        print(f"3. Manual run: ./run_paper_trading_now.sh")
    else:
        print(f"\n❌ RELEASE TAGGING FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
