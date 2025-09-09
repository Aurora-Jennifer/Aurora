"""
Feature whitelist protection and CI enforcement.

Prevents unauthorized features from entering production.
"""
import hashlib
import json
import os
from typing import List, Set
import warnings


def compute_content_hash(feature_list: List[str]) -> str:
    """
    Compute SHA256 hash of feature whitelist content.
    
    Args:
        feature_list: List of feature names
        
    Returns:
        Hexadecimal hash string
    """
    # Sort for deterministic hashing
    sorted_features = sorted(feature_list)
    content = json.dumps(sorted_features, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def validate_feature_names(features: List[str]) -> tuple[bool, List[str]]:
    """
    Validate feature names against forbidden patterns.
    
    Args:
        features: List of feature names to validate
        
    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    FORBIDDEN_PATTERNS = [
        'fwd', 'forward', 'future', 'label', 'target', 
        'excess_ret_fwd', 'ret_fwd', 'oof', 'lookahead',
        'next_', 'ahead', '_fwd_'
    ]
    
    violations = []
    
    for feature in features:
        feature_lower = feature.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in feature_lower:
                violations.append(f"{feature} (contains: {pattern})")
                break
    
    return len(violations) == 0, violations


def load_whitelist_with_validation(whitelist_path: str) -> tuple[List[str], str]:
    """
    Load feature whitelist and validate its content.
    
    Args:
        whitelist_path: Path to whitelist JSON file
        
    Returns:
        Tuple of (feature_list, content_hash)
        
    Raises:
        ValueError: If whitelist fails validation
    """
    if not os.path.exists(whitelist_path):
        raise FileNotFoundError(f"Feature whitelist not found: {whitelist_path}")
    
    with open(whitelist_path, 'r') as f:
        features = json.load(f)
    
    if not isinstance(features, list):
        raise ValueError("Feature whitelist must be a list")
    
    # Validate feature names
    is_valid, violations = validate_feature_names(features)
    if not is_valid:
        raise ValueError(f"Forbidden features detected: {violations}")
    
    # Compute hash
    content_hash = compute_content_hash(features)
    
    return features, content_hash


def save_whitelist_with_hash(features: List[str], whitelist_path: str, 
                           hash_path: str = None) -> str:
    """
    Save feature whitelist with content hash.
    
    Args:
        features: List of feature names
        whitelist_path: Path to save whitelist JSON
        hash_path: Optional path to save hash (defaults to whitelist_path + .hash)
        
    Returns:
        Content hash string
    """
    # Validate before saving
    is_valid, violations = validate_feature_names(features)
    if not is_valid:
        raise ValueError(f"Cannot save whitelist with forbidden features: {violations}")
    
    # Save whitelist
    with open(whitelist_path, 'w') as f:
        json.dump(sorted(features), f, indent=2)
    
    # Compute and save hash
    content_hash = compute_content_hash(features)
    
    if hash_path is None:
        hash_path = whitelist_path + '.hash'
    
    with open(hash_path, 'w') as f:
        f.write(content_hash)
    
    return content_hash


def verify_whitelist_integrity(whitelist_path: str, expected_hash: str = None,
                              hash_path: str = None) -> tuple[bool, str]:
    """
    Verify whitelist integrity against expected hash.
    
    Args:
        whitelist_path: Path to whitelist JSON file
        expected_hash: Expected hash (if provided)
        hash_path: Path to hash file (if expected_hash not provided)
        
    Returns:
        Tuple of (is_valid, current_hash)
    """
    features, current_hash = load_whitelist_with_validation(whitelist_path)
    
    if expected_hash is None:
        if hash_path is None:
            hash_path = whitelist_path + '.hash'
        
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
        else:
            warnings.warn(f"No expected hash available for verification")
            return True, current_hash
    
    is_valid = current_hash == expected_hash
    
    if not is_valid:
        warnings.warn(f"Whitelist integrity check failed: {current_hash} != {expected_hash}")
    
    return is_valid, current_hash


def ci_whitelist_guard(whitelist_path: str, reference_hash_path: str = None) -> bool:
    """
    CI guard function to prevent unauthorized whitelist changes.
    
    Args:
        whitelist_path: Path to feature whitelist
        reference_hash_path: Path to reference hash file
        
    Returns:
        True if whitelist passes all checks
        
    Raises:
        SystemExit: If whitelist fails any check (for CI failure)
    """
    print(f"🔒 CI WHITELIST GUARD: Validating {whitelist_path}")
    
    try:
        # Load and validate whitelist
        features, current_hash = load_whitelist_with_validation(whitelist_path)
        print(f"✅ Whitelist loaded: {len(features)} features")
        
        # Verify integrity if reference hash exists
        if reference_hash_path and os.path.exists(reference_hash_path):
            is_valid, _ = verify_whitelist_integrity(whitelist_path, hash_path=reference_hash_path)
            if not is_valid:
                print(f"❌ WHITELIST INTEGRITY FAILED")
                print(f"   Current hash: {current_hash}")
                print(f"   Reference: {reference_hash_path}")
                raise SystemExit(1)
            print(f"✅ Whitelist integrity verified")
        
        # Check feature count (sanity)
        if len(features) == 0:
            print(f"❌ EMPTY WHITELIST")
            raise SystemExit(1)
        
        if len(features) > 200:  # Reasonable upper bound
            print(f"⚠️ LARGE WHITELIST: {len(features)} features (review recommended)")
        
        # All checks passed
        print(f"✅ WHITELIST GUARD PASSED")
        print(f"   Features: {len(features)}")
        print(f"   Hash: {current_hash[:12]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ WHITELIST GUARD FAILED: {e}")
        raise SystemExit(1)


def create_protected_whitelist(features: List[str], base_path: str) -> dict:
    """
    Create a protected whitelist with hash and CI integration.
    
    Args:
        features: List of feature names
        base_path: Base path for files (e.g., 'results/production/')
        
    Returns:
        Dict with file paths and hash
    """
    os.makedirs(base_path, exist_ok=True)
    
    whitelist_path = os.path.join(base_path, 'features_whitelist.json')
    hash_path = os.path.join(base_path, 'features_whitelist.json.hash')
    
    # Save with protection
    content_hash = save_whitelist_with_hash(features, whitelist_path, hash_path)
    
    # Create CI script
    ci_script_path = os.path.join(base_path, 'ci_whitelist_check.py')
    with open(ci_script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
CI script to validate feature whitelist integrity.
Auto-generated by whitelist protection system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.whitelist_guard import ci_whitelist_guard

if __name__ == "__main__":
    success = ci_whitelist_guard("{whitelist_path}", "{hash_path}")
    sys.exit(0 if success else 1)
''')
    
    # Make CI script executable
    os.chmod(ci_script_path, 0o755)
    
    return {
        'whitelist_path': whitelist_path,
        'hash_path': hash_path,
        'ci_script_path': ci_script_path,
        'content_hash': content_hash
    }


if __name__ == "__main__":
    print("Feature whitelist protection module loaded successfully")
