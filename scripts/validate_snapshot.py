#!/usr/bin/env python3
"""Validate frozen golden snapshot integrity."""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def validate_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    """Validate snapshot manifest and files."""
    manifest_path = snapshot_path / "manifest.json"
    
    if not manifest_path.exists():
        return {"valid": False, "error": "manifest.json not found"}
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON in manifest: {e}"}
    
    # Validate each file in manifest
    for filename, file_info in manifest.get("files", {}).items():
        file_path = snapshot_path / filename
        expected_hash = file_info.get("sha256")
        
        if not file_path.exists():
            return {"valid": False, "error": f"File {filename} not found"}
        
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            return {
                "valid": False, 
                "error": f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            }
    
    return {
        "valid": True,
        "snapshot_id": manifest.get("snapshot_id"),
        "version": manifest.get("version"),
        "files_validated": len(manifest.get("files", {}))
    }


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_snapshot.py <snapshot_path>")
        sys.exit(1)
    
    snapshot_path = Path(sys.argv[1])
    if not snapshot_path.exists():
        print(f"Error: Snapshot path {snapshot_path} does not exist")
        sys.exit(1)
    
    result = validate_snapshot(snapshot_path)
    
    if result["valid"]:
        print(f"✅ Snapshot {result['snapshot_id']} v{result['version']} is valid")
        print(f"   Files validated: {result['files_validated']}")
        sys.exit(0)
    else:
        print(f"❌ Snapshot validation failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
