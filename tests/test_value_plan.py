#!/usr/bin/env python3
"""
Test Value Plan - Enforce Reality Check Documentation and Flag Structure
======================================================================

These tests ensure the reality check is properly documented and the
improvement roadmap is tracked through feature flags.
"""

import json
import yaml
from pathlib import Path


def test_reality_check_doc_exists_and_linked():
    """Test that reality check documentation exists and is linked from README."""
    # Check reality check doc exists
    reality_check_path = Path("docs/REALITY_CHECK.md")
    assert reality_check_path.exists(), "docs/REALITY_CHECK.md must exist"
    
    # Check README links to it
    readme_content = Path("README.md").read_text()
    assert "REALITY_CHECK.md" in readme_content, "README.md must link to REALITY_CHECK.md"
    assert "reality_check" in readme_content.lower(), "README.md must mention reality check"


def test_flags_declared_and_default_off():
    """Test that all improvement roadmap flags are declared and default to False."""
    # Load config
    config_path = Path("config/base.yaml")
    assert config_path.exists(), "config/base.yaml must exist"
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check flags section exists
    flags = cfg.get("flags", {})
    assert isinstance(flags, dict), "flags section must be a dictionary"
    
    # Required flags from roadmap
    required_flags = [
        "show_reality_check_on_start",
        "enable_realtime_data", 
        "enable_oms_v1",
        "enable_risk_v2",
        "enable_bt_gates",
        "features_v2",
        "model_zoo",
        "exec_sim_v2",
        "altdata_v1",
        "pm_v1"
    ]
    
    # Check all required flags exist and default to False
    for flag in required_flags:
        assert flag in flags, f"Flag '{flag}' must be declared in config/base.yaml"
        assert flags[flag] is False, f"Flag '{flag}' must default to False"


def test_reality_check_content_structure():
    """Test that reality check document has required sections."""
    reality_check_path = Path("docs/REALITY_CHECK.md")
    content = reality_check_path.read_text()
    
    # Required sections
    required_sections = [
        "Current State Assessment",
        "To Be Actually Valuable",
        "Honest Assessment", 
        "Roadmap",
        "Definition of Done",
        "Warning"
    ]
    
    for section in required_sections:
        assert section in content, f"REALITY_CHECK.md must contain '{section}' section"


def test_roadmap_flags_match_documentation():
    """Test that flags in config match the roadmap in documentation."""
    # Load config flags
    with open("config/base.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    config_flags = set(cfg.get("flags", {}).keys())
    
    # Load reality check content
    reality_check_content = Path("docs/REALITY_CHECK.md").read_text()
    
    # Extract flags mentioned in roadmap
    roadmap_flags = set()
    for line in reality_check_content.split('\n'):
        if 'flag:' in line:
            flag_name = line.split('flag:')[1].strip().strip('`')
            roadmap_flags.add(flag_name)
    
    # Check that all roadmap flags are in config
    missing_flags = roadmap_flags - config_flags
    assert not missing_flags, f"Flags mentioned in roadmap but missing from config: {missing_flags}"


def test_reality_check_banner_integration():
    """Test that reality check banner can be enabled in paper runner."""
    # Check that paper runner has reality check banner code
    paper_runner_path = Path("scripts/paper_runner.py")
    assert paper_runner_path.exists(), "scripts/paper_runner.py must exist"
    
    paper_runner_content = paper_runner_path.read_text()
    assert "show_reality_check_on_start" in paper_runner_content, "Paper runner must check reality check flag"
    assert "REALITY_CHECK.md" in paper_runner_content, "Paper runner must reference reality check doc"


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_reality_check_doc_exists_and_linked,
        test_flags_declared_and_default_off,
        test_reality_check_content_structure,
        test_roadmap_flags_match_documentation,
        test_reality_check_banner_integration
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            raise
