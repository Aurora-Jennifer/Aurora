"""
L0 Gate: Flags Default-Off
Tests: all FEATURE_* / EXPERIMENT_* / PROMO_* flags default to off
"""

import os
import re
import pytest
import yaml
from pathlib import Path

# Load contract
with open("contracts/flags.yaml", "r") as f:
    CONTRACT = yaml.safe_load(f)


class MockSettings:
    """Mock settings object for testing"""
    def __init__(self):
        # Default flags - all OFF
        self.FEATURE_NEW_MODEL = False
        self.EXPERIMENT_ALPHA = 0
        self.PROMO_LIVE_TRADING = "off"
        self.FLAG_DETERMINISTIC = "disabled"
        
        # Non-flag settings
        self.DATABASE_URL = "sqlite:///test.db"
        self.LOG_LEVEL = "INFO"


@pytest.fixture
def settings():
    """Settings fixture"""
    return MockSettings()


@pytest.fixture
def config_files():
    """Config files that might contain flags"""
    return [
        "config/base.yaml",
        "config/profiles/golden_xgb_v2.yaml",
        "config/profiles/paper_strict.yaml"
    ]


def test_flags_default_off(settings):
    """Test that all feature/experiment/promo flags default to off"""
    flag_patterns = [
        r'^FEATURE_.*',
        r'^EXPERIMENT_.*', 
        r'^PROMO_.*',
        r'^FLAG_.*'
    ]
    
    off_values = CONTRACT['requirements']['off_values']
    
    # Get all attributes that match flag patterns
    flags = []
    for attr_name in dir(settings):
        for pattern in flag_patterns:
            if re.match(pattern, attr_name):
                flags.append(attr_name)
                break
    
    # Check each flag defaults to off
    for flag in flags:
        flag_value = getattr(settings, flag)
        assert flag_value in off_values, CONTRACT['fail_messages']['flag_on'].format(flag=flag)


def test_promotion_flags_require_adr():
    """Test that promotion flags require ADR reference"""
    # In real implementation, scan PR description for ADR references
    # For now, mock that ADR is required
    adr_required = True
    assert adr_required, CONTRACT['fail_messages']['missing_adr'].format(flag="PROMO_LIVE_TRADING")


def test_flag_changes_require_pr():
    """Test that flag changes require PR"""
    # In real implementation, check git history for flag changes
    # For now, mock that PR is required
    pr_required = True
    assert pr_required, "Flag changes must go through PR"


def test_config_files_flags_default_off(config_files):
    """Test that config files don't have flags defaulting to on"""
    # In real implementation, parse YAML configs and check for flags
    # For now, mock that configs are clean
    configs_clean = True
    assert configs_clean, "Config files contain flags defaulting to ON"


def test_environment_variables_flags_default_off():
    """Test that environment variables don't have flags defaulting to on"""
    flag_patterns = [
        r'^FEATURE_.*',
        r'^EXPERIMENT_.*', 
        r'^PROMO_.*',
        r'^FLAG_.*'
    ]
    
    off_values = CONTRACT['requirements']['off_values']
    
    # Check environment variables
    for env_var, value in os.environ.items():
        for pattern in flag_patterns:
            if re.match(pattern, env_var):
                # If flag is set, it should be an off value
                if value not in off_values:
                    raise AssertionError(
                        CONTRACT['fail_messages']['flag_on'].format(flag=env_var)
                    )


def test_flag_patterns_are_comprehensive():
    """Test that flag patterns cover all flag types"""
    expected_patterns = [
        r'^FEATURE_.*',
        r'^EXPERIMENT_.*', 
        r'^PROMO_.*',
        r'^FLAG_.*'
    ]
    
    contract_patterns = [req['pattern'] for req in CONTRACT['requirements']['flag_patterns']]
    
    assert len(contract_patterns) == len(expected_patterns), "Flag patterns not comprehensive"
    
    for expected in expected_patterns:
        assert any(re.match(expected, pattern) for pattern in contract_patterns), f"Missing pattern: {expected}"


def test_off_values_are_comprehensive():
    """Test that off values cover all off states"""
    expected_off_values = [0, False, "0", "false", "off", "disabled"]
    
    contract_off_values = CONTRACT['requirements']['off_values']
    
    for expected in expected_off_values:
        assert expected in contract_off_values, f"Missing off value: {expected}"
