# utils/env_check.py
"""
Environment capability checking for honest system reporting
"""

def check_capabilities():
    """Check which ML libraries are available in the environment."""
    caps = {"python": True}
    
    for lib, key in (("lightgbm", "lightgbm"), ("xgboost", "xgboost")):
        try:
            __import__(lib)
            caps[key] = True
        except Exception:
            caps[key] = False
    
    return caps
