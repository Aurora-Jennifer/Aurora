"""
Configuration loading with CLI overrides
"""
import yaml
import argparse
from typing import Any


def load_config(path: str, cli_overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config with CLI overrides"""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    
    # Apply CLI overrides like +decision.tau_search.0=0.30
    for kv in (cli_overrides or []):
        if not kv.startswith("+"):
            continue
        k, v = kv[1:].split("=", 1)
        cursor = cfg
        keys = k.split(".")
        
        # Navigate to the parent of the target key
        for kk in keys[:-1]:
            if kk not in cursor:
                cursor[kk] = {}
            cursor = cursor[kk]
        
        # Coerce value type
        target_key = keys[-1]
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                # Try int first, then float
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                # Keep as string
                pass
        
        cursor[target_key] = v
    
    return cfg


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments for config overrides"""
    parser = argparse.ArgumentParser(description="Walk-forward validation with config overrides")
    parser.add_argument("--config", default="configs/wfv.yaml", help="Config file path")
    parser.add_argument("overrides", nargs="*", help="Config overrides like +train.batch_size=2048")
    return parser.parse_args()


def get_config() -> dict[str, Any]:
    """Get config with CLI overrides applied"""
    args = parse_cli_args()
    return load_config(args.config, args.overrides)
