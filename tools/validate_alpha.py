#!/usr/bin/env python3
"""
Validate Alpha evaluation results against promotion gates.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import jsonschema

def load_schema() -> dict:
    """Load alpha evaluation schema."""
    schema_path = Path("reports/alpha.schema.json")
    return json.loads(schema_path.read_text())

def validate_schema(results: dict) -> bool:
    """Validate results against schema."""
    try:
        schema = load_schema()
        jsonschema.validate(instance=results, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def validate_promotion_gates(results: dict) -> tuple[bool, list[str]]:
    """
    Validate promotion gates.
    
    Returns:
        (passed, list of failures)
    """
    failures = []
    overall_metrics = results.get('overall_metrics', {})
    
    # Check IC threshold
    mean_ic = overall_metrics.get('mean_ic', 0.0)
    if mean_ic < 0.02:
        failures.append(f"IC {mean_ic:.4f} < 0.02 threshold")
    
    # Check hit rate threshold
    mean_hit_rate = overall_metrics.get('mean_hit_rate', 0.0)
    if mean_hit_rate < 0.52:
        failures.append(f"Hit rate {mean_hit_rate:.4f} < 0.52 threshold")
    
    # Check for reasonable turnover (not too high)
    mean_turnover = overall_metrics.get('mean_turnover', 0.0)
    if mean_turnover > 2.0:  # 200% daily turnover is excessive
        failures.append(f"Turnover {mean_turnover:.4f} > 2.0 threshold")
    
    # Check for sufficient test samples
    fold_summaries = results.get('fold_summaries', [])
    total_predictions = sum(fold.get('n_predictions', 0) for fold in fold_summaries)
    if total_predictions < 100:
        failures.append(f"Total predictions {total_predictions} < 100 minimum")
    
    return len(failures) == 0, failures

def main():
    parser = argparse.ArgumentParser(description="Validate Alpha evaluation results")
    parser.add_argument("eval_file", help="Path to alpha evaluation JSON file")
    parser.add_argument("--ic-threshold", type=float, default=0.02, help="IC threshold")
    parser.add_argument("--hit-rate-threshold", type=float, default=0.52, help="Hit rate threshold")
    args = parser.parse_args()
    
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"❌ Evaluation file not found: {eval_file}")
        sys.exit(1)
    
    try:
        with open(eval_file, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)
    
    print(f"Validating {eval_file}")
    print("=" * 50)
    
    # Validate schema
    if not validate_schema(results):
        print("❌ Schema validation failed")
        sys.exit(1)
    
    print("✅ Schema validation passed")
    
    # Validate promotion gates
    passed, failures = validate_promotion_gates(results)
    
    # Print metrics
    overall_metrics = results.get('overall_metrics', {})
    print(f"IC (Spearman): {overall_metrics.get('mean_ic', 0.0):.4f} ± {overall_metrics.get('std_ic', 0.0):.4f}")
    print(f"Hit Rate: {overall_metrics.get('mean_hit_rate', 0.0):.4f} ± {overall_metrics.get('std_hit_rate', 0.0):.4f}")
    print(f"Turnover: {overall_metrics.get('mean_turnover', 0.0):.4f}")
    print(f"Return (with costs): {overall_metrics.get('mean_return_with_costs', 0.0):.4f}")
    
    # Print fold summary
    fold_summaries = results.get('fold_summaries', [])
    print(f"\nFold Summary ({len(fold_summaries)} folds):")
    for fold in fold_summaries:
        print(f"  Fold {fold['fold']}: IC={fold['ic_spearman']:.4f}, Hit={fold['hit_rate']:.4f}, N={fold['n_predictions']}")
    
    # Check promotion gates
    if passed:
        print("\n🎉 PROMOTION GATES PASSED!")
        print("Model is ready for promotion to paper trading.")
        sys.exit(0)
    else:
        print(f"\n❌ PROMOTION GATES FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nModel needs improvement before promotion.")
        sys.exit(1)

if __name__ == "__main__":
    main()
