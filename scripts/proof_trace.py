#!/usr/bin/env python3
"""
Proof Trace Script
Prints each deployment flag and the file that justified it
"""

import json
import os
import glob
import pandas as pd
import argparse
from pathlib import Path


def print_proof_trace(deployment_report_path: str, output_dir: str = None):
    """Print proof trace for deployment report."""
    
    if not os.path.exists(deployment_report_path):
        print(f"❌ Deployment report not found: {deployment_report_path}")
        return
    
    # Load deployment report
    with open(deployment_report_path, 'r') as f:
        report = json.load(f)
    
    print("=== PROOF TRACE ===")
    print(f"Deployment Report: {deployment_report_path}")
    print(f"Summary: {report.get('summary', 'UNKNOWN')}")
    print()
    
    # Print component status
    component_status = report.get('component_status', {})
    for component, status in component_status.items():
        icon = "✅" if status else "❌"
        print(f"{component:15} -> {icon} {'OK' if status else 'FAIL'}")
    
    # Print verification messages if available
    verification_messages = report.get('verification_messages', {})
    if verification_messages:
        print("\n=== VERIFICATION DETAILS ===")
        for component, result in verification_messages.items():
            if isinstance(result, tuple) and len(result) == 2:
                success, message = result
                icon = "✅" if success else "❌"
                print(f"{component:15} -> {icon} {message}")
            else:
                print(f"{component:15} -> ⚠️ {result}")
    
    # Print capabilities
    capabilities = report.get('capabilities', {})
    if capabilities:
        print("\n=== CAPABILITIES ===")
        for lib, available in capabilities.items():
            icon = "✅" if available else "❌"
            print(f"{lib:15} -> {icon} {'Available' if available else 'Missing'}")
    
    # Find and print baseline contexts
    if output_dir:
        baseline_contexts = glob.glob(
            os.path.join(output_dir, "**", "baseline_context.json"), 
            recursive=True
        )
        print(f"\n=== BASELINE CONTEXTS ===")
        if baseline_contexts:
            for ctx_path in baseline_contexts:
                print(f"Found: {ctx_path}")
                try:
                    with open(ctx_path, 'r') as f:
                        context = json.load(f)
                    print("Context:")
                    for key, value in context.items():
                        print(f"  {key}: {value}")
                    print()
                except Exception as e:
                    print(f"  Error reading context: {e}")
        else:
            print("No baseline contexts found")
    
    # Find and print key artifacts
    if output_dir:
        print(f"\n=== KEY ARTIFACTS ===")
        
        # Check for portfolio weights
        portfolio_files = glob.glob(
            os.path.join(output_dir, "**", "*portfolio*weights*.csv"), 
            recursive=True
        )
        if portfolio_files:
            print(f"Portfolio weights: {len(portfolio_files)} files")
            for pf in portfolio_files:
                try:
                    df = pd.read_csv(pf)
                    print(f"  {pf}: {len(df)} strategies")
                except Exception as e:
                    print(f"  {pf}: Error reading - {e}")
        else:
            print("Portfolio weights: Not found")
        
        # Check for grid results
        grid_files = glob.glob(
            os.path.join(output_dir, "**", "*grid*.csv"), 
            recursive=True
        )
        if grid_files:
            print(f"Grid results: {len(grid_files)} files")
            total_strategies = 0
            for gf in grid_files:
                try:
                    df = pd.read_csv(gf)
                    strategies = len(df)
                    total_strategies += strategies
                    print(f"  {gf}: {strategies} strategies")
                except Exception as e:
                    print(f"  {gf}: Error reading - {e}")
            print(f"Total strategies across all grids: {total_strategies}")
        else:
            print("Grid results: Not found")
        
        # Check for ablation results
        ablation_files = glob.glob(
            os.path.join(output_dir, "**", "*ablation*.csv"), 
            recursive=True
        )
        if ablation_files:
            print(f"Ablation results: {len(ablation_files)} files")
            for af in ablation_files:
                try:
                    df = pd.read_csv(af)
                    print(f"  {af}: {len(df)} feature comparisons")
                except Exception as e:
                    print(f"  {af}: Error reading - {e}")
        else:
            print("Ablation results: Not found")
        
        # Check for monitoring artifacts
        monitoring_files = glob.glob(
            os.path.join(output_dir, "**", "*monitoring*.json"), 
            recursive=True
        )
        if monitoring_files:
            print(f"Monitoring artifacts: {len(monitoring_files)} files")
            for mf in monitoring_files:
                print(f"  {mf}")
        else:
            print("Monitoring artifacts: Not found")


def main():
    parser = argparse.ArgumentParser(description='Print proof trace for deployment report')
    parser.add_argument('--report', required=True, help='Path to deployment_report.json')
    parser.add_argument('--output-dir', help='Output directory to search for artifacts')
    
    args = parser.parse_args()
    
    print_proof_trace(args.report, args.output_dir)


if __name__ == "__main__":
    main()
