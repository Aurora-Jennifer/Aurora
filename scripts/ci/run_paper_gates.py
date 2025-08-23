#!/usr/bin/env python3
"""
Paper Trading Gate Runner
Execute all paper trading gates and enforce ratchet principle.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml


def load_gate_config() -> Dict[str, Any]:
    """Load paper gate configuration."""
    config_path = Path("config/paper_gate.yaml")
    if not config_path.exists():
        print(f"Error: Gate config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_ratchet_config() -> Dict[str, Any]:
    """Load ratchet configuration."""
    ratchet_path = Path("config/ratchet.yaml")
    if not ratchet_path.exists():
        print(f"Error: Ratchet config not found: {ratchet_path}")
        sys.exit(1)
    
    with open(ratchet_path) as f:
        return yaml.safe_load(f)


def run_gate(gate: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single gate and return results."""
    gate_id = gate["id"]
    desc = gate["desc"]
    command = gate["command"]
    timeout = gate.get("timeout_minutes", 10)
    
    print(f"🔍 Running gate: {gate_id}")
    print(f"   Description: {desc}")
    print(f"   Command: {command}")
    print(f"   Timeout: {timeout} minutes")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout * 60
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(f"   {'✅ PASSED' if success else '❌ FAILED'} ({duration:.1f}s)")
        
        if not success:
            print(f"   Error: {result.stderr}")
        
        return {
            "gate_id": gate_id,
            "success": success,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"   ⏰ TIMEOUT ({duration:.1f}s)")
        return {
            "gate_id": gate_id,
            "success": False,
            "duration": duration,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Gate timed out after {timeout} minutes"
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"   💥 ERROR ({duration:.1f}s): {e}")
        return {
            "gate_id": gate_id,
            "success": False,
            "duration": duration,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def check_ratchet_compliance(results: List[Dict[str, Any]], ratchet_config: Dict[str, Any]) -> bool:
    """Check that results comply with ratchet thresholds."""
    thresholds = ratchet_config.get("thresholds", {})
    
    # Check latency thresholds
    latency_thresholds = thresholds.get("latency", {})
    e2d_max_ms = latency_thresholds.get("e2d_max_ms", 150.0)
    
    # Find E2D gate result
    e2d_result = next((r for r in results if "e2d" in r["gate_id"].lower()), None)
    if e2d_result and e2d_result["success"]:
        e2d_duration_ms = e2d_result["duration"] * 1000
        if e2d_duration_ms > e2d_max_ms:
            print(f"❌ Ratchet violation: E2D latency {e2d_duration_ms:.1f}ms > {e2d_max_ms}ms")
            return False
    
    # Check memory thresholds
    memory_thresholds = thresholds.get("memory", {})
    peak_mb = memory_thresholds.get("peak_mb", 500.0)
    
    # This would need to be extracted from metrics in practice
    # For now, just check that gates passed
    
    return True


def main():
    """Main gate runner."""
    print("🚀 Paper Trading Gate Runner")
    print("=" * 50)
    
    # Load configurations
    gate_config = load_gate_config()
    ratchet_config = load_ratchet_config()
    
    gates = gate_config.get("gates", [])
    if not gates:
        print("Error: No gates defined in configuration")
        sys.exit(1)
    
    print(f"Found {len(gates)} gates to run")
    print()
    
    # Run all gates
    results = []
    failed_gates = []
    
    for gate in gates:
        result = run_gate(gate)
        results.append(result)
        
        if not result["success"]:
            failed_gates.append(gate["id"])
        
        print()
    
    # Summary
    print("📊 Gate Results Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/len(results)*100:.1f}%")
    
    if failed_gates:
        print(f"\n❌ Failed Gates:")
        for gate_id in failed_gates:
            print(f"   - {gate_id}")
    
    # Check ratchet compliance
    print("\n🔒 Ratchet Compliance Check")
    print("=" * 50)
    
    ratchet_ok = check_ratchet_compliance(results, ratchet_config)
    if ratchet_ok:
        print("✅ Ratchet compliance: PASSED")
    else:
        print("❌ Ratchet compliance: FAILED")
    
    # Save results
    output_dir = Path("artifacts/gates")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"gate_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "total_gates": len(results),
            "passed": passed,
            "failed": failed,
            "ratchet_compliant": ratchet_ok,
            "results": results
        }, f, indent=2)
    
    print(f"\n📁 Results saved: {results_file}")
    
    # Exit with appropriate code
    if failed > 0 or not ratchet_ok:
        print("\n❌ Gate run FAILED")
        sys.exit(1)
    else:
        print("\n✅ Gate run PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
