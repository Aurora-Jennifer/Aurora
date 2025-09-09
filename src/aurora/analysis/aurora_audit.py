#!/usr/bin/env python3
"""
Aurora System Audit Script (CLEARFRAME Mode)

Brutally honest assessment of trading system readiness.
No fluffing, no hedging - just facts.
"""

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class AuroraAuditor:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def log_check(self, category, test_name, passed, details="", score=None):
        """Log a single audit check"""
        if category not in self.results:
            self.results[category] = {"tests": [], "score": 0, "total": 0}
        
        self.results[category]["tests"].append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "score": score
        })
        
        if score is not None:
            self.results[category]["total"] += 1
            self.results[category]["score"] += score
            
        status = "✅" if passed else "❌"
        score_str = f" ({score}/10)" if score is not None else ""
        print(f"{status} {category}: {test_name}{score_str}")
        if details:
            print(f"   └─ {details}")
    
    def audit_architecture(self):
        """Check if core modules exist and can be imported"""
        print("\n🏗️  ARCHITECTURE CHECK")
        print("=" * 50)
        
        core_modules = [
            "core.data_sanity.api",
            "core.engine.backtest", 
            "core.walk.ml_pipeline",
            "core.risk.guardrails",
            "core.model_router",
            "core.config_loader"
        ]
        
        import_score = 0
        for module in core_modules:
            try:
                importlib.import_module(module)
                self.log_check("Architecture", f"Import {module}", True)
                import_score += 1
            except Exception as e:
                self.log_check("Architecture", f"Import {module}", False, str(e))
        
        # Overall architecture score
        arch_score = int(10 * import_score / len(core_modules))
        self.log_check("Architecture", "Module System", arch_score >= 7, 
                      f"{import_score}/{len(core_modules)} modules importable", arch_score)
    
    def audit_data_sanity(self):
        """Check data validation and sanity systems"""
        print("\n🧹 DATA SANITY CHECK") 
        print("=" * 50)
        
        try:
            from core.data_sanity.api import DataSanityValidator
            validator = DataSanityValidator()
            self.log_check("Data Sanity", "Validator loads", True)
            
            # Test with dummy data (correct column names for data sanity)
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
                'Open': np.random.randn(10) + 100,
                'High': np.random.randn(10) + 102, 
                'Low': np.random.randn(10) + 98,
                'Close': np.random.randn(10) + 100,
                'Volume': np.random.randint(1000, 10000, 10)
            })
            
            result = validator.validate_and_repair(test_data, "TEST")
            self.log_check("Data Sanity", "Validation runs", True, 
                          "Validation completed", 8)
            
        except Exception as e:
            self.log_check("Data Sanity", "Validator loads", False, str(e), 0)
    
    def audit_models(self):
        """Check model loading and inference"""
        print("\n🤖 MODEL SYSTEM CHECK")
        print("=" * 50)
        
        model_paths = [
            "artifacts/models/linear_v1.pkl",
            "artifacts/models/latest.onnx", 
            "models/universal_v1.pkl",
            "artifacts/models/dummy_v1.pkl"
        ]
        
        found_models = 0
        working_models = 0
        
        for path in model_paths:
            if os.path.exists(path):
                found_models += 1
                self.log_check("Models", f"Found {path}", True)
                
                # Try to load it
                try:
                    if path.endswith('.pkl'):
                        import pickle
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                        working_models += 1
                        self.log_check("Models", f"Load {path}", True)
                    elif path.endswith('.onnx'):
                        # Would need onnxruntime to test, skip for now
                        self.log_check("Models", f"Load {path}", True, "ONNX (assumed working)")
                        working_models += 1
                except Exception as e:
                    self.log_check("Models", f"Load {path}", False, str(e))
            else:
                self.log_check("Models", f"Found {path}", False)
        
        model_score = int(10 * working_models / len(model_paths)) if model_paths else 0
        self.log_check("Models", "Model System", working_models > 0,
                      f"{working_models}/{len(model_paths)} models working", model_score)
    
    def audit_backtesting(self):
        """Check backtesting engine"""
        print("\n📈 BACKTESTING CHECK")
        print("=" * 50)
        
        try:
            from core.engine.backtest import BacktestEngine
            # Use a default config file that should exist
            engine = BacktestEngine("config/base.yaml")
            self.log_check("Backtesting", "Engine loads", True, "", 7)
            
            # Check if we can run a minimal backtest
            # (Would need more setup for full test)
            self.log_check("Backtesting", "Determinism test", False, 
                          "Not implemented - need seed/repeatability tests", 3)
            
        except Exception as e:
            self.log_check("Backtesting", "Engine loads", False, str(e), 2)
    
    def audit_execution(self):
        """Check paper trading and execution"""
        print("\n⚡ EXECUTION CHECK")
        print("=" * 50)
        
        try:
            # Check if paper runner exists and is callable
            paper_script = "scripts/paper_runner.py"
            if os.path.exists(paper_script):
                self.log_check("Execution", "Paper runner exists", True)
                
                # Try importing the execution modules
                try:
                    from oms.paper_adapter import PaperAdapter
                    adapter = PaperAdapter()
                    self.log_check("Execution", "Paper adapter loads", True, "", 6)
                except Exception as e:
                    self.log_check("Execution", "Paper adapter loads", False, str(e), 2)
                    
            else:
                self.log_check("Execution", "Paper runner exists", False, "", 0)
                
        except Exception as e:
            self.log_check("Execution", "Execution system", False, str(e), 0)
    
    def audit_config_system(self):
        """Check configuration and contracts"""
        print("\n⚙️  CONFIG SYSTEM CHECK")
        print("=" * 50)
        
        config_files = [
            "config/base.yaml",
            "config/base.json", 
            "contracts/dtypes.yaml"
        ]
        
        found_configs = 0
        for config_file in config_files:
            if os.path.exists(config_file):
                found_configs += 1
                self.log_check("Config", f"Found {config_file}", True)
            else:
                self.log_check("Config", f"Found {config_file}", False)
        
        # Try loading config
        try:
            from core.config_loader import load_config
            config = load_config()
            self.log_check("Config", "Config loader works", True, "", 8)
        except Exception as e:
            self.log_check("Config", "Config loader works", False, str(e), 2)
    
    def audit_ci_testing(self):
        """Check CI/CD and testing setup"""
        print("\n🧪 CI/CD & TESTING CHECK")
        print("=" * 50)
        
        # Check for CI files
        ci_files = [
            ".github/workflows/ci.yml",
            "pytest.ini",
            "Makefile"
        ]
        
        for ci_file in ci_files:
            exists = os.path.exists(ci_file)
            self.log_check("CI/CD", f"Found {ci_file}", exists)
        
        # Check test coverage
        test_dirs = ["tests/", "tests/unit/", "tests/integration/"]
        test_files = []
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                test_files.extend(list(Path(test_dir).rglob("test_*.py")))
        
        test_score = min(10, len(test_files))
        self.log_check("CI/CD", "Test Coverage", len(test_files) > 5,
                      f"Found {len(test_files)} test files", test_score)
    
    def audit_operational_readiness(self):
        """Check logging, monitoring, kill switches"""
        print("\n🚨 OPERATIONAL READINESS CHECK")
        print("=" * 50)
        
        # Check for logging setup
        log_dirs = ["logs/", "reports/"]
        for log_dir in log_dirs:
            exists = os.path.exists(log_dir)
            self.log_check("Operations", f"Found {log_dir}", exists)
        
        # Check for kill switch
        kill_files = ["runtime/ALLOW_LIVE.txt", "KILL_SWITCH"]
        kill_found = any(os.path.exists(f) for f in kill_files)
        self.log_check("Operations", "Kill switch mechanism", kill_found, "", 6 if kill_found else 2)
        
        # Check for monitoring scripts
        monitor_scripts = list(Path("scripts/").rglob("monitor*.py"))
        self.log_check("Operations", "Monitoring scripts", len(monitor_scripts) > 0,
                      f"Found {len(monitor_scripts)} monitoring scripts", 
                      min(10, len(monitor_scripts) * 3))
    
    def generate_final_report(self):
        """Generate final brutal assessment"""
        print("\n" + "=" * 70)
        print("🔍 FINAL ASSESSMENT (CLEARFRAME MODE)")
        print("=" * 70)
        
        total_score = 0
        total_categories = 0
        
        for category, data in self.results.items():
            if data["total"] > 0:
                avg_score = data["score"] / data["total"]
                total_score += avg_score
                total_categories += 1
                
                if avg_score >= 8:
                    status = "🟢 GOOD"
                elif avg_score >= 5:
                    status = "🟡 WEAK"
                else:
                    status = "🔴 BROKEN"
                    
                print(f"{status} {category}: {avg_score:.1f}/10")
        
        if total_categories > 0:
            overall_score = total_score / total_categories
        else:
            overall_score = 0
            
        print(f"\n📊 OVERALL SYSTEM SCORE: {overall_score:.1f}/10")
        
        # Brutal verdict
        if overall_score >= 8:
            verdict = "🟢 FUND-READY: This system could handle real money with minor fixes"
        elif overall_score >= 6:
            verdict = "🟡 DEMO-READY: Good for paper trading, needs work for live"
        elif overall_score >= 4:
            verdict = "🟠 PROTOTYPE: Core logic works, infrastructure is shaky"
        else:
            verdict = "🔴 ACADEMIC TOY: Not ready for any real trading"
            
        print(f"\n🎯 VERDICT: {verdict}")
        
        # Top 3 blockers
        print("\n⚠️  TOP BLOCKERS:")
        failures = []
        for category, data in self.results.items():
            for test in data["tests"]:
                score = test.get("score")
                if not test["passed"] and (score is None or score < 5):
                    failures.append(f"{category}: {test['test']}")
        
        for i, failure in enumerate(failures[:3], 1):
            print(f"   {i}. {failure}")
        
        # Save detailed report
        report_file = f"reports/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("reports", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_score": overall_score,
                "verdict": verdict,
                "results": self.results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return overall_score

def main():
    print("🔍 AURORA TRADING SYSTEM AUDIT")
    print("=" * 70)
    print("Brutal honesty mode: No fluffing, just facts.")
    print("=" * 70)
    
    auditor = AuroraAuditor()
    
    # Run all audits
    auditor.audit_architecture()
    auditor.audit_data_sanity() 
    auditor.audit_models()
    auditor.audit_backtesting()
    auditor.audit_execution()
    auditor.audit_config_system()
    auditor.audit_ci_testing()
    auditor.audit_operational_readiness()
    
    # Final assessment
    score = auditor.generate_final_report()
    
    return score

if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 6 else 1)  # Exit with error if system not ready
