#!/usr/bin/env python3
"""
Production Training Scheduler - 1 Hour Budget

Orchestrates GPU and CPU tasks for large-scale cross-sectional training:
- GPU tasks run serially (one at a time)
- CPU tasks run in parallel (8 workers)
- Optimized for 3080 + i7-11700K + 32GB
"""

import argparse
import logging
import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_environment():
    """Set environment variables for optimal performance."""
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    os.environ["XGBOOST_NUM_THREADS"] = "8"
    
    logger.info("Environment variables set for optimal performance")


def run_command(cmd, timeout=3600):
    """Run a command with timeout."""
    logger.info(f">> {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            timeout=timeout, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Command completed in {duration:.2f}s")
        
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout[:500]}...")
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr[:500]}...")
            
        return True, duration
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Command timed out after {timeout}s")
        return False, timeout
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"❌ Command failed after {duration:.2f}s: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False, duration


def gpu_worker(gpu_queue, results):
    """GPU worker thread - processes tasks serially."""
    logger.info("🚀 GPU worker started")
    
    while True:
        try:
            task = gpu_queue.get(timeout=1)
            if task is None:  # Shutdown signal
                break
            
            task_name, cmd = task
            logger.info(f"🎯 Processing GPU task: {task_name}")
            
            success, duration = run_command(cmd, timeout=3600)
            results[task_name] = {
                'success': success,
                'duration': duration,
                'type': 'gpu'
            }
            
            gpu_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"GPU worker error: {e}")
            gpu_queue.task_done()


def cpu_worker(cpu_queue, results, worker_id):
    """CPU worker function."""
    logger.info(f"💻 CPU worker {worker_id} started")
    
    while True:
        try:
            task = cpu_queue.get(timeout=1)
            if task is None:  # Shutdown signal
                break
            
            task_name, cmd = task
            logger.info(f"🎯 Processing CPU task: {task_name}")
            
            success, duration = run_command(cmd, timeout=1800)
            results[task_name] = {
                'success': success,
                'duration': duration,
                'type': 'cpu',
                'worker_id': worker_id
            }
            
            cpu_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"CPU worker {worker_id} error: {e}")
            cpu_queue.task_done()


def run_production_training(universe_cfg, xgb_cfg, catboost_cfg, output_dir):
    """Run production training pipeline."""
    logger.info("=== PRODUCTION TRAINING PIPELINE ===")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment
    set_environment()
    
    # Task queues
    gpu_queue = queue.Queue()
    cpu_queue = queue.Queue()
    results = {}
    
    # GPU tasks (run serially)
    gpu_tasks = [
        ("xgb_global", [
            "python", "scripts/run_universe.py",
            "--universe-cfg", universe_cfg,
            "--grid-cfg", xgb_cfg,
            "--out-dir", str(output_dir / "xgb_global")
        ]),
        ("catboost_global", [
            "python", "scripts/run_universe.py",
            "--universe-cfg", universe_cfg,
            "--grid-cfg", catboost_cfg,
            "--out-dir", str(output_dir / "catboost_global")
        ])
    ]
    
    # CPU tasks (run in parallel)
    cpu_tasks = [
        ("ridge_per_asset", [
            "python", "scripts/run_universe.py",
            "--universe-cfg", universe_cfg,
            "--grid-cfg", "config/grids/ridge_per_asset.yaml",
            "--out-dir", str(output_dir / "ridge_per_asset")
        ]),
        ("feature_engineering", [
            "python", "scripts/build_features.py",
            "--universe-cfg", universe_cfg,
            "--out-dir", str(output_dir / "features")
        ]),
        ("baseline_comparison", [
            "python", "scripts/baseline_comparison.py",
            "--input-dir", str(output_dir),
            "--out-dir", str(output_dir / "baselines")
        ])
    ]
    
    # Add tasks to queues
    for task in gpu_tasks:
        gpu_queue.put(task)
    
    for task in cpu_tasks:
        cpu_queue.put(task)
    
    # Start GPU worker
    gpu_thread = threading.Thread(target=gpu_worker, args=(gpu_queue, results))
    gpu_thread.daemon = True
    gpu_thread.start()
    
    # Start CPU workers
    cpu_workers = []
    for i in range(8):  # 8 CPU workers
        worker_thread = threading.Thread(target=cpu_worker, args=(cpu_queue, results, i))
        worker_thread.daemon = True
        worker_thread.start()
        cpu_workers.append(worker_thread)
    
    # Wait for GPU tasks to complete
    logger.info("⏳ Waiting for GPU tasks...")
    gpu_queue.join()
    
    # Wait for CPU tasks to complete
    logger.info("⏳ Waiting for CPU tasks...")
    cpu_queue.join()
    
    # Shutdown workers
    gpu_queue.put(None)
    for _ in range(8):
        cpu_queue.put(None)
    
    # Wait for threads to finish
    gpu_thread.join()
    for worker in cpu_workers:
        worker.join()
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, output_dir)
    
    logger.info("=== PRODUCTION TRAINING COMPLETE ===")


def print_results(results):
    """Print training results summary."""
    print("\n🎯 **PRODUCTION TRAINING RESULTS**")
    print("=" * 60)
    
    gpu_tasks = [r for r in results.values() if r.get('type') == 'gpu']
    cpu_tasks = [r for r in results.values() if r.get('type') == 'cpu']
    
    if gpu_tasks:
        print(f"\n🚀 **GPU Tasks**")
        total_gpu_time = sum(r.get('duration', 0) for r in gpu_tasks)
        for task_name, result in results.items():
            if result.get('type') == 'gpu':
                status = "✅" if result.get('success') else "❌"
                duration = result.get('duration', 0)
                print(f"   {status} {task_name}: {duration:.2f}s")
        print(f"   Total GPU time: {total_gpu_time:.2f}s")
    
    if cpu_tasks:
        print(f"\n💻 **CPU Tasks**")
        total_cpu_time = sum(r.get('duration', 0) for r in cpu_tasks)
        for task_name, result in results.items():
            if result.get('type') == 'cpu':
                status = "✅" if result.get('success') else "❌"
                duration = result.get('duration', 0)
                worker_id = result.get('worker_id', '?')
                print(f"   {status} {task_name} (worker {worker_id}): {duration:.2f}s")
        print(f"   Total CPU time: {total_cpu_time:.2f}s")
    
    # Overall stats
    total_time = max(r.get('duration', 0) for r in results.values())
    successful_tasks = sum(1 for r in results.values() if r.get('success'))
    total_tasks = len(results)
    
    print(f"\n📊 **Overall**")
    print(f"   Total wall time: {total_time:.2f}s")
    print(f"   Successful tasks: {successful_tasks}/{total_tasks}")
    print(f"   Success rate: {successful_tasks/total_tasks*100:.1f}%")


def save_results(results, output_dir):
    """Save training results to file."""
    results_file = output_dir / "training_results.json"
    
    # Add metadata
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_tasks': len(results),
            'successful_tasks': sum(1 for r in results.values() if r.get('success')),
            'total_wall_time': max(r.get('duration', 0) for r in results.values()),
            'gpu_time': sum(r.get('duration', 0) for r in results.values() if r.get('type') == 'gpu'),
            'cpu_time': sum(r.get('duration', 0) for r in results.values() if r.get('type') == 'cpu')
        }
    }
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Production Training Scheduler')
    parser.add_argument('--universe-cfg', type=str, 
                       default='config/universe_production.yaml',
                       help='Universe configuration file')
    parser.add_argument('--xgb-cfg', type=str,
                       default='config/grids/cs_xgb_gpu_hour.yaml',
                       help='XGBoost GPU configuration file')
    parser.add_argument('--catboost-cfg', type=str,
                       default='config/grids/cs_cat_gpu_light.yaml',
                       help='CatBoost GPU configuration file')
    parser.add_argument('--output-dir', type=str,
                       default=f'results/production_{datetime.now().strftime("%Y%m%d_%H%M")}',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run production training
    run_production_training(
        args.universe_cfg,
        args.xgb_cfg,
        args.catboost_cfg,
        args.output_dir
    )


if __name__ == '__main__':
    main()
