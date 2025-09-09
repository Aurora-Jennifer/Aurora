#!/usr/bin/env python3
"""
Hardware-Optimized Training Scheduler

Orchestrates GPU and CPU tasks to maximize hardware utilization:
- GPU tasks run serially (one at a time)
- CPU tasks run in parallel (6-8 workers)
- Proper thread management and memory optimization
"""

import argparse
import logging
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Callable, Any
import yaml
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import GPUtil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HardwareMonitor:
    """Monitor hardware utilization."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = len(GPUtil.getGPUs()) > 0
        
        logger.info(f"Hardware detected:")
        logger.info(f"  CPU cores: {self.cpu_count}")
        logger.info(f"  Memory: {self.memory_gb:.1f} GB")
        logger.info(f"  GPU available: {self.gpu_available}")
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def get_gpu_usage(self) -> Dict[str, float]:
        """Get GPU utilization and memory usage."""
        if not self.gpu_available:
            return {}
        
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {}
        
        gpu = gpus[0]  # Use first GPU
        return {
            'utilization': gpu.load * 100,
            'memory_used': gpu.memoryUsed,
            'memory_total': gpu.memoryTotal,
            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
            'temperature': gpu.temperature
        }


class TaskScheduler:
    """Hardware-optimized task scheduler."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set environment variables
        self._set_environment()
        
        # Initialize hardware monitor
        self.monitor = HardwareMonitor()
        
        # Task queues
        self.gpu_queue = queue.Queue()
        self.cpu_queue = queue.Queue()
        
        # Results storage
        self.results = {}
        self.results_lock = threading.Lock()
        
        # Worker pools
        self.cpu_executor = None
        self.gpu_worker = None
        
        logger.info("Task scheduler initialized")
    
    def _set_environment(self):
        """Set environment variables for optimal performance."""
        runtime_config = self.config.get('runtime', {})
        
        # Thread limits
        os.environ['OMP_NUM_THREADS'] = str(runtime_config.get('omp_num_threads', 8))
        os.environ['MKL_NUM_THREADS'] = str(runtime_config.get('mkl_num_threads', 8))
        os.environ['OPENBLAS_NUM_THREADS'] = str(runtime_config.get('openblas_num_threads', 8))
        os.environ['NUMEXPR_NUM_THREADS'] = str(runtime_config.get('numexpr_num_threads', 8))
        os.environ['XGBOOST_NUM_THREADS'] = str(runtime_config.get('xgboost_num_threads', 8))
        
        # GPU settings
        if runtime_config.get('gpu_allow_growth', True):
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        logger.info("Environment variables set for optimal performance")
    
    def add_gpu_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Add a GPU task to the queue."""
        self.gpu_queue.put({
            'id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs
        })
        logger.info(f"Added GPU task: {task_id}")
    
    def add_cpu_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Add a CPU task to the queue."""
        self.cpu_queue.put({
            'id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs
        })
        logger.info(f"Added CPU task: {task_id}")
    
    def _gpu_worker_thread(self):
        """GPU worker thread - processes tasks serially."""
        logger.info("GPU worker thread started")
        
        while True:
            try:
                task = self.gpu_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                task_id = task['id']
                task_func = task['func']
                args = task['args']
                kwargs = task['kwargs']
                
                logger.info(f"Processing GPU task: {task_id}")
                start_time = time.time()
                
                # Execute task
                result = task_func(*args, **kwargs)
                
                # Store result
                with self.results_lock:
                    self.results[task_id] = {
                        'result': result,
                        'duration': time.time() - start_time,
                        'type': 'gpu'
                    }
                
                logger.info(f"Completed GPU task: {task_id} in {time.time() - start_time:.2f}s")
                self.gpu_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU task {task_id} failed: {e}")
                with self.results_lock:
                    self.results[task_id] = {
                        'error': str(e),
                        'type': 'gpu'
                    }
                self.gpu_queue.task_done()
    
    def _cpu_worker_thread(self, task):
        """CPU worker function."""
        task_id = task['id']
        task_func = task['func']
        args = task['args']
        kwargs = task['kwargs']
        
        logger.info(f"Processing CPU task: {task_id}")
        start_time = time.time()
        
        try:
            # Execute task
            result = task_func(*args, **kwargs)
            
            # Store result
            with self.results_lock:
                self.results[task_id] = {
                    'result': result,
                    'duration': time.time() - start_time,
                    'type': 'cpu'
                }
            
            logger.info(f"Completed CPU task: {task_id} in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"CPU task {task_id} failed: {e}")
            with self.results_lock:
                self.results[task_id] = {
                    'error': str(e),
                    'type': 'cpu'
                }
            raise e
    
    def start_workers(self):
        """Start GPU and CPU workers."""
        # Start GPU worker thread
        if self.monitor.gpu_available:
            self.gpu_worker = threading.Thread(target=self._gpu_worker_thread)
            self.gpu_worker.daemon = True
            self.gpu_worker.start()
            logger.info("GPU worker started")
        
        # Start CPU worker pool
        cpu_workers = self.config.get('runtime', {}).get('cpu_workers', 6)
        self.cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
        logger.info(f"CPU worker pool started with {cpu_workers} workers")
    
    def submit_cpu_tasks(self):
        """Submit all CPU tasks to the worker pool."""
        cpu_tasks = []
        while not self.cpu_queue.empty():
            task = self.cpu_queue.get()
            cpu_tasks.append(task)
        
        if cpu_tasks:
            logger.info(f"Submitting {len(cpu_tasks)} CPU tasks")
            futures = [self.cpu_executor.submit(self._cpu_worker_thread, task) for task in cpu_tasks]
            return futures
        
        return []
    
    def wait_for_completion(self, timeout: int = 3600):
        """Wait for all tasks to complete."""
        logger.info("Waiting for task completion...")
        
        # Wait for GPU tasks
        if self.gpu_worker:
            self.gpu_queue.join()
        
        # Wait for CPU tasks
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        
        logger.info("All tasks completed")
    
    def get_results(self) -> Dict[str, Any]:
        """Get all task results."""
        with self.results_lock:
            return self.results.copy()
    
    def print_performance_summary(self):
        """Print performance summary."""
        results = self.get_results()
        
        gpu_tasks = [r for r in results.values() if r.get('type') == 'gpu']
        cpu_tasks = [r for r in results.values() if r.get('type') == 'cpu']
        
        print("\n🎯 **TRAINING PERFORMANCE SUMMARY**")
        print("=" * 60)
        
        if gpu_tasks:
            gpu_duration = sum(r.get('duration', 0) for r in gpu_tasks)
            print(f"\n🚀 **GPU Tasks**")
            print(f"   Total tasks: {len(gpu_tasks)}")
            print(f"   Total duration: {gpu_duration:.2f}s")
            print(f"   Average per task: {gpu_duration/len(gpu_tasks):.2f}s")
        
        if cpu_tasks:
            cpu_duration = sum(r.get('duration', 0) for r in cpu_tasks)
            print(f"\n💻 **CPU Tasks**")
            print(f"   Total tasks: {len(cpu_tasks)}")
            print(f"   Total duration: {cpu_duration:.2f}s")
            print(f"   Average per task: {cpu_duration/len(cpu_tasks):.2f}s")
        
        # Hardware utilization
        print(f"\n📊 **Hardware Utilization**")
        print(f"   CPU usage: {self.monitor.get_cpu_usage():.1f}%")
        print(f"   Memory usage: {self.monitor.get_memory_usage():.1f}%")
        
        if self.monitor.gpu_available:
            gpu_stats = self.monitor.get_gpu_usage()
            if gpu_stats:
                print(f"   GPU utilization: {gpu_stats['utilization']:.1f}%")
                print(f"   GPU memory: {gpu_stats['memory_percent']:.1f}%")
                print(f"   GPU temperature: {gpu_stats['temperature']}°C")
        
        print(f"\n✅ **Training Complete**")


def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency."""
    optimized_df = df.copy()
    
    # Convert float64 to float32
    float_cols = optimized_df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        optimized_df[col] = optimized_df[col].astype('float32')
    
    # Convert int64 to int32
    int_cols = optimized_df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        optimized_df[col] = optimized_df[col].astype('int32')
    
    # Convert object to category if low cardinality
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if optimized_df[col].nunique() < 1000:
            optimized_df[col] = optimized_df[col].astype('category')
    
    logger.info(f"Data types optimized: {len(float_cols)} float64→float32, {len(int_cols)} int64→int32")
    return optimized_df


def main():
    parser = argparse.ArgumentParser(description='Hardware-Optimized Training Scheduler')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Data file to process')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = TaskScheduler(args.config)
    
    # Load and optimize data
    logger.info(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file)
    df = optimize_data_types(df)
    
    # Add tasks (example)
    # scheduler.add_gpu_task('global_xgb', train_global_xgb, df, args.output_dir)
    # scheduler.add_cpu_task('per_asset_ridge', train_per_asset_ridge, df, args.output_dir)
    
    # Start workers
    scheduler.start_workers()
    
    # Submit CPU tasks
    cpu_futures = scheduler.submit_cpu_tasks()
    
    # Wait for completion
    scheduler.wait_for_completion()
    
    # Print performance summary
    scheduler.print_performance_summary()
    
    logger.info("Training scheduler completed")


if __name__ == '__main__':
    main()
