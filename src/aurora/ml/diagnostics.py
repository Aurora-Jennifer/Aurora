"""
PyTorch GPU utilization diagnostics.
Identifies bottlenecks in training pipeline.
"""
import contextlib
import time
from collections import defaultdict

import torch


def diagnose_pytorch_step(model, loader, device=None, steps=50):
    """
    Diagnose GPU under-utilization and identify bottlenecks.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device to use (default: cuda if available)
        steps: Number of steps to measure
        
    Returns:
        dict: Timing breakdown and GPU utilization stats
    """
    assert steps > 5, "use >5 for stable averages"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    # Try to expose real pipeline limits
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    it = iter(loader)
    stats = defaultdict(float)

    def cuda_event():
        return torch.cuda.Event(enable_timing=True)

    @contextlib.contextmanager
    def section(name):
        if str(device).startswith("cuda"):
            s, e = cuda_event(), cuda_event()
            torch.cuda.synchronize()
            s.record()
            yield
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e)
            stats[name] += ms / 1000.0  # seconds
        else:
            t0 = time.perf_counter()
            yield
            stats[name] += (time.perf_counter() - t0)

    # Warmup one batch to build kernels/graphs
    try:
        batch_warm = next(it)
        if len(batch_warm) == 2:
            x_warm, y_warm = batch_warm
        else:
            x_warm, y_warm, _ = batch_warm  # Handle 3+ tensor datasets
    except StopIteration:
        it = iter(loader)
        batch_warm = next(it)
        if len(batch_warm) == 2:
            x_warm, y_warm = batch_warm
        else:
            x_warm, y_warm, _ = batch_warm
        
    with section("h2d"):
        x_warm = x_warm.to(device, non_blocking=True)
        y_warm = y_warm.to(device, non_blocking=True)
    with section("fwd"):
        out = model(x_warm)
        loss = out.mean() if not hasattr(out, "loss") else out.loss
    with section("bwd"):
        loss.backward()
    with section("opt"):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None  # simulate zero_grad for harness

    # Timed steps
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    t0 = time.perf_counter()
    for i in range(steps):
        with section("load"):
            try:
                batch = next(it)
                if len(batch) == 2:
                    xb, yb = batch
                else:
                    xb, yb, _ = batch  # Handle 3+ tensor datasets
            except StopIteration:
                it = iter(loader)
                batch = next(it)
                if len(batch) == 2:
                    xb, yb = batch
                else:
                    xb, yb, _ = batch

        with section("h2d"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

        with section("fwd"):
            out = model(xb)
            loss = out.mean() if not hasattr(out, "loss") else out.loss

        with section("bwd"):
            loss.backward()

        with section("opt"):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

    total = time.perf_counter() - t0
    # Collect instantaneous GPU stats if available
    gpu_util = None
    if torch.cuda.is_available():
        try:
            import shlex
            import subprocess
            q = "nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits"
            line = subprocess.check_output(shlex.split(q)).decode().strip().splitlines()[0]
            util, power, mem = [float(v) for v in line.split(",")]
            gpu_util = {"util_%": util, "power_W": power, "mem_MB": mem}
        except Exception:
            pass

    avg = {k: stats[k]/steps for k in ("load","h2d","fwd","bwd","opt")}
    step_s = sum(avg.values())
    result = {
        "avg_seconds": avg,
        "per_step_seconds": step_s,
        "throughput_batches_per_s": 1.0/step_s if step_s else None,
        "timeline_share_%": {k: round(100*avg[k]/step_s, 1) for k in avg},
        "gpu_sample": gpu_util,
        "steps": steps,
        "notes": {
            "bottleneck_rule": "load>> means CPU/disk; h2d>> means transfer; fwd/bwd/opt>> compute; small fwd with low util => increase batch",
            "tuning_knobs": "batch_size, num_workers, pin_memory, prefetch_factor, non_blocking, AMP, gradient accumulation"
        }
    }
    print(result)
    return result


def get_system_info():
    """Get system and GPU information"""
    import platform
    import subprocess
    
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader", 
            shell=True
        ).decode().strip()
        info["gpu_details"] = gpu_info
    except Exception as e:
        info["gpu_error"] = str(e)
        
    return info
