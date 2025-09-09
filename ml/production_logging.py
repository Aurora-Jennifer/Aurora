"""
Production logging setup with UTF-8 safety and worker queue handling.

Replaces print statements with proper logging for paper trading.
"""
import logging
import logging.handlers
import queue
import sys
import os
import re
from typing import Optional, Dict
from pathlib import Path
import threading
import time


def clean_unicode_sequences(text: str) -> str:
    """
    Clean problematic Unicode sequences from log messages.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text safe for logging
    """
    # Remove emoji variation selectors and other problematic sequences
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)  # Variation selectors
    text = re.sub(r'[\u200B-\u200D]', '', text)  # Zero-width characters
    text = re.sub(r'[\u2060-\u206F]', '', text)  # Word joiners etc
    
    # Ensure printable ASCII for critical logs
    if any(ord(c) > 127 for c in text):
        # Keep basic unicode but escape problematic ones
        text = text.encode('ascii', 'replace').decode('ascii')
    
    return text


class ProductionFormatter(logging.Formatter):
    """Custom formatter with Unicode safety and structured output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Clean the message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = clean_unicode_sequences(record.msg)
        
        # Clean any args
        if hasattr(record, 'args') and record.args:
            cleaned_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    cleaned_args.append(clean_unicode_sequences(arg))
                else:
                    cleaned_args.append(arg)
            record.args = tuple(cleaned_args)
        
        return super().format(record)


def setup_production_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_log_size_mb: int = 50,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup production logging with file rotation and UTF-8 safety.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Whether to log to files
        enable_console_logging: Whether to log to console
        max_log_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = ProductionFormatter()
    
    # File handler with rotation
    if enable_file_logging:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "trader.log",
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    # Console handler with UTF-8 safety
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        
        # Ensure UTF-8 encoding
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        
        root_logger.addHandler(console_handler)
    
    # Set up queue handler for worker processes
    setup_worker_logging_queue()
    
    return root_logger


def setup_worker_logging_queue():
    """Setup queue-based logging for worker processes."""
    # Create a queue for worker log messages
    log_queue = queue.Queue(-1)
    
    # Queue handler for workers
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    
    # Start queue listener in a separate thread
    def process_log_queue():
        while True:
            try:
                record = log_queue.get(timeout=1)
                if record is None:
                    break
                
                # Process the record through main handlers
                for handler in root_logger.handlers:
                    if not isinstance(handler, logging.handlers.QueueHandler):
                        handler.handle(record)
                        
            except queue.Empty:
                continue
            except Exception as e:
                # Fallback to print if logging fails
                print(f"Logging error: {e}", file=sys.stderr)
    
    # Start background thread
    thread = threading.Thread(target=process_log_queue, daemon=True)
    thread.start()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with production settings.
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        Configured logger
    """
    if name is None:
        # Auto-detect caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


def log_performance_metrics(logger: logging.Logger, 
                           metrics: Dict, 
                           prefix: str = "PERF") -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Log message prefix
    """
    # Clean and format metrics
    cleaned_metrics = {}
    for key, value in metrics.items():
        clean_key = clean_unicode_sequences(str(key))
        if isinstance(value, (int, float)):
            cleaned_metrics[clean_key] = value
        else:
            cleaned_metrics[clean_key] = clean_unicode_sequences(str(value))
    
    # Log as structured message
    metric_str = " | ".join([f"{k}={v}" for k, v in cleaned_metrics.items()])
    logger.info(f"{prefix}: {metric_str}")


def replace_print_with_logging():
    """
    Replace print statements with logging calls (for legacy code).
    
    WARNING: This is a global monkey patch. Use carefully.
    """
    original_print = print
    logger = get_logger('print_replacement')
    
    def logged_print(*args, **kwargs):
        # Convert print to log message
        message = ' '.join(str(arg) for arg in args)
        message = clean_unicode_sequences(message)
        
        # Determine log level based on content
        if any(word in message.lower() for word in ['error', 'failed', 'exception']):
            logger.error(message)
        elif any(word in message.lower() for word in ['warning', 'warn']):
            logger.warning(message)
        elif any(word in message.lower() for word in ['debug', '🔍']):
            logger.debug(message)
        else:
            logger.info(message)
    
    # Replace built-in print
    import builtins
    builtins.print = logged_print
    
    return original_print  # Return original for restoration


def configure_environment_logging():
    """Configure environment variables for consistent logging."""
    # Force UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Disable buffering for immediate output
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Set logging level from environment
    log_level = os.environ.get('TRADER_LOG_LEVEL', 'INFO')
    
    return log_level


def create_paper_trading_logger() -> logging.Logger:
    """
    Create specialized logger for paper trading with all safety features.
    
    Returns:
        Configured paper trading logger
    """
    # Configure environment
    log_level = configure_environment_logging()
    
    # Setup production logging
    setup_production_logging(
        log_dir="logs/paper_trading",
        log_level=log_level,
        enable_file_logging=True,
        enable_console_logging=True,
        max_log_size_mb=100,  # Larger files for paper trading
        backup_count=10
    )
    
    # Get specialized logger
    logger = get_logger('paper_trading')
    
    logger.info("📋 Paper trading logger initialized")
    logger.info(f"   Log level: {log_level}")
    logger.info(f"   UTF-8 encoding: {os.environ.get('PYTHONIOENCODING', 'not set')}")
    logger.info(f"   Unbuffered output: {os.environ.get('PYTHONUNBUFFERED', 'not set')}")
    
    return logger


def test_logging_setup():
    """Test logging setup with various message types."""
    print("🧪 TESTING PRODUCTION LOGGING SETUP")
    print("="*50)
    
    # Setup logging
    logger = create_paper_trading_logger()
    
    # Test different log levels
    logger.debug("🔍 Debug message with unicode: π ≈ 3.14159")
    logger.info("📊 Info message with emoji: Strategy performance ✅")
    logger.warning("⚠️ Warning message with problematic unicode: \uFE0F")
    logger.error("❌ Error message with mixed content: Failed α = 0.05")
    
    # Test metrics logging
    metrics = {
        'sharpe_ratio': 0.324,
        'ic_rank': 0.0174,
        'turnover': 1.85,
        'status': 'running ✅'
    }
    
    log_performance_metrics(logger, metrics, "DAILY_METRICS")
    
    # Test Unicode cleaning
    dirty_text = "Test with \uFE0F emoji variation and \u200B zero-width"
    clean_text = clean_unicode_sequences(dirty_text)
    logger.info(f"Unicode cleaning: '{dirty_text}' → '{clean_text}'")
    
    print(f"\n✅ Logging test completed - check logs/paper_trading/trader.log")


if __name__ == "__main__":
    test_logging_setup()
