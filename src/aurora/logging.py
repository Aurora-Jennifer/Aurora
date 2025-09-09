"""
Centralized Logging Utilities

Provides consistent logging setup across the ML trading system.
All loggers use the same formatter and configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Global logger configuration
_logger_config = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_handler": None,
    "console_handler": None,
    "formatter": None,
}


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None,
    date_format: str | None = None,
    force: bool = False,
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        date_format: Optional custom date format
        force: Force reconfiguration even if already setup
    """
    global _logger_config

    if _logger_config["formatter"] is not None and not force:
        return  # Already configured

    # Set configuration
    _logger_config["level"] = level
    if format_string:
        _logger_config["format"] = format_string
    if date_format:
        _logger_config["date_format"] = date_format

    # Create formatter
    _logger_config["formatter"] = logging.Formatter(
        _logger_config["format"], datefmt=_logger_config["date_format"]
    )

    # Create console handler
    _logger_config["console_handler"] = logging.StreamHandler(sys.stdout)
    _logger_config["console_handler"].setFormatter(_logger_config["formatter"])
    _logger_config["console_handler"].setLevel(level)

    # Create file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger_config["file_handler"] = logging.FileHandler(log_file)
        _logger_config["file_handler"].setFormatter(_logger_config["formatter"])
        _logger_config["file_handler"].setLevel(level)


def get_logger(name: str, level: int | None = None, propagate: bool = True) -> logging.Logger:
    """
    Get a logger with consistent configuration.

    Args:
        name: Logger name (usually __name__)
        level: Optional custom level for this logger
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger
    """
    # Setup logging if not already done
    if _logger_config["formatter"] is None:
        setup_logging()

    # Get logger
    logger = logging.getLogger(name)
    logger.propagate = propagate

    # Set level
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(_logger_config["level"])

    # Add handlers if not already added
    if not logger.handlers:
        # Add console handler
        if _logger_config["console_handler"]:
            logger.addHandler(_logger_config["console_handler"])

        # Add file handler
        if _logger_config["file_handler"]:
            logger.addHandler(_logger_config["file_handler"])

    return logger


def set_log_level(level: int) -> None:
    """
    Set global log level.

    Args:
        level: New log level
    """
    global _logger_config

    _logger_config["level"] = level

    # Update existing handlers
    if _logger_config["console_handler"]:
        _logger_config["console_handler"].setLevel(level)
    if _logger_config["file_handler"]:
        _logger_config["file_handler"].setLevel(level)

    # Update root logger
    logging.getLogger().setLevel(level)


def add_file_handler(log_file: str, level: int | None = None) -> None:
    """
    Add a file handler to existing loggers.

    Args:
        log_file: Log file path
        level: Optional custom level for file handler
    """
    global _logger_config

    if _logger_config["formatter"] is None:
        setup_logging()

    # Create file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_logger_config["formatter"])

    if level is not None:
        file_handler.setLevel(level)
    else:
        file_handler.setLevel(_logger_config["level"])

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def remove_file_handler(log_file: str) -> None:
    """
    Remove a file handler from existing loggers.

    Args:
        log_file: Log file path to remove
    """
    root_logger = logging.getLogger()

    # Find and remove handlers for this file
    handlers_to_remove = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)


def get_log_config() -> dict[str, Any]:
    """
    Get current logging configuration.

    Returns:
        Dictionary with current configuration
    """
    return _logger_config.copy()


def log_function_call(
    logger: logging.Logger,
    func_name: str,
    args: tuple = (),
    kwargs: dict = None,
    level: int = logging.DEBUG,
) -> None:
    """
    Log function call details.

    Args:
        logger: Logger instance
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
        level: Log level
    """
    if kwargs is None:
        kwargs = {}

    args_str = ", ".join([str(arg) for arg in args])
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

    params = []
    if args_str:
        params.append(args_str)
    if kwargs_str:
        params.append(kwargs_str)

    params_str = ", ".join(params)
    logger.log(level, f"CALL: {func_name}({params_str})")


def log_function_result(
    logger: logging.Logger, func_name: str, result: Any, level: int = logging.DEBUG
) -> None:
    """
    Log function result.

    Args:
        logger: Logger instance
        func_name: Function name
        result: Function result
        level: Log level
    """
    logger.log(level, f"RESULT: {func_name} -> {result}")


def log_performance(
    logger: logging.Logger, operation: str, duration: float, level: int = logging.INFO
) -> None:
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        level: Log level
    """
    if duration < 1.0:
        duration_str = f"{duration * 1000:.1f}ms"
    elif duration < 60.0:
        duration_str = f"{duration:.2f}s"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        duration_str = f"{minutes}m {seconds:.1f}s"

    logger.log(level, f"PERFORMANCE: {operation} completed in {duration_str}")


def log_data_info(
    logger: logging.Logger,
    data_name: str,
    shape: tuple,
    memory_usage: int | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Log data information.

    Args:
        logger: Logger instance
        data_name: Data name/identifier
        shape: Data shape
        memory_usage: Memory usage in bytes (optional)
        level: Log level
    """
    shape_str = " x ".join(str(dim) for dim in shape)
    msg = f"DATA: {data_name} shape={shape_str}"

    if memory_usage is not None:
        if memory_usage < 1024:
            mem_str = f"{memory_usage}B"
        elif memory_usage < 1024**2:
            mem_str = f"{memory_usage / 1024:.1f}KB"
        elif memory_usage < 1024**3:
            mem_str = f"{memory_usage / 1024**2:.1f}MB"
        else:
            mem_str = f"{memory_usage / 1024**3:.1f}GB"
        msg += f", memory={mem_str}"

    logger.log(level, msg)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str = "",
    level: int = logging.ERROR,
) -> None:
    """
    Log error with context information.

    Args:
        logger: Logger instance
        error: Exception to log
        context: Context information
        level: Log level
    """
    error_msg = f"ERROR: {type(error).__name__}: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"

    logger.log(level, error_msg, exc_info=True)


# Initialize default logging
setup_logging()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing logging utilities...")

    # Get logger
    logger = get_logger(__name__)
    logger.info("Logger setup successful")

    # Test different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test function logging
    log_function_call(logger, "test_function", (1, 2), {"param": "value"})
    log_function_result(logger, "test_function", "result_value")

    # Test performance logging
    log_performance(logger, "data_processing", 1.5)

    # Test data info logging
    log_data_info(logger, "price_data", (1000, 5), 40000)

    print("All logging tests completed!")
