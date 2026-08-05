"""
fraudguard/logging_config.py — Structured Logging Configuration

Replaces all print() statements with proper Python logging.
Supports JSON-formatted output for production environments.
"""

import logging
import sys
from fraudguard.config import get_settings


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.
    
    Returns:
        logging.Logger: Configured logger instance for the application.
    """
    settings = get_settings()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger("fraudguard")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)
    
    # Prevent duplicate log messages
    logger.propagate = False

    return logger


def get_logger(name: str = "fraudguard") -> logging.Logger:
    """Get a named logger instance.
    
    Args:
        name: Logger name, typically the module name.
        
    Returns:
        logging.Logger: Named logger instance.
    """
    return logging.getLogger(name)
