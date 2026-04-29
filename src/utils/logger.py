"""
Logging configuration for Agentic RAG System.

This module provides centralized logging setup with consistent formatting
across all components. Supports multiple log levels and handlers.

Features:
- Consistent timestamp format
- Component-level logging (agent.planner, retrieval.vector, etc)
- Configurable log levels
- Console and file output support
- Colored output for better readability (optional)

Usage:
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger("agent.planner")
    >>> logger.info("Planning started")
    >>> logger.error("Planning failed", exc_info=True)
"""

import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure a logger with consistent formatting.
    
    Creates a logger with the specified name and configuration.
    Supports both console and file output with customizable formatting.
    
    Args:
        name: Logger name (e.g., 'agent.planner', 'retrieval.vector')
              Use hierarchical names for better organization
        level: Log level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               Default is 'INFO'
        log_file: Optional path to log file. If provided, logs will be written to file
        format_string: Optional custom log format. Uses DEFAULT_FORMAT if not provided
        date_format: Optional custom date format. Uses DEFAULT_DATE_FORMAT if not provided
    
    Returns:
        Configured logger instance
    
    Raises:
        ValueError: If invalid log level is provided
    
    Example:
        >>> logger = setup_logger("agent.planner", level="DEBUG")
        >>> logger.debug("This is a debug message")
        >>> logger.info("This is an info message")
        
        >>> # With file output
        >>> logger = setup_logger("retrieval", level="INFO", log_file="logs/retrieval.log")
    """
    # Validate log level
    if level.upper() not in LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Must be one of: {', '.join(LOG_LEVELS.keys())}"
        )
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[level.upper()])
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Use default formats if not provided
    fmt = format_string or DEFAULT_FORMAT
    date_fmt = date_format or DEFAULT_DATE_FORMAT
    
    # Create formatter
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVELS[level.upper()])
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVELS[level.upper()])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger
