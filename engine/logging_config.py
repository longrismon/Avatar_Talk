"""
Structured logging configuration for Avatar Agent engine.

Uses structlog with JSON output. All log entries include:
  - timestamp_ms: Unix epoch milliseconds
  - level: log level string
  - module: the logger's bound module name

Usage:
    from engine.logging_config import setup_logging, get_logger

    setup_logging(level="INFO", log_file="./logs/agent.log")
    log = get_logger("orchestrator")
    log.info("state_transition", from_state="IDLE", to_state="PLANNING")
"""

import logging
import logging.handlers
import os
import sys
import time
from typing import Optional

import structlog


def _add_timestamp_ms(logger, method_name, event_dict):
    """structlog processor: add Unix timestamp in milliseconds."""
    event_dict["timestamp_ms"] = int(time.time() * 1000)
    return event_dict


def _reorder_keys(logger, method_name, event_dict):
    """structlog processor: put standard fields first for readability."""
    ordered = {}
    for key in ("timestamp_ms", "level", "module", "event"):
        if key in event_dict:
            ordered[key] = event_dict.pop(key)
    ordered.update(event_dict)
    return ordered


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 50,
    backup_count: int = 5,
) -> None:
    """Configure structlog and the standard library logging handlers.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. If None, logs to stdout only.
        max_size_mb: Max log file size in MB before rotation.
        backup_count: Number of rotated backup files to keep.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Set up stdlib handlers
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(message)s",  # structlog renders the full JSON line
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            _add_timestamp_ms,
            _reorder_keys,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(module: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given module name.

    Args:
        module: Module identifier, e.g. "orchestrator", "stt", "browser".

    Returns:
        A bound logger with module pre-set on every log call.

    Example:
        log = get_logger("browser")
        log.info("action_complete", action="search_contact", duration_ms=320)
        # → {"timestamp_ms": 1718000000123, "level": "info", "module": "browser",
        #     "event": "action_complete", "action": "search_contact", "duration_ms": 320}
    """
    return structlog.get_logger().bind(module=module)
