"""Logging configuration and custom handlers for Visionmate application.

This module provides custom logging handlers and configuration utilities
for the Visionmate application, including a handler that emits signals
for UI integration.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


class LogSignalEmitter(QObject):
    """Qt signal emitter for log messages.

    This class is used by LogConsoleHandler to emit Qt signals
    without causing method name conflicts with QObject.emit().

    Signals:
        log_message: Emitted when a log record is processed
            Args:
                level (str): Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                message (str): Formatted log message
                record (logging.LogRecord): Original log record
    """

    # Signal emitted when a log message is received
    log_message = Signal(str, str, object)  # level, message, record


class MillisecondFormatter(logging.Formatter):
    """Custom formatter that includes milliseconds in timestamps.

    This formatter extends logging.Formatter to include milliseconds
    in the timestamp, even when a custom datefmt is specified.
    """

    def formatTime(self, record, datefmt=None):
        """Format the time with milliseconds.

        Args:
            record: LogRecord instance
            datefmt: Date format string (ignored, uses internal format)

        Returns:
            Formatted timestamp string with milliseconds
        """
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        # Add milliseconds
        s = f"{s}.{int(record.msecs):03d}"
        return s


class LogConsoleHandler(logging.Handler):
    """Custom logging handler that emits Qt signals for UI integration.

    This handler extends logging.Handler and uses a separate QObject
    for signal emission to avoid method name conflicts. It supports
    log level filtering and emits formatted log records as signals.

    Requirements: 17.2, 17.3, 17.5
    """

    def __init__(self, level: int = logging.NOTSET):
        """Initialize the handler.

        Args:
            level: Minimum log level to handle (default: NOTSET, handles all levels)
        """
        super().__init__(level=level)
        self._signal_emitter = LogSignalEmitter()

    @property
    def log_message(self):
        """Get the log_message signal for connecting to UI components.

        Returns:
            Signal that emits (level: str, message: str, record: LogRecord)
        """
        return self._signal_emitter.log_message

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record and emit a signal.

        This method is called by the logging system when a log record
        needs to be handled. It formats the record and emits a signal
        that can be connected to UI components.

        Args:
            record: The log record to process

        Requirements: 17.2, 17.3
        """
        try:
            # Format the log record
            message = self.format(record)

            # Emit signal with level name, formatted message, and original record
            self._signal_emitter.log_message.emit(record.levelname, message, record)

        except Exception:
            # Handle errors in emit to prevent logging system from breaking
            self.handleError(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_handler: Optional[LogConsoleHandler] = None,
) -> None:
    """Configure application-wide logging.

    Sets up logging with file, console, and optional UI handlers.
    Configures log format with timestamps and appropriate log levels.

    Log format: YYYY-MM-DD HH:MM:SS.mmm [ThreadName     ] LEVEL  filename.py:line - message
    - Timestamp: Date and time with millisecond precision (for performance analysis)
    - Thread Name: Name of the thread (fixed-width 15 chars, left-aligned)
    - Level: Log level (fixed-width 5 chars: DEBUG, INFO, WARN, ERROR, FATAL)
    - Source: Filename and line number where the log was generated
    - Message: The actual log message

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (default: visionmate.log in current directory)
        console_handler: Optional LogConsoleHandler for UI integration

    Requirements: 25.1, 25.2, 25.3, 25.4, 25.5, 25.6
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create custom formatter with specified format
    # Format: YYYY-MM-DD HH:MM:SS.mmm [ThreadName     ] LEVEL  filename.py:line - message
    detailed_formatter = MillisecondFormatter(
        fmt="%(asctime)s [%(threadName)-15s] %(levelname)-5s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console formatter - same format for consistency
    console_formatter = MillisecondFormatter(
        fmt="%(asctime)s [%(threadName)-15s] %(levelname)-5s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create handlers list
    handlers: list[logging.Handler] = []

    # File handler - detailed logging
    if log_file is None:
        log_file = Path("visionmate.log")

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    except (OSError, PermissionError) as e:
        # If we can't create the log file, print a warning but continue
        print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)

    # Console handler - less verbose
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(console_formatter)
    handlers.append(stream_handler)

    # UI console handler - if provided
    if console_handler is not None:
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(detailed_formatter)
        handlers.append(console_handler)

    # Configure root logger
    # Clear existing handlers and configure manually for better control
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)  # Root logger captures everything

    # Add all handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log initial message (after handlers are configured)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    This is a convenience function that wraps logging.getLogger
    to provide consistent logger creation across the application.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)
