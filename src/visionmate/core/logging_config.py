"""Logging configuration for VisionMate application."""

import logging
import sys
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter with fixed-width fields for better readability."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured layout.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Format timestamp with milliseconds
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        msecs = f"{int(record.msecs):03d}"
        timestamp_with_ms = f"{timestamp}.{msecs}"

        # Format thread name with fixed width (15 chars, left-aligned)
        thread_name = f"[{record.threadName:<15s}]"

        # Format level name with fixed width (5 chars, left-aligned)
        level_name = f"{record.levelname:<5s}"

        # Format source location
        source = f"{record.filename}:{record.lineno}"

        # Format message
        message = record.getMessage()

        # Handle exceptions
        if record.exc_info:
            if not message.endswith("\n"):
                message += "\n"
            message += self.formatException(record.exc_info)

        return f"{timestamp_with_ms} {thread_name} {level_name} {source} - {message}"


class LoggingConfig:
    """Configure logging for the application."""

    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def setup_logging(
        level: str = "INFO",
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        log_to_console: bool = True,
    ) -> None:
        """Set up logging configuration.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to a file
            log_file_path: Path to log file (default: logs/visionmate.log)
            log_to_console: Whether to log to console (stdout)
        """
        # Convert level string to logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create custom formatter
        formatter = StructuredFormatter(datefmt=LoggingConfig.DEFAULT_DATE_FORMAT)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if requested
        if log_to_file:
            if log_file_path is None:
                # Default log file path
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                log_file_path = str(log_dir / "visionmate.log")

            # Ensure log directory exists
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            root_logger.info(f"Logging to file: {log_file_path}")

        root_logger.info(f"Logging initialized at level: {level}")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        return logging.getLogger(name)
