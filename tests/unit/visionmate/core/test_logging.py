"""Unit tests for logging module."""

import logging
import tempfile
from pathlib import Path

from visionmate.core.logging import LogConsoleHandler, get_logger, setup_logging


class TestLogConsoleHandler:
    """Test LogConsoleHandler class."""

    def test_handler_creation(self):
        """Test that handler can be created."""
        handler = LogConsoleHandler()
        assert handler is not None
        assert hasattr(handler, "log_message")

    def test_handler_with_level(self):
        """Test handler creation with specific log level."""
        handler = LogConsoleHandler(level=logging.WARNING)
        assert handler.level == logging.WARNING

    def test_emit_signal(self, qtbot):
        """Test that handler emits signals when processing log records."""
        handler = LogConsoleHandler()

        # Track emitted signals
        emitted_signals = []

        def capture_signal(level, message, record):
            emitted_signals.append((level, message, record))

        handler.log_message.connect(capture_signal)

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Emit the record
        handler.emit(record)

        # Verify signal was emitted
        assert len(emitted_signals) == 1
        level, message, emitted_record = emitted_signals[0]
        assert level == "INFO"
        assert "Test message" in message
        assert emitted_record is record

    def test_emit_with_formatting(self, qtbot):
        """Test that handler formats log records correctly."""
        handler = LogConsoleHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(threadName)-15s] %(levelname)-5s %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        emitted_signals = []

        def capture_signal(level, message, record):
            emitted_signals.append((level, message, record))

        handler.log_message.connect(capture_signal)

        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        assert len(emitted_signals) == 1
        level, message, _ = emitted_signals[0]
        assert level == "ERROR"
        # Verify format includes thread name, level, filename:line
        assert "[MainThread" in message or "[" in message  # Thread name
        assert "ERROR" in message
        assert "test.py:42" in message
        assert "Error occurred" in message


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test logging setup with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_file)

            # Verify log file was created
            assert log_file.exists()

            # Test logging
            logger = get_logger(__name__)
            logger.info("Test message")

            # Verify message was written to file with correct format
            log_content = log_file.read_text()
            assert "Test message" in log_content
            assert "INFO" in log_content
            # Verify format: YYYY-MM-DD HH:MM:SS.mmm [ThreadName] LEVEL filename.py:line - message
            assert "[MainThread" in log_content or "[" in log_content  # Thread name
            assert "test_logging.py:" in log_content  # filename:line
            assert " - Test message" in log_content  # message separator

    def test_setup_logging_with_level(self):
        """Test logging setup with specific log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_level="DEBUG", log_file=log_file)

            logger = get_logger(__name__)
            logger.debug("Debug message")

            log_content = log_file.read_text()
            assert "Debug message" in log_content
            assert "DEBUG" in log_content

    def test_setup_logging_with_console_handler(self, qtbot):
        """Test logging setup with UI console handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            console_handler = LogConsoleHandler()

            emitted_signals = []

            def capture_signal(level, message, record):
                emitted_signals.append((level, message, record))

            console_handler.log_message.connect(capture_signal)

            setup_logging(log_file=log_file, console_handler=console_handler)

            logger = get_logger(__name__)
            logger.info("UI test message")

            # Verify signal was emitted
            assert len(emitted_signals) > 0
            # Find our message
            found = False
            for level, message, _ in emitted_signals:
                if "UI test message" in message:
                    found = True
                    assert level == "INFO"
                    break
            assert found, "Expected log message not found in emitted signals"

    def test_setup_logging_invalid_file(self):
        """Test logging setup handles invalid file path gracefully."""
        # Use an invalid path (directory that doesn't exist)
        invalid_path = Path("/nonexistent/directory/test.log")

        # Should not raise an exception
        setup_logging(log_file=invalid_path)

        # Logging should still work (to console)
        logger = get_logger(__name__)
        logger.info("Test message")  # Should not raise


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2
