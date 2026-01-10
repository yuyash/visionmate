"""
Unit tests for LogConsoleDialog.

Tests the log console dialog functionality including:
- Log message display
- Log level filtering
- Clear functionality
- Save to file functionality
"""

import logging
import tempfile
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from visionmate.core.logging import LogConsoleHandler
from visionmate.desktop.dialogs import LogConsoleDialog


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def log_handler():
    """Create a LogConsoleHandler for testing."""
    handler = LogConsoleHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    return handler


@pytest.fixture
def dialog(qapp, log_handler):
    """Create a LogConsoleDialog for testing."""
    dialog = LogConsoleDialog(log_handler)
    yield dialog
    dialog.close()


def test_dialog_initialization(dialog):
    """Test that dialog initializes correctly.

    Requirements: 17.1
    """
    assert dialog.windowTitle() == "Log Console"
    assert dialog._log_table is not None
    assert dialog._level_combo is not None
    assert dialog._clear_button is not None
    assert dialog._save_button is not None
    # Check default log level is INFO
    assert dialog._current_filter_level == logging.INFO
    assert dialog._level_combo.currentText() == "INFO"


def test_buffered_logs_display(qapp, log_handler):
    """Test that buffered logs are displayed when dialog opens.

    This verifies that logs from before the dialog was opened are shown.
    """
    # Create a test logger and emit some messages BEFORE creating dialog
    logger = logging.getLogger("test_buffer_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # Emit messages before dialog exists
    logger.info("Message 1 before dialog")
    logger.warning("Message 2 before dialog")
    logger.error("Message 3 before dialog")

    QApplication.processEvents()

    # Now create the dialog
    dialog = LogConsoleDialog(log_handler)

    # Check that buffered messages are displayed in table
    assert dialog._log_table.rowCount() == 3

    # Check message content
    messages = []
    for row in range(dialog._log_table.rowCount()):
        message_item = dialog._log_table.item(row, dialog.COL_MESSAGE)
        if message_item:
            messages.append(message_item.text())

    assert "Message 1 before dialog" in messages
    assert "Message 2 before dialog" in messages
    assert "Message 3 before dialog" in messages

    dialog.close()


def test_log_message_display(dialog, log_handler):
    """Test that log messages are displayed in the dialog.

    Requirements: 17.3
    """
    # Create a test log record
    logger = logging.getLogger("test_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # Get initial row count
    initial_rows = dialog._log_table.rowCount()

    # Emit a log message
    test_message = "Test log message"
    logger.info(test_message)

    # Process Qt events to allow signal to be delivered
    QApplication.processEvents()

    # Check that a new row was added
    assert dialog._log_table.rowCount() == initial_rows + 1

    # Check that message appears in the last row
    last_row = dialog._log_table.rowCount() - 1
    message_item = dialog._log_table.item(last_row, dialog.COL_MESSAGE)
    assert test_message in message_item.text()

    # Check that level is INFO
    level_item = dialog._log_table.item(last_row, dialog.COL_LEVEL)
    assert level_item.text() == "INFO"


def test_log_level_filtering(dialog, log_handler):
    """Test that log level filtering works correctly.

    Requirements: 17.5
    """
    # Create a test logger
    logger = logging.getLogger("test_filter_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # Set filter to WARNING level
    dialog._level_combo.setCurrentText("WARNING")
    QApplication.processEvents()

    # Clear any existing messages
    dialog._log_table.setRowCount(0)

    # Emit messages at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Process Qt events
    QApplication.processEvents()

    # Check that only WARNING and above are displayed
    assert dialog._log_table.rowCount() == 2  # WARNING and ERROR

    # Check messages
    messages = []
    for row in range(dialog._log_table.rowCount()):
        message_item = dialog._log_table.item(row, dialog.COL_MESSAGE)
        messages.append(message_item.text())

    assert "Debug message" not in messages
    assert "Info message" not in messages
    assert "Warning message" in messages
    assert "Error message" in messages


def test_clear_button(dialog, log_handler):
    """Test that clear button clears the log display.

    Requirements: 17.4
    """
    # Add some log messages
    logger = logging.getLogger("test_clear_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # Emit log message
    logger.info("Message before clear")

    QApplication.processEvents()

    # Verify message is displayed
    assert dialog._log_table.rowCount() > 0

    # Click clear button
    dialog._clear_button.click()
    QApplication.processEvents()

    # Verify table is cleared
    assert dialog._log_table.rowCount() == 0


def test_save_to_file(dialog, log_handler, monkeypatch):
    """Test that save to file functionality works.

    Requirements: 17.4
    """
    # Add some log messages
    logger = logging.getLogger("test_save_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
    logger.info("Message to save")

    QApplication.processEvents()

    # Verify message is in the dialog
    assert dialog._log_table.rowCount() > 0

    # Create a temporary file for saving
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Mock the file dialog to return our temp file path
        from PySide6.QtWidgets import QFileDialog

        def mock_get_save_filename(*args, **kwargs):
            return tmp_path, "Text Files (*.txt)"

        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_filename)

        # Click save button
        dialog._save_button.click()
        QApplication.processEvents()

        # Verify file was created and contains the log message
        saved_content = Path(tmp_path).read_text()
        assert "Message to save" in saved_content

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def test_log_level_colors(dialog):
    """Test that different log levels have different colors.

    Requirements: 17.1
    """
    # Test color mapping
    assert dialog._get_level_color("DEBUG") == "#808080"
    assert dialog._get_level_color("INFO") == "#4ec9b0"
    assert dialog._get_level_color("WARNING") == "#dcdcaa"
    assert dialog._get_level_color("ERROR") == "#f48771"
    assert dialog._get_level_color("CRITICAL") == "#f44747"
    assert dialog._get_level_color("UNKNOWN") == "#d4d4d4"  # Default


def test_dialog_is_scrollable(dialog):
    """Test that the log table is scrollable.

    Requirements: 17.4
    """
    # Check that table has scroll bars
    assert dialog._log_table.verticalScrollBar() is not None
    assert dialog._log_table.horizontalScrollBar() is not None


def test_multiple_log_messages(dialog, log_handler):
    """Test handling multiple log messages in sequence.

    Requirements: 17.3
    """
    # Create a test logger
    logger = logging.getLogger("test_multiple_logger")
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    # Set filter to DEBUG to see all messages
    dialog._level_combo.setCurrentText("DEBUG")
    QApplication.processEvents()

    # Clear any existing messages
    dialog._log_table.setRowCount(0)

    # Emit multiple messages
    messages = [
        ("DEBUG", "Debug message 1"),
        ("INFO", "Info message 1"),
        ("WARNING", "Warning message 1"),
        ("ERROR", "Error message 1"),
    ]

    for level, message in messages:
        getattr(logger, level.lower())(message)

    # Process Qt events
    QApplication.processEvents()

    # Check that all messages appear
    assert dialog._log_table.rowCount() == len(messages)

    # Check each message
    table_messages = []
    for row in range(dialog._log_table.rowCount()):
        message_item = dialog._log_table.item(row, dialog.COL_MESSAGE)
        if message_item:
            table_messages.append(message_item.text())

    for _, message in messages:
        assert message in table_messages


def test_auto_scroll_to_bottom(dialog, log_handler):
    """Test that new messages auto-scroll to bottom.

    Requirements: 17.4
    """
    # Create a test logger
    logger = logging.getLogger("test_scroll_logger")
    logger.addHandler(log_handler)

    # Clear existing messages
    dialog._log_table.setRowCount(0)

    # Add many messages to trigger scrolling
    for i in range(100):
        logger.info(f"Message {i}")

    QApplication.processEvents()

    # Check that scrollbar is at the bottom
    scrollbar = dialog._log_table.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()
