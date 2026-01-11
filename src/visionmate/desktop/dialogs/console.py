"""
Log Console Dialog for Visionmate desktop application.

This module provides a dialog window for viewing application logs
with filtering, clearing, and saving capabilities.

"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.logging import LogConsoleHandler

logger = logging.getLogger(__name__)


class LogConsoleDialog(QDialog):
    """Dialog for viewing and managing application logs.

    Provides a table-based log viewer with:
    - Columns: Timestamp, Level, Thread, Source, Message
    - Log level filtering
    - Clear button to clear displayed logs
    - Save to file button to export logs

    """

    # Column indices
    COL_TIMESTAMP = 0
    COL_LEVEL = 1
    COL_THREAD = 2
    COL_SOURCE = 3
    COL_MESSAGE = 4

    def __init__(
        self,
        log_handler: LogConsoleHandler,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the log console dialog.

        Args:
            log_handler: LogConsoleHandler instance to receive log messages from
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing LogConsoleDialog")

        self._log_handler = log_handler
        self._current_filter_level = logging.INFO  # Default to INFO level

        # Setup UI
        self.setWindowTitle("Log Console")
        self.setMinimumSize(800, 600)

        self._setup_ui()
        self._connect_signals()

        # Load buffered logs from handler
        self._load_buffered_logs()

        logger.debug("LogConsoleDialog initialized")

    def _load_buffered_logs(self) -> None:
        """Load and display buffered logs from the handler.

        This displays all logs that were captured before the dialog was opened,
        including logs from application startup.
        """
        buffered_logs = self._log_handler.get_buffered_logs()
        logger.debug(f"Loading {len(buffered_logs)} buffered log messages")

        for level, _message, record in buffered_logs:
            # Apply the same filtering logic as _on_log_message
            if record.levelno >= self._current_filter_level:
                self._add_log_row(level, record)

        # Auto-scroll to bottom
        if buffered_logs:
            self._log_table.scrollToBottom()

    def _setup_ui(self) -> None:
        """Setup the dialog UI components."""
        # Main layout
        layout = QVBoxLayout(self)

        # Top toolbar with filter and buttons
        toolbar_layout = QHBoxLayout()

        # Log level filter
        filter_label = QLabel("Log Level:")
        toolbar_layout.addWidget(filter_label)

        self._level_combo = QComboBox()
        self._level_combo.addItem("DEBUG", logging.DEBUG)
        self._level_combo.addItem("INFO", logging.INFO)
        self._level_combo.addItem("WARNING", logging.WARNING)
        self._level_combo.addItem("ERROR", logging.ERROR)
        self._level_combo.addItem("CRITICAL", logging.CRITICAL)
        self._level_combo.setCurrentIndex(1)  # Default to INFO
        self._level_combo.currentIndexChanged.connect(self._on_filter_changed)
        toolbar_layout.addWidget(self._level_combo)

        toolbar_layout.addStretch()

        # Clear button
        self._clear_button = QPushButton("Clear")
        self._clear_button.setToolTip("Clear all displayed logs")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        toolbar_layout.addWidget(self._clear_button)

        # Save to file button
        self._save_button = QPushButton("Save to File...")
        self._save_button.setToolTip("Save logs to a file")
        self._save_button.clicked.connect(self._on_save_clicked)
        toolbar_layout.addWidget(self._save_button)

        layout.addLayout(toolbar_layout)

        # Log table display (scrollable)
        self._log_table = QTableWidget()
        self._log_table.setColumnCount(5)
        self._log_table.setHorizontalHeaderLabels(
            ["Timestamp", "Level", "Thread", "Source", "Message"]
        )

        # Configure table behavior
        self._log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._log_table.setAlternatingRowColors(True)
        self._log_table.verticalHeader().setVisible(False)

        # Configure column widths
        header = self._log_table.horizontalHeader()
        header.setSectionResizeMode(self.COL_TIMESTAMP, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_LEVEL, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_THREAD, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SOURCE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_MESSAGE, QHeaderView.ResizeMode.Stretch)

        # Apply dark theme styling
        self._log_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                gridline-color: #3e3e3e;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #264f78;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #d4d4d4;
                padding: 6px;
                border: 1px solid #3e3e3e;
                font-weight: bold;
            }
            """
        )
        layout.addWidget(self._log_table)

        # Close button at bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        logger.debug("Log console UI setup complete")

    def _connect_signals(self) -> None:
        """Connect log handler signals to dialog slots."""
        # Connect to log handler's signal
        self._log_handler.log_message.connect(self._on_log_message)
        logger.debug("Connected to log handler signals")

    @Slot(str, str, object)
    def _on_log_message(self, level: str, message: str, record: logging.LogRecord) -> None:
        """Handle incoming log message from handler.

        Args:
            level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Formatted log message
            record: Original log record

        """
        # Filter by log level
        if record.levelno < self._current_filter_level:
            return

        # Add row to table
        self._add_log_row(level, record)

        # Auto-scroll to bottom
        self._log_table.scrollToBottom()

    def _add_log_row(self, level: str, record: logging.LogRecord) -> None:
        """Add a log record as a row in the table.

        Args:
            level: Log level name
            record: Log record to add
        """
        # Insert new row at the end
        row = self._log_table.rowCount()
        self._log_table.insertRow(row)

        # Format timestamp
        import datetime

        timestamp = datetime.datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]  # Remove last 3 digits to show milliseconds

        # Get thread name (handle None case)
        thread_name = record.threadName if record.threadName else "Unknown"

        # Create items for each column
        timestamp_item = QTableWidgetItem(timestamp)
        level_item = QTableWidgetItem(level)
        thread_item = QTableWidgetItem(thread_name)
        source_item = QTableWidgetItem(f"{record.filename}:{record.lineno}")
        message_item = QTableWidgetItem(record.getMessage())

        # Apply color to level column based on log level
        level_color = self._get_level_qcolor(level)
        level_item.setForeground(level_color)

        # Set items in table
        self._log_table.setItem(row, self.COL_TIMESTAMP, timestamp_item)
        self._log_table.setItem(row, self.COL_LEVEL, level_item)
        self._log_table.setItem(row, self.COL_THREAD, thread_item)
        self._log_table.setItem(row, self.COL_SOURCE, source_item)
        self._log_table.setItem(row, self.COL_MESSAGE, message_item)

    def _get_level_color(self, level: str) -> str:
        """Get HTML color for log level.

        Args:
            level: Log level name

        Returns:
            HTML color code
        """
        colors = {
            "DEBUG": "#808080",  # Gray
            "INFO": "#4ec9b0",  # Cyan
            "WARNING": "#dcdcaa",  # Yellow
            "ERROR": "#f48771",  # Orange
            "CRITICAL": "#f44747",  # Red
        }
        return colors.get(level, "#d4d4d4")  # Default white

    def _get_level_qcolor(self, level: str) -> QColor:
        """Get QColor for log level.

        Args:
            level: Log level name

        Returns:
            QColor instance
        """
        color_str = self._get_level_color(level)
        return QColor(color_str)

    def _on_filter_changed(self, index: int) -> None:
        """Handle log level filter change.

        Args:
            index: Selected combo box index

        """
        self._current_filter_level = self._level_combo.itemData(index)
        logger.debug(f"Log filter changed to: {logging.getLevelName(self._current_filter_level)}")

        # Note: This only affects NEW log messages
        # Existing messages in the display are not filtered retroactively

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self._log_table.setRowCount(0)
        logger.debug("Log console cleared")

    def _on_save_clicked(self) -> None:
        """Handle save to file button click."""
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            str(Path.home() / "visionmate_logs.txt"),
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)",
        )

        if not file_path:
            logger.debug("Save cancelled by user")
            return

        try:
            # Build log content from table
            lines = []
            for row in range(self._log_table.rowCount()):
                timestamp_item = self._log_table.item(row, self.COL_TIMESTAMP)
                level_item = self._log_table.item(row, self.COL_LEVEL)
                thread_item = self._log_table.item(row, self.COL_THREAD)
                source_item = self._log_table.item(row, self.COL_SOURCE)
                message_item = self._log_table.item(row, self.COL_MESSAGE)

                # Skip if any item is None (shouldn't happen, but be safe)
                if not all([timestamp_item, level_item, thread_item, source_item, message_item]):
                    continue

                # Type assertions for mypy/pyright
                assert timestamp_item is not None
                assert level_item is not None
                assert thread_item is not None
                assert source_item is not None
                assert message_item is not None

                timestamp = timestamp_item.text()
                level = level_item.text()
                thread = thread_item.text()
                source = source_item.text()
                message = message_item.text()

                # Format as tab-separated values
                line = f"{timestamp}\t{level}\t{thread}\t{source}\t{message}"
                lines.append(line)

            log_content = "\n".join(lines)

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(log_content)

            logger.info(f"Logs saved to: {file_path}")

            # Show success message in status (if parent has status bar)
            parent = self.parent()
            if parent and hasattr(parent, "statusBar"):
                # Type assertion for MainWindow

                from PySide6.QtWidgets import QMainWindow

                if isinstance(parent, QMainWindow):
                    parent.statusBar().showMessage(f"Logs saved to {file_path}", 3000)

        except Exception as e:
            logger.error(f"Failed to save logs: {e}", exc_info=True)

            # Show error message
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save logs to file:\n{e}",
            )
