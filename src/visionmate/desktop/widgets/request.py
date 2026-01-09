"""Request input widget for text instructions.

This module provides the RequestWidget that allows users to input
additional text instructions alongside video/audio inputs.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class RequestWidget(QWidget):
    """Widget for inputting text instructions.

    Provides a text input area for users to add additional instructions
    that will be included with video/audio inputs.
    """

    # Signal emitted when request is submitted
    request_submitted = Signal(str)  # request_text

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the RequestWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing RequestWidget")

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create group box
        group_box = QGroupBox("Text Instructions")
        group_box.setFlat(True)
        group_layout = QVBoxLayout(group_box)

        # Label
        label = QLabel("Additional instructions (optional):")
        group_layout.addWidget(label)

        # Text input area
        self._text_input = QTextEdit()
        self._text_input.setPlaceholderText("Enter additional instructions here...")
        self._text_input.setMaximumHeight(100)
        self._text_input.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            """
        )
        group_layout.addWidget(self._text_input)

        # Submit button
        self._submit_button = QPushButton("Submit Instructions")
        self._submit_button.setToolTip("Submit text instructions")
        self._submit_button.clicked.connect(self._on_submit_clicked)
        group_layout.addWidget(self._submit_button)

        # Clear button
        self._clear_button = QPushButton("Clear")
        self._clear_button.setToolTip("Clear text input")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        group_layout.addWidget(self._clear_button)

        # Add group box to main layout
        layout.addWidget(group_box)

        logger.debug("RequestWidget UI setup complete")

    def _on_submit_clicked(self) -> None:
        """Handle submit button click."""
        text = self._text_input.toPlainText().strip()
        if text:
            logger.info(f"Request submitted: {text[:50]}...")
            self.request_submitted.emit(text)
        else:
            logger.debug("Empty request - not submitting")

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self._text_input.clear()
        logger.debug("Request text cleared")

    def get_text(self) -> str:
        """Get the current text input.

        Returns:
            Current text input
        """
        return self._text_input.toPlainText().strip()

    def set_text(self, text: str) -> None:
        """Set the text input.

        Args:
            text: Text to set
        """
        self._text_input.setPlainText(text)

    def clear(self) -> None:
        """Clear the text input."""
        self._text_input.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the widget.

        Args:
            enabled: Whether to enable the widget
        """
        self._text_input.setEnabled(enabled)
        self._submit_button.setEnabled(enabled)
        self._clear_button.setEnabled(enabled)
