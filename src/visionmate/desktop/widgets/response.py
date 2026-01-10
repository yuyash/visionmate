"""Response display widget for VLM responses.

This module provides the ResponseWidget that displays the current question,
current response, and response history with timestamps.
"""

import logging
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.recognition import VLMResponse

logger = logging.getLogger(__name__)


class ResponseWidget(QWidget):
    """Widget for displaying VLM responses.

    Displays:
    - Current question
    - Current response (direct answer, follow-up questions, supplementary info)
    - Response history with timestamps (scrollable)

    Requirements: 14.1, 14.2, 14.3, 14.4, 10.8
    """

    # Signals
    status_message = Signal(str, int)  # message, timeout

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the ResponseWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing ResponseWidget")

        self._current_question: Optional[str] = None
        self._current_response: Optional[VLMResponse] = None
        self._response_history: list[tuple[datetime, VLMResponse]] = []

        self._setup_ui()
        logger.debug("ResponseWidget initialized")

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("VLM Responses")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        # Current question section
        question_label = QLabel("Current Question:")
        question_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(question_label)

        self._question_text = QLabel("No question detected yet")
        self._question_text.setWordWrap(True)
        self._question_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 4px;
                color: #666;
            }
            """
        )
        layout.addWidget(self._question_text)

        # Current response section
        response_label = QLabel("Current Response:")
        response_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(response_label)

        self._response_text = QLabel("Waiting for response...")
        self._response_text.setWordWrap(True)
        self._response_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._response_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #e8f4f8;
                border-radius: 4px;
                color: #333;
            }
            """
        )
        layout.addWidget(self._response_text)

        # Response history section
        history_label = QLabel("Response History:")
        history_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(history_label)

        # Scrollable history area
        self._history_scroll = QScrollArea()
        self._history_scroll.setWidgetResizable(True)
        self._history_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        self._history_scroll.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            """
        )

        # History container widget
        self._history_container = QWidget()
        self._history_layout = QVBoxLayout(self._history_container)
        self._history_layout.setContentsMargins(8, 8, 8, 8)
        self._history_layout.setSpacing(8)
        self._history_layout.addStretch()

        self._history_scroll.setWidget(self._history_container)
        layout.addWidget(self._history_scroll, stretch=1)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_current_question(self, question: str) -> None:
        """Set the current question.

        Args:
            question: Question text

        Requirements: 14.1, 8.4
        """
        self._current_question = question
        self._question_text.setText(question)
        self._question_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #fff3cd;
                border-radius: 4px;
                color: #333;
                font-weight: bold;
            }
            """
        )
        logger.debug(f"Current question set: {question[:50]}...")

    def set_current_response(self, response: VLMResponse) -> None:
        """Set the current response.

        Args:
            response: VLM response object

        Requirements: 14.1, 14.2, 21.4
        """
        self._current_response = response

        # Format response text
        response_parts = []

        if response.direct_answer:
            response_parts.append(f"<b>Answer:</b> {response.direct_answer}")

        if response.follow_up_questions:
            response_parts.append("<b>Follow-up Questions:</b>")
            for i, question in enumerate(response.follow_up_questions, 1):
                response_parts.append(f"{i}. {question}")

        if response.supplementary_info:
            response_parts.append(f"<b>Additional Info:</b> {response.supplementary_info}")

        if response.is_partial:
            response_parts.append("<i>(Partial response - more coming...)</i>")

        response_text = "<br><br>".join(response_parts) if response_parts else "No response content"

        self._response_text.setText(response_text)
        self._response_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #d4edda;
                border-radius: 4px;
                color: #333;
            }
            """
        )

        # Add to history if not partial
        if not response.is_partial:
            self._add_to_history(response)

        logger.debug(f"Current response set (partial={response.is_partial})")

    def _add_to_history(self, response: VLMResponse) -> None:
        """Add response to history.

        Args:
            response: VLM response object

        Requirements: 14.3, 14.4
        """
        # Add to history list
        timestamp = datetime.fromtimestamp(response.timestamp)
        self._response_history.append((timestamp, response))

        # Create history entry widget
        entry_widget = self._create_history_entry(timestamp, response)

        # Insert at the top of history (most recent first)
        self._history_layout.insertWidget(0, entry_widget)

        # Limit history to 50 entries
        if len(self._response_history) > 50:
            self._response_history.pop(0)
            # Remove oldest widget (last in layout before stretch)
            item = self._history_layout.takeAt(self._history_layout.count() - 2)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        logger.debug(f"Added response to history (total: {len(self._response_history)})")

    def _create_history_entry(self, timestamp: datetime, response: VLMResponse) -> QWidget:
        """Create a history entry widget.

        Args:
            timestamp: Response timestamp
            response: VLM response object

        Returns:
            History entry widget

        Requirements: 14.3, 14.4
        """
        entry = QFrame()
        entry.setFrameShape(QFrame.Shape.StyledPanel)
        entry.setStyleSheet(
            """
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
            """
        )

        entry_layout = QVBoxLayout(entry)
        entry_layout.setContentsMargins(8, 8, 8, 8)
        entry_layout.setSpacing(4)

        # Timestamp
        time_label = QLabel(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        time_label.setStyleSheet("font-size: 10px; color: #6c757d;")
        entry_layout.addWidget(time_label)

        # Question
        if response.question:
            question_label = QLabel(f"<b>Q:</b> {response.question}")
            question_label.setWordWrap(True)
            question_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            entry_layout.addWidget(question_label)

        # Answer
        if response.direct_answer:
            answer_label = QLabel(f"<b>A:</b> {response.direct_answer}")
            answer_label.setWordWrap(True)
            answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            entry_layout.addWidget(answer_label)

        # Follow-up questions
        if response.follow_up_questions:
            followup_text = "<b>Follow-ups:</b> " + "; ".join(response.follow_up_questions)
            followup_label = QLabel(followup_text)
            followup_label.setWordWrap(True)
            followup_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            followup_label.setStyleSheet("font-size: 11px; color: #495057;")
            entry_layout.addWidget(followup_label)

        return entry

    def clear_current_question(self) -> None:
        """Clear the current question display.

        Requirements: 14.1
        """
        self._current_question = None
        self._question_text.setText("No question detected yet")
        self._question_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 4px;
                color: #666;
            }
            """
        )
        logger.debug("Current question cleared")

    def clear_current_response(self) -> None:
        """Clear the current response display.

        Requirements: 14.1, 14.2
        """
        self._current_response = None
        self._response_text.setText("Waiting for response...")
        self._response_text.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #e8f4f8;
                border-radius: 4px;
                color: #333;
            }
            """
        )
        logger.debug("Current response cleared")

    def clear_history(self) -> None:
        """Clear the response history.

        Requirements: 14.3
        """
        self._response_history.clear()

        # Remove all history widgets except the stretch
        while self._history_layout.count() > 1:
            item = self._history_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        logger.debug("Response history cleared")

    def get_history_count(self) -> int:
        """Get the number of responses in history.

        Returns:
            Number of history entries

        Requirements: 14.3
        """
        return len(self._response_history)
