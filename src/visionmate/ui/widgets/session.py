"""Session control buttons widget."""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SessionState:
    """Session state constants."""

    IDLE = "idle"
    CAPTURING = "capturing"
    RECOGNIZING = "recognizing"


class SessionControlWidget(QWidget):
    """Widget for session control buttons."""

    # Signals emitted when buttons are clicked
    start_capture_clicked = Signal()
    stop_capture_clicked = Signal()
    start_recognition_clicked = Signal()
    stop_recognition_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self):
        """Initialize session control widget."""
        super().__init__()

        self._current_state = SessionState.IDLE

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Create session control group
        control_group = QGroupBox("Session Control")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(8)

        # Create buttons
        self._start_capture_btn = QPushButton("Start Capture")
        self._start_capture_btn.setToolTip("Start capturing video and audio input")
        self._start_capture_btn.clicked.connect(self._on_start_capture)

        self._stop_capture_btn = QPushButton("Stop Capture")
        self._stop_capture_btn.setToolTip("Stop all input capture")
        self._stop_capture_btn.clicked.connect(self._on_stop_capture)

        self._start_recognition_btn = QPushButton("Start Recognition")
        self._start_recognition_btn.setToolTip("Start VLM recognition and answer generation")
        self._start_recognition_btn.clicked.connect(self._on_start_recognition)

        self._stop_recognition_btn = QPushButton("Stop Recognition")
        self._stop_recognition_btn.setToolTip("Stop recognition while keeping capture active")
        self._stop_recognition_btn.clicked.connect(self._on_stop_recognition)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset question understanding and clear context")
        self._reset_btn.clicked.connect(self._on_reset)

        # Add buttons to layout
        control_layout.addWidget(self._start_capture_btn)
        control_layout.addWidget(self._stop_capture_btn)
        control_layout.addWidget(self._start_recognition_btn)
        control_layout.addWidget(self._stop_recognition_btn)
        control_layout.addWidget(self._reset_btn)

        # Add group to main layout
        layout.addWidget(control_group)

        # Set initial button states
        self._update_button_states()

    def set_session_state(self, state: str) -> None:
        """Set the current session state and update button states.

        Args:
            state: Session state (IDLE, CAPTURING, RECOGNIZING)
        """
        if state not in (SessionState.IDLE, SessionState.CAPTURING, SessionState.RECOGNIZING):
            logger.warning(f"Invalid session state: {state}")
            return

        self._current_state = state
        self._update_button_states()
        logger.debug(f"Session state set to {state}")

    def get_session_state(self) -> str:
        """Get the current session state.

        Returns:
            Current session state
        """
        return self._current_state

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on current session state."""
        if self._current_state == SessionState.IDLE:
            # IDLE: Can only start capture
            self._start_capture_btn.setEnabled(True)
            self._stop_capture_btn.setEnabled(False)
            self._start_recognition_btn.setEnabled(False)
            self._stop_recognition_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)

        elif self._current_state == SessionState.CAPTURING:
            # CAPTURING: Can stop capture or start recognition
            self._start_capture_btn.setEnabled(False)
            self._stop_capture_btn.setEnabled(True)
            self._start_recognition_btn.setEnabled(True)
            self._stop_recognition_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)

        elif self._current_state == SessionState.RECOGNIZING:
            # RECOGNIZING: Can stop recognition, stop capture, or reset
            self._start_capture_btn.setEnabled(False)
            self._stop_capture_btn.setEnabled(True)
            self._start_recognition_btn.setEnabled(False)
            self._stop_recognition_btn.setEnabled(True)
            self._reset_btn.setEnabled(True)

    def _on_start_capture(self) -> None:
        """Handle start capture button click."""
        logger.info("Start capture button clicked")
        self.start_capture_clicked.emit()

    def _on_stop_capture(self) -> None:
        """Handle stop capture button click."""
        logger.info("Stop capture button clicked")
        self.stop_capture_clicked.emit()

    def _on_start_recognition(self) -> None:
        """Handle start recognition button click."""
        logger.info("Start recognition button clicked")
        self.start_recognition_clicked.emit()

    def _on_stop_recognition(self) -> None:
        """Handle stop recognition button click."""
        logger.info("Stop recognition button clicked")
        self.stop_recognition_clicked.emit()

    def _on_reset(self) -> None:
        """Handle reset button click."""
        logger.info("Reset button clicked")
        self.reset_clicked.emit()
