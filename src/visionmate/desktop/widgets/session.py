"""Session control widget for managing session state.

This module provides the SessionControlWidget that displays Start, Stop,
and Reset buttons for controlling the application session.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import SessionState

logger = logging.getLogger(__name__)


class SessionControlWidget(QWidget):
    """Widget for session control buttons.

    Provides Start, Stop, and Reset buttons for controlling the session state.
    Button states are managed based on session state and device selection.

    """

    # Signals
    start_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the SessionControlWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing SessionControlWidget")

        # Buttons
        self._start_button: Optional[QPushButton] = None
        self._stop_button: Optional[QPushButton] = None
        self._reset_button: Optional[QPushButton] = None

        # State tracking
        self._has_devices = False
        self._session_state = SessionState.IDLE

        self._setup_ui()
        self._update_button_states()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Add title label
        title_label = QLabel("Session Control")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # Create button layout (vertical stack)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        # Create Start button
        self._start_button = QPushButton("Start")
        self._start_button.setToolTip("Start capture and recognition")
        self._start_button.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self._start_button)

        # Create Stop button
        self._stop_button = QPushButton("Stop")
        self._stop_button.setToolTip("Stop capture and recognition")
        self._stop_button.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self._stop_button)

        # Create Reset button
        self._reset_button = QPushButton("Reset")
        self._reset_button.setToolTip("Reset question understanding (keep capturing)")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self._reset_button)

        main_layout.addLayout(button_layout)

        logger.debug("SessionControlWidget UI setup complete")

    def _on_start_clicked(self) -> None:
        """Handle Start button click."""
        logger.info("Start button clicked")
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        """Handle Stop button click."""
        logger.info("Stop button clicked")
        self.stop_requested.emit()

    def _on_reset_clicked(self) -> None:
        """Handle Reset button click."""
        logger.info("Reset button clicked")
        self.reset_requested.emit()

    def set_session_state(self, state: SessionState) -> None:
        """Set the session state and update button states.

        Args:
            state: Current session state

        """
        self._session_state = state
        self._update_button_states()
        logger.debug(f"Session state updated: {state.value}")

    def set_has_devices(self, has_devices: bool) -> None:
        """Set whether devices are selected.

        Args:
            has_devices: True if at least one device is selected

        """
        self._has_devices = has_devices
        self._update_button_states()
        logger.debug(f"Device selection updated: has_devices={has_devices}")

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on session state and device selection.

        Button state logic:
        - Start: Enabled when IDLE and devices are selected
        - Stop: Enabled when ACTIVE
        - Reset: Enabled when ACTIVE

        """
        if not self._start_button or not self._stop_button or not self._reset_button:
            return

        if self._session_state == SessionState.IDLE:
            # IDLE state: Enable Start if devices selected, disable Stop and Reset
            self._start_button.setEnabled(self._has_devices)
            self._stop_button.setEnabled(False)
            self._reset_button.setEnabled(False)
        elif self._session_state == SessionState.ACTIVE:
            # ACTIVE state: Disable Start, enable Stop and Reset
            self._start_button.setEnabled(False)
            self._stop_button.setEnabled(True)
            self._reset_button.setEnabled(True)

        logger.debug(
            f"Button states updated: Start={self._start_button.isEnabled()}, "
            f"Stop={self._stop_button.isEnabled()}, Reset={self._reset_button.isEnabled()}"
        )
