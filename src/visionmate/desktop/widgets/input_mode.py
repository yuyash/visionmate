"""
Input mode selection widget.

This module provides the InputModeWidget for selecting between
Video+Audio, Video Only, and Audio Only input modes.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import InputMode

logger = logging.getLogger(__name__)


class InputModeWidget(QWidget):
    """Widget for selecting input mode.

    Provides radio buttons for selecting between:
    - Video + Audio
    - Video Only
    - Audio Only

    Requirements: 10.6, 3.1
    """

    # Signal emitted when input mode changes
    mode_changed = Signal(InputMode)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the InputModeWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing InputModeWidget")

        self._current_mode = InputMode.VIDEO_AUDIO
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create group box (global stylesheet applied)
        group_box = QGroupBox("Input Mode")
        group_box.setFlat(True)
        group_layout = QVBoxLayout(group_box)

        # Create radio buttons (global stylesheet applied)
        self._video_audio_radio = QRadioButton("Video + Audio")
        self._video_only_radio = QRadioButton("Video Only")
        self._audio_only_radio = QRadioButton("Audio Only")

        # Set default selection
        self._video_audio_radio.setChecked(True)

        # Create button group for mutual exclusivity
        self._button_group = QButtonGroup(self)
        self._button_group.addButton(self._video_audio_radio, 0)
        self._button_group.addButton(self._video_only_radio, 1)
        self._button_group.addButton(self._audio_only_radio, 2)

        # Connect signals
        self._video_audio_radio.toggled.connect(self._on_mode_changed)
        self._video_only_radio.toggled.connect(self._on_mode_changed)
        self._audio_only_radio.toggled.connect(self._on_mode_changed)

        # Add radio buttons to layout
        group_layout.addWidget(self._video_audio_radio)
        group_layout.addWidget(self._video_only_radio)
        group_layout.addWidget(self._audio_only_radio)

        # Add group box to main layout
        layout.addWidget(group_box)

        logger.debug("InputModeWidget UI setup complete")

    def _on_mode_changed(self, checked: bool) -> None:
        """Handle input mode change.

        Args:
            checked: Whether the radio button is checked
        """
        if not checked:
            return

        # Determine which mode was selected
        if self._video_audio_radio.isChecked():
            new_mode = InputMode.VIDEO_AUDIO
        elif self._video_only_radio.isChecked():
            new_mode = InputMode.VIDEO_ONLY
        else:
            new_mode = InputMode.AUDIO_ONLY

        if new_mode != self._current_mode:
            self._current_mode = new_mode
            logger.info(f"Input mode changed to: {new_mode.value}")
            self.mode_changed.emit(new_mode)

    def get_mode(self) -> InputMode:
        """Get the currently selected input mode.

        Returns:
            Current InputMode
        """
        return self._current_mode

    def set_mode(self, mode: InputMode) -> None:
        """Set the input mode.

        Args:
            mode: InputMode to set
        """
        if mode == self._current_mode:
            return

        self._current_mode = mode

        # Update radio button selection
        if mode == InputMode.VIDEO_AUDIO:
            self._video_audio_radio.setChecked(True)
        elif mode == InputMode.VIDEO_ONLY:
            self._video_only_radio.setChecked(True)
        else:
            self._audio_only_radio.setChecked(True)

        logger.debug(f"Input mode set to: {mode.value}")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the widget.

        Args:
            enabled: Whether to enable the widget
        """
        self._video_audio_radio.setEnabled(enabled)
        self._video_only_radio.setEnabled(enabled)
        self._audio_only_radio.setEnabled(enabled)
