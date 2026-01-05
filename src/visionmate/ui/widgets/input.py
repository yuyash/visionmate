"""Input mode selection controls."""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.input import InputMode

logger = logging.getLogger(__name__)


class InputModeWidget(QWidget):
    """Widget for input mode selection controls."""

    # Signal emitted when input mode changes
    input_mode_changed = Signal(InputMode)

    def __init__(self, initial_mode: InputMode = InputMode.VIDEO_AUDIO):
        """Initialize input mode widget.

        Args:
            initial_mode: Initial input mode (default: VIDEO_AUDIO)
        """
        super().__init__()

        self._current_mode = initial_mode
        self._is_capturing = False

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Create input mode selection group
        mode_group = QGroupBox("Input Mode")
        mode_layout = QVBoxLayout(mode_group)

        # Create radio buttons for each mode
        self._button_group = QButtonGroup(self)

        self._video_audio_radio = QRadioButton("Video + Audio")
        self._video_audio_radio.setToolTip("Capture both video and audio input")
        self._button_group.addButton(self._video_audio_radio, 0)

        self._video_only_radio = QRadioButton("Video Only")
        self._video_only_radio.setToolTip("Capture video input only")
        self._button_group.addButton(self._video_only_radio, 1)

        self._audio_only_radio = QRadioButton("Audio Only")
        self._audio_only_radio.setToolTip("Capture audio input only")
        self._button_group.addButton(self._audio_only_radio, 2)

        # Add radio buttons to layout
        mode_layout.addWidget(self._video_audio_radio)
        mode_layout.addWidget(self._video_only_radio)
        mode_layout.addWidget(self._audio_only_radio)

        # Create status label
        self._status_label = QLabel()
        self._status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        mode_layout.addWidget(self._status_label)

        # Add group to main layout
        layout.addWidget(mode_group)

        # Connect signal
        self._button_group.buttonClicked.connect(self._on_mode_changed)

        # Set initial mode
        self.set_input_mode(initial_mode)

    def set_input_mode(self, mode: InputMode) -> None:
        """Set the current input mode.

        Args:
            mode: Input mode to set
        """
        self._current_mode = mode

        # Update radio button selection
        if mode == InputMode.VIDEO_AUDIO:
            self._video_audio_radio.setChecked(True)
        elif mode == InputMode.VIDEO_ONLY:
            self._video_only_radio.setChecked(True)
        elif mode == InputMode.AUDIO_ONLY:
            self._audio_only_radio.setChecked(True)

        # Update status label
        self._update_status_label()

        logger.debug(f"Input mode set to {mode}")

    def get_input_mode(self) -> InputMode:
        """Get the current input mode.

        Returns:
            Current input mode
        """
        return self._current_mode

    def set_capture_active(self, active: bool) -> None:
        """Set capture active state and enable/disable mode controls.

        Args:
            active: True if capture is active, False otherwise
        """
        self._is_capturing = active

        # Disable mode selection during capture
        self._video_audio_radio.setEnabled(not active)
        self._video_only_radio.setEnabled(not active)
        self._audio_only_radio.setEnabled(not active)

        # Update status label
        self._update_status_label()

        logger.debug(f"Capture active state set to {active}")

    def is_capture_active(self) -> bool:
        """Check if capture is currently active.

        Returns:
            True if capture is active, False otherwise
        """
        return self._is_capturing

    def _on_mode_changed(self) -> None:
        """Handle input mode radio button change."""
        # Determine which mode was selected
        if self._video_audio_radio.isChecked():
            new_mode = InputMode.VIDEO_AUDIO
        elif self._video_only_radio.isChecked():
            new_mode = InputMode.VIDEO_ONLY
        elif self._audio_only_radio.isChecked():
            new_mode = InputMode.AUDIO_ONLY
        else:
            return

        # Only emit signal if mode actually changed
        if new_mode != self._current_mode:
            self._current_mode = new_mode
            self._update_status_label()
            self.input_mode_changed.emit(new_mode)
            logger.info(f"Input mode changed to {new_mode}")

    def _update_status_label(self) -> None:
        """Update the status label text."""
        if self._is_capturing:
            status_text = f"Current mode: {self._current_mode}"
        else:
            status_text = f"Mode: {self._current_mode}"

        self._status_label.setText(status_text)
