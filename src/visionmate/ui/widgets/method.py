"""Capture method selection widget."""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.input import CaptureMethod

logger = logging.getLogger(__name__)


class CaptureMethodWidget(QWidget):
    """Widget for selecting capture method (OS-native vs UVC)."""

    # Signal emitted when capture method changes
    capture_method_changed = Signal(CaptureMethod)

    def __init__(self, initial_method: CaptureMethod = CaptureMethod.OS_NATIVE):
        """Initialize capture method widget.

        Args:
            initial_method: Initial capture method
        """
        super().__init__()

        self._current_method = initial_method
        self._is_capturing = False

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Create group box
        group_box = QGroupBox("Screen Capture Method")
        group_layout = QVBoxLayout(group_box)

        # Create radio buttons
        self._os_native_radio = QRadioButton(str(CaptureMethod.OS_NATIVE))
        self._uvc_device_radio = QRadioButton(str(CaptureMethod.UVC_DEVICE))

        # Create button group for mutual exclusivity
        self._button_group = QButtonGroup(self)
        self._button_group.addButton(self._os_native_radio, 0)
        self._button_group.addButton(self._uvc_device_radio, 1)

        # Set initial selection
        if initial_method == CaptureMethod.OS_NATIVE:
            self._os_native_radio.setChecked(True)
        else:
            self._uvc_device_radio.setChecked(True)

        # Connect signals
        self._os_native_radio.toggled.connect(self._on_method_changed)
        self._uvc_device_radio.toggled.connect(self._on_method_changed)

        # Add radio buttons to layout
        group_layout.addWidget(self._os_native_radio)
        group_layout.addWidget(self._uvc_device_radio)

        # Add group box to main layout
        layout.addWidget(group_box)

    def _on_method_changed(self, checked: bool) -> None:
        """Handle capture method change.

        Args:
            checked: Whether the radio button is checked
        """
        if not checked:
            # Only respond to the button being checked, not unchecked
            return

        # Determine which method is selected
        if self._os_native_radio.isChecked():
            new_method = CaptureMethod.OS_NATIVE
        else:
            new_method = CaptureMethod.UVC_DEVICE

        # Only emit if method actually changed
        if new_method != self._current_method:
            logger.info(f"Capture method changed from {self._current_method} to {new_method}")
            self._current_method = new_method
            self.capture_method_changed.emit(new_method)

    def get_capture_method(self) -> CaptureMethod:
        """Get the currently selected capture method.

        Returns:
            Current capture method
        """
        return self._current_method

    def set_capture_method(self, method: CaptureMethod) -> None:
        """Set the capture method programmatically.

        Args:
            method: Capture method to set
        """
        if method == self._current_method:
            return

        self._current_method = method

        # Update radio button selection
        if method == CaptureMethod.OS_NATIVE:
            self._os_native_radio.setChecked(True)
        else:
            self._uvc_device_radio.setChecked(True)

    def set_capture_active(self, active: bool) -> None:
        """Set capture active state and enable/disable controls.

        Args:
            active: True if capture is active, False otherwise
        """
        self._is_capturing = active

        # Disable method selection during capture
        self._os_native_radio.setEnabled(not active)
        self._uvc_device_radio.setEnabled(not active)

    def is_capture_active(self) -> bool:
        """Check if capture is currently active.

        Returns:
            True if capture is active, False otherwise
        """
        return self._is_capturing
