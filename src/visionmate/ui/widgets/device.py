"""Device selection controls for video and audio input."""

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.capture.audio import AudioCaptureInterface
from visionmate.capture.screen import ScreenCaptureInterface


class DeviceControlsWidget(QWidget):
    """Widget for device selection controls."""

    # Signals emitted when devices are selected
    video_device_changed = Signal(int)  # Emits device ID
    audio_device_changed = Signal(int)  # Emits device ID

    def __init__(
        self,
        screen_capture: Optional[ScreenCaptureInterface] = None,
        audio_capture: Optional[AudioCaptureInterface] = None,
    ):
        """Initialize device controls widget.

        Args:
            screen_capture: Screen capture interface for video device enumeration
            audio_capture: Audio capture interface for audio device enumeration
        """
        super().__init__()

        self._screen_capture = screen_capture
        self._audio_capture = audio_capture
        self._is_capturing = False  # Track capture state

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Create video device selection group
        video_group = QGroupBox("Video Input Device")
        video_layout = QVBoxLayout(video_group)

        self._video_device_combo = QComboBox()
        self._video_device_combo.currentIndexChanged.connect(self._on_video_device_changed)

        self._video_refresh_button = QPushButton("Refresh Video Devices")
        self._video_refresh_button.clicked.connect(self._refresh_video_devices)

        video_layout.addWidget(QLabel("Select video capture device:"))
        video_layout.addWidget(self._video_device_combo)
        video_layout.addWidget(self._video_refresh_button)

        # Create audio device selection group
        audio_group = QGroupBox("Audio Input Device")
        audio_layout = QVBoxLayout(audio_group)

        self._audio_device_combo = QComboBox()
        self._audio_device_combo.currentIndexChanged.connect(self._on_audio_device_changed)

        self._audio_refresh_button = QPushButton("Refresh Audio Devices")
        self._audio_refresh_button.clicked.connect(self._refresh_audio_devices)

        audio_layout.addWidget(QLabel("Select audio input device:"))
        audio_layout.addWidget(self._audio_device_combo)
        audio_layout.addWidget(self._audio_refresh_button)

        # Add groups to main layout
        layout.addWidget(video_group)
        layout.addWidget(audio_group)

        # Initial device enumeration
        self._refresh_video_devices()
        self._refresh_audio_devices()

    def set_screen_capture(self, capture: ScreenCaptureInterface) -> None:
        """Set the screen capture interface.

        Args:
            capture: Screen capture interface
        """
        self._screen_capture = capture
        self._refresh_video_devices()

    def set_audio_capture(self, capture: AudioCaptureInterface) -> None:
        """Set the audio capture interface.

        Args:
            capture: Audio capture interface
        """
        self._audio_capture = capture
        self._refresh_audio_devices()

    def _refresh_video_devices(self) -> None:
        """Refresh the list of video devices."""
        if self._screen_capture is None:
            return

        # Clear current items
        self._video_device_combo.clear()

        # Get available devices
        devices = self._screen_capture.list_devices()

        # Populate combo box
        for device in devices:
            device_id = device.get("id", 0)
            device_name = device.get("name", f"Device {device_id}")
            width = device.get("width", 0)
            height = device.get("height", 0)

            # Format display text
            display_text = f"{device_name} ({width}x{height})"

            # Add item with device ID as user data
            self._video_device_combo.addItem(display_text, device_id)

        # If no devices found, add placeholder
        if self._video_device_combo.count() == 0:
            self._video_device_combo.addItem("No devices found", -1)
            self._video_device_combo.setEnabled(False)
        else:
            self._video_device_combo.setEnabled(True)

    def _refresh_audio_devices(self) -> None:
        """Refresh the list of audio devices."""
        if self._audio_capture is None:
            return

        # Clear current items
        self._audio_device_combo.clear()

        # Get available devices
        devices = self._audio_capture.list_devices()

        # Populate combo box
        for device in devices:
            device_id = device.get("id", 0)
            device_name = device.get("name", f"Device {device_id}")
            channels = device.get("channels", 0)
            sample_rate = device.get("sample_rate", 0)

            # Format display text
            display_text = f"{device_name} ({channels}ch, {sample_rate}Hz)"

            # Add item with device ID as user data
            self._audio_device_combo.addItem(display_text, device_id)

        # If no devices found, add placeholder
        if self._audio_device_combo.count() == 0:
            self._audio_device_combo.addItem("No devices found", -1)
            self._audio_device_combo.setEnabled(False)
        else:
            self._audio_device_combo.setEnabled(True)

    def _on_video_device_changed(self, index: int) -> None:
        """Handle video device selection change.

        Args:
            index: Selected combo box index
        """
        if index < 0:
            return

        device_id = self._video_device_combo.itemData(index)

        if device_id is not None and device_id >= 0:
            self.video_device_changed.emit(device_id)

    def _on_audio_device_changed(self, index: int) -> None:
        """Handle audio device selection change.

        Args:
            index: Selected combo box index
        """
        if index < 0:
            return

        device_id = self._audio_device_combo.itemData(index)

        if device_id is not None and device_id >= 0:
            self.audio_device_changed.emit(device_id)

    def get_selected_video_device(self) -> Optional[int]:
        """Get the currently selected video device ID.

        Returns:
            Device ID or None if no valid device selected
        """
        index = self._video_device_combo.currentIndex()
        if index < 0:
            return None

        device_id = self._video_device_combo.itemData(index)
        return device_id if device_id >= 0 else None

    def get_selected_audio_device(self) -> Optional[int]:
        """Get the currently selected audio device ID.

        Returns:
            Device ID or None if no valid device selected
        """
        index = self._audio_device_combo.currentIndex()
        if index < 0:
            return None

        device_id = self._audio_device_combo.itemData(index)
        return device_id if device_id >= 0 else None

    def set_capture_active(self, active: bool) -> None:
        """Set capture active state and enable/disable device controls.

        Args:
            active: True if capture is active, False otherwise
        """
        self._is_capturing = active

        # Disable device selection and refresh buttons during capture
        self._video_device_combo.setEnabled(not active)
        self._video_refresh_button.setEnabled(not active)
        self._audio_device_combo.setEnabled(not active)
        self._audio_refresh_button.setEnabled(not active)

    def is_capture_active(self) -> bool:
        """Check if capture is currently active.

        Returns:
            True if capture is active, False otherwise
        """
        return self._is_capturing
