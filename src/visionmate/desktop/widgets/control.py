"""Control container widget for managing input controls.

This module provides the ControlContainer widget that manages the control panel
with input mode selection, video input configuration, and device management.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.manager import CaptureManager
from visionmate.core.models import DeviceMetadata
from visionmate.desktop.widgets.input import AudioInputWidget, VideoInputWidget
from visionmate.desktop.widgets.mode import InputModeWidget

logger = logging.getLogger(__name__)


class ControlContainer(QWidget):
    """Container for input control widgets.

    Manages:
    - Input mode selection (Video+Audio, Video Only, Audio Only)
    - Video input configuration (Screen, UVC, RTSP)
    - Device enumeration and selection
    - Device refresh functionality

    Requirements: 10.4, 10.5, 10.6
    """

    # Signals
    source_type_changed = Signal(str)  # source_type
    refresh_requested = Signal(str)  # source_type
    device_selected = Signal(str, str)  # source_type, device_id
    selection_changed = Signal(list)  # selected_device_ids
    window_capture_mode_changed = Signal(str, list)  # mode, selected_titles
    audio_device_selected = Signal(str)  # device_id
    audio_refresh_requested = Signal()  # no args
    status_message = Signal(str, int)  # message, timeout_ms

    def __init__(
        self,
        capture_manager: CaptureManager,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the ControlContainer.

        Args:
            capture_manager: CaptureManager instance
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing ControlContainer")

        self._capture_manager = capture_manager

        # Device cache - populated on startup
        self._device_cache: dict[str, list[DeviceMetadata]] = {
            "screen": [],
            "uvc": [],
            "rtsp": [],
            "audio": [],
        }

        # Widgets
        self._input_mode_widget: Optional[InputModeWidget] = None
        self._video_input_widget: Optional[VideoInputWidget] = None
        self._audio_input_widget: Optional[AudioInputWidget] = None

        self._setup_ui()
        self._connect_signals()

        # Scan devices on startup
        self._scan_all_devices()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Set fixed width for control panel
        self.setFixedWidth(320)
        self.setStyleSheet(
            """
            QWidget#controlContainer {
                background-color: #f8f8f8;
            }
            """
        )
        self.setObjectName("controlContainer")

        # Create scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        # Create controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(8, 8, 0, 8)
        controls_layout.setSpacing(12)

        # Add Input Mode widget
        self._input_mode_widget = InputModeWidget()
        controls_layout.addWidget(self._input_mode_widget)

        # Add Video Input widget
        self._video_input_widget = VideoInputWidget()
        controls_layout.addWidget(self._video_input_widget)

        # Add Audio Input widget
        self._audio_input_widget = AudioInputWidget()
        controls_layout.addWidget(self._audio_input_widget)

        # Add stretch to push controls to top
        controls_layout.addStretch()

        # Set controls widget in scroll area
        scroll_area.setWidget(controls_widget)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        logger.debug("ControlContainer UI setup complete")

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        if self._input_mode_widget is not None:
            # Connect input mode change signal
            self._input_mode_widget.mode_changed.connect(self._on_input_mode_changed)

        if self._video_input_widget is not None:
            # Connect and forward signals
            self._video_input_widget.source_type_changed.connect(self._on_source_type_changed)
            self._video_input_widget.refresh_requested.connect(self._on_refresh_requested)
            self._video_input_widget.device_selected.connect(self._on_device_selected)
            self._video_input_widget.selection_changed.connect(self._on_selection_changed)
            self._video_input_widget.window_capture_mode_changed.connect(
                self._on_window_capture_mode_changed
            )

        if self._audio_input_widget is not None:
            # Connect audio widget signals
            self._audio_input_widget.device_selected.connect(self._on_audio_device_selected)
            self._audio_input_widget.refresh_requested.connect(self._on_audio_refresh_requested)

        logger.debug("ControlContainer signals connected")

    def _scan_all_devices(self) -> None:
        """Scan all device types on startup and populate cache."""
        logger.info("Scanning all devices on startup...")

        try:
            # Scan screens
            self._device_cache["screen"] = self._capture_manager.get_screens()
            logger.info(f"Cached {len(self._device_cache['screen'])} screen device(s)")

            # Scan UVC devices
            self._device_cache["uvc"] = self._capture_manager.get_uvc_devices()
            logger.info(f"Cached {len(self._device_cache['uvc'])} UVC device(s)")

            # Scan audio devices
            self._device_cache["audio"] = self._capture_manager.get_audio_devices()
            logger.info(f"Cached {len(self._device_cache['audio'])} audio device(s)")

            # Update audio device list
            if self._audio_input_widget is not None:
                self._audio_input_widget.update_device_list(self._device_cache["audio"])

            # RTSP doesn't have enumerable devices
            self._device_cache["rtsp"] = []

        except Exception as e:
            logger.error(f"Error scanning devices: {e}", exc_info=True)

    def _on_source_type_changed(self, source_type: str) -> None:
        """Handle source type change.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
        """
        logger.debug(f"Source type changed to: {source_type}")

        # Update device list from cache
        self._update_device_list_from_cache(source_type)

        # Forward signal
        self.source_type_changed.emit(source_type)

    def _update_device_list_from_cache(self, source_type: str) -> None:
        """Update device list from cached data.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp", or "")
        """
        if not self._video_input_widget or not source_type:
            return

        # Get devices from cache
        devices = self._device_cache.get(source_type, [])

        # Update the device list
        self._video_input_widget.update_device_list(devices)

        logger.debug(f"Updated device list from cache: {len(devices)} {source_type} device(s)")

    def _on_refresh_requested(self, source_type: str) -> None:
        """Handle device refresh request.

        Args:
            source_type: Type of source to refresh ("screen", "uvc", "rtsp", or "")
        """
        if not source_type:
            return

        logger.info(f"Refreshing devices for source type: {source_type}")

        try:
            # Re-scan devices based on source type
            if source_type == "screen":
                devices = self._capture_manager.get_screens()
            elif source_type == "uvc":
                devices = self._capture_manager.get_uvc_devices()
            elif source_type == "rtsp":
                devices = []
            else:
                logger.warning(f"Unknown source type: {source_type}")
                devices = []

            # Update cache
            self._device_cache[source_type] = devices

            # Update the device list
            if self._video_input_widget is not None:
                self._video_input_widget.update_device_list(devices)

            logger.info(f"Refreshed {len(devices)} {source_type} device(s)")

            # Emit status message
            self.status_message.emit(f"Found {len(devices)} {source_type} device(s)", 3000)

            # Forward signal
            self.refresh_requested.emit(source_type)

        except Exception as e:
            logger.error(f"Error refreshing device list: {e}", exc_info=True)
            self.status_message.emit(f"Error: {e}", 5000)

    def _on_device_selected(self, source_type: str, device_id: str) -> None:
        """Handle device selection.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier
        """
        logger.info(f"Device selected: {device_id} (type: {source_type})")

        # Forward signal
        self.device_selected.emit(source_type, device_id)

    def _on_selection_changed(self, selected_device_ids: list[str]) -> None:
        """Handle selection change (for multiple selection).

        Args:
            selected_device_ids: List of selected device IDs
        """
        logger.debug(f"Selection changed: {len(selected_device_ids)} device(s) selected")

        # Forward signal
        self.selection_changed.emit(selected_device_ids)

    def _on_window_capture_mode_changed(self, mode: str, selected_titles: list[str]) -> None:
        """Handle window capture mode change.

        Args:
            mode: Capture mode
            selected_titles: List of selected window titles
        """
        logger.info(f"Window capture mode changed: {mode}")

        # Forward signal
        self.window_capture_mode_changed.emit(mode, selected_titles)

    def get_current_source_type(self) -> str:
        """Get the current source type.

        Returns:
            Current source type ("screen", "uvc", "rtsp", or "")
        """
        if self._video_input_widget is not None:
            return self._video_input_widget.get_current_source_type()
        return ""

    def get_window_capture_mode(self) -> str:
        """Get the current window capture mode.

        Returns:
            Current window capture mode
        """
        if self._video_input_widget is not None:
            return self._video_input_widget.get_window_capture_mode()
        return "full_screen"

    def get_fps(self) -> int:
        """Get the current FPS setting.

        Returns:
            FPS value (1-240)
        """
        if self._video_input_widget is not None:
            return self._video_input_widget.get_fps()
        return 1  # Default

    def clear_selection(self) -> None:
        """Clear device selection."""
        if self._video_input_widget is not None:
            self._video_input_widget.clear_selection()
            logger.debug("Cleared device selection")

    def get_device_cache(self) -> dict[str, list[DeviceMetadata]]:
        """Get the device cache.

        Returns:
            Device cache dictionary
        """
        return self._device_cache.copy()

    def _on_audio_device_selected(self, device_id: str) -> None:
        """Handle audio device selection.

        Args:
            device_id: Audio device identifier

        Requirements: 12.1
        """
        logger.info(f"Audio device selected: {device_id}")

        # Forward signal
        self.audio_device_selected.emit(device_id)

    def _on_audio_refresh_requested(self) -> None:
        """Handle audio device refresh request.

        Requirements: 12.1
        """
        logger.info("Refreshing audio devices")

        try:
            # Re-scan audio devices
            devices = self._capture_manager.get_audio_devices()

            # Update cache
            self._device_cache["audio"] = devices

            # Update the device list
            if self._audio_input_widget is not None:
                self._audio_input_widget.update_device_list(devices)

            logger.info(f"Refreshed {len(devices)} audio device(s)")

            # Forward signal
            self.audio_refresh_requested.emit()

        except Exception as e:
            logger.error(f"Error refreshing audio device list: {e}", exc_info=True)

    def get_selected_audio_device_id(self) -> Optional[str]:
        """Get the currently selected audio device ID.

        Returns:
            Audio device ID, or None if no device is selected
        """
        if self._audio_input_widget is not None:
            return self._audio_input_widget.get_selected_device_id()
        return None

    def _on_input_mode_changed(self, mode) -> None:
        """Handle input mode change.

        Args:
            mode: InputMode enum value

        Requirements: 10.6
        """
        from visionmate.core.models import InputMode

        logger.info(f"Input mode changed to: {mode.value}")

        # Show/hide widgets based on input mode
        if mode == InputMode.VIDEO_AUDIO:
            # Show both video and audio controls
            if self._video_input_widget is not None:
                self._video_input_widget.show()
            if self._audio_input_widget is not None:
                self._audio_input_widget.show()
        elif mode == InputMode.VIDEO_ONLY:
            # Show only video controls
            if self._video_input_widget is not None:
                self._video_input_widget.show()
            if self._audio_input_widget is not None:
                self._audio_input_widget.hide()
        elif mode == InputMode.AUDIO_ONLY:
            # Show only audio controls
            if self._video_input_widget is not None:
                self._video_input_widget.hide()
            if self._audio_input_widget is not None:
                self._audio_input_widget.show()

        logger.debug(f"Updated widget visibility for mode: {mode.value}")
