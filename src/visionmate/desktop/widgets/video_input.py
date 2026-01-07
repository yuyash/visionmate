"""
Video input configuration widget.

This module provides the VideoInputWidget for configuring video input sources
including source type selection, device list, and metadata display.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import DeviceMetadata

logger = logging.getLogger(__name__)


class VideoInputWidget(QWidget):
    """Widget for configuring video input sources.

    Provides controls for:
    - Source type selection (Screen, UVC, RTSP)
    - Device list with refresh button
    - Device metadata display

    Requirements: 11.1, 11.2, 11.7
    """

    # Signal emitted when a device is selected
    device_selected = Signal(str, str)  # (source_type, device_id)

    # Signal emitted when refresh is requested
    refresh_requested = Signal(str)  # source_type

    # Signal emitted when source type changes
    source_type_changed = Signal(str)  # source_type

    # Signal emitted when selection changes (for multiple selection)
    selection_changed = Signal(list)  # List of selected device_ids

    # Signal emitted when window detection mode changes
    window_detection_changed = Signal(bool)  # enabled

    # Signal emitted when window capture mode changes
    window_capture_mode_changed = Signal(str, list)  # (mode, selected_titles)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the VideoInputWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing VideoInputWidget")

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create group box (global stylesheet applied)
        group_box = QGroupBox("Video Input")
        group_box.setFlat(True)
        group_layout = QVBoxLayout(group_box)

        # Source type selector with refresh button
        source_type_layout = QHBoxLayout()
        source_type_label = QLabel("Source Type:")
        # Global stylesheet handles label styling

        self._source_type_combo = QComboBox()
        self._source_type_combo.addItem("-- Select --", "")  # Shorter default text
        self._source_type_combo.addItem("Screen Capture", "screen")
        self._source_type_combo.addItem("UVC Device", "uvc")
        self._source_type_combo.addItem("RTSP Stream", "rtsp")
        # Set minimum width for dropdown list to prevent truncation
        self._source_type_combo.view().setMinimumWidth(200)
        # Global stylesheet handles combo box styling
        self._source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)

        # Refresh button (global stylesheet applied)
        self._refresh_button = QPushButton("↻")
        self._refresh_button.setFixedSize(40, 30)  # Match combo box height
        self._refresh_button.setToolTip("Refresh device list")
        self._refresh_button.clicked.connect(self._on_refresh_clicked)

        source_type_layout.addWidget(source_type_label)
        source_type_layout.addWidget(self._source_type_combo, stretch=1)
        source_type_layout.addWidget(self._refresh_button)
        group_layout.addLayout(source_type_layout)

        # Window capture mode (only for screen capture)
        self._window_mode_widget = QWidget()
        window_mode_layout = QVBoxLayout(self._window_mode_widget)
        window_mode_layout.setContentsMargins(0, 5, 0, 5)
        window_mode_layout.setSpacing(5)

        mode_label = QLabel("Capture Mode:")
        window_mode_layout.addWidget(mode_label)

        self._full_screen_radio = QRadioButton("Full Screen")
        self._full_screen_radio.setChecked(True)  # Default
        self._full_screen_radio.toggled.connect(self._on_capture_mode_changed)
        window_mode_layout.addWidget(self._full_screen_radio)

        self._active_window_radio = QRadioButton("Active Window Only")
        self._active_window_radio.toggled.connect(self._on_capture_mode_changed)
        window_mode_layout.addWidget(self._active_window_radio)

        self._selected_windows_radio = QRadioButton("Selected Windows")
        self._selected_windows_radio.toggled.connect(self._on_capture_mode_changed)
        window_mode_layout.addWidget(self._selected_windows_radio)

        group_layout.addWidget(self._window_mode_widget)
        self._window_mode_widget.hide()  # Hidden until screen capture is selected

        # Device list (full width, global stylesheet applied)
        self._device_list = QListWidget()
        self._device_list.setFrameShape(QListWidget.Shape.NoFrame)
        self._device_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        # Global stylesheet handles list widget styling
        self._device_list.itemSelectionChanged.connect(self._on_device_selected)
        group_layout.addWidget(self._device_list)

        # Button to select windows (only visible in selected windows mode)
        # Placed after device list for natural flow
        self._select_windows_button = QPushButton("Select Windows...")
        self._select_windows_button.clicked.connect(self._on_select_windows_clicked)
        self._select_windows_button.hide()  # Hidden by default
        group_layout.addWidget(self._select_windows_button)

        # Add group box to main layout
        layout.addWidget(group_box)

        logger.debug("VideoInputWidget UI setup complete")

    def _on_source_type_changed(self, index: int) -> None:
        """Handle source type change.

        Args:
            index: Index of selected source type
        """
        source_type = self._source_type_combo.currentData()
        logger.debug(f"Source type changed to: {source_type}")

        # Clear device list
        self._device_list.clear()

        # Show/hide window mode widget based on source type
        # Only show for screen capture
        if source_type == "screen":
            self._window_mode_widget.show()
            self._device_list.setEnabled(True)
        elif source_type == "":
            # Empty selection - hide everything
            self._window_mode_widget.hide()
            self._device_list.setEnabled(False)
        else:
            self._window_mode_widget.hide()
            self._device_list.setEnabled(True)

        # Emit signal to notify parent
        self.source_type_changed.emit(source_type)

    def _on_refresh_clicked(self) -> None:
        """Handle refresh button click."""
        source_type = self._source_type_combo.currentData()
        logger.debug(f"Refresh requested for source type: {source_type}")
        self.refresh_requested.emit(source_type)

    def _on_window_detection_changed(self, state: int) -> None:
        """Handle window detection checkbox state change.

        Args:
            state: Qt.CheckState value
        """
        enabled = state == 2  # Qt.CheckState.Checked
        logger.debug(f"Window detection changed: {enabled}")
        self.window_detection_changed.emit(enabled)

    def _on_capture_mode_changed(self) -> None:
        """Handle capture mode radio button change."""
        if self._full_screen_radio.isChecked():
            mode = "full_screen"
            self._select_windows_button.hide()
        elif self._active_window_radio.isChecked():
            mode = "active_window"
            self._select_windows_button.hide()
        elif self._selected_windows_radio.isChecked():
            mode = "selected_windows"
            self._select_windows_button.show()
        else:
            return

        logger.debug(f"Capture mode changed: {mode}")
        # Emit with empty list (will be updated when windows are selected via button)
        self.window_capture_mode_changed.emit(mode, [])

    def _on_select_windows_clicked(self) -> None:
        """Handle select windows button click."""
        logger.debug("Select windows button clicked")
        # Emit signal to parent to show window selector dialog
        # Use a special signal value to indicate button was clicked
        self.window_capture_mode_changed.emit("show_selector", [])

    def _on_device_selected(self) -> None:
        """Handle device selection change."""
        selected_items = self._device_list.selectedItems()
        source_type = self._source_type_combo.currentData()

        # Get list of selected device IDs
        selected_device_ids = [item.data(1) for item in selected_items]

        logger.debug(f"Selection changed: {len(selected_device_ids)} device(s) selected")

        # Emit selection changed signal with all selected devices
        self.selection_changed.emit(selected_device_ids)

        # For backward compatibility, emit device_selected for single selection
        if len(selected_device_ids) == 1:
            self.device_selected.emit(source_type, selected_device_ids[0])

    def update_device_list(self, devices: list) -> None:
        """Update the device list with new devices.

        Args:
            devices: List of DeviceMetadata objects
        """
        logger.debug(f"Updating device list with {len(devices)} devices")

        self._device_list.clear()

        for device in devices:
            if not isinstance(device, DeviceMetadata):
                logger.warning(f"Invalid device object: {device}")
                continue

            # Create list item
            item = QListWidgetItem(device.name)
            item.setData(1, device.device_id)  # Store device ID in user role

            # Set tooltip with device metadata
            tooltip = self._format_device_tooltip(device)
            item.setToolTip(tooltip)

            self._device_list.addItem(item)

    def _format_device_tooltip(self, metadata: DeviceMetadata) -> str:
        """Format device metadata as tooltip text.

        Args:
            metadata: DeviceMetadata object

        Returns:
            Formatted tooltip string
        """
        lines = [metadata.name]

        if metadata.device_type.value != "audio":
            # Video device metadata
            if metadata.current_resolution:
                lines.append(f"Resolution: {metadata.current_resolution}")
            if metadata.current_fps:
                lines.append(f"FPS: {metadata.current_fps}")
            if metadata.native_fps:
                lines.append(f"Native FPS: {metadata.native_fps}")
            if metadata.supported_resolutions:
                res_count = len(metadata.supported_resolutions)
                lines.append(f"Supported Resolutions: {res_count}")

        return "\n".join(lines)

    def get_current_source_type(self) -> str:
        """Get the currently selected source type.

        Returns:
            Source type string ("screen", "uvc", or "rtsp")
        """
        return self._source_type_combo.currentData()

    def get_window_capture_mode(self) -> str:
        """Get the current window capture mode.

        Returns:
            Mode string ("full_screen", "active_window", or "selected_windows")
        """
        if self._full_screen_radio.isChecked():
            return "full_screen"
        elif self._active_window_radio.isChecked():
            return "active_window"
        elif self._selected_windows_radio.isChecked():
            return "selected_windows"
        return "full_screen"

    def is_window_detection_enabled(self) -> bool:
        """Get the current window detection state.

        Returns:
            True if window detection is enabled, False otherwise
        """
        # Window detection is enabled for active_window and selected_windows modes
        return not self._full_screen_radio.isChecked()

    def clear_selection(self) -> None:
        """Clear all device selections in the device list."""
        self._device_list.clearSelection()
        logger.debug("Cleared device list selection")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the widget.

        Args:
            enabled: Whether to enable the widget
        """
        self._source_type_combo.setEnabled(enabled)
        self._device_list.setEnabled(enabled)
        self._refresh_button.setEnabled(enabled)
