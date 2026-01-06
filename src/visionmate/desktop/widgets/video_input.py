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
        self._source_type_combo.addItem("Screen Capture", "screen")
        self._source_type_combo.addItem("UVC Device", "uvc")
        self._source_type_combo.addItem("RTSP Stream", "rtsp")
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

        # Device list (full width, global stylesheet applied)
        self._device_list = QListWidget()
        self._device_list.setFrameShape(QListWidget.Shape.NoFrame)
        self._device_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        # Global stylesheet handles list widget styling
        self._device_list.itemSelectionChanged.connect(self._on_device_selected)
        group_layout.addWidget(self._device_list)

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

        # Emit signal to notify parent
        self.source_type_changed.emit(source_type)

    def _on_refresh_clicked(self) -> None:
        """Handle refresh button click."""
        source_type = self._source_type_combo.currentData()
        logger.debug(f"Refresh requested for source type: {source_type}")
        self.refresh_requested.emit(source_type)

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

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the widget.

        Args:
            enabled: Whether to enable the widget
        """
        self._source_type_combo.setEnabled(enabled)
        self._device_list.setEnabled(enabled)
        self._refresh_button.setEnabled(enabled)
