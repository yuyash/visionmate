"""
Main window for Visionmate desktop application.

This module provides the main application window with control panel,
preview area, and status bar.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.device import DeviceManager
from visionmate.desktop.dialogs import AboutDialog
from visionmate.desktop.widgets import InputModeWidget, VideoInputWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for Visionmate.

    Provides the primary user interface with:
    - Control panel on the left (collapsible)
    - Preview area in the center
    - Status bar at the bottom
    - Menu bar at the top
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.info("Initializing MainWindow")

        self.setWindowTitle("Visionmate")
        self.setMinimumSize(800, 600)

        # Store app info for About dialog
        from visionmate.__main__ import APP_NAME, APP_VERSION

        self._app_name = APP_NAME
        self._app_version = APP_VERSION

        # Control panel widgets
        self._control_panel: Optional[QWidget] = None
        self._drawer_button: Optional[QPushButton] = None
        self._input_mode_widget: Optional[InputModeWidget] = None
        self._video_input_widget: Optional[VideoInputWidget] = None

        # Device manager
        self._device_manager = DeviceManager()

        # Device cache - populated on startup
        self._device_cache: dict[str, list] = {
            "screen": [],
            "uvc": [],
            "rtsp": [],
        }

        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Wire up signals
        self._connect_signals()

        # Scan devices on startup and populate cache
        self._scan_all_devices()

        # Populate device lists from cache
        self._populate_device_lists()

        logger.info("MainWindow initialized successfully")

    def _setup_menu_bar(self) -> None:
        """Setup the menu bar with all menus."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        # Placeholder actions will be added later
        view_placeholder = QAction("(View options coming soon)", self)
        view_placeholder.setEnabled(False)
        view_menu.addAction(view_placeholder)

        # Session menu
        session_menu = menu_bar.addMenu("&Session")
        # Placeholder actions will be added later
        session_placeholder = QAction("(Session controls coming soon)", self)
        session_placeholder.setEnabled(False)
        session_menu.addAction(session_placeholder)

        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")
        # Placeholder actions will be added later
        settings_placeholder = QAction("(Settings coming soon)", self)
        settings_placeholder.setEnabled(False)
        settings_menu.addAction(settings_placeholder)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.setStatusTip("About Visionmate")
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar setup complete")

    def _show_about_dialog(self) -> None:
        """Show the About dialog."""
        logger.debug("Showing About dialog")
        dialog = AboutDialog(self._app_name, self._app_version, self)
        dialog.exec()

    def _setup_central_widget(self) -> None:
        """Setup the central widget with control panel and preview area.

        Requirements: 10.4, 10.5, 10.6
        """
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create control panel (collapsible)
        self._control_panel = self._create_control_panel()
        main_layout.addWidget(self._control_panel)

        # Create drawer toggle button
        drawer_container = QWidget()
        drawer_layout = QVBoxLayout(drawer_container)
        drawer_layout.setContentsMargins(5, 10, 5, 10)
        drawer_layout.setSpacing(0)

        self._drawer_button = QPushButton("◀")
        self._drawer_button.setFixedSize(30, 60)
        self._drawer_button.setToolTip("Toggle control panel")
        self._drawer_button.clicked.connect(self._toggle_control_panel)
        self._drawer_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            """
        )

        drawer_layout.addWidget(self._drawer_button)
        drawer_layout.addStretch()

        main_layout.addWidget(drawer_container)

        # Create placeholder for preview area
        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        placeholder_label = QLabel("Preview Area\n\nSelect a video source to begin")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet(
            """
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 40px;
            }
            """
        )
        preview_layout.addWidget(placeholder_label)

        # Add preview area to main layout
        main_layout.addWidget(preview_area, stretch=1)

        logger.debug("Central widget setup complete")

    def _create_control_panel(self) -> QWidget:
        """Create the control panel with input controls.

        Returns:
            Control panel widget

        Requirements: 10.4, 10.5, 10.6
        """
        # Create control panel container
        panel = QWidget()
        panel.setFixedWidth(320)  # Increased from 300 to prevent clipping
        panel.setStyleSheet(
            """
            QWidget#controlPanel {
                background-color: #f8f8f8;
            }
            """
        )
        panel.setObjectName("controlPanel")

        # Create scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        # Create controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(15, 10, 10, 10)
        controls_layout.setSpacing(15)

        # Add Input Mode widget
        self._input_mode_widget = InputModeWidget()
        controls_layout.addWidget(self._input_mode_widget)

        # Add Video Input widget
        self._video_input_widget = VideoInputWidget()
        controls_layout.addWidget(self._video_input_widget)

        # Add stretch to push controls to top
        controls_layout.addStretch()

        # Set controls widget in scroll area
        scroll_area.setWidget(controls_widget)

        # Create panel layout
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll_area)

        logger.debug("Control panel created")
        return panel

    def _toggle_control_panel(self) -> None:
        """Toggle the visibility of the control panel."""
        if not self._control_panel or not self._drawer_button:
            return

        if self._control_panel.isVisible():
            self._control_panel.hide()
            self._drawer_button.setText("▶")
            logger.debug("Control panel hidden")
        else:
            self._control_panel.show()
            self._drawer_button.setText("◀")
            logger.debug("Control panel shown")

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        status_bar = self.statusBar()
        status_bar.showMessage("Ready")
        logger.debug("Status bar setup complete")

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers.

        Requirements: 9.4, 1.7
        """
        if self._video_input_widget:
            # Connect source type change signal
            self._video_input_widget.source_type_changed.connect(self._on_source_type_changed)

            # Connect refresh signal
            self._video_input_widget.refresh_requested.connect(self._on_refresh_devices)

            # Connect device selection signal
            self._video_input_widget.device_selected.connect(self._on_device_selected)

            # Connect selection changed signal (for multiple selection)
            self._video_input_widget.selection_changed.connect(self._on_selection_changed)

        logger.debug("Signals connected")

    def _scan_all_devices(self) -> None:
        """Scan all device types on startup and populate cache.

        This is called once during initialization to avoid repeated scanning.
        """
        logger.info("Scanning all devices on startup...")

        try:
            # Scan screens
            self._device_cache["screen"] = self._device_manager.enumerate_screens()
            logger.info(f"Cached {len(self._device_cache['screen'])} screen device(s)")

            # Scan UVC devices
            self._device_cache["uvc"] = self._device_manager.enumerate_uvc_devices()
            logger.info(f"Cached {len(self._device_cache['uvc'])} UVC device(s)")

            # RTSP doesn't have enumerable devices
            self._device_cache["rtsp"] = []

        except Exception as e:
            logger.error(f"Error scanning devices: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error scanning devices: {e}", 5000)

    def _populate_device_lists(self) -> None:
        """Populate device lists from cache.

        Requirements: 9.4, 1.7
        """
        if not self._video_input_widget:
            return

        # Get current source type
        source_type = self._video_input_widget.get_current_source_type()

        # Populate from cache
        self._update_device_list_from_cache(source_type)

        logger.debug("Device lists populated from cache")

    def _on_source_type_changed(self, source_type: str) -> None:
        """Handle source type change.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
        """
        logger.debug(f"Source type changed to: {source_type}")

        # Update device list from cache (no scanning)
        self._update_device_list_from_cache(source_type)

    def _update_device_list_from_cache(self, source_type: str) -> None:
        """Update device list from cached data.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
        """
        if not self._video_input_widget:
            return

        # Get devices from cache
        devices = self._device_cache.get(source_type, [])

        # Update the device list
        self._video_input_widget.update_device_list(devices)

        logger.debug(f"Updated device list from cache: {len(devices)} {source_type} device(s)")

    def _on_refresh_devices(self, source_type: str) -> None:
        """Handle device refresh request.

        This re-scans devices and updates the cache.

        Args:
            source_type: Type of source to refresh ("screen", "uvc", "rtsp")

        Requirements: 9.4, 1.7
        """
        logger.info(f"Refreshing devices for source type: {source_type}")
        self.statusBar().showMessage(f"Scanning {source_type} devices...", 0)

        try:
            # Re-scan devices based on source type
            if source_type == "screen":
                devices = self._device_manager.enumerate_screens()
            elif source_type == "uvc":
                devices = self._device_manager.enumerate_uvc_devices()
            elif source_type == "rtsp":
                # RTSP doesn't have enumerable devices
                devices = []
            else:
                logger.warning(f"Unknown source type: {source_type}")
                devices = []

            # Update cache
            self._device_cache[source_type] = devices

            # Update the device list
            if self._video_input_widget:
                self._video_input_widget.update_device_list(devices)

            logger.info(f"Refreshed {len(devices)} {source_type} device(s)")
            self.statusBar().showMessage(f"Found {len(devices)} {source_type} device(s)", 3000)

        except Exception as e:
            logger.error(f"Error refreshing device list: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error refreshing devices: {e}", 5000)

    def _on_device_selected(self, source_type: str, device_id: str) -> None:
        """Handle device selection.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier

        Requirements: 9.4, 1.7
        """
        logger.info(f"Device selected: {device_id} (type: {source_type})")

        try:
            # Get device metadata
            metadata = self._device_manager.get_device_metadata(device_id)

            # Update status bar
            self.statusBar().showMessage(f"Selected: {metadata.name}", 3000)

        except Exception as e:
            logger.error(f"Error getting device metadata: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

    def _on_selection_changed(self, selected_device_ids: list[str]) -> None:
        """Handle selection change (for multiple selection).

        Args:
            selected_device_ids: List of selected device IDs
        """
        count = len(selected_device_ids)
        logger.debug(f"Selection changed: {count} device(s) selected")

        if count == 0:
            self.statusBar().showMessage("No devices selected", 2000)
        elif count == 1:
            # Single selection - status already updated by _on_device_selected
            pass
        else:
            self.statusBar().showMessage(f"{count} devices selected", 3000)
