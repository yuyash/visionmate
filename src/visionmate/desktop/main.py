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
from visionmate.desktop.dialogs import AboutDialog, WindowSelectorDialog
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

        # Preview area widgets
        self._preview_area: Optional[QWidget] = None
        self._preview_placeholder: Optional[QLabel] = None
        self._preview_container: Optional[QWidget] = None
        self._preview_container_layout: Optional[QVBoxLayout] = None

        # Active video captures and previews
        # Import here to avoid circular dependency
        from visionmate.core.capture.video import VideoCaptureInterface
        from visionmate.desktop.widgets.video_preview import VideoPreviewWidget

        self._active_captures: dict[str, VideoCaptureInterface] = {}
        self._active_previews: dict[str, VideoPreviewWidget] = {}

        # Track selected screen devices for window selection
        self._selected_screen_devices: set[str] = set()

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

        # Don't populate device lists automatically - wait for user to select source type
        # self._populate_device_lists()

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

        # Create drawer toggle button container
        drawer_container = QWidget()
        drawer_layout = QVBoxLayout(drawer_container)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)

        # Add stretch to center the button vertically
        drawer_layout.addStretch()

        self._drawer_button = QPushButton("◀")
        self._drawer_button.setFixedSize(10, 72)  # Narrow width, tall height
        self._drawer_button.setToolTip("Toggle control panel")
        self._drawer_button.clicked.connect(self._toggle_control_panel)
        self._drawer_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                font-size: 10px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            """
        )

        drawer_layout.addWidget(self._drawer_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add stretch to center the button vertically
        drawer_layout.addStretch()

        main_layout.addWidget(drawer_container)

        # Create preview area
        self._preview_area = self._create_preview_area()
        main_layout.addWidget(self._preview_area, stretch=1)

        logger.debug("Central widget setup complete")

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

    def _create_preview_area(self) -> QWidget:
        """Create the preview area with placeholder.

        Returns:
            Preview area widget

        Requirements: 10.3, 13.6
        """
        # Create preview area container
        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create placeholder label
        self._preview_placeholder = QLabel("Preview Area\n\nSelect a video source to begin")
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setStyleSheet(
            """
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 40px;
            }
            """
        )
        preview_layout.addWidget(self._preview_placeholder)

        # Create container for video previews (initially hidden)
        self._preview_container = QWidget()
        self._preview_container_layout = QVBoxLayout(self._preview_container)
        self._preview_container_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_container_layout.setSpacing(10)
        # Set equal stretch for all items (will be updated when previews are added)
        preview_layout.addWidget(self._preview_container)
        self._preview_container.hide()  # Hidden until sources are added

        logger.debug("Preview area created")
        return preview_area

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
        controls_layout.setContentsMargins(8, 8, 0, 8)
        controls_layout.setSpacing(12)

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

            # Connect window capture mode changed signal
            self._video_input_widget.window_capture_mode_changed.connect(
                self._on_window_capture_mode_changed
            )

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
            source_type: Type of source ("screen", "uvc", "rtsp", or "")
        """
        if not self._video_input_widget:
            return

        # Skip if empty source type
        if not source_type:
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
            source_type: Type of source to refresh ("screen", "uvc", "rtsp", or "")

        Requirements: 9.4, 1.7
        """
        # Skip if empty source type
        if not source_type:
            return

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

        Requirements: 9.4, 1.7, 11.6, 28.9
        """
        logger.info(f"Device selected: {device_id} (type: {source_type})")

        try:
            # Get device metadata
            metadata = self._device_manager.get_device_metadata(device_id)

            # Update status bar
            self.statusBar().showMessage(f"Selected: {metadata.name}", 3000)

            # Start capture and create preview
            self._start_capture_and_preview(source_type, device_id)

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

        # Get currently active device IDs
        current_device_ids = set(self._active_captures.keys())
        selected_device_ids_set = set(selected_device_ids)

        # Find devices that were deselected (in current but not in selected)
        deselected_device_ids = current_device_ids - selected_device_ids_set

        # Close previews for deselected devices
        for device_id in deselected_device_ids:
            logger.info(f"Device deselected, closing preview: {device_id}")
            self._close_preview(device_id)

        # Update status bar
        if count == 0:
            self.statusBar().showMessage("No devices selected", 2000)
        elif count == 1:
            # Single selection - status already updated by _on_device_selected
            pass
        else:
            self.statusBar().showMessage(f"{count} devices selected", 3000)

    def _on_window_detection_changed(self, enabled: bool) -> None:
        """Handle window detection mode change.

        Args:
            enabled: True for active window mode, False for full screen mode
        """
        logger.info(f"Window detection mode changed: {enabled}")

        # Update all active screen captures
        for device_id, capture in self._active_captures.items():
            if device_id.startswith("screen_"):
                capture.set_window_detection(enabled)

        mode = "active window" if enabled else "full screen"
        self.statusBar().showMessage(f"Capture mode: {mode}", 3000)

    def _on_window_capture_mode_changed(self, mode: str, selected_titles: list[str]) -> None:
        """Handle window capture mode change.

        Args:
            mode: Capture mode ("full_screen", "active_window", "selected_windows", "show_selector")
            selected_titles: List of selected window titles (for selected_windows mode)
        """
        logger.info(f"Window capture mode changed: {mode}")

        # If show_selector, open the window selector dialog
        if mode == "show_selector":
            self._show_window_selector()
            return

        # If selected_windows mode, just close existing window previews
        # Don't show dialog until user clicks "Select Windows..." button
        if mode == "selected_windows":
            self._close_all_window_previews()
            self.statusBar().showMessage("Click 'Select Windows...' to choose windows", 3000)
            return

        # For full_screen and active_window modes
        # Close all existing previews first
        all_device_ids = list(self._active_captures.keys())
        for device_id in all_device_ids:
            if device_id.startswith("screen_"):
                logger.info(f"Closing existing preview: {device_id}")
                self._close_preview(device_id)

        # Create new previews for selected screen devices
        from visionmate.core.capture.video import WindowCaptureMode

        if mode == "full_screen":
            capture_mode = WindowCaptureMode.FULL_SCREEN
            enable_window_detection = False
        elif mode == "active_window":
            capture_mode = WindowCaptureMode.ACTIVE_WINDOW
            enable_window_detection = True
        else:
            return

        # Create preview for each selected screen device
        for base_device_id in self._selected_screen_devices:
            try:
                from visionmate.core.capture.video import ScreenCapture

                capture = ScreenCapture(device_manager=self._device_manager)

                # Start capture
                capture.start_capture(
                    device_id=base_device_id, fps=1, enable_window_detection=enable_window_detection
                )
                # Set capture mode
                capture.set_window_capture_mode(capture_mode)

                # Store capture instance
                self._active_captures[base_device_id] = capture

                # Create preview widget
                from visionmate.desktop.widgets import VideoPreviewWidget

                preview = VideoPreviewWidget(
                    source_id=base_device_id, capture=capture, parent=self._preview_container
                )

                # Connect preview signals
                preview.close_requested.connect(self._on_preview_close_requested)
                preview.info_requested.connect(self._on_preview_info_requested)

                # Store preview widget
                self._active_previews[base_device_id] = preview

                # Add preview to container
                if self._preview_container_layout:
                    self._preview_container_layout.addWidget(preview)

                # Show preview container and hide placeholder
                if self._preview_container and self._preview_placeholder:
                    self._preview_container.show()
                    self._preview_placeholder.hide()

                # Update layout
                self._update_preview_layout()

                logger.info(f"Created preview for {base_device_id} in {mode} mode")

            except Exception as e:
                logger.error(f"Error creating preview: {e}", exc_info=True)

        self.statusBar().showMessage(f"Capture mode: {mode.replace('_', ' ')}", 3000)

    def _close_all_window_previews(self) -> None:
        """Close all window-specific previews (keeping base screen previews)."""
        # Find all window-specific device IDs
        window_device_ids = [
            device_id for device_id in list(self._active_captures.keys()) if "_window_" in device_id
        ]

        # Close each window preview
        for device_id in window_device_ids:
            logger.info(f"Closing window preview: {device_id}")
            self._close_preview(device_id)

        # Update layout after closing
        self._update_preview_layout()

    def _update_preview_layout(self) -> None:
        """Update preview layout to distribute space equally among previews."""
        if not self._preview_container_layout:
            return

        # Get number of previews
        preview_count = len(self._active_previews)

        if preview_count == 0:
            return

        # Set equal stretch for all preview widgets
        for i in range(self._preview_container_layout.count()):
            item = self._preview_container_layout.itemAt(i)
            if item and item.widget():
                # Set stretch factor to 1 for equal distribution
                self._preview_container_layout.setStretch(i, 1)

        logger.debug(f"Updated preview layout for {preview_count} previews")

    def _show_window_selector(self) -> None:
        """Show window selector dialog for selected windows mode."""
        # Find base screen device from selected devices
        base_device_id = None

        # Use the first selected screen device
        if self._selected_screen_devices:
            base_device_id = next(iter(self._selected_screen_devices))
        else:
            # Fallback: try to find from active captures
            for device_id in self._active_captures.keys():
                if device_id.startswith("screen_") and "_window_" not in device_id:
                    base_device_id = device_id
                    break

        if not base_device_id:
            logger.warning("No screen device selected for window selection")
            self.statusBar().showMessage("Please select a screen device first", 3000)
            return

        try:
            # Create temporary capture to get available windows
            from visionmate.core.capture.video import ScreenCapture

            temp_capture = ScreenCapture(device_manager=self._device_manager)
            temp_capture.start_capture(device_id=base_device_id, fps=1)

            available_windows = temp_capture.get_available_windows()

            # Stop temporary capture
            temp_capture.stop_capture()

            if not available_windows:
                logger.warning("No windows available for selection")
                self.statusBar().showMessage("No windows found on screen", 3000)
                return

            # Get currently selected window IDs
            current_window_ids = [
                device_id
                for device_id in self._active_captures.keys()
                if device_id.startswith(f"{base_device_id}_window_")
            ]

            current_titles = []
            for wid in current_window_ids:
                cap = self._active_captures[wid]
                if isinstance(cap, ScreenCapture):
                    titles = cap.get_selected_window_titles()
                    if titles:
                        current_titles.append(titles[0])

            dialog = WindowSelectorDialog(available_windows, current_titles, self)

            if dialog.exec():
                # User clicked OK
                selected_titles = dialog.get_selected_titles()

                # Close ALL existing previews (including base screen previews)
                all_device_ids = list(self._active_captures.keys())
                for device_id in all_device_ids:
                    if device_id.startswith("screen_"):
                        logger.info(f"Closing existing preview: {device_id}")
                        self._close_preview(device_id)

                if not selected_titles:
                    logger.warning("No windows selected")
                    self.statusBar().showMessage("No windows selected", 3000)
                    return

                # Create previews for all selected windows
                for title in selected_titles:
                    self._create_window_capture(base_device_id, title)

                logger.info(f"Updated window selection: {len(selected_titles)} windows")
                self.statusBar().showMessage(
                    f"Capturing {len(selected_titles)} selected window(s)", 3000
                )

        except Exception as e:
            logger.error(f"Error showing window selector: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

    def _create_window_capture(self, base_device_id: str, window_title: str) -> None:
        """Create a capture and preview for a specific window.

        Args:
            base_device_id: Base screen device ID (e.g., "screen_1")
            window_title: Title of the window to capture
        """
        try:
            # Create unique device ID for this window
            # Use hash of title to create unique but consistent ID
            window_hash = abs(hash(window_title)) % 10000
            device_id = f"{base_device_id}_window_{window_hash}"

            # Check if already exists
            if device_id in self._active_captures:
                logger.debug(f"Window capture already exists: {device_id}")
                return

            from visionmate.core.capture.video import ScreenCapture, WindowCaptureMode

            # Create new capture instance
            capture = ScreenCapture(device_manager=self._device_manager)

            # Start capture
            capture.start_capture(device_id=base_device_id, fps=1, enable_window_detection=True)

            # Set to selected windows mode with this specific window
            capture.set_window_capture_mode(WindowCaptureMode.SELECTED_WINDOWS, [window_title])

            # Store capture instance
            self._active_captures[device_id] = capture

            # Create preview widget
            from visionmate.desktop.widgets import VideoPreviewWidget

            preview = VideoPreviewWidget(
                source_id=device_id, capture=capture, parent=self._preview_container
            )

            # Connect preview signals
            preview.close_requested.connect(self._on_preview_close_requested)
            preview.info_requested.connect(self._on_preview_info_requested)

            # Store preview widget
            self._active_previews[device_id] = preview

            # Add preview to container
            if self._preview_container_layout:
                self._preview_container_layout.addWidget(preview)

            # Show preview container and hide placeholder
            if self._preview_container and self._preview_placeholder:
                self._preview_container.show()
                self._preview_placeholder.hide()

            # Update layout to distribute space equally
            self._update_preview_layout()

            logger.info(f"Created window capture for: {window_title}")

        except Exception as e:
            logger.error(f"Error creating window capture: {e}", exc_info=True)

    def _start_capture_and_preview(self, source_type: str, device_id: str) -> None:
        """Start capture for a device and create preview widget.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier

        Requirements: 11.6, 28.9
        """
        # Check if already capturing from this device
        if device_id in self._active_captures:
            logger.warning(f"Already capturing from device: {device_id}")
            return

        # Track selected screen devices
        if source_type == "screen":
            self._selected_screen_devices.add(device_id)

        try:
            # Create capture instance based on source type
            if source_type == "screen":
                from visionmate.core.capture.video import ScreenCapture, WindowCaptureMode

                capture = ScreenCapture(device_manager=self._device_manager)

                # Get window capture mode from UI
                mode_str = (
                    self._video_input_widget.get_window_capture_mode()
                    if self._video_input_widget
                    else "full_screen"
                )

                if mode_str == "full_screen":
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False
                elif mode_str == "active_window":
                    capture_mode = WindowCaptureMode.ACTIVE_WINDOW
                    enable_window_detection = True
                elif mode_str == "selected_windows":
                    # For selected windows mode, don't create preview
                    # User will select windows via "Select Windows..." button
                    logger.info("Selected windows mode - waiting for window selection")
                    self.statusBar().showMessage(
                        "Click 'Select Windows...' to choose windows", 3000
                    )
                    return
                else:
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False

            elif source_type == "uvc":
                # UVC capture not yet implemented
                logger.warning("UVC capture not yet implemented")
                self.statusBar().showMessage("UVC capture coming soon", 3000)
                return
            elif source_type == "rtsp":
                # RTSP capture not yet implemented
                logger.warning("RTSP capture not yet implemented")
                self.statusBar().showMessage("RTSP capture coming soon", 3000)
                return
            else:
                logger.error(f"Unknown source type: {source_type}")
                return

            # Start capture with default settings
            # Window detection is configurable for screen capture
            if source_type == "screen":
                capture.start_capture(
                    device_id=device_id, fps=1, enable_window_detection=enable_window_detection
                )
                # Set capture mode after starting
                capture.set_window_capture_mode(capture_mode)
            else:
                capture.start_capture(device_id=device_id, fps=1)

            # Store capture instance
            self._active_captures[device_id] = capture

            # Create preview widget
            from visionmate.desktop.widgets import VideoPreviewWidget

            preview = VideoPreviewWidget(
                source_id=device_id, capture=capture, parent=self._preview_container
            )

            # Connect preview signals
            preview.close_requested.connect(self._on_preview_close_requested)
            preview.info_requested.connect(self._on_preview_info_requested)

            # Store preview widget
            self._active_previews[device_id] = preview

            # Add preview to container
            if self._preview_container_layout:
                self._preview_container_layout.addWidget(preview)

            # Show preview container and hide placeholder
            if self._preview_container and self._preview_placeholder:
                self._preview_container.show()
                self._preview_placeholder.hide()

            # Update layout to distribute space equally
            self._update_preview_layout()

            logger.info(f"Started capture and preview for device: {device_id}")
            self.statusBar().showMessage(f"Preview started for {device_id}", 3000)

        except Exception as e:
            logger.error(f"Error starting capture and preview: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error starting preview: {e}", 5000)

            # Cleanup on error
            if device_id in self._active_captures:
                try:
                    self._active_captures[device_id].stop_capture()
                except Exception:
                    pass
                del self._active_captures[device_id]

    def _on_preview_close_requested(self, source_id: str) -> None:
        """Handle preview close request.

        Args:
            source_id: Source identifier

        Requirements: 11.9
        """
        logger.info(f"Close requested for preview: {source_id}")
        self._close_preview(source_id)

    def _close_preview(self, source_id: str) -> None:
        """Close preview and stop capture for a source.

        Args:
            source_id: Source identifier
        """
        try:
            # Stop capture
            if source_id in self._active_captures:
                capture = self._active_captures[source_id]
                capture.stop_capture()
                del self._active_captures[source_id]

            # Remove and cleanup preview widget
            if source_id in self._active_previews:
                preview = self._active_previews[source_id]
                preview.cleanup()

                # Remove from layout
                if self._preview_container_layout:
                    self._preview_container_layout.removeWidget(preview)

                # Delete widget
                preview.deleteLater()
                del self._active_previews[source_id]

            # If no more previews, show placeholder
            if not self._active_previews:
                if self._preview_container and self._preview_placeholder:
                    self._preview_container.hide()
                    self._preview_placeholder.show()

                # Deselect all devices in the device list
                if self._video_input_widget:
                    self._video_input_widget.clear_selection()
                    logger.debug("Cleared device selection (no more previews)")
            else:
                # Update layout to redistribute space
                self._update_preview_layout()

            # Remove from selected screen devices if it's a base screen device
            if source_id.startswith("screen_") and "_window_" not in source_id:
                self._selected_screen_devices.discard(source_id)

            logger.info(f"Closed preview for source: {source_id}")
            self.statusBar().showMessage(f"Preview closed for {source_id}", 3000)

        except Exception as e:
            logger.error(f"Error closing preview: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error closing preview: {e}", 5000)

    def _on_preview_info_requested(self, source_id: str) -> None:
        """Handle preview info request.

        Args:
            source_id: Source identifier

        Requirements: 11.8, 11.10
        """
        logger.info(f"Info requested for preview: {source_id}")

        try:
            # For window-specific captures, extract base device ID
            if "_window_" in source_id:
                base_device_id = source_id.split("_window_")[0]
                metadata = self._device_manager.get_device_metadata(base_device_id)

                # Get window title from capture
                capture = self._active_captures.get(source_id)
                from visionmate.core.capture.video import ScreenCapture

                if isinstance(capture, ScreenCapture):
                    window_titles = capture.get_selected_window_titles()
                    window_title = window_titles[0] if window_titles else "Unknown"
                    info_text = (
                        f"{metadata.name} - Window: {window_title} - "
                        f"{metadata.current_resolution} @ {metadata.current_fps}fps"
                    )
                else:
                    info_text = f"{metadata.name} - {metadata.current_resolution}"
            else:
                # Regular device
                metadata = self._device_manager.get_device_metadata(source_id)
                info_text = (
                    f"{metadata.name} - {metadata.current_resolution} @ {metadata.current_fps}fps"
                )

            self.statusBar().showMessage(info_text, 5000)

        except Exception as e:
            logger.error(f"Error getting device info: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)
