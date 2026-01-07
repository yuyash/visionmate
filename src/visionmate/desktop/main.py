"""
Main window for Visionmate desktop application (refactored).

This module provides the main application window with delegated control
and preview management.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.manager import CaptureManager
from visionmate.desktop.dialogs import AboutDialog
from visionmate.desktop.widgets import ControlContainer, PreviewContainer

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

        # Capture manager
        self._capture_manager = CaptureManager()

        # Control and preview containers
        self._control_container: Optional[ControlContainer] = None
        self._preview_container: Optional[PreviewContainer] = None
        self._drawer_button: Optional[QPushButton] = None

        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Wire up signals
        self._connect_signals()

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
        view_placeholder = QAction("(View options coming soon)", self)
        view_placeholder.setEnabled(False)
        view_menu.addAction(view_placeholder)

        # Session menu
        session_menu = menu_bar.addMenu("&Session")
        session_placeholder = QAction("(Session controls coming soon)", self)
        session_placeholder.setEnabled(False)
        session_menu.addAction(session_placeholder)

        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")
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

        # Create control container
        self._control_container = ControlContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        main_layout.addWidget(self._control_container)

        # Create drawer toggle button container
        drawer_container = QWidget()
        drawer_layout = QVBoxLayout(drawer_container)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)
        drawer_layout.addStretch()

        self._drawer_button = QPushButton("◀")
        self._drawer_button.setFixedSize(10, 72)
        self._drawer_button.setToolTip("Toggle control panel")
        self._drawer_button.clicked.connect(self._toggle_control_panel)
        self._drawer_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 4px;
                font-size: 10px;
                padding: 0px;
            }
            """
        )
        drawer_layout.addWidget(self._drawer_button, alignment=Qt.AlignmentFlag.AlignCenter)
        drawer_layout.addStretch()

        main_layout.addWidget(drawer_container)

        # Create preview container
        self._preview_container = PreviewContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        main_layout.addWidget(self._preview_container, stretch=1)

        logger.debug("Central widget setup complete")

    def _toggle_control_panel(self) -> None:
        """Toggle the visibility of the control panel."""
        if not self._control_container or not self._drawer_button:
            return

        if self._control_container.isVisible():
            self._control_container.hide()
            self._drawer_button.setText("▶")
            logger.debug("Control panel hidden")
        else:
            self._control_container.show()
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
        if self._control_container:
            # Connect control container signals
            self._control_container.source_type_changed.connect(self._on_source_type_changed)
            self._control_container.refresh_requested.connect(self._on_refresh_requested)
            self._control_container.device_selected.connect(self._on_device_selected)
            self._control_container.selection_changed.connect(self._on_selection_changed)
            self._control_container.window_capture_mode_changed.connect(
                self._on_window_capture_mode_changed
            )

        if self._preview_container is not None:
            # Connect preview container signals
            self._preview_container.preview_close_requested.connect(
                self._on_preview_close_requested
            )
            self._preview_container.preview_info_requested.connect(self._on_preview_info_requested)

        logger.debug("Signals connected")

    def _on_source_type_changed(self, source_type: str) -> None:
        """Handle source type change.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
        """
        logger.debug(f"Source type changed to: {source_type}")
        # Status update handled by control container

    def _on_refresh_requested(self, source_type: str) -> None:
        """Handle device refresh request.

        Args:
            source_type: Type of source to refresh
        """
        logger.info(f"Devices refreshed for source type: {source_type}")

        # Get device count from control container cache
        if self._control_container:
            device_cache = self._control_container.get_device_cache()
            devices = device_cache.get(source_type, [])
            self.statusBar().showMessage(f"Found {len(devices)} {source_type} device(s)", 3000)

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
            metadata = self._capture_manager.get_device_metadata(device_id)
            self.statusBar().showMessage(f"Selected: {metadata.name}", 3000)

            # Get window capture mode from control container
            window_capture_mode = (
                self._control_container.get_window_capture_mode()
                if self._control_container
                else "full_screen"
            )

            logger.debug(f"Window capture mode: {window_capture_mode}")
            logger.debug(f"Preview container exists: {self._preview_container is not None}")

            # Start capture and preview via preview container
            if self._preview_container is not None:
                logger.debug("Calling start_capture_and_preview...")
                self._preview_container.start_capture_and_preview(
                    source_type=source_type,
                    device_id=device_id,
                    fps=1,
                    window_capture_mode=window_capture_mode,
                )
                logger.debug("start_capture_and_preview returned")
            else:
                logger.error("Preview container is None!")

        except Exception as e:
            logger.error(f"Error handling device selection: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

    def _on_selection_changed(self, selected_device_ids: list[str]) -> None:
        """Handle selection change (for multiple selection).

        Args:
            selected_device_ids: List of selected device IDs
        """
        if self._preview_container is None:
            return

        # Delegate to preview container
        count = self._preview_container.handle_selection_change(selected_device_ids)

        # Update status bar
        if count == 0:
            self.statusBar().showMessage("No devices selected", 2000)
        elif count == 1:
            pass  # Status already updated by _on_device_selected
        else:
            self.statusBar().showMessage(f"{count} devices selected", 3000)

    def _on_window_capture_mode_changed(self, mode: str, selected_titles: list[str]) -> None:
        """Handle window capture mode change.

        Args:
            mode: Capture mode
            selected_titles: List of selected window titles
        """
        if self._preview_container is None:
            return

        # Delegate to preview container
        action, message = self._preview_container.handle_window_capture_mode_change(
            mode, selected_titles
        )

        # Handle action
        if action == "show_selector":
            self._show_window_selector()
        elif message:
            self.statusBar().showMessage(message, 3000)

    def _show_window_selector(self) -> None:
        """Show window selector dialog for selected windows mode."""
        if self._preview_container is None:
            return

        # Delegate dialog display to preview container
        selected_titles = self._preview_container.show_window_selector_dialog(self)

        if selected_titles is None:
            # Dialog cancelled or error
            if not self._preview_container.get_selected_screen_devices():
                self.statusBar().showMessage("Please select a screen device first", 3000)
            return

        if not selected_titles:
            self.statusBar().showMessage("No windows selected", 3000)
            return

        # Get base device ID
        selected_screen_devices = self._preview_container.get_selected_screen_devices()
        if not selected_screen_devices:
            return

        base_device_id = next(iter(selected_screen_devices))

        # Create window captures via preview container
        success_count = self._preview_container.create_window_captures(
            base_device_id, selected_titles
        )

        if success_count > 0:
            self.statusBar().showMessage(f"Capturing {success_count} selected window(s)", 3000)
        else:
            self.statusBar().showMessage("Failed to create window captures", 3000)

    def _on_preview_close_requested(self, source_id: str) -> None:
        """Handle preview close request.

        Args:
            source_id: Source identifier

        Requirements: 11.9
        """
        logger.info(f"Close requested for preview: {source_id}")

        # Close preview via preview container
        if self._preview_container is not None:
            self._preview_container.close_preview(source_id)

        # Clear selection if no more previews
        if self._capture_manager.get_video_source_count() == 0:
            if self._control_container:
                self._control_container.clear_selection()

        self.statusBar().showMessage(f"Preview closed for {source_id}", 3000)

    def _on_preview_info_requested(self, source_id: str) -> None:
        """Handle preview info request.

        Args:
            source_id: Source identifier

        Requirements: 11.8, 11.10
        """
        if self._preview_container is None:
            return

        # Delegate to preview container
        info_text = self._preview_container.get_device_info_text(source_id)
        self.statusBar().showMessage(info_text, 5000)
