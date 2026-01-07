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
            self._control_container.device_selected.connect(self._on_device_selected)
            self._control_container.selection_changed.connect(self._on_selection_changed)
            self._control_container.window_capture_mode_changed.connect(
                self._on_window_capture_mode_changed
            )
            self._control_container.audio_device_selected.connect(self._on_audio_device_selected)
            # Connect status message signal
            self._control_container.status_message.connect(self._update_status_bar)

            # Connect input mode change signal
            if self._control_container._input_mode_widget is not None:
                self._control_container._input_mode_widget.mode_changed.connect(
                    self._on_input_mode_changed
                )

        if self._preview_container is not None:
            # Connect preview container signals
            self._preview_container.preview_close_requested.connect(
                self._on_preview_close_requested
            )
            # Connect status message signal
            self._preview_container.status_message.connect(self._update_status_bar)

        logger.debug("Signals connected")

    def _update_status_bar(self, message: str, timeout: int = 0) -> None:
        """Update status bar with a message.

        Args:
            message: Status message to display
            timeout: Timeout in milliseconds (0 = permanent)
        """
        self.statusBar().showMessage(message, timeout)

    def _on_source_type_changed(self, source_type: str) -> None:
        """Handle source type change.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
        """
        logger.debug(f"Source type changed to: {source_type}")

    def _on_device_selected(self, source_type: str, device_id: str) -> None:
        """Handle device selection.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier

        Requirements: 9.4, 1.7, 11.6, 28.9
        """
        logger.info(f"Device selected: {device_id} (type: {source_type})")

        try:
            # Get window capture mode from control container
            window_capture_mode = (
                self._control_container.get_window_capture_mode()
                if self._control_container
                else "full_screen"
            )

            # Get FPS setting
            fps = self._control_container.get_fps() if self._control_container else 1

            # Start capture and preview via preview container
            if self._preview_container is not None:
                self._preview_container.start_capture_and_preview(
                    source_type=source_type,
                    device_id=device_id,
                    fps=fps,
                    window_capture_mode=window_capture_mode,
                )

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
        self._preview_container.handle_selection_change(selected_device_ids)

    def _on_window_capture_mode_changed(self, mode: str, selected_titles: list[str]) -> None:
        """Handle window capture mode change.

        Args:
            mode: Capture mode
            selected_titles: List of selected window titles
        """
        if self._preview_container is None:
            return

        # Delegate to preview container
        action = self._preview_container.handle_window_capture_mode_change(mode, selected_titles)

        # Handle action
        if action == "show_selector":
            self._show_window_selector()

    def _show_window_selector(self) -> None:
        """Show window selector dialog for selected windows mode."""
        if self._preview_container is None:
            return

        # Delegate to preview container
        self._preview_container.show_window_selector_and_create_captures(self)

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

    def _on_input_mode_changed(self, mode) -> None:
        """Handle input mode change.

        Args:
            mode: InputMode enum value

        Requirements: 10.6
        """
        from visionmate.core.models import InputMode

        logger.info(f"Input mode changed to: {mode.value}")

        if self._preview_container is None:
            return

        # Stop and remove previews based on mode
        if mode == InputMode.VIDEO_ONLY:
            # Remove audio previews
            self._preview_container.remove_audio_previews()
            logger.debug("Removed audio previews for VIDEO_ONLY mode")
        elif mode == InputMode.AUDIO_ONLY:
            # Remove video previews
            self._preview_container.remove_video_previews()
            logger.debug("Removed video previews for AUDIO_ONLY mode")
        # VIDEO_AUDIO mode: keep all previews

    def _on_audio_device_selected(self, device_id: str) -> None:
        """Handle audio device selection.

        Args:
            device_id: Audio device identifier

        Requirements: 12.1
        """
        logger.info(f"Audio device selected: {device_id}")

        try:
            # Start audio capture and preview via preview container
            if self._preview_container is not None:
                self._preview_container.start_audio_capture_and_preview(device_id)

        except Exception as e:
            logger.error(f"Error handling audio device selection: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)
