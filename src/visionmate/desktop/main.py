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
from visionmate.desktop.widgets import ControlContainer, PreviewContainer, SessionControlWidget

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

        # Store app info for About dialog
        from visionmate.__main__ import APP_NAME, APP_VERSION

        self.setWindowTitle(f"{APP_NAME.capitalize()} v{APP_VERSION}")
        self.setMinimumSize(1024, 768)

        self._app_name = APP_NAME
        self._app_version = APP_VERSION

        # Capture manager
        self._capture_manager = CaptureManager()

        # Session manager
        from visionmate.core.session import SessionManager

        self._session_manager = SessionManager()

        # Control and preview containers
        self._control_container: Optional[ControlContainer] = None
        self._preview_container: Optional[PreviewContainer] = None
        self._drawer_button: Optional[QPushButton] = None
        self._session_control_widget: Optional[SessionControlWidget] = None

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

        Requirements: 10.4, 10.5, 10.6, 10.7
        """
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main horizontal layout (left: control panel, right: session + preview)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left side: Control container (full height)
        self._control_container = ControlContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        main_layout.addWidget(self._control_container)

        # Drawer toggle button container
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

        # Right side: Vertical stack of session control and preview
        right_side_widget = QWidget()
        right_side_layout = QVBoxLayout(right_side_widget)
        right_side_layout.setContentsMargins(0, 8, 0, 0)  # Add top margin
        right_side_layout.setSpacing(8)

        # Session control widget at the top
        self._session_control_widget = SessionControlWidget()
        right_side_layout.addWidget(self._session_control_widget)

        # Preview container below (takes remaining space)
        self._preview_container = PreviewContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        right_side_layout.addWidget(self._preview_container, stretch=1)

        main_layout.addWidget(right_side_widget, stretch=1)

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

        Requirements: 9.4, 1.7, 9.1, 9.2, 9.3
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
            self._preview_container.preview_settings_requested.connect(
                self._on_preview_settings_requested
            )
            # Connect status message signal
            self._preview_container.status_message.connect(self._update_status_bar)

        if self._session_control_widget is not None:
            # Connect session control signals
            self._session_control_widget.start_requested.connect(self._on_start_requested)
            self._session_control_widget.stop_requested.connect(self._on_stop_requested)
            self._session_control_widget.reset_requested.connect(self._on_reset_requested)

        # Connect session manager callbacks
        self._session_manager.register_callback("state_changed", self._on_session_state_changed)
        self._session_manager.register_callback("error_occurred", self._on_session_error)

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

            # For Selected Windows mode, get selected window titles
            selected_window_titles: list[str] = []
            if window_capture_mode == "selected_windows" and self._control_container:
                if self._control_container._video_input_widget is not None:
                    selected_window_titles = (
                        self._control_container._video_input_widget.get_selected_windows()
                    )

            # Start capture and preview via preview container
            if self._preview_container is not None:
                # For Selected Windows mode, create window captures
                if window_capture_mode == "selected_windows" and selected_window_titles:
                    self._preview_container.create_window_captures(
                        device_id, selected_window_titles
                    )
                else:
                    self._preview_container.start_capture_and_preview(
                        source_type=source_type,
                        device_id=device_id,
                        fps=fps,
                        window_capture_mode=window_capture_mode,
                    )

            # Update session control state
            self._update_session_control_state()

        except Exception as e:
            logger.error(f"Error handling device selection: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

    def _on_selection_changed(self, selected_device_ids: list[str]) -> None:
        """Handle selection change (for multiple selection).

        Args:
            selected_device_ids: List of selected device IDs
        """
        # Note: With the new "Add to Preview" button workflow, we don't automatically
        # close previews when devices are deselected. Users must explicitly close
        # previews using the close button on each preview.
        # This method is kept for potential future use or status updates.

        # Emit status message for selection count
        count = len(selected_device_ids)
        if count == 0:
            self.statusBar().showMessage("No devices selected", 2000)
        elif count == 1:
            self.statusBar().showMessage("1 device selected", 2000)
        elif count > 1:
            self.statusBar().showMessage(f"{count} devices selected", 2000)

    def _on_window_capture_mode_changed(self, mode: str, selected_titles: list[str]) -> None:
        """Handle window capture mode change.

        Args:
            mode: Capture mode
            selected_titles: List of selected window titles
        """
        # Note: With the new "Add to Preview" button workflow, changing the capture mode
        # only affects NEW previews that will be added. Existing previews are not modified.
        # This ensures that once a preview is added, it remains independent of control changes.

        if mode == "show_selector":
            # Special case: show window selector dialog
            self._show_window_selector()

    def _show_window_selector(self) -> None:
        """Show window selector dialog for selected windows mode."""
        if self._preview_container is None or self._control_container is None:
            return

        # Show window selector and update control container with selected windows
        selected_titles = self._preview_container.show_window_selector(self)

        if selected_titles is not None:
            # Update VideoInputWidget with selected windows
            if self._control_container._video_input_widget is not None:
                self._control_container._video_input_widget.set_selected_windows(selected_titles)

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

        # Update session control state
        self._update_session_control_state()

    def _on_preview_settings_requested(self, source_id: str) -> None:
        """Handle preview settings request.

        Args:
            source_id: Source identifier
        """
        logger.info(f"Settings requested for preview: {source_id}")

        try:
            # Get current capture instance
            capture = self._capture_manager.get_video_source(source_id)
            if not capture:
                logger.warning(f"Capture not found for source: {source_id}")
                return

            # Get current FPS from capture
            current_fps = 1  # Default
            if hasattr(capture, "get_fps") and callable(getattr(capture, "get_fps", None)):
                current_fps = capture.get_fps()  # type: ignore[attr-defined]

            # Show settings dialog
            from visionmate.desktop.dialogs import SettingsDialog

            dialog = SettingsDialog(current_fps=current_fps, parent=self)

            if dialog.exec():
                # User clicked OK - apply new FPS
                new_fps = dialog.get_fps()
                logger.info(f"Applying new FPS: {new_fps} for source: {source_id}")

                # Update FPS on capture instance
                if hasattr(capture, "set_fps") and callable(getattr(capture, "set_fps", None)):
                    capture.set_fps(new_fps)  # type: ignore[attr-defined]
                    self.statusBar().showMessage(f"FPS updated to {new_fps}", 3000)
                else:
                    logger.warning(f"Capture does not support set_fps: {type(capture)}")
                    self.statusBar().showMessage("FPS update not supported for this source", 3000)
            else:
                # User clicked Cancel - do nothing
                logger.debug("Settings dialog cancelled")

        except Exception as e:
            logger.error(f"Error showing settings dialog: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

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

            # Update session control state
            self._update_session_control_state()

        except Exception as e:
            logger.error(f"Error handling audio device selection: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error: {e}", 5000)

    # ========================================================================
    # Session Control Handlers
    # ========================================================================

    def _on_start_requested(self) -> None:
        """Handle Start button click.

        Requirements: 9.1, 9.6
        """
        logger.info("Start session requested")

        try:
            # Add all active video sources to session manager
            for source_id in self._capture_manager.get_video_source_ids():
                capture = self._capture_manager.get_video_source(source_id)
                if capture:
                    # Extract source type and device_id from source_id
                    # Format: "{source_type}_{device_id}"
                    parts = source_id.split("_", 1)
                    if len(parts) == 2:
                        source_type_str, device_id = parts
                        from visionmate.core.models import VideoSourceType

                        try:
                            _source_type = VideoSourceType(source_type_str)
                            # Note: We're not re-adding sources to session manager
                            # because they're already managed by capture_manager
                            # The session manager will coordinate their lifecycle
                        except ValueError:
                            logger.warning(f"Unknown source type: {source_type_str}")

            # Add audio source if configured
            audio_device_id = (
                self._control_container.get_selected_audio_device_id()
                if self._control_container
                else None
            )
            if audio_device_id:
                # Assume device audio for now
                # Note: Similar to video, audio is already managed by capture_manager
                pass

            # Start the session
            self._session_manager.start()

            self.statusBar().showMessage("Session started", 3000)

        except Exception as e:
            logger.error(f"Failed to start session: {e}", exc_info=True)
            self.statusBar().showMessage(f"Failed to start: {e}", 5000)

    def _on_stop_requested(self) -> None:
        """Handle Stop button click.

        Requirements: 9.2, 9.7
        """
        logger.info("Stop session requested")

        try:
            # Stop the session
            self._session_manager.stop()

            self.statusBar().showMessage("Session stopped", 3000)

        except Exception as e:
            logger.error(f"Failed to stop session: {e}", exc_info=True)
            self.statusBar().showMessage(f"Failed to stop: {e}", 5000)

    def _on_reset_requested(self) -> None:
        """Handle Reset button click.

        Requirements: 9.3, 9.8
        """
        logger.info("Reset session requested")

        try:
            # Reset the session
            self._session_manager.reset()

            self.statusBar().showMessage("Session reset", 3000)

        except Exception as e:
            logger.error(f"Failed to reset session: {e}", exc_info=True)
            self.statusBar().showMessage(f"Failed to reset: {e}", 5000)

    def _on_session_state_changed(self, data: dict) -> None:
        """Handle session state change event.

        Args:
            data: Event data containing "state" key

        Requirements: 9.5, 9.9, 9.10
        """
        from visionmate.core.models import SessionState

        state = data.get("state")
        if not isinstance(state, SessionState):
            return

        logger.info(f"Session state changed: {state.value}")

        # Update session control widget
        if self._session_control_widget:
            self._session_control_widget.set_session_state(state)

        # Update control container (enable/disable device selection)
        if self._control_container:
            if state == SessionState.ACTIVE:
                # Disable device selection during active session
                self._control_container.setEnabled(False)
            else:
                # Enable device selection when stopped
                self._control_container.setEnabled(True)

        # Update status bar
        if state == SessionState.ACTIVE:
            self.statusBar().showMessage("Session active - capturing and processing")
        else:
            self.statusBar().showMessage("Session idle")

    def _on_session_error(self, data: dict) -> None:
        """Handle session error event.

        Args:
            data: Event data containing "error" key
        """
        error = data.get("error", "Unknown error")
        logger.error(f"Session error: {error}")
        self.statusBar().showMessage(f"Session error: {error}", 5000)

    def _update_session_control_state(self) -> None:
        """Update session control widget based on device selection.

        Requirements: 9.5
        """
        if not self._session_control_widget:
            return

        # Check if we have any devices selected
        has_video = self._capture_manager.get_video_source_count() > 0
        has_audio = self._capture_manager.get_audio_source_count() > 0
        has_devices = has_video or has_audio

        self._session_control_widget.set_has_devices(has_devices)
