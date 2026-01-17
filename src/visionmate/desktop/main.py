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

from visionmate.__main__ import APP_NAME, APP_VERSION
from visionmate.core import AppSettings
from visionmate.core.capture.manager import CaptureManager
from visionmate.core.capture.video import WindowCaptureMode
from visionmate.core.logging import LogConsoleHandler
from visionmate.core.models import VideoSourceType, WindowGeometry
from visionmate.core.session import SessionManager
from visionmate.core.settings import SettingsManager
from visionmate.desktop.dialogs import AboutDialog, LogConsoleDialog
from visionmate.desktop.widgets import (
    ActionContainer,
    ControlContainer,
    PreviewContainer,
)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for Visionmate.

    Provides the primary user interface with:
    - Control panel on the left (collapsible)
    - Preview area in the center
    - Action panel on the right (collapsible)
    - Status bar at the bottom
    - Menu bar at the top
    """

    def __init__(
        self,
        log_console_handler: Optional[LogConsoleHandler] = None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the main window.

        Args:
            log_console_handler: Optional LogConsoleHandler to use for log console.
                                If provided, logs from application startup will be available.
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.info("Initializing MainWindow")

        self._app_name: str = APP_NAME
        self._app_version: str = APP_VERSION

        self.setWindowTitle(f"{self._app_name.capitalize()} v{self._app_version}")
        self.setMinimumSize(1200, 800)

        # Settings manager
        self._settings_manager = SettingsManager()
        self._settings: AppSettings = self._settings_manager.load_settings()
        logger.info("Settings loaded successfully")

        # Create or use provided log console handler
        if log_console_handler is not None:
            # Use the handler that was created at startup (contains all logs)
            self._log_console_handler = log_console_handler
            logger.info("Using provided log console handler (contains startup logs)")
        else:
            # Create new handler and add to logging system
            self._log_console_handler = LogConsoleHandler()
            logging.getLogger().addHandler(self._log_console_handler)
            logger.info("Created new log console handler")

        # Log console dialog (created on demand)
        self._log_console_dialog: Optional[LogConsoleDialog] = None

        # Capture manager
        self._capture_manager = CaptureManager()
        self._session_manager = SessionManager(settings=self._settings)

        # Control and preview containers
        self._control_container: Optional[ControlContainer] = None
        self._preview_container: Optional[PreviewContainer] = None
        self._action_container: Optional[ActionContainer] = None
        self._control_drawer_button: Optional[QPushButton] = None
        self._action_drawer_button: Optional[QPushButton] = None

        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # Wire up signals
        self._connect_signals()

        # Apply loaded settings to components
        self._apply_settings()

        logger.info("MainWindow initialized successfully")

    def _apply_settings(self) -> None:
        """Apply loaded settings to components."""
        logger.info("Applying settings to components")

        try:
            # Apply window geometry if available
            if self._settings.window_geometry:
                geom = self._settings.window_geometry
                self.setGeometry(geom.x, geom.y, geom.width, geom.height)
                logger.debug(
                    f"Applied window geometry: {geom.x}, {geom.y}, {geom.width}x{geom.height}"
                )

            # Apply input mode
            if self._control_container and self._control_container._input_mode_widget:
                self._control_container._input_mode_widget.set_mode(self._settings.input_mode)
                logger.debug(f"Applied input mode: {self._settings.input_mode.value}")

            # Apply audio mode to session manager
            self._session_manager.set_audio_mode(self._settings.audio_mode)
            logger.debug(f"Applied audio mode: {self._settings.audio_mode.value}")

            logger.info("Settings applied successfully")
        except Exception as e:
            logger.error(f"Error applying settings: {e}", exc_info=True)

    def _save_settings(self) -> None:
        """Save current settings to storage."""
        try:
            # Update window geometry
            geom = self.geometry()
            self._settings.window_geometry = WindowGeometry(
                x=geom.x(),
                y=geom.y(),
                width=geom.width(),
                height=geom.height(),
            )

            # Update input mode
            if self._control_container and self._control_container._input_mode_widget:
                self._settings.input_mode = self._control_container._input_mode_widget.get_mode()

            # Update default FPS (currently fixed at 1, but save for future use)
            self._settings.default_fps = 1

            # Note: Preview layout settings will be saved when preview container is enhanced

            # Save to disk
            self._settings_manager.save_settings(self._settings)
            logger.info("Settings saved successfully")

        except Exception as e:
            logger.error(f"Error saving settings: {e}", exc_info=True)

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

        log_console_action = QAction("&Log Console", self)
        log_console_action.setShortcut("Ctrl+L")
        log_console_action.setStatusTip("Open log console")
        log_console_action.triggered.connect(self._show_log_console)
        view_menu.addAction(log_console_action)

        # Session menu
        session_menu = menu_bar.addMenu("&Session")
        session_placeholder = QAction("(Session controls coming soon)", self)
        session_placeholder.setEnabled(False)
        session_menu.addAction(session_placeholder)

        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")

        settings_action = QAction("&Preferences...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self._show_settings_dialog)
        settings_menu.addAction(settings_action)

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

    def _show_log_console(self) -> None:
        """Show the Log Console dialog."""
        logger.debug("Showing Log Console dialog")

        # Create dialog if it doesn't exist
        if self._log_console_dialog is None:
            self._log_console_dialog = LogConsoleDialog(
                log_handler=self._log_console_handler,
                parent=self,
            )

        # Show the dialog (non-modal so user can interact with main window)
        self._log_console_dialog.show()
        self._log_console_dialog.raise_()
        self._log_console_dialog.activateWindow()

    def _show_settings_dialog(self) -> None:
        """Show the Settings dialog."""
        logger.debug("Showing Settings dialog")

        from visionmate.desktop.dialogs import SettingsDialog

        dialog = SettingsDialog(
            settings_manager=self._settings_manager,
            current_fps=self._settings.default_fps,
            parent=self,
        )

        if dialog.exec():
            # Settings were saved, reload them
            self._settings = self._settings_manager.load_settings()
            self._apply_settings()
            self.statusBar().showMessage("Settings saved successfully", 3000)
            logger.info("Settings updated and applied")
        else:
            logger.debug("Settings dialog cancelled")

    def _setup_central_widget(self) -> None:
        """Setup the central widget with control panel, preview area, and action panel."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main horizontal layout (left: control, center: preview, right: action)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left side: Control container
        self._control_container = ControlContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        main_layout.addWidget(self._control_container)

        # Control drawer toggle button container
        control_drawer_container = QWidget()
        control_drawer_layout = QVBoxLayout(control_drawer_container)
        control_drawer_layout.setContentsMargins(0, 0, 0, 0)
        control_drawer_layout.setSpacing(0)
        control_drawer_layout.addStretch()

        self._control_drawer_button = QPushButton("◀")
        self._control_drawer_button.setFixedSize(10, 72)
        self._control_drawer_button.setToolTip("Toggle control panel")
        self._control_drawer_button.clicked.connect(self._toggle_control_panel)
        self._control_drawer_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 4px;
                font-size: 10px;
                padding: 0px;
            }
            """
        )
        control_drawer_layout.addWidget(
            self._control_drawer_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        control_drawer_layout.addStretch()

        main_layout.addWidget(control_drawer_container)

        # Center: Preview container
        self._preview_container = PreviewContainer(
            capture_manager=self._capture_manager,
            parent=self,
        )
        main_layout.addWidget(self._preview_container, stretch=1)

        # Action drawer toggle button container
        action_drawer_container = QWidget()
        action_drawer_layout = QVBoxLayout(action_drawer_container)
        action_drawer_layout.setContentsMargins(0, 0, 0, 0)
        action_drawer_layout.setSpacing(0)
        action_drawer_layout.addStretch()

        self._action_drawer_button = QPushButton("▶")
        self._action_drawer_button.setFixedSize(10, 72)
        self._action_drawer_button.setToolTip("Toggle action panel")
        self._action_drawer_button.clicked.connect(self._toggle_action_panel)
        self._action_drawer_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 4px;
                font-size: 10px;
                padding: 0px;
            }
            """
        )
        action_drawer_layout.addWidget(
            self._action_drawer_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        action_drawer_layout.addStretch()

        main_layout.addWidget(action_drawer_container)

        # Right side: Action container
        self._action_container = ActionContainer(parent=self)
        main_layout.addWidget(self._action_container)

        logger.debug("Central widget setup complete")

    def _toggle_control_panel(self) -> None:
        """Toggle the visibility of the control panel."""
        if not self._control_container or not self._control_drawer_button:
            return

        if self._control_container.isVisible():
            self._control_container.hide()
            self._control_drawer_button.setText("▶")
            logger.debug("Control panel hidden")
        else:
            self._control_container.show()
            self._control_drawer_button.setText("◀")
            logger.debug("Control panel shown")

    def _toggle_action_panel(self) -> None:
        """Toggle the visibility of the action panel."""
        if not self._action_container or not self._action_drawer_button:
            return

        if self._action_container.isVisible():
            self._action_container.hide()
            self._action_drawer_button.setText("◀")
            logger.debug("Action panel hidden")
        else:
            self._action_container.show()
            self._action_drawer_button.setText("▶")
            logger.debug("Action panel shown")

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        status_bar = self.statusBar()
        status_bar.showMessage("Ready")
        logger.debug("Status bar setup complete")

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
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

        if self._action_container is not None:
            # Connect action container signals
            self._action_container.start_requested.connect(self._on_start_requested)
            self._action_container.stop_requested.connect(self._on_stop_requested)
            self._action_container.reset_requested.connect(self._on_reset_requested)
            self._action_container.request_submitted.connect(self._on_request_submitted)
            self._action_container.status_message.connect(self._update_status_bar)

        # Connect session manager callbacks
        self._session_manager.register_callback("state_changed", self._on_session_state_changed)
        self._session_manager.register_callback("error_occurred", self._on_session_error)
        self._session_manager.register_callback("question_detected", self._on_question_detected)
        self._session_manager.register_callback("response_generated", self._on_response_generated)
        self._session_manager.register_callback("session_reset", self._on_session_reset)

        # Setup metrics widget callback
        if self._action_container:
            metrics_widget = self._action_container.get_metrics_widget()
            if metrics_widget:
                metrics_widget.set_metrics_callback(self._get_metrics)

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
        """
        logger.info(f"Device selected: {device_id} (type: {source_type})")

        try:
            source_type_enum = VideoSourceType(source_type)
        except ValueError:
            logger.warning("Unknown source type '%s'", source_type)
            return

        try:
            # Get window capture mode from control container
            window_capture_mode = WindowCaptureMode.FULL_SCREEN
            if self._control_container:
                mode_value = self._control_container.get_window_capture_mode()
                try:
                    window_capture_mode = WindowCaptureMode(mode_value)
                except ValueError:
                    logger.warning(
                        "Unknown window capture mode '%s', defaulting to full_screen", mode_value
                    )

            # Get FPS setting
            fps = self._control_container.get_fps() if self._control_container else 1

            # For Selected Windows mode, get selected window titles
            selected_window_titles: list[str] = []
            if window_capture_mode == WindowCaptureMode.SELECTED_WINDOWS and self._control_container:
                if self._control_container._video_input_widget is not None:
                    selected_window_titles = (
                        self._control_container._video_input_widget.get_selected_windows()
                    )

            # Start capture and preview via preview container
            if self._preview_container is not None:
                # For Selected Windows mode, create window captures
                if window_capture_mode == WindowCaptureMode.SELECTED_WINDOWS and selected_window_titles:
                    self._preview_container.create_window_captures(
                        device_id, selected_window_titles
                    )
                else:
                    self._preview_container.start_video_capture_and_preview(
                        source_type=source_type_enum,
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

            # Show comprehensive settings dialog
            from visionmate.desktop.dialogs import SettingsDialog

            dialog = SettingsDialog(
                settings_manager=self._settings_manager,
                current_fps=current_fps,
                parent=self,
            )

            if dialog.exec():
                # User clicked OK - apply new FPS to this specific capture
                new_fps = dialog.get_fps()
                logger.info(f"Applying new FPS: {new_fps} for source: {source_id}")

                # Update FPS on capture instance
                if hasattr(capture, "set_fps") and callable(getattr(capture, "set_fps", None)):
                    capture.set_fps(new_fps)  # type: ignore[attr-defined]
                    self.statusBar().showMessage(f"FPS updated to {new_fps}", 3000)
                else:
                    logger.warning(f"Capture does not support set_fps: {type(capture)}")
                    self.statusBar().showMessage("FPS update not supported for this source", 3000)

                # Reload settings in case other settings were changed
                self._settings = self._settings_manager.load_settings()
                self._apply_settings()
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

    def _on_request_submitted(self, request_text: str) -> None:
        """Handle request text submission.

        Args:
            request_text: Submitted text instructions
        """
        logger.info(f"Request submitted: {request_text[:50]}...")
        # TODO: Integrate text instructions with session manager
        self.statusBar().showMessage(f"Request received: {request_text[:50]}...", 3000)

    # ========================================================================
    # Session Control Handlers
    # ========================================================================

    def _on_start_requested(self) -> None:
        """Handle Start button click."""
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
        """Handle Stop button click."""
        logger.info("Stop session requested")

        try:
            # Stop the session
            self._session_manager.stop()

            self.statusBar().showMessage("Session stopped", 3000)

        except Exception as e:
            logger.error(f"Failed to stop session: {e}", exc_info=True)
            self.statusBar().showMessage(f"Failed to stop: {e}", 5000)

    def _on_reset_requested(self) -> None:
        """Handle Reset button click."""
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
        """
        from visionmate.core.models import SessionState

        state = data.get("state")
        if not isinstance(state, SessionState):
            return

        logger.info(f"Session state changed: {state.value}")

        # Update session control widget via action container
        if self._action_container:
            session_control = self._action_container.get_session_control_widget()
            if session_control:
                session_control.set_session_state(state)

            # Start/stop metrics auto-refresh based on session state
            metrics_widget = self._action_container.get_metrics_widget()
            if metrics_widget:
                if state == SessionState.ACTIVE:
                    metrics_widget.start_auto_refresh(interval_ms=1000)
                else:
                    metrics_widget.stop_auto_refresh()

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

    def _on_question_detected(self, data: dict) -> None:
        """Handle question detected event.

        Args:
            data: Event data containing "question", "confidence", "timestamp"
        """
        question = data.get("question", "")
        confidence = data.get("confidence", 0.0)

        logger.info(f"Question detected: {question} (confidence: {confidence:.2f})")

        # Update response widget via action container
        if self._action_container:
            response_widget = self._action_container.get_response_widget()
            if response_widget:
                response_widget.set_current_question(question)

        # Update status bar
        self.statusBar().showMessage(f"Question detected: {question[:50]}...", 3000)

    def _on_response_generated(self, data: dict) -> None:
        """Handle response generated event.

        Args:
            data: Event data containing "response" and other fields
        """
        from visionmate.core.recognition import VLMResponse

        response = data.get("response")
        if not isinstance(response, VLMResponse):
            logger.warning("Invalid response object in event data")
            return

        logger.info(
            f"Response generated: answer={response.direct_answer[:50] if response.direct_answer else None}..."
        )

        # Update response widget via action container
        if self._action_container:
            response_widget = self._action_container.get_response_widget()
            if response_widget:
                response_widget.set_current_response(response)

        # Update status bar
        if response.is_partial:
            self.statusBar().showMessage("Receiving response...", 1000)
        else:
            self.statusBar().showMessage("Response received", 3000)

    def _on_session_reset(self, data: dict) -> None:
        """Handle session reset event.

        Args:
            data: Event data (empty for reset)
        """
        logger.info("Session reset - clearing current question and response")

        # Clear current question and response via action container
        if self._action_container:
            response_widget = self._action_container.get_response_widget()
            if response_widget:
                response_widget.clear_current_question()
                response_widget.clear_current_response()

        # Update status bar
        self.statusBar().showMessage("Session reset - ready for new question", 3000)

    def _update_session_control_state(self) -> None:
        """Update session control widget based on device selection."""
        if not self._action_container:
            return

        session_control = self._action_container.get_session_control_widget()
        if not session_control:
            return

        # Check if we have any devices selected
        has_video = self._capture_manager.get_video_source_count() > 0
        has_audio = self._capture_manager.get_audio_source_count() > 0
        has_devices = has_video or has_audio

        session_control.set_has_devices(has_devices)

    def _get_metrics(self):
        """Get current metrics from session manager.

        Returns:
            ManagerMetrics object

        """
        from visionmate.core.models import ManagerMetrics

        try:
            return self._session_manager.get_metrics()
        except Exception as e:
            logger.error(f"Error getting metrics: {e}", exc_info=True)
            # Return empty metrics on error
            return ManagerMetrics()

    def closeEvent(self, event) -> None:
        """Handle window close event.

        Save settings before closing.
        """
        logger.info("Closing main window")

        # Save settings
        self._save_settings()

        # Stop session if active
        from visionmate.core.models import SessionState

        if self._session_manager.get_state() == SessionState.ACTIVE:
            logger.info("Stopping active session before closing")
            self._session_manager.stop()

        # Close log console dialog if open
        if self._log_console_dialog is not None:
            self._log_console_dialog.close()

        # Note: We don't remove the log console handler here because:
        # 1. If it was provided from __main__, it's managed externally
        # 2. If we created it, it will be cleaned up when the app exits
        # This ensures logs continue to be captured until the very end

        # Accept the close event
        event.accept()
        logger.info("Main window closed")
