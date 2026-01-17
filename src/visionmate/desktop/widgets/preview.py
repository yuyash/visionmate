"""Preview container widget for managing multiple video previews.

This module provides the PreviewContainer widget that supports multiple
layout modes (horizontal, vertical, grid) and drag-and-drop reordering.
It also manages video source lifecycle and preview widgets.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal

from visionmate.core.models import VideoSourceType

if TYPE_CHECKING:
    from visionmate.desktop.widgets import AudioPreviewWidget, VideoPreviewWidget
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.manager import CaptureManager
from visionmate.core.capture.video import ScreenCapture, VideoCaptureInterface, WindowCaptureMode

logger = logging.getLogger(__name__)


class PreviewLayout(Enum):
    """Preview layout modes."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


class PreviewContainer(QWidget):
    """Container for multiple video previews with flexible layouts.

    Supports:
    - Horizontal layout (side-by-side)
    - Vertical layout (stacked)
    - Grid layout (automatic grid arrangement)
    - Drag-and-drop reordering (future implementation)

    """

    # Signal emitted when layout mode changes
    layout_changed = Signal(str)  # layout mode

    # Signal emitted when preview close is requested
    preview_close_requested = Signal(str)  # source_id

    # Signal emitted when preview info is requested
    preview_info_requested = Signal(str)  # source_id

    # Signal emitted when preview settings is requested
    preview_settings_requested = Signal(str)  # source_id

    # Signal emitted for status bar updates
    status_message = Signal(str, int)  # message, timeout_ms

    def __init__(
        self,
        capture_manager: CaptureManager,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the PreviewContainer.

        Args:
            capture_manager: CaptureManager instance
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing PreviewContainer")

        self._capture_manager = capture_manager
        self._layout_mode = PreviewLayout.VERTICAL
        self._video_previews: dict[str, "VideoPreviewWidget"] = {}  # type: ignore
        self._audio_previews: dict[str, "AudioPreviewWidget"] = {}  # type: ignore
        self._placeholder: Optional[QLabel] = None

        # Separate containers for video and audio (initialized in _setup_ui)
        self._video_container: QWidget
        self._audio_container: QWidget
        self._video_layout: QVBoxLayout
        self._audio_layout: QVBoxLayout

        # Track selected screen devices for window selection
        self._selected_screen_devices: set[str] = set()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout (vertical: video on top, audio on bottom)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(10)

        # Create video container
        self._video_container = QWidget()
        self._video_layout = QVBoxLayout(self._video_container)
        self._video_layout.setContentsMargins(0, 0, 0, 0)
        self._video_layout.setSpacing(10)

        # Create audio container
        self._audio_container = QWidget()
        self._audio_layout = QVBoxLayout(self._audio_container)
        self._audio_layout.setContentsMargins(0, 0, 0, 0)
        self._audio_layout.setSpacing(10)

        # Add containers to main layout
        self._main_layout.addWidget(self._video_container, stretch=3)  # Video gets more space
        self._main_layout.addWidget(self._audio_container, stretch=1)  # Audio gets less space

        # Create placeholder
        self._placeholder = QLabel("No video sources\n\nSelect a video source to begin")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            """
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 40px;
            }
            """
        )
        self._video_layout.addWidget(self._placeholder)

        # Initially hide audio container
        self._audio_container.hide()

        logger.debug("PreviewContainer UI setup complete")

    def add_preview(self, source_id: str, capture: VideoCaptureInterface) -> None:
        """Add a video preview widget for a video source.

        Args:
            source_id: Unique identifier for the source
            capture: VideoCaptureInterface instance

        """
        if source_id in self._video_previews:
            logger.warning(f"Video preview already exists for source: {source_id}")
            return

        # Import here to avoid circular dependency
        from visionmate.desktop.widgets.video import VideoPreviewWidget

        # Create preview widget
        preview = VideoPreviewWidget(
            source_id=source_id,
            capture=capture,
            parent=self,
        )

        # Connect preview signals
        preview.close_requested.connect(self._on_preview_close_requested)
        preview.settings_requested.connect(self._on_preview_settings_requested)

        # Store preview
        self._video_previews[source_id] = preview

        # Hide placeholder if this is the first preview
        if len(self._video_previews) == 1 and self._placeholder:
            self._placeholder.hide()

        # Add to video layout
        self._video_layout.addWidget(preview)

        # Show video container
        self._video_container.show()

        # Update container visibility
        self.update_container_visibility()

        logger.info(
            f"Added video preview for source: {source_id} "
            f"(total: {len(self._video_previews)} video, {len(self._audio_previews)} audio)"
        )

        # Emit status message
        try:
            metadata = self._capture_manager.get_device_metadata(source_id)
            self.status_message.emit(f"Preview added: {metadata.name}", 3000)
        except Exception:
            pass

    def add_audio_preview(self, source_id: str, capture) -> None:
        """Add an audio preview widget for an audio source.

        Args:
            source_id: Unique identifier for the source
            capture: AudioCaptureInterface instance

        """
        if source_id in self._audio_previews:
            logger.warning(f"Audio preview already exists for source: {source_id}")
            return

        # Import here to avoid circular dependency
        from visionmate.desktop.widgets.audio import AudioPreviewWidget

        # Create preview widget
        preview = AudioPreviewWidget(
            source_id=source_id,
            capture=capture,
            parent=self,
        )

        # Connect preview signals
        preview.close_requested.connect(self._on_preview_close_requested)

        # Store preview
        self._audio_previews[source_id] = preview

        # Add to audio layout
        self._audio_layout.addWidget(preview)

        # Show audio container
        self._audio_container.show()

        # Update container visibility
        self.update_container_visibility()

        logger.info(
            f"Added audio preview for source: {source_id} "
            f"(total: {len(self._video_previews)} video, {len(self._audio_previews)} audio)"
        )

        # Emit status message
        try:
            metadata = self._capture_manager.get_device_metadata(source_id)
            self.status_message.emit(f"Audio preview added: {metadata.name}", 3000)
        except Exception:
            pass

    def remove_preview(self, source_id: str) -> None:
        """Remove a preview widget (video or audio).

        Args:
            source_id: Source identifier

        """
        # Try video preview first
        if source_id in self._video_previews:
            preview = self._video_previews[source_id]

            # Cleanup preview
            preview.cleanup()

            # Remove from layout
            self._video_layout.removeWidget(preview)

            # Delete widget
            preview.deleteLater()

            # Remove from dict
            del self._video_previews[source_id]

            # Hide video container if no more video previews
            if len(self._video_previews) == 0:
                self._video_container.hide()
                # Show placeholder if no audio previews either
                if len(self._audio_previews) == 0 and self._placeholder:
                    self._placeholder.show()

            # Update container visibility
            self.update_container_visibility()

            logger.info(
                f"Removed video preview for source: {source_id} "
                f"(remaining: {len(self._video_previews)} video, {len(self._audio_previews)} audio)"
            )

        # Try audio preview
        elif source_id in self._audio_previews:
            preview = self._audio_previews[source_id]

            # Cleanup preview
            preview.cleanup()

            # Remove from layout
            self._audio_layout.removeWidget(preview)

            # Delete widget
            preview.deleteLater()

            # Remove from dict
            del self._audio_previews[source_id]

            # Hide audio container if no more audio previews
            if len(self._audio_previews) == 0:
                self._audio_container.hide()

            # Update container visibility
            self.update_container_visibility()

            logger.info(
                f"Removed audio preview for source: {source_id} "
                f"(remaining: {len(self._video_previews)} video, {len(self._audio_previews)} audio)"
            )

        else:
            logger.warning(f"Preview not found for source: {source_id}")
            return

        # Emit status message
        self.status_message.emit(f"Preview closed for {source_id}", 3000)

    def clear_previews(self) -> None:
        """Remove all preview widgets from the container."""
        # Remove all video previews
        for source_id in list(self._video_previews.keys()):
            self.remove_preview(source_id)

        # Remove all audio previews
        for source_id in list(self._audio_previews.keys()):
            self.remove_preview(source_id)

        logger.info("Cleared all previews from container")

    def _on_preview_close_requested(self, source_id: str) -> None:
        """Handle preview close request.

        Args:
            source_id: Source identifier
        """
        logger.info(f"Close requested for preview: {source_id}")
        self.preview_close_requested.emit(source_id)

    def _on_preview_settings_requested(self, source_id: str) -> None:
        """Handle preview settings request.

        Args:
            source_id: Source identifier
        """
        logger.info(f"Settings requested for preview: {source_id}")
        self.preview_settings_requested.emit(source_id)

    def get_preview_count(self) -> int:
        """Get the number of previews in the container.

        Returns:
            Number of previews (video + audio)
        """
        return len(self._video_previews) + len(self._audio_previews)

    def set_layout_mode(self, mode: PreviewLayout) -> None:
        """Set the layout mode for previews.

        Note: With separated video/audio containers, layout mode is less relevant.
        This method is kept for compatibility but doesn't change the layout structure.

        Args:
            mode: Layout mode (HORIZONTAL, VERTICAL, GRID)

        """
        if self._layout_mode == mode:
            return

        logger.info(f"Layout mode changed from {self._layout_mode.value} to {mode.value}")
        self._layout_mode = mode

        # Emit signal
        self.layout_changed.emit(mode.value)

    def get_layout_mode(self) -> PreviewLayout:
        """Get the current layout mode.

        Returns:
            Current layout mode
        """
        return self._layout_mode

    def enable_drag_drop(self, enabled: bool = True) -> None:
        """Enable or disable drag-and-drop reordering.

        Args:
            enabled: True to enable drag-and-drop, False to disable


        Note: This is a placeholder for future implementation.
        Drag-and-drop reordering requires implementing custom drag/drop
        event handlers and preview reordering logic.
        """
        if enabled:
            logger.info("Drag-and-drop reordering requested (not yet implemented)")
            # TODO: Implement drag-and-drop reordering
            # - Set accept drops on container
            # - Implement dragEnterEvent, dragMoveEvent, dropEvent
            # - Track drag source and drop target
            # - Reorder previews list and rebuild layout
        else:
            logger.debug("Drag-and-drop reordering disabled")

    def __len__(self) -> int:
        """Get the number of previews in the container.

        Returns:
            Number of previews (video + audio)
        """
        return len(self._video_previews) + len(self._audio_previews)

    def __repr__(self) -> str:
        """Get string representation of the container.

        Returns:
            String representation
        """
        return (
            f"PreviewContainer(mode={self._layout_mode.value}, "
            f"video={len(self._video_previews)}, audio={len(self._audio_previews)})"
        )

    def start_video_capture_and_preview(
        self,
        source_type: VideoSourceType,
        device_id: str,
        fps: int = 1,
        window_capture_mode: WindowCaptureMode = WindowCaptureMode.FULL_SCREEN,
    ) -> None:
        """Start capture for a device and create preview.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier
            fps: Frame rate (default: 1)
            window_capture_mode: Window capture mode (default: WindowCaptureMode.FULL_SCREEN)

        """
        if isinstance(window_capture_mode, str):
            try:
                window_capture_mode = WindowCaptureMode(window_capture_mode)
            except ValueError:
                logger.warning(
                    "Unknown window capture mode '%s', defaulting to full_screen",
                    window_capture_mode,
                )
                window_capture_mode = WindowCaptureMode.FULL_SCREEN

        logger.debug(
            f"start_video_capture_and_preview called: source_type={source_type}, "
            f"device_id={device_id}, mode={window_capture_mode.value}"
        )

        # For screen capture, create a unique device ID based on capture mode
        # This allows multiple previews of the same screen with different modes
        if source_type == VideoSourceType.SCREEN:
            if window_capture_mode == WindowCaptureMode.FULL_SCREEN:
                unique_device_id = f"{device_id}_fullscreen"
            elif window_capture_mode == WindowCaptureMode.ACTIVE_WINDOW:
                unique_device_id = f"{device_id}_activewindow"
            elif window_capture_mode == WindowCaptureMode.SELECTED_WINDOWS:
                # For selected windows mode, don't create preview yet
                logger.info("Selected windows mode - waiting for window selection")
                return
            else:
                unique_device_id = f"{device_id}_fullscreen"
        else:
            unique_device_id = device_id

        # Check if already capturing with this exact configuration
        if unique_device_id in self._capture_manager:
            logger.warning(f"Already capturing from device with this mode: {unique_device_id}")
            return

        # Track selected screen devices (use base device_id)
        if source_type == VideoSourceType.SCREEN:
            self._selected_screen_devices.add(device_id)

        try:
            # Create capture instance based on source type
            if source_type == VideoSourceType.SCREEN:
                capture = ScreenCapture(device_manager=self._capture_manager.get_device_manager())

                # Determine capture mode
                if window_capture_mode == WindowCaptureMode.FULL_SCREEN:
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False
                elif window_capture_mode == WindowCaptureMode.ACTIVE_WINDOW:
                    capture_mode = WindowCaptureMode.ACTIVE_WINDOW
                    enable_window_detection = True
                else:
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False

                # Start capture with base device_id
                capture.start_capture(
                    device_id=device_id,
                    fps=fps,
                    enable_window_detection=enable_window_detection,
                )
                capture.set_window_capture_mode(capture_mode)

            elif source_type == VideoSourceType.UVC:
                logger.warning("UVC capture not yet implemented")
                return
            elif source_type == VideoSourceType.RTSP:
                logger.warning("RTSP capture not yet implemented")
                return
            else:
                logger.error(f"Unknown source type: {source_type}")
                return

            # Add to video source manager with unique device ID
            self._capture_manager.add_video_source(unique_device_id, capture)

            # Create and add preview with unique device ID
            self.add_preview(unique_device_id, capture)

            logger.info(f"Started capture and preview for device: {unique_device_id}")

            # Emit status message for device selection
            try:
                metadata = self._capture_manager.get_device_metadata(device_id)
                mode_text = window_capture_mode.value.replace("_", " ").title()
                self.status_message.emit(f"Added: {metadata.name} ({mode_text})", 3000)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error starting capture and preview: {e}", exc_info=True)

            # Cleanup on error
            if unique_device_id in self._capture_manager:
                try:
                    capture = self._capture_manager.get_video_source(unique_device_id)
                    if capture:
                        capture.stop_capture()
                    self._capture_manager.remove_video_source(unique_device_id)
                except Exception:
                    pass

    def close_preview(self, source_id: str, keep_selection: bool = False) -> None:
        """Close preview and stop capture for a source (video or audio).

        Args:
            source_id: Source identifier
            keep_selection: If True, keep device in selected_screen_devices (for mode changes)
        """
        try:
            # Check if it's a video source
            if source_id in self._capture_manager.get_video_source_ids():
                # Stop video capture
                capture = self._capture_manager.get_video_source(source_id)
                if capture:
                    capture.stop_capture()
                self._capture_manager.remove_video_source(source_id)

                # Remove preview
                self.remove_preview(source_id)

                # Remove from selected screen devices if it's a base screen device
                # Only remove if not keeping selection (e.g., user explicitly closed, not mode change)
                if not keep_selection:
                    if source_id.startswith("screen_") and "_window_" not in source_id:
                        self._selected_screen_devices.discard(source_id)

                logger.info(
                    f"Closed video preview for source: {source_id} (keep_selection={keep_selection})"
                )

            # Check if it's an audio source
            elif source_id in self._capture_manager.get_audio_source_ids():
                self.close_audio_preview(source_id)

            else:
                logger.warning(f"Source not found: {source_id}")

        except Exception as e:
            logger.error(f"Error closing preview: {e}", exc_info=True)

    def get_selected_screen_devices(self) -> set[str]:
        """Get selected screen devices.

        Returns:
            Set of selected screen device IDs
        """
        return self._selected_screen_devices.copy()

    def handle_selection_change(self, selected_device_ids: list[str]) -> None:
        """Handle device selection change.

        Closes previews for deselected devices.

        Args:
            selected_device_ids: List of selected device IDs
        """
        # Get currently active device IDs
        current_device_ids = set(self._capture_manager.get_video_source_ids())
        selected_device_ids_set = set(selected_device_ids)

        # Find devices that were deselected
        deselected_device_ids = current_device_ids - selected_device_ids_set

        # Close previews for deselected devices
        for device_id in deselected_device_ids:
            logger.info(f"Device deselected, closing preview: {device_id}")
            self.close_preview(device_id)

        # Emit status message
        count = len(selected_device_ids)
        if count == 0:
            self.status_message.emit("No devices selected", 2000)
        elif count > 1:
            self.status_message.emit(f"{count} devices selected", 3000)

    def _close_all_window_previews(self) -> None:
        """Close all window-specific previews."""
        window_device_ids = [
            device_id
            for device_id in self._capture_manager.get_video_source_ids()
            if "_window_" in device_id
        ]

        for device_id in window_device_ids:
            logger.info(f"Closing window preview: {device_id}")
            self.close_preview(device_id)

    def show_window_selector(self, parent: QWidget) -> Optional[list[str]]:
        """Show window selector dialog and return selected window titles.

        Args:
            parent: Parent widget for the dialog

        Returns:
            List of selected window titles, or None if cancelled
        """
        # Get base screen device from MainWindow's control container
        # Import here to avoid circular dependency
        from visionmate.desktop.main import MainWindow

        if isinstance(parent, MainWindow):
            if parent._control_container and parent._control_container._video_input_widget:
                video_widget = parent._control_container._video_input_widget
                selected_items = video_widget._device_list.selectedItems()

                if not selected_items:
                    logger.warning("No screen device selected for window selection")
                    self.status_message.emit("Please select a screen device first", 3000)
                    return None

                # Get the first selected device
                base_device_id = selected_items[0].data(1)
            else:
                logger.warning("Control container or video input widget not available")
                self.status_message.emit("Please select a screen device first", 3000)
                return None
        else:
            # Fallback to old behavior
            if not self._selected_screen_devices:
                logger.warning("No screen device selected for window selection")
                self.status_message.emit("Please select a screen device first", 3000)
                return None
            base_device_id = next(iter(self._selected_screen_devices))

        try:
            # Create temporary capture to get available windows
            temp_capture = ScreenCapture(device_manager=self._capture_manager.get_device_manager())
            temp_capture.start_capture(device_id=base_device_id, fps=1)

            available_windows = temp_capture.get_available_windows()
            temp_capture.stop_capture()

            if not available_windows:
                logger.warning("No windows available for selection")
                self.status_message.emit("No windows available for selection", 3000)
                return None

            # Get currently selected window titles from VideoInputWidget
            # (This will be empty on first selection)
            current_titles: list[str] = []

            # Show dialog
            from visionmate.desktop.dialogs import WindowSelectorDialog

            dialog = WindowSelectorDialog(available_windows, current_titles, parent)

            if not dialog.exec():
                # Dialog cancelled
                return None

            selected_titles = dialog.get_selected_titles()
            if not selected_titles:
                self.status_message.emit("No windows selected", 3000)
                return None

            logger.info(f"Selected {len(selected_titles)} window(s)")
            return selected_titles

        except Exception as e:
            logger.error(f"Error showing window selector: {e}", exc_info=True)
            self.status_message.emit(f"Error: {e}", 5000)
            return None

    def show_window_selector_and_create_captures(self, parent: QWidget) -> None:
        """Show window selector dialog and create captures for selected windows.

        Args:
            parent: Parent widget for the dialog
        """
        # Get base screen device
        if not self._selected_screen_devices:
            logger.warning("No screen device selected for window selection")
            self.status_message.emit("Please select a screen device first", 3000)
            return

        base_device_id = next(iter(self._selected_screen_devices))

        try:
            # Create temporary capture to get available windows
            temp_capture = ScreenCapture(device_manager=self._capture_manager.get_device_manager())
            temp_capture.start_capture(device_id=base_device_id, fps=1)

            available_windows = temp_capture.get_available_windows()
            temp_capture.stop_capture()

            if not available_windows:
                logger.warning("No windows available for selection")
                self.status_message.emit("No windows available for selection", 3000)
                return

            # Get currently selected window titles
            current_window_ids = [
                device_id
                for device_id in self._capture_manager.get_video_source_ids()
                if device_id.startswith(f"{base_device_id}_window_")
            ]

            current_titles = []
            for wid in current_window_ids:
                cap = self._capture_manager.get_video_source(wid)
                if cap and isinstance(cap, ScreenCapture):
                    titles = cap.get_selected_window_titles()
                    if titles:
                        current_titles.append(titles[0])

            # Show dialog
            from visionmate.desktop.dialogs import WindowSelectorDialog

            dialog = WindowSelectorDialog(available_windows, current_titles, parent)

            if not dialog.exec():
                # Dialog cancelled
                return

            selected_titles = dialog.get_selected_titles()
            if not selected_titles:
                self.status_message.emit("No windows selected", 3000)
                return

            # Create window captures
            success_count = self.create_window_captures(base_device_id, selected_titles)

            if success_count > 0:
                self.status_message.emit(f"Capturing {success_count} selected window(s)", 3000)
            else:
                self.status_message.emit("Failed to create window captures", 3000)

        except Exception as e:
            logger.error(f"Error showing window selector: {e}", exc_info=True)
            self.status_message.emit(f"Error: {e}", 5000)

    def create_window_captures(self, base_device_id: str, window_titles: list[str]) -> int:
        """Create captures and previews for selected windows.

        Args:
            base_device_id: Base screen device ID
            window_titles: List of window titles to capture

        Returns:
            Number of windows successfully created
        """
        # Note: With the new "Add to Preview" workflow, we don't close existing previews.
        # Each preview is independent and users can have multiple previews with different modes.

        # Create previews for selected windows
        success_count = 0
        for title in window_titles:
            try:
                self._create_window_capture(base_device_id, title)
                success_count += 1
            except Exception as e:
                logger.error(f"Error creating window capture for '{title}': {e}", exc_info=True)

        logger.info(f"Created {success_count}/{len(window_titles)} window captures")
        return success_count

    def _create_window_capture(self, base_device_id: str, window_title: str) -> None:
        """Create a capture and preview for a specific window.

        Args:
            base_device_id: Base screen device ID
            window_title: Title of the window to capture
        """
        # Create unique device ID for this window
        window_hash = abs(hash(window_title)) % 10000
        device_id = f"{base_device_id}_window_{window_hash}"

        # Check if already exists
        if device_id in self._capture_manager:
            logger.debug(f"Window capture already exists: {device_id}")
            return

        # Create capture
        capture = ScreenCapture(device_manager=self._capture_manager.get_device_manager())
        capture.start_capture(device_id=base_device_id, fps=1, enable_window_detection=True)
        capture.set_window_capture_mode(WindowCaptureMode.SELECTED_WINDOWS, [window_title])

        # Add to video source manager
        self._capture_manager.add_video_source(device_id, capture)

        # Add preview
        self.add_preview(device_id, capture)

        logger.info(f"Created window capture for: {window_title}")

    def get_device_info_text(self, source_id: str) -> str:
        """Get device information text for display.

        Args:
            source_id: Source identifier

        Returns:
            Formatted device information text
        """
        try:
            # Extract base device ID for special cases
            base_device_id = source_id

            # For window-specific captures, extract base device ID
            if "_window_" in source_id:
                base_device_id = source_id.split("_window_")[0]
            # For mode-specific captures, extract base device ID
            elif "_fullscreen" in source_id:
                base_device_id = source_id.replace("_fullscreen", "")
            elif "_activewindow" in source_id:
                base_device_id = source_id.replace("_activewindow", "")

            # Get metadata using base device ID
            metadata = self._capture_manager.get_device_metadata(base_device_id)

            # For window-specific captures
            if "_window_" in source_id:
                # Get window title from capture
                capture = self._capture_manager.get_video_source(source_id)
                if capture and isinstance(capture, ScreenCapture):
                    window_titles = capture.get_selected_window_titles()
                    window_title = window_titles[0] if window_titles else "Unknown"
                    return (
                        f"{metadata.name} - Window: {window_title} - "
                        f"{metadata.resolution} @ {metadata.fps}Hz"
                    )
                else:
                    return f"{metadata.name} - {metadata.resolution}"
            else:
                # Check if it's an audio device
                if metadata.device_type.value == "audio":
                    return (
                        f"{metadata.name} - {metadata.sample_rate}Hz, {metadata.current_channels}ch"
                    )
                else:
                    return f"{metadata.name} - {metadata.resolution} @ {metadata.fps}Hz"

        except Exception as e:
            logger.error(f"Error getting device info: {e}", exc_info=True)
            return f"Error: {e}"

    def start_audio_capture_and_preview(self, device_id: str) -> None:
        """Start audio capture for a device and create preview.

        Args:
            device_id: Audio device identifier

        """
        logger.debug(f"start_audio_capture_and_preview called: device_id={device_id}")

        # Check if already capturing
        if device_id in self._capture_manager.get_audio_source_ids():
            logger.warning(f"Already capturing from audio device: {device_id}")
            return

        try:
            # Import audio capture
            from visionmate.core.capture.audio import DeviceAudioCapture

            # Create audio capture instance
            capture = DeviceAudioCapture(chunk_duration=0.5)

            # Start capture
            capture.start_capture(device_id=device_id)

            # Add to audio source manager
            self._capture_manager.add_audio_source(device_id, capture)

            # Create and add preview
            self.add_audio_preview(device_id, capture)

            logger.info(f"Started audio capture and preview for device: {device_id}")

            # Emit status message for device selection
            try:
                metadata = self._capture_manager.get_device_metadata(device_id)
                self.status_message.emit(f"Selected: {metadata.name}", 3000)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error starting audio capture and preview: {e}", exc_info=True)

            # Cleanup on error
            if device_id in self._capture_manager.get_audio_source_ids():
                try:
                    capture_obj = self._capture_manager.get_audio_source(device_id)
                    if capture_obj and hasattr(capture_obj, "stop_capture"):
                        capture_obj.stop_capture()  # type: ignore
                    self._capture_manager.remove_audio_source(device_id)
                except Exception:
                    pass

    def close_audio_preview(self, source_id: str) -> None:
        """Close audio preview and stop capture for a source.

        Args:
            source_id: Source identifier
        """
        try:
            # Stop capture
            if source_id in self._capture_manager.get_audio_source_ids():
                capture = self._capture_manager.get_audio_source(source_id)
                if capture and hasattr(capture, "stop_capture"):
                    capture.stop_capture()  # type: ignore
                self._capture_manager.remove_audio_source(source_id)

            # Remove preview
            self.remove_preview(source_id)

            logger.info(f"Closed audio preview for source: {source_id}")

        except Exception as e:
            logger.error(f"Error closing audio preview: {e}", exc_info=True)

    def remove_video_previews(self) -> None:
        """Remove all video previews."""
        # Get all video source IDs
        video_source_ids = list(self._capture_manager.get_video_source_ids())

        # Close all video previews
        for source_id in video_source_ids:
            logger.info(f"Removing video preview: {source_id}")
            self.close_preview(source_id)

        logger.info(f"Removed {len(video_source_ids)} video preview(s)")

    def remove_audio_previews(self) -> None:
        """Remove all audio previews."""
        # Get all audio source IDs (copy to avoid modification during iteration)
        audio_source_ids = list(self._capture_manager.get_audio_source_ids())

        # Close all audio previews
        for source_id in audio_source_ids:
            logger.info(f"Removing audio preview: {source_id}")
            self.close_preview(source_id)

        logger.info(f"Removed {len(audio_source_ids)} audio preview(s)")

    def update_container_visibility(self) -> None:
        """Update container visibility based on Input Mode.

        This method adjusts the visibility and stretch factors of video and audio
        containers based on which previews are active.
        """
        has_video = len(self._video_previews) > 0
        has_audio = len(self._audio_previews) > 0

        if has_video and has_audio:
            # Both: video gets more space (3:1 ratio)
            self._video_container.show()
            self._audio_container.show()
            self._main_layout.setStretch(0, 3)  # Video container
            self._main_layout.setStretch(1, 1)  # Audio container
        elif has_video:
            # Video only: video gets all space
            self._video_container.show()
            self._audio_container.hide()
            self._main_layout.setStretch(0, 1)
            self._main_layout.setStretch(1, 0)
        elif has_audio:
            # Audio only: audio gets all space
            self._video_container.hide()
            self._audio_container.show()
            self._main_layout.setStretch(0, 0)
            self._main_layout.setStretch(1, 1)
        else:
            # None: show placeholder in video container
            self._video_container.show()
            self._audio_container.hide()
            if self._placeholder:
                self._placeholder.show()

        logger.debug(f"Updated container visibility: video={has_video}, audio={has_audio}")
