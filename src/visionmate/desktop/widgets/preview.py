"""Preview container widget for managing multiple video previews.

This module provides the PreviewContainer widget that supports multiple
layout modes (horizontal, vertical, grid) and drag-and-drop reordering.
It also manages video source lifecycle and preview widgets.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal

if TYPE_CHECKING:
    from visionmate.desktop.widgets import VideoPreviewWidget
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
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

    Requirements: 13.1-13.5
    """

    # Signal emitted when layout mode changes
    layout_changed = Signal(str)  # layout mode

    # Signal emitted when preview close is requested
    preview_close_requested = Signal(str)  # source_id

    # Signal emitted when preview info is requested
    preview_info_requested = Signal(str)  # source_id

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
        self._previews: dict[str, "VideoPreviewWidget"] = {}  # type: ignore
        self._placeholder: Optional[QLabel] = None

        # Track selected screen devices for window selection
        self._selected_screen_devices: set[str] = set()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout (will be replaced based on mode)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(10)

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
        self._main_layout.addWidget(self._placeholder)

        logger.debug("PreviewContainer UI setup complete")

    def add_preview(self, source_id: str, capture: VideoCaptureInterface) -> None:
        """Add a preview widget for a video source.

        Args:
            source_id: Unique identifier for the source
            capture: VideoCaptureInterface instance

        Requirements: 11.5, 11.6
        """
        if source_id in self._previews:
            logger.warning(f"Preview already exists for source: {source_id}")
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
        preview.info_requested.connect(self._on_preview_info_requested)

        # Store preview
        self._previews[source_id] = preview

        # Hide placeholder if this is the first preview
        if len(self._previews) == 1 and self._placeholder:
            self._placeholder.hide()

        # Add to layout
        self._add_preview_to_layout(preview)

        # Update layout distribution
        self._update_layout_distribution()

        logger.info(f"Added preview for source: {source_id} (total: {len(self._previews)})")

    def remove_preview(self, source_id: str) -> None:
        """Remove a preview widget for a video source.

        Args:
            source_id: Source identifier

        Requirements: 11.9
        """
        if source_id not in self._previews:
            logger.warning(f"Preview not found for source: {source_id}")
            return

        preview = self._previews[source_id]

        # Cleanup preview
        preview.cleanup()

        # Remove from layout
        self._main_layout.removeWidget(preview)

        # Delete widget
        preview.deleteLater()

        # Remove from dict
        del self._previews[source_id]

        # Show placeholder if no more previews
        if len(self._previews) == 0 and self._placeholder:
            self._placeholder.show()
        else:
            # Update layout distribution
            self._update_layout_distribution()

        logger.info(f"Removed preview for source: {source_id} (remaining: {len(self._previews)})")

    def clear_previews(self) -> None:
        """Remove all preview widgets from the container.

        Requirements: 11.9
        """
        # Remove all previews
        for source_id in list(self._previews.keys()):
            self.remove_preview(source_id)

        logger.info("Cleared all previews from container")

    def _on_preview_close_requested(self, source_id: str) -> None:
        """Handle preview close request.

        Args:
            source_id: Source identifier
        """
        logger.info(f"Close requested for preview: {source_id}")
        self.preview_close_requested.emit(source_id)

    def _on_preview_info_requested(self, source_id: str) -> None:
        """Handle preview info request.

        Args:
            source_id: Source identifier
        """
        logger.info(f"Info requested for preview: {source_id}")
        self.preview_info_requested.emit(source_id)

    def get_preview_count(self) -> int:
        """Get the number of previews in the container.

        Returns:
            Number of previews
        """
        return len(self._previews)

    def set_layout_mode(self, mode: PreviewLayout) -> None:
        """Set the layout mode for previews.

        Args:
            mode: Layout mode (HORIZONTAL, VERTICAL, GRID)

        Requirements: 13.1, 13.2, 13.3, 13.4
        """
        if self._layout_mode == mode:
            return

        logger.info(f"Changing layout mode from {self._layout_mode.value} to {mode.value}")

        self._layout_mode = mode

        # Rebuild layout with new mode
        self._rebuild_layout()

        # Emit signal
        self.layout_changed.emit(mode.value)

    def get_layout_mode(self) -> PreviewLayout:
        """Get the current layout mode.

        Returns:
            Current layout mode
        """
        return self._layout_mode

    def _add_preview_to_layout(self, preview: QWidget) -> None:  # type: ignore
        """Add a preview widget to the current layout.

        Args:
            preview: Preview widget to add
        """
        if self._layout_mode == PreviewLayout.GRID:
            # Grid layout requires special handling
            self._rebuild_layout()
        else:
            # For horizontal and vertical, just add to layout
            self._main_layout.addWidget(preview)

    def _rebuild_layout(self) -> None:
        """Rebuild the layout with current mode and previews."""
        # Remove all widgets from current layout
        while self._main_layout.count():
            item = self._main_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        # Delete old layout
        old_layout = self._main_layout
        if old_layout:
            # Transfer ownership to None to allow deletion
            QWidget().setLayout(old_layout)

        # Create new layout based on mode
        if self._layout_mode == PreviewLayout.HORIZONTAL:
            self._main_layout = QHBoxLayout(self)
            self._main_layout.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setSpacing(10)

            # Add all previews
            for preview in self._previews.values():
                self._main_layout.addWidget(preview)

        elif self._layout_mode == PreviewLayout.VERTICAL:
            self._main_layout = QVBoxLayout(self)
            self._main_layout.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setSpacing(10)

            # Add all previews
            for preview in self._previews.values():
                self._main_layout.addWidget(preview)

        elif self._layout_mode == PreviewLayout.GRID:
            self._main_layout = QGridLayout(self)
            self._main_layout.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setSpacing(10)

            # Calculate grid dimensions
            count = len(self._previews)
            if count == 0:
                cols = 1
            elif count <= 2:
                cols = 2
            elif count <= 4:
                cols = 2
            elif count <= 6:
                cols = 3
            else:
                cols = 3

            # Add previews to grid
            for i, preview in enumerate(self._previews.values()):
                row = i // cols
                col = i % cols
                self._main_layout.addWidget(preview, row, col)

        # Add placeholder if no previews
        if len(self._previews) == 0 and self._placeholder:
            self._main_layout.addWidget(self._placeholder)
            self._placeholder.show()
        elif self._placeholder:
            self._placeholder.hide()

        # Update layout distribution
        self._update_layout_distribution()

        logger.debug(f"Rebuilt layout in {self._layout_mode.value} mode")

    def _update_layout_distribution(self) -> None:
        """Update layout to distribute space equally among previews."""
        if len(self._previews) == 0:
            return

        # Set equal stretch for all preview widgets
        if isinstance(self._main_layout, (QHBoxLayout, QVBoxLayout)):
            for i in range(self._main_layout.count()):
                item = self._main_layout.itemAt(i)
                if item and item.widget() and item.widget() != self._placeholder:
                    # Set stretch factor to 1 for equal distribution
                    self._main_layout.setStretch(i, 1)

        logger.debug(f"Updated layout distribution for {len(self._previews)} previews")

    def enable_drag_drop(self, enabled: bool = True) -> None:
        """Enable or disable drag-and-drop reordering.

        Args:
            enabled: True to enable drag-and-drop, False to disable

        Requirements: 13.5

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
            Number of previews
        """
        return len(self._previews)

    def __repr__(self) -> str:
        """Get string representation of the container.

        Returns:
            String representation
        """
        return f"PreviewContainer(mode={self._layout_mode.value}, previews={len(self._previews)})"

    def start_capture_and_preview(
        self,
        source_type: str,
        device_id: str,
        fps: int = 1,
        window_capture_mode: str = "full_screen",
    ) -> None:
        """Start capture for a device and create preview.

        Args:
            source_type: Type of source ("screen", "uvc", "rtsp")
            device_id: Device identifier
            fps: Frame rate (default: 1)
            window_capture_mode: Window capture mode (default: "full_screen")

        Requirements: 11.6, 28.9
        """
        logger.debug(
            f"start_capture_and_preview called: source_type={source_type}, "
            f"device_id={device_id}, mode={window_capture_mode}"
        )

        # Check if already capturing
        if device_id in self._capture_manager:
            logger.warning(f"Already capturing from device: {device_id}")
            return

        # Track selected screen devices
        if source_type == "screen":
            self._selected_screen_devices.add(device_id)

        try:
            # Create capture instance based on source type
            if source_type == "screen":
                capture = ScreenCapture(device_manager=self._capture_manager.get_device_manager())

                # Determine capture mode
                if window_capture_mode == "full_screen":
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False
                elif window_capture_mode == "active_window":
                    capture_mode = WindowCaptureMode.ACTIVE_WINDOW
                    enable_window_detection = True
                elif window_capture_mode == "selected_windows":
                    # For selected windows mode, don't create preview yet
                    logger.info("Selected windows mode - waiting for window selection")
                    return
                else:
                    capture_mode = WindowCaptureMode.FULL_SCREEN
                    enable_window_detection = False

                # Start capture
                capture.start_capture(
                    device_id=device_id,
                    fps=fps,
                    enable_window_detection=enable_window_detection,
                )
                capture.set_window_capture_mode(capture_mode)

            elif source_type == "uvc":
                logger.warning("UVC capture not yet implemented")
                return
            elif source_type == "rtsp":
                logger.warning("RTSP capture not yet implemented")
                return
            else:
                logger.error(f"Unknown source type: {source_type}")
                return

            # Add to video source manager
            self._capture_manager.add_video_source(device_id, capture)

            # Create and add preview
            self.add_preview(device_id, capture)

            logger.info(f"Started capture and preview for device: {device_id}")

        except Exception as e:
            logger.error(f"Error starting capture and preview: {e}", exc_info=True)

            # Cleanup on error
            if device_id in self._capture_manager:
                try:
                    capture = self._capture_manager.get_video_source(device_id)
                    if capture:
                        capture.stop_capture()
                    self._capture_manager.remove_video_source(device_id)
                except Exception:
                    pass

    def close_preview(self, source_id: str, keep_selection: bool = False) -> None:
        """Close preview and stop capture for a source.

        Args:
            source_id: Source identifier
            keep_selection: If True, keep device in selected_screen_devices (for mode changes)
        """
        try:
            # Stop capture
            if source_id in self._capture_manager:
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

            logger.info(f"Closed preview for source: {source_id} (keep_selection={keep_selection})")

        except Exception as e:
            logger.error(f"Error closing preview: {e}", exc_info=True)

    def get_selected_screen_devices(self) -> set[str]:
        """Get selected screen devices.

        Returns:
            Set of selected screen device IDs
        """
        return self._selected_screen_devices.copy()

    def handle_selection_change(self, selected_device_ids: list[str]) -> int:
        """Handle device selection change.

        Closes previews for deselected devices.

        Args:
            selected_device_ids: List of selected device IDs

        Returns:
            Number of currently selected devices
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

        return len(selected_device_ids)

    def handle_window_capture_mode_change(
        self,
        mode: str,
        selected_titles: Optional[list[str]] = None,
    ) -> tuple[str, Optional[str]]:
        """Handle window capture mode change.

        Args:
            mode: Capture mode
            selected_titles: List of selected window titles (unused, for compatibility)

        Returns:
            Tuple of (action, message) where action is "show_selector", "wait", or "recreate"
        """
        # Handle show_selector mode
        if mode == "show_selector":
            return ("show_selector", None)

        # Handle selected_windows mode
        if mode == "selected_windows":
            self._close_all_window_previews()
            return ("wait", "Click 'Select Windows...' to choose windows")

        # For full_screen and active_window modes
        # Close all existing screen previews (but keep selection)
        all_device_ids = list(self._capture_manager.get_video_source_ids())
        for device_id in all_device_ids:
            if device_id.startswith("screen_"):
                logger.info(f"Closing existing preview: {device_id}")
                self.close_preview(device_id, keep_selection=True)

        # Recreate previews with new mode
        for device_id in self._selected_screen_devices:
            self.start_capture_and_preview(
                source_type="screen",
                device_id=device_id,
                fps=1,
                window_capture_mode=mode,
            )

        return ("recreate", f"Capture mode: {mode.replace('_', ' ')}")

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

    def show_window_selector_dialog(self, parent: QWidget) -> Optional[list[str]]:
        """Show window selector dialog and return selected window titles.

        Args:
            parent: Parent widget for the dialog

        Returns:
            List of selected window titles, or None if cancelled
        """
        # Get base screen device
        if not self._selected_screen_devices:
            logger.warning("No screen device selected for window selection")
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
                return None

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

            if dialog.exec():
                return dialog.get_selected_titles()

            return None

        except Exception as e:
            logger.error(f"Error showing window selector: {e}", exc_info=True)
            return None

    def create_window_captures(self, base_device_id: str, window_titles: list[str]) -> int:
        """Create captures and previews for selected windows.

        Args:
            base_device_id: Base screen device ID
            window_titles: List of window titles to capture

        Returns:
            Number of windows successfully created
        """
        # Close all existing screen previews first (but keep selection)
        all_device_ids = list(self._capture_manager.get_video_source_ids())
        for device_id in all_device_ids:
            if device_id.startswith("screen_"):
                logger.info(f"Closing existing preview: {device_id}")
                self.close_preview(device_id, keep_selection=True)

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
            # For window-specific captures, extract base device ID
            if "_window_" in source_id:
                base_device_id = source_id.split("_window_")[0]
                metadata = self._capture_manager.get_device_metadata(base_device_id)

                # Get window title from capture
                capture = self._capture_manager.get_video_source(source_id)
                if capture and isinstance(capture, ScreenCapture):
                    window_titles = capture.get_selected_window_titles()
                    window_title = window_titles[0] if window_titles else "Unknown"
                    return (
                        f"{metadata.name} - Window: {window_title} - "
                        f"{metadata.current_resolution} @ {metadata.current_fps}fps"
                    )
                else:
                    return f"{metadata.name} - {metadata.current_resolution}"
            else:
                # Regular device
                metadata = self._capture_manager.get_device_metadata(source_id)
                return (
                    f"{metadata.name} - {metadata.current_resolution} @ {metadata.current_fps}fps"
                )

        except Exception as e:
            logger.error(f"Error getting device info: {e}", exc_info=True)
            return f"Error: {e}"
