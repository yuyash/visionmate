"""
Video preview widget for displaying captured video frames.

This module provides the VideoPreviewWidget for displaying video frames
with metadata overlay, info icon, and close button.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import VideoFrame

logger = logging.getLogger(__name__)


class VideoPreviewWidget(QWidget):
    """Widget for displaying video preview with controls.

    Provides:
    - Video display using QLabel
    - Info icon button for metadata
    - Close button to remove source
    - Real-time frame updates using QTimer

    Requirements: 11.6, 11.7, 11.8, 11.9
    """

    # Signal emitted when close button is clicked
    close_requested = Signal(str)  # source_id

    # Signal emitted when info button is clicked
    info_requested = Signal(str)  # source_id

    def __init__(
        self,
        source_id: str,
        capture: VideoCaptureInterface,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the VideoPreviewWidget.

        Args:
            source_id: Unique identifier for the video source
            capture: VideoCaptureInterface instance for frame retrieval
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug(f"Initializing VideoPreviewWidget for source: {source_id}")

        self._source_id = source_id
        self._capture = capture
        self._update_timer: Optional[QTimer] = None

        self._setup_ui()
        self._start_updates()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Create container for video and overlay buttons
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)

        # Create video display label
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setStyleSheet(
            """
            QLabel {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
            }
            """
        )
        self._video_label.setText("Loading...")
        video_container_layout.addWidget(self._video_label)

        # Create overlay widget for buttons (positioned on top of video)
        self._overlay_widget = QWidget(self._video_label)
        overlay_layout = QHBoxLayout(self._overlay_widget)
        overlay_layout.setContentsMargins(5, 5, 5, 5)
        overlay_layout.setSpacing(5)

        # Info button (top-left)
        self._info_button = QPushButton("ⓘ")
        self._info_button.setFixedSize(32, 32)
        self._info_button.setToolTip("Show device metadata")
        self._info_button.clicked.connect(self._on_info_clicked)
        self._info_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 200);
            }
            """
        )
        overlay_layout.addWidget(self._info_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Spacer
        overlay_layout.addStretch()

        # Close button (top-right)
        self._close_button = QPushButton("⊗")
        self._close_button.setFixedSize(32, 32)
        self._close_button.setToolTip("Remove this source")
        self._close_button.clicked.connect(self._on_close_clicked)
        self._close_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(200, 0, 0, 200);
            }
            """
        )
        overlay_layout.addWidget(self._close_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Position overlay at top of video label
        self._update_overlay_position()
        self._overlay_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Install event filter to track video label resize
        self._video_label.installEventFilter(self)

        layout.addWidget(video_container)

        # Create status label for window detection indicator
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            """
            QLabel {
                color: #666666;
                font-size: 11px;
                padding: 2px;
            }
            """
        )
        layout.addWidget(self._status_label)

        logger.debug("VideoPreviewWidget UI setup complete")

    def _start_updates(self) -> None:
        """Start the frame update timer.

        Updates at 30 FPS for smooth preview.
        """
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_frame)
        self._update_timer.start(33)  # ~30 FPS (1000ms / 30 = 33ms)
        logger.debug("Frame update timer started")

    def _stop_updates(self) -> None:
        """Stop the frame update timer."""
        if self._update_timer:
            self._update_timer.stop()
            logger.debug("Frame update timer stopped")

    def _update_overlay_position(self) -> None:
        """Update overlay widget position to match video label size."""
        if hasattr(self, "_overlay_widget") and hasattr(self, "_video_label"):
            # Set overlay to cover the full width of video label
            self._overlay_widget.setGeometry(0, 0, self._video_label.width(), 42)

    def eventFilter(self, obj, event) -> bool:
        """Event filter to track video label resize events.

        Args:
            obj: Object that triggered the event
            event: Event object

        Returns:
            False to allow event to propagate
        """
        from PySide6.QtCore import QEvent

        if obj == self._video_label and event.type() == QEvent.Type.Resize:
            # Update overlay position when video label is resized
            self._update_overlay_position()

        return False

    def _update_frame(self) -> None:
        """Update the displayed frame from capture source."""
        try:
            # Get latest frame from capture
            frame = self._capture.get_frame()

            if frame is None:
                # No frame available yet
                if self._video_label.pixmap() is None:
                    self._video_label.setText("Waiting for frames...")
                return

            # Convert frame to QPixmap
            pixmap = self._frame_to_pixmap(frame)

            if pixmap:
                # Scale pixmap to fit label while maintaining aspect ratio
                # Use FastTransformation for instant resize (not smooth)
                scaled_pixmap = pixmap.scaled(
                    self._video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                self._video_label.setPixmap(scaled_pixmap)

                # Update status label with window detection info
                self._update_status_label(frame)

        except Exception as e:
            logger.error(f"Error updating frame: {e}", exc_info=True)
            self._video_label.setText(f"Error: {e}")

    def _frame_to_pixmap(self, frame: VideoFrame) -> Optional[QPixmap]:
        """Convert VideoFrame to QPixmap.

        Args:
            frame: VideoFrame object

        Returns:
            QPixmap object, or None if conversion fails
        """
        try:
            # Get image data (RGB format)
            img = frame.image

            # Ensure array is C-contiguous (required by QImage)
            if not img.flags["C_CONTIGUOUS"]:
                img = img.copy()

            # Get dimensions
            height, width, channels = img.shape

            # Create QImage from numpy array
            bytes_per_line = channels * width
            q_image = QImage(
                img.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

            # Convert to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            return pixmap

        except Exception as e:
            logger.error(f"Error converting frame to pixmap: {e}", exc_info=True)
            return None

    def _update_status_label(self, frame: VideoFrame) -> None:
        """Update status label with window detection info.

        Args:
            frame: VideoFrame object

        Requirements: 28.9
        """
        if frame.is_cropped and frame.active_region:
            region = frame.active_region
            status_text = (
                f"🪟 Window detected: {region.width}×{region.height} "
                f"(confidence: {region.confidence:.0%})"
            )
            self._status_label.setText(status_text)
        else:
            self._status_label.setText("")

    def _on_info_clicked(self) -> None:
        """Handle info button click.

        Requirements: 11.8
        """
        logger.debug(f"Info button clicked for source: {self._source_id}")
        # Get metadata and show as tooltip
        try:
            metadata_text = self._get_metadata_text()
            if metadata_text:
                # Show tooltip at button position
                from PySide6.QtGui import QCursor
                from PySide6.QtWidgets import QToolTip

                QToolTip.showText(
                    QCursor.pos(), metadata_text, self._info_button, self.rect(), 5000
                )
        except Exception as e:
            logger.error(f"Error showing metadata tooltip: {e}", exc_info=True)

    def _get_metadata_text(self) -> str:
        """Get metadata text for display.

        Returns:
            Formatted metadata string
        """
        try:
            # Get metadata from capture
            metadata = self._capture.get_source_info()

            lines = [f"<b>{metadata.name}</b>"]

            if metadata.current_resolution:
                lines.append(f"Resolution: {metadata.current_resolution}")
            if metadata.current_fps:
                lines.append(f"FPS: {metadata.current_fps}")
            if metadata.native_fps:
                lines.append(f"Native FPS: {metadata.native_fps}")

            # Check if this is a window-specific capture
            if "_window_" in self._source_id:
                from visionmate.core.capture.video import ScreenCapture

                if isinstance(self._capture, ScreenCapture):
                    window_titles = self._capture.get_selected_window_titles()
                    if window_titles:
                        lines.append(f"<br><b>Window:</b> {window_titles[0]}")

            return "<br>".join(lines)

        except Exception as e:
            logger.error(f"Error getting metadata: {e}", exc_info=True)
            return f"Error: {e}"

    def _on_close_clicked(self) -> None:
        """Handle close button click.

        Requirements: 11.9
        """
        logger.debug(f"Close button clicked for source: {self._source_id}")
        self.close_requested.emit(self._source_id)

    def get_source_id(self) -> str:
        """Get the source ID for this preview.

        Returns:
            Source ID string
        """
        return self._source_id

    def cleanup(self) -> None:
        """Cleanup resources when widget is destroyed."""
        logger.debug(f"Cleaning up VideoPreviewWidget for source: {self._source_id}")
        self._stop_updates()
