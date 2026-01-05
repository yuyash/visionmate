"""Video preview widget for displaying captured frames."""

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from visionmate.capture.screen_capture import ScreenCaptureInterface


class VideoPreviewWidget(QWidget):
    """Widget for displaying video preview from screen capture."""

    def __init__(self, capture: Optional[ScreenCaptureInterface] = None):
        """Initialize video preview widget.

        Args:
            capture: Screen capture interface to get frames from
        """
        super().__init__()

        self._capture = capture
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_frame)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create label for video display
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("QLabel { background-color: black; }")
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setScaledContents(False)

        layout.addWidget(self._video_label)

        # Set placeholder text
        self._video_label.setText("No video feed")
        self._video_label.setStyleSheet(
            "QLabel { background-color: black; color: white; font-size: 16px; }"
        )

    def set_capture(self, capture: ScreenCaptureInterface) -> None:
        """Set the screen capture interface.

        Args:
            capture: Screen capture interface to get frames from
        """
        self._capture = capture

    def start_preview(self, fps: int = 30) -> None:
        """Start updating the preview at specified FPS.

        Args:
            fps: Frame rate for preview updates (default 30 FPS)
        """
        if self._capture is None:
            return

        # Calculate update interval in milliseconds
        interval_ms = int(1000 / fps)
        self._update_timer.start(interval_ms)

    def stop_preview(self) -> None:
        """Stop updating the preview."""
        self._update_timer.stop()

    def _update_frame(self) -> None:
        """Update the displayed frame from capture."""
        if self._capture is None:
            return

        # Get frame with highlight overlay
        frame = self._capture.get_frame_with_highlight()

        if frame is None:
            return

        # Convert numpy array to QPixmap
        pixmap = self._numpy_to_qpixmap(frame)

        if pixmap is not None:
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._video_label.setPixmap(scaled_pixmap)

    def _numpy_to_qpixmap(self, frame: np.ndarray) -> Optional[QPixmap]:
        """Convert numpy array (BGR format) to QPixmap.

        Args:
            frame: Frame as numpy array in BGR format

        Returns:
            QPixmap or None if conversion fails
        """
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1].copy()

            height, width, channels = rgb_frame.shape

            # Create QImage from numpy array
            bytes_per_line = channels * width
            q_image = QImage(
                rgb_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

            # Convert QImage to QPixmap
            return QPixmap.fromImage(q_image)

        except Exception as e:
            print(f"Error converting frame to QPixmap: {e}")
            return None
