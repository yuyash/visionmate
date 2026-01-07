"""Audio preview widget for displaying audio waveform.

This module provides the AudioPreviewWidget for displaying real-time
audio waveform visualization.
"""

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.capture.audio import AudioCaptureInterface

logger = logging.getLogger(__name__)


class AudioPreviewWidget(QWidget):
    """Widget for displaying audio waveform preview.

    Displays real-time audio waveform using QGraphicsView.
    Updates at regular intervals to show current audio levels.

    Requirements: 12.2, 12.3
    """

    # Signal emitted when close is requested
    close_requested = Signal(str)  # source_id

    def __init__(
        self,
        source_id: str,
        capture: AudioCaptureInterface,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the AudioPreviewWidget.

        Args:
            source_id: Unique identifier for the audio source
            capture: AudioCaptureInterface instance
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug(f"Initializing AudioPreviewWidget for source: {source_id}")

        self._source_id = source_id
        self._capture = capture
        self._update_timer: Optional[QTimer] = None

        self._setup_ui()
        self._start_updates()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create group box
        group_box = QGroupBox("Audio Preview")
        group_box.setFlat(True)
        group_layout = QVBoxLayout(group_box)

        # Device name label
        try:
            device_info = self._capture.get_source_info()
            device_name = device_info.name
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            device_name = "Unknown Device"

        self._device_label = QLabel(device_name)
        self._device_label.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(self._device_label)

        # Waveform display using QGraphicsView
        self._waveform_view = QGraphicsView()
        self._waveform_view.setMinimumHeight(100)
        self._waveform_view.setMaximumHeight(150)
        self._waveform_view.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._waveform_view.setStyleSheet(
            """
            QGraphicsView {
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
            """
        )

        # Create scene for waveform
        self._waveform_scene = QGraphicsScene()
        self._waveform_view.setScene(self._waveform_scene)

        group_layout.addWidget(self._waveform_view)

        # Level meter label
        self._level_label = QLabel("Level: --")
        self._level_label.setStyleSheet("color: #888888;")
        group_layout.addWidget(self._level_label)

        # Add group box to main layout
        layout.addWidget(group_box)

        logger.debug("AudioPreviewWidget UI setup complete")

    def _start_updates(self) -> None:
        """Start the update timer for waveform display."""
        # Update at 30 FPS
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_waveform)
        self._update_timer.start(33)  # ~30 FPS

        logger.debug("Started audio preview updates")

    def _update_waveform(self) -> None:
        """Update the waveform display with latest audio data.

        Requirements: 12.3
        """
        if not self._capture.is_capturing():
            return

        try:
            # Get latest audio chunk
            chunk = self._capture.get_chunk()
            if chunk is None:
                return

            # Calculate audio level (RMS)
            audio_data = chunk.data
            if len(audio_data) == 0:
                return

            # Convert to mono if stereo
            if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            elif audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_data**2))
            db_level = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)

            # Update level label
            self._level_label.setText(f"Level: {db_level:.1f} dB")

            # Draw waveform
            self._draw_waveform(audio_data)

        except Exception as e:
            logger.error(f"Error updating waveform: {e}", exc_info=True)

    def _draw_waveform(self, audio_data: np.ndarray) -> None:
        """Draw waveform in the graphics view.

        Args:
            audio_data: Audio data array (mono)

        Requirements: 12.2
        """
        # Clear scene
        self._waveform_scene.clear()

        # Get view dimensions
        view_width = self._waveform_view.viewport().width()
        view_height = self._waveform_view.viewport().height()

        if view_width <= 0 or view_height <= 0:
            return

        # Downsample audio data to fit view width
        num_samples = len(audio_data)
        if num_samples == 0:
            return

        samples_per_pixel = max(1, num_samples // view_width)
        downsampled_data = audio_data[::samples_per_pixel]

        # Normalize to view height
        if len(downsampled_data) == 0:
            return

        max_amplitude = np.max(np.abs(downsampled_data))
        if max_amplitude > 0:
            normalized_data = downsampled_data / max_amplitude
        else:
            normalized_data = downsampled_data

        # Scale to view height (leave some margin)
        margin = 10
        scale = (view_height - 2 * margin) / 2
        center_y = view_height / 2

        # Draw waveform
        pen = QPen()
        pen.setColor("#4a9eff")  # Blue color
        pen.setWidth(1)

        for i in range(len(normalized_data) - 1):
            x1 = i * (view_width / len(normalized_data))
            y1 = center_y - normalized_data[i] * scale
            x2 = (i + 1) * (view_width / len(normalized_data))
            y2 = center_y - normalized_data[i + 1] * scale

            self._waveform_scene.addLine(x1, y1, x2, y2, pen)

        # Draw center line
        center_pen = QPen()
        center_pen.setColor("#3a3a3a")
        center_pen.setWidth(1)
        self._waveform_scene.addLine(0, center_y, view_width, center_y, center_pen)

    def cleanup(self) -> None:
        """Cleanup resources when widget is being destroyed."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None

        logger.debug(f"Cleaned up AudioPreviewWidget for source: {self._source_id}")

    def get_source_id(self) -> str:
        """Get the source ID for this preview.

        Returns:
            Source ID string
        """
        return self._source_id

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation
        """
        return f"AudioPreviewWidget(source_id={self._source_id})"
