"""Audio waveform preview widget for displaying audio input."""

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QGraphicsView, QVBoxLayout, QWidget

from visionmate.capture.audio_capture import AudioCaptureInterface


class AudioWaveformWidget(QWidget):
    """Widget for displaying audio waveform preview."""

    def __init__(self, capture: Optional[AudioCaptureInterface] = None):
        """Initialize audio waveform widget.

        Args:
            capture: Audio capture interface to get audio from
        """
        super().__init__()

        self._capture = capture
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_waveform)
        self._waveform_data: Optional[np.ndarray] = None

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create graphics view for waveform display
        self._waveform_view = WaveformGraphicsView()
        self._waveform_view.setMinimumHeight(100)
        self._waveform_view.setMaximumHeight(150)

        layout.addWidget(self._waveform_view)

    def set_capture(self, capture: AudioCaptureInterface) -> None:
        """Set the audio capture interface.

        Args:
            capture: Audio capture interface to get audio from
        """
        self._capture = capture

    def start_preview(self, fps: int = 30) -> None:
        """Start updating the waveform preview at specified FPS.

        Args:
            fps: Frame rate for waveform updates (default 30 FPS)
        """
        if self._capture is None:
            return

        # Calculate update interval in milliseconds
        interval_ms = int(1000 / fps)
        self._update_timer.start(interval_ms)

    def stop_preview(self) -> None:
        """Stop updating the waveform preview."""
        self._update_timer.stop()

    def _update_waveform(self) -> None:
        """Update the waveform display from audio capture."""
        if self._capture is None:
            return

        # Get latest audio chunk
        audio_chunk = self._capture.get_audio_chunk()

        if audio_chunk is None:
            return

        # Flatten to 1D if multi-channel
        if audio_chunk.ndim > 1:
            audio_data = audio_chunk.mean(axis=1)
        else:
            audio_data = audio_chunk.flatten()

        # Store waveform data
        self._waveform_data = audio_data

        # Update the graphics view
        self._waveform_view.set_waveform_data(audio_data)


class WaveformGraphicsView(QGraphicsView):
    """Graphics view for rendering audio waveform."""

    def __init__(self):
        """Initialize waveform graphics view."""
        super().__init__()

        self._waveform_data: Optional[np.ndarray] = None

        # Set view properties
        self.setStyleSheet("QGraphicsView { background-color: black; border: 1px solid #333; }")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_waveform_data(self, data: np.ndarray) -> None:
        """Set waveform data and trigger repaint.

        Args:
            data: Audio waveform data as 1D numpy array
        """
        self._waveform_data = data
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the waveform.

        Args:
            event: Paint event
        """
        super().paintEvent(event)

        if self._waveform_data is None or len(self._waveform_data) == 0:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get view dimensions
        width = self.viewport().width()
        height = self.viewport().height()
        center_y = height / 2

        # Set pen for waveform
        pen = QPen(Qt.GlobalColor.green)
        pen.setWidth(2)
        painter.setPen(pen)

        # Downsample waveform data to fit width
        data = self._waveform_data
        num_samples = len(data)
        samples_per_pixel = max(1, num_samples // width)

        # Draw waveform
        prev_x = 0
        prev_y = center_y

        for x in range(width):
            # Get sample index
            sample_idx = x * samples_per_pixel

            if sample_idx >= num_samples:
                break

            # Get sample value (average over samples_per_pixel)
            end_idx = min(sample_idx + samples_per_pixel, num_samples)
            sample_value = np.mean(data[sample_idx:end_idx])

            # Normalize to [-1, 1] range (assuming audio is already normalized)
            # Scale to view height
            y = center_y - (sample_value * center_y * 0.8)  # 0.8 for some padding

            # Draw line from previous point
            painter.drawLine(int(prev_x), int(prev_y), int(x), int(y))

            prev_x = x
            prev_y = y

        painter.end()
