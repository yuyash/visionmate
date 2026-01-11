"""Audio preview widget for displaying audio input visualization.

This module provides the AudioPreviewWidget that displays real-time
audio level meters and waveform visualization.
"""

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class AudioLevelMeter(QWidget):
    """Audio level meter widget.

    Displays a vertical bar showing the current audio level.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the audio level meter.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._level = 0.0  # 0.0 to 1.0
        self.setMinimumSize(40, 100)
        self.setMaximumWidth(60)

    def set_level(self, level: float) -> None:
        """Set the audio level.

        Args:
            level: Audio level from 0.0 to 1.0
        """
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the level meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        # Calculate bar height
        bar_height = int(self.height() * self._level)
        bar_rect = self.rect()
        bar_rect.setTop(self.height() - bar_height)

        # Choose color based on level
        if self._level > 0.9:
            color = Qt.GlobalColor.red
        elif self._level > 0.7:
            color = Qt.GlobalColor.yellow
        else:
            color = Qt.GlobalColor.green

        # Draw level bar
        painter.fillRect(bar_rect, color)

        # Draw border
        painter.setPen(QPen(Qt.GlobalColor.gray, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class AudioPreviewWidget(QFrame):
    """Widget for previewing audio input.

    Displays:
    - Device name
    - Audio level meter
    - Sample rate and channel information
    - Close button

    """

    # Signal emitted when close button is clicked
    close_requested = Signal(str)  # source_id

    # Signal emitted when info button is clicked
    info_requested = Signal(str)  # source_id

    def __init__(
        self,
        source_id: str,
        capture,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the audio preview widget.

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
        # Set frame style
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            """
        )

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Header with device name and close button
        header_layout = QHBoxLayout()

        # Device name label
        try:
            metadata = self._capture.get_source_info()
            device_name = metadata.name
        except Exception:
            device_name = self._source_id

        self._name_label = QLabel(device_name)
        self._name_label.setStyleSheet(
            """
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            """
        )
        header_layout.addWidget(self._name_label)

        header_layout.addStretch()

        # Info button
        info_button = QPushButton("ℹ")
        info_button.setFixedSize(24, 24)
        info_button.setToolTip("Show device information")
        info_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            """
        )
        info_button.clicked.connect(self._on_info_clicked)
        header_layout.addWidget(info_button)

        # Close button
        close_button = QPushButton("×")
        close_button.setFixedSize(24, 24)
        close_button.setToolTip("Close preview")
        close_button.setStyleSheet(
            """
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
            """
        )
        close_button.clicked.connect(self._on_close_clicked)
        header_layout.addWidget(close_button)

        main_layout.addLayout(header_layout)

        # Content area with level meter and info
        content_layout = QHBoxLayout()

        # Audio level meter
        self._level_meter = AudioLevelMeter()
        content_layout.addWidget(self._level_meter)

        # Info labels
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self._level_label = QLabel("Level: 0%")
        self._level_label.setStyleSheet("QLabel { color: white; font-size: 11px; }")
        info_layout.addWidget(self._level_label)

        self._sample_rate_label = QLabel("Sample Rate: --")
        self._sample_rate_label.setStyleSheet("QLabel { color: #aaaaaa; font-size: 10px; }")
        info_layout.addWidget(self._sample_rate_label)

        self._channels_label = QLabel("Channels: --")
        self._channels_label.setStyleSheet("QLabel { color: #aaaaaa; font-size: 10px; }")
        info_layout.addWidget(self._channels_label)

        self._status_label = QLabel("Status: Capturing")
        self._status_label.setStyleSheet("QLabel { color: #4caf50; font-size: 10px; }")
        info_layout.addWidget(self._status_label)

        info_layout.addStretch()

        content_layout.addLayout(info_layout, stretch=1)

        main_layout.addLayout(content_layout)

        # Update metadata labels
        self._update_metadata_labels()

        logger.debug("AudioPreviewWidget UI setup complete")

    def _update_metadata_labels(self) -> None:
        """Update metadata labels with device information."""
        try:
            metadata = self._capture.get_source_info()

            if metadata.current_sample_rate:
                self._sample_rate_label.setText(f"Sample Rate: {metadata.current_sample_rate} Hz")

            if metadata.current_channels:
                self._channels_label.setText(f"Channels: {metadata.current_channels}")

        except Exception as e:
            logger.error(f"Error updating metadata labels: {e}", exc_info=True)

    def _start_updates(self) -> None:
        """Start periodic updates of audio level."""
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_audio_level)
        self._update_timer.start(50)  # Update every 50ms (20 FPS)
        logger.debug("Started audio level updates")

    def _update_audio_level(self) -> None:
        """Update the audio level display."""
        try:
            # Get latest audio chunk
            chunk = self._capture.get_chunk()

            if chunk is not None and chunk.data is not None:
                # Calculate RMS level
                rms = np.sqrt(np.mean(chunk.data**2))

                # Normalize to 0.0-1.0 range (assuming 16-bit audio)
                level = min(1.0, rms * 10)  # Scale factor for visibility

                # Update meter
                self._level_meter.set_level(level)

                # Update level label
                level_percent = int(level * 100)
                self._level_label.setText(f"Level: {level_percent}%")

                # Update status
                if self._capture.is_capturing():
                    self._status_label.setText("Status: Capturing")
                    self._status_label.setStyleSheet("QLabel { color: #4caf50; font-size: 10px; }")
                else:
                    self._status_label.setText("Status: Stopped")
                    self._status_label.setStyleSheet("QLabel { color: #f44336; font-size: 10px; }")

        except Exception as e:
            logger.error(f"Error updating audio level: {e}", exc_info=True)

    def _on_close_clicked(self) -> None:
        """Handle close button click."""
        logger.debug(f"Close requested for audio preview: {self._source_id}")
        self.close_requested.emit(self._source_id)

    def _on_info_clicked(self) -> None:
        """Handle info button click."""
        logger.debug(f"Info button clicked for audio source: {self._source_id}")
        # Get metadata and show as tooltip
        try:
            metadata_text = self._get_metadata_text()
            if metadata_text:
                # Show tooltip at button position
                from PySide6.QtGui import QCursor
                from PySide6.QtWidgets import QToolTip

                QToolTip.showText(QCursor.pos(), metadata_text, self, self.rect(), 5000)
        except Exception as e:
            logger.error(f"Error showing metadata tooltip: {e}", exc_info=True)

    def _get_metadata_text(self) -> str:
        """Get metadata text for display.

        Returns:
            Formatted metadata string
        """
        try:
            metadata = self._capture.get_source_info()
            lines = [f"Device: {metadata.name}"]

            if metadata.sample_rate:
                lines.append(f"Sample Rate: {metadata.sample_rate} Hz")

            if metadata.current_channels:
                lines.append(f"Channels: {metadata.current_channels}")

            lines.append(f"Device ID: {self._source_id}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error getting metadata: {e}", exc_info=True)
            return f"Error: {e}"

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
        logger.debug(f"Cleaned up AudioPreviewWidget for source: {self._source_id}")

    def get_source_id(self) -> str:
        """Get the source identifier.

        Returns:
            Source identifier string
        """
        return self._source_id
