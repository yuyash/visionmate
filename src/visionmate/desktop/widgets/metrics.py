"""Metrics display widget for observability.

This module provides the MetricsWidget that displays real-time metrics
from the multimedia manager including buffer usage, throughput, and errors.
"""

import logging
from collections.abc import Callable
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import ManagerMetrics

logger = logging.getLogger(__name__)


class MetricsWidget(QWidget):
    """Widget for displaying multimedia manager metrics.

    Shows:
    - Buffer usage (segments buffered, dropped, memory)
    - Throughput (segments sent, requests sent)
    - Error counts (send errors, STT errors, connection errors)
    - Latencies (segment latency, STT duration)

    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the MetricsWidget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing MetricsWidget")

        # Labels for metrics
        self._segments_buffered_label: Optional[QLabel] = None
        self._segments_dropped_label: Optional[QLabel] = None
        self._buffer_memory_label: Optional[QLabel] = None
        self._segments_sent_label: Optional[QLabel] = None
        self._requests_sent_label: Optional[QLabel] = None
        self._send_errors_label: Optional[QLabel] = None
        self._stt_errors_label: Optional[QLabel] = None
        self._connection_errors_label: Optional[QLabel] = None
        self._avg_latency_label: Optional[QLabel] = None
        self._avg_stt_duration_label: Optional[QLabel] = None

        # Auto-refresh timer
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._request_metrics_update)
        self._refresh_interval_ms = 1000  # Update every second

        # Callback for requesting metrics
        self._metrics_callback: Optional[Callable] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Create group box
        group_box = QGroupBox("Metrics")
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(8)

        # Buffer metrics section
        buffer_frame = self._create_section_frame("Buffer")
        buffer_layout = QGridLayout(buffer_frame)
        buffer_layout.setContentsMargins(4, 4, 4, 4)
        buffer_layout.setSpacing(4)

        buffer_layout.addWidget(QLabel("Buffered:"), 0, 0)
        self._segments_buffered_label = QLabel("0")
        buffer_layout.addWidget(self._segments_buffered_label, 0, 1)

        buffer_layout.addWidget(QLabel("Dropped:"), 1, 0)
        self._segments_dropped_label = QLabel("0")
        buffer_layout.addWidget(self._segments_dropped_label, 1, 1)

        buffer_layout.addWidget(QLabel("Memory:"), 2, 0)
        self._buffer_memory_label = QLabel("0.0 MB")
        buffer_layout.addWidget(self._buffer_memory_label, 2, 1)

        group_layout.addWidget(buffer_frame)

        # Throughput metrics section
        throughput_frame = self._create_section_frame("Throughput")
        throughput_layout = QGridLayout(throughput_frame)
        throughput_layout.setContentsMargins(4, 4, 4, 4)
        throughput_layout.setSpacing(4)

        throughput_layout.addWidget(QLabel("Segments:"), 0, 0)
        self._segments_sent_label = QLabel("0")
        throughput_layout.addWidget(self._segments_sent_label, 0, 1)

        throughput_layout.addWidget(QLabel("Requests:"), 1, 0)
        self._requests_sent_label = QLabel("0")
        throughput_layout.addWidget(self._requests_sent_label, 1, 1)

        group_layout.addWidget(throughput_frame)

        # Error metrics section
        error_frame = self._create_section_frame("Errors")
        error_layout = QGridLayout(error_frame)
        error_layout.setContentsMargins(4, 4, 4, 4)
        error_layout.setSpacing(4)

        error_layout.addWidget(QLabel("Send:"), 0, 0)
        self._send_errors_label = QLabel("0")
        error_layout.addWidget(self._send_errors_label, 0, 1)

        error_layout.addWidget(QLabel("STT:"), 1, 0)
        self._stt_errors_label = QLabel("0")
        error_layout.addWidget(self._stt_errors_label, 1, 1)

        error_layout.addWidget(QLabel("Connection:"), 2, 0)
        self._connection_errors_label = QLabel("0")
        error_layout.addWidget(self._connection_errors_label, 2, 1)

        group_layout.addWidget(error_frame)

        # Latency metrics section
        latency_frame = self._create_section_frame("Latency")
        latency_layout = QGridLayout(latency_frame)
        latency_layout.setContentsMargins(4, 4, 4, 4)
        latency_layout.setSpacing(4)

        latency_layout.addWidget(QLabel("Segment:"), 0, 0)
        self._avg_latency_label = QLabel("0.0 ms")
        latency_layout.addWidget(self._avg_latency_label, 0, 1)

        latency_layout.addWidget(QLabel("STT:"), 1, 0)
        self._avg_stt_duration_label = QLabel("0.0 ms")
        latency_layout.addWidget(self._avg_stt_duration_label, 1, 1)

        group_layout.addWidget(latency_frame)

        main_layout.addWidget(group_box)

        logger.debug("MetricsWidget UI setup complete")

    def _create_section_frame(self, title: str) -> QFrame:
        """Create a section frame with title.

        Args:
            title: Section title

        Returns:
            QFrame with title label
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            """
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        return frame

    def set_metrics_callback(self, callback: Callable) -> None:
        """Set callback for requesting metrics updates.

        Args:
            callback: Function that returns ManagerMetrics
        """
        self._metrics_callback = callback
        logger.debug("Metrics callback registered")

    def start_auto_refresh(self, interval_ms: int = 1000) -> None:
        """Start automatic metrics refresh.

        Args:
            interval_ms: Refresh interval in milliseconds (default: 1000)
        """
        self._refresh_interval_ms = interval_ms
        self._refresh_timer.start(interval_ms)
        logger.debug(f"Auto-refresh started with interval: {interval_ms}ms")

    def stop_auto_refresh(self) -> None:
        """Stop automatic metrics refresh."""
        self._refresh_timer.stop()
        logger.debug("Auto-refresh stopped")

    def _request_metrics_update(self) -> None:
        """Request metrics update from callback."""
        if self._metrics_callback:
            try:
                metrics = self._metrics_callback()
                self.update_metrics(metrics)
            except Exception as e:
                logger.error(f"Error requesting metrics update: {e}", exc_info=True)

    def update_metrics(self, metrics: ManagerMetrics) -> None:
        """Update displayed metrics.

        Args:
            metrics: ManagerMetrics object with current values

        """
        # Update buffer metrics
        if self._segments_buffered_label:
            self._segments_buffered_label.setText(str(metrics.segments_buffered))

        if self._segments_dropped_label:
            self._segments_dropped_label.setText(str(metrics.segments_dropped))
            # Highlight if dropped segments
            if metrics.segments_dropped > 0:
                self._segments_dropped_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self._segments_dropped_label.setStyleSheet("")

        if self._buffer_memory_label:
            self._buffer_memory_label.setText(f"{metrics.buffer_memory_mb:.1f} MB")

        # Update throughput metrics
        if self._segments_sent_label:
            self._segments_sent_label.setText(str(metrics.segments_sent))

        if self._requests_sent_label:
            self._requests_sent_label.setText(str(metrics.requests_sent))

        # Update error metrics
        if self._send_errors_label:
            self._send_errors_label.setText(str(metrics.send_errors))
            # Highlight if errors
            if metrics.send_errors > 0:
                self._send_errors_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self._send_errors_label.setStyleSheet("")

        if self._stt_errors_label:
            self._stt_errors_label.setText(str(metrics.stt_errors))
            # Highlight if errors
            if metrics.stt_errors > 0:
                self._stt_errors_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self._stt_errors_label.setStyleSheet("")

        if self._connection_errors_label:
            self._connection_errors_label.setText(str(metrics.connection_errors))
            # Highlight if errors
            if metrics.connection_errors > 0:
                self._connection_errors_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self._connection_errors_label.setStyleSheet("")

        # Update latency metrics
        if self._avg_latency_label:
            self._avg_latency_label.setText(f"{metrics.avg_segment_latency_ms:.1f} ms")

        if self._avg_stt_duration_label:
            self._avg_stt_duration_label.setText(f"{metrics.avg_stt_duration_ms:.1f} ms")
