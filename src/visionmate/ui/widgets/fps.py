"""FPS control widget for adjusting capture frame rate."""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class FPSControlWidget(QWidget):
    """Widget for FPS control with slider."""

    # Signal emitted when FPS changes
    fps_changed = Signal(int)

    def __init__(self, initial_fps: int = 30, min_fps: int = 1, max_fps: int = 60):
        """Initialize FPS control widget.

        Args:
            initial_fps: Initial FPS value (default: 30)
            min_fps: Minimum FPS value (default: 1)
            max_fps: Maximum FPS value (default: 60)
        """
        super().__init__()

        self._current_fps = initial_fps
        self._min_fps = min_fps
        self._max_fps = max_fps
        self._is_capturing = False

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Create FPS control group
        fps_group = QGroupBox("Capture Frame Rate")
        fps_layout = QVBoxLayout(fps_group)

        # Create horizontal layout for slider and value label
        slider_layout = QHBoxLayout()

        # Create FPS slider
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setMinimum(min_fps)
        self._fps_slider.setMaximum(max_fps)
        self._fps_slider.setValue(initial_fps)
        self._fps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._fps_slider.setTickInterval(10)
        self._fps_slider.setToolTip(f"Adjust capture frame rate ({min_fps}-{max_fps} FPS)")

        # Create value label
        self._value_label = QLabel(f"{initial_fps} FPS")
        self._value_label.setMinimumWidth(60)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setStyleSheet("QLabel { font-weight: bold; }")

        # Add slider and label to horizontal layout
        slider_layout.addWidget(self._fps_slider, stretch=1)
        slider_layout.addWidget(self._value_label, stretch=0)

        # Create info label
        self._info_label = QLabel("Higher FPS = smoother capture, more CPU usage")
        self._info_label.setStyleSheet(
            "QLabel { color: #666; font-style: italic; font-size: 10px; }"
        )
        self._info_label.setWordWrap(True)

        # Add widgets to group layout
        fps_layout.addLayout(slider_layout)
        fps_layout.addWidget(self._info_label)

        # Add group to main layout
        layout.addWidget(fps_group)

        # Connect signal
        self._fps_slider.valueChanged.connect(self._on_fps_changed)

        logger.debug(f"FPS control widget initialized with FPS={initial_fps}")

    def set_fps(self, fps: int) -> None:
        """Set the current FPS value.

        Args:
            fps: FPS value to set (will be clamped to min/max range)
        """
        fps = max(self._min_fps, min(self._max_fps, fps))

        if fps == self._current_fps:
            return

        self._current_fps = fps

        # Update slider (this will trigger valueChanged signal)
        self._fps_slider.blockSignals(True)
        self._fps_slider.setValue(fps)
        self._fps_slider.blockSignals(False)

        # Update label
        self._value_label.setText(f"{fps} FPS")

        logger.debug(f"FPS set to {fps}")

    def get_fps(self) -> int:
        """Get the current FPS value.

        Returns:
            Current FPS value
        """
        return self._current_fps

    def set_capture_active(self, active: bool) -> None:
        """Set capture active state and enable/disable FPS control.

        Args:
            active: True if capture is active, False otherwise
        """
        self._is_capturing = active

        # Keep slider enabled during capture to allow dynamic FPS adjustment
        # This is intentional - FPS can be changed while capturing
        self._fps_slider.setEnabled(True)

        logger.debug(f"Capture active state set to {active}")

    def is_capture_active(self) -> bool:
        """Check if capture is currently active.

        Returns:
            True if capture is active, False otherwise
        """
        return self._is_capturing

    def _on_fps_changed(self, value: int) -> None:
        """Handle FPS slider value change.

        Args:
            value: New FPS value from slider
        """
        if value == self._current_fps:
            return

        self._current_fps = value

        # Update label
        self._value_label.setText(f"{value} FPS")

        # Emit signal
        self.fps_changed.emit(value)

        logger.info(f"FPS changed to {value}")
