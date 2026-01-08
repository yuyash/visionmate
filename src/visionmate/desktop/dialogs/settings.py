"""Settings dialog for video capture configuration.

This module provides the SettingsDialog for configuring video capture settings
such as FPS (frames per second).
"""

from logging import Logger, getLogger
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger: Logger = getLogger(name=__name__)


class SettingsDialog(QDialog):
    """Dialog for configuring video capture settings.

    Provides controls for:
    - FPS (frames per second) setting
    """

    def __init__(
        self,
        current_fps: int = 1,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the SettingsDialog.

        Args:
            current_fps: Current FPS value
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug(f"Initializing SettingsDialog with FPS={current_fps}")

        self._current_fps = current_fps
        self._new_fps = current_fps

        self.setWindowTitle("Capture Settings")
        self.setModal(True)
        self.setMinimumWidth(300)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Add description label
        description = QLabel(
            "Configure video capture settings.\nChanges will be applied when you click OK."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(description)

        # Create form layout for settings
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # FPS setting
        fps_container = QWidget()
        fps_layout = QHBoxLayout(fps_container)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.setSpacing(5)

        self._fps_spinbox = QSpinBox()
        self._fps_spinbox.setMinimum(1)
        self._fps_spinbox.setMaximum(240)
        self._fps_spinbox.setValue(self._current_fps)
        self._fps_spinbox.setToolTip(
            "Frame capture rate (1-240 fps)\nHigher values capture more frames per second"
        )
        self._fps_spinbox.valueChanged.connect(self._on_fps_changed)

        fps_unit_label = QLabel("fps")
        fps_unit_label.setStyleSheet("color: #666666;")

        fps_layout.addWidget(self._fps_spinbox)
        fps_layout.addWidget(fps_unit_label)
        fps_layout.addStretch()

        fps_label = QLabel("Capture FPS:")
        fps_label.setToolTip("How often to capture frames (1 = once per second)")

        form_layout.addRow(fps_label, fps_container)

        layout.addLayout(form_layout)

        # Add stretch to push buttons to bottom
        layout.addStretch()

        # Add dialog buttons (OK and Cancel)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        logger.debug("SettingsDialog UI setup complete")

    def _on_fps_changed(self, value: int) -> None:
        """Handle FPS value change.

        Args:
            value: New FPS value
        """
        self._new_fps = value
        logger.debug(f"FPS changed to: {value}")

    def get_fps(self) -> int:
        """Get the selected FPS value.

        Returns:
            FPS value (1-240)
        """
        return self._new_fps

    def exec(self) -> bool:
        """Execute the dialog and return whether OK was clicked.

        Returns:
            True if OK was clicked, False if Cancel was clicked
        """
        result = super().exec()
        return result == QDialog.DialogCode.Accepted
