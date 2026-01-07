"""
Dialogs for Visionmate desktop application.

This module provides various dialog windows including About dialog,
Settings dialog, and other modal dialogs.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.desktop.dialogs.window_selector import WindowSelectorDialog

logger = logging.getLogger(__name__)


class AboutDialog(QDialog):
    """About dialog displaying application information.

    Shows:
    - Application name
    - Version number
    - Copyright information
    - License information
    """

    def __init__(self, app_name: str, app_version: str, parent: Optional[QWidget] = None):
        """Initialize the About dialog.

        Args:
            app_name: Application name
            app_version: Application version
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing AboutDialog")

        self.setWindowTitle(f"About {app_name}")
        self.setModal(True)
        self.setMinimumWidth(400)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # Application name
        name_label = QLabel(f"<h1>{app_name}</h1>")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # Version
        version_label = QLabel(f"<p>Version {app_version}</p>")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #666666;")
        layout.addWidget(version_label)

        # Description
        description_label = QLabel(
            "<p>A multi-modal assistant that continuously observes video and audio "
            "and streams into Vision Language Models.</p>"
        )
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

        # Copyright
        copyright_label = QLabel("<p>Copyright © 2025 Visionmate Contributors</p>")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(copyright_label)

        # License
        license_label = QLabel(
            "<p>Licensed under the MIT License<br>See LICENSE file for details</p>"
        )
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(license_label)

        # Close button
        close_button = QPushButton("Close")
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

        logger.debug("AboutDialog initialized successfully")


__all__ = ["AboutDialog", "WindowSelectorDialog"]
