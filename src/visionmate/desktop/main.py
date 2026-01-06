"""
Main window for Visionmate desktop application.

This module provides the main application window with control panel,
preview area, and status bar.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from visionmate.desktop.dialogs import AboutDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for Visionmate.

    Provides the primary user interface with:
    - Control panel on the left (collapsible)
    - Preview area in the center
    - Status bar at the bottom
    - Menu bar at the top
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the main window.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.info("Initializing MainWindow")

        self.setWindowTitle("Visionmate")
        self.setMinimumSize(800, 600)

        # Store app info for About dialog
        from visionmate.__main__ import APP_NAME, APP_VERSION

        self._app_name = APP_NAME
        self._app_version = APP_VERSION

        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        logger.info("MainWindow initialized successfully")

    def _setup_menu_bar(self) -> None:
        """Setup the menu bar with all menus."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        # Placeholder actions will be added later
        view_placeholder = QAction("(View options coming soon)", self)
        view_placeholder.setEnabled(False)
        view_menu.addAction(view_placeholder)

        # Session menu
        session_menu = menu_bar.addMenu("&Session")
        # Placeholder actions will be added later
        session_placeholder = QAction("(Session controls coming soon)", self)
        session_placeholder.setEnabled(False)
        session_menu.addAction(session_placeholder)

        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")
        # Placeholder actions will be added later
        settings_placeholder = QAction("(Settings coming soon)", self)
        settings_placeholder.setEnabled(False)
        settings_menu.addAction(settings_placeholder)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.setStatusTip("About Visionmate")
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        logger.debug("Menu bar setup complete")

    def _show_about_dialog(self) -> None:
        """Show the About dialog."""
        logger.debug("Showing About dialog")
        dialog = AboutDialog(self._app_name, self._app_version, self)
        dialog.exec()

    def _setup_central_widget(self) -> None:
        """Setup the central widget with basic layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create placeholder for preview area
        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        placeholder_label = QLabel("Preview Area\n\nSelect a video source to begin")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet(
            """
            QLabel {
                color: #888888;
                font-size: 16px;
                padding: 40px;
            }
            """
        )
        preview_layout.addWidget(placeholder_label)

        # Add preview area to main layout
        main_layout.addWidget(preview_area, stretch=1)

        logger.debug("Central widget setup complete")

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        status_bar = self.statusBar()
        status_bar.showMessage("Ready")
        logger.debug("Status bar setup complete")
