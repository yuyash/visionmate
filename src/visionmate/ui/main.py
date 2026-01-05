"""Main window for the Desktop UI."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Main application window for VisionMate Desktop UI."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.setWindowTitle("VisionMate - Real-time QA Assistant")
        self.setMinimumSize(1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create control panel area (left side)
        self.control_panel = QWidget()
        self.control_panel.setMaximumWidth(400)
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        self.control_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.control_panel_layout.setSpacing(10)
        self.control_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create preview panel area (right side)
        self.preview_panel = QWidget()
        self.preview_panel_layout = QVBoxLayout(self.preview_panel)
        self.preview_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_panel_layout.setSpacing(10)

        # Add panels to main layout
        main_layout.addWidget(self.control_panel, stretch=0)
        main_layout.addWidget(self.preview_panel, stretch=1)

    def get_control_panel_layout(self) -> QVBoxLayout:
        """Get the control panel layout for adding widgets.

        Returns:
            QVBoxLayout for control panel
        """
        return self.control_panel_layout

    def get_preview_panel_layout(self) -> QVBoxLayout:
        """Get the preview panel layout for adding widgets.

        Returns:
            QVBoxLayout for preview panel
        """
        return self.preview_panel_layout
