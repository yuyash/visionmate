"""
Window selector dialog for selecting specific windows to capture.

This module provides a dialog for users to select which windows to capture
when using the "Selected Windows" capture mode.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import WindowRegion

logger = logging.getLogger(__name__)


class WindowSelectorDialog(QDialog):
    """Dialog for selecting windows to capture.

    Displays a list of available windows on the screen and allows
    the user to select multiple windows for capture.
    """

    def __init__(
        self,
        available_windows: list[WindowRegion],
        selected_titles: Optional[list[str]] = None,
        parent: Optional[QDialog | QWidget] = None,
    ):
        """Initialize the WindowSelectorDialog.

        Args:
            available_windows: List of available WindowRegion objects
            selected_titles: List of currently selected window titles
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing WindowSelectorDialog")

        self._available_windows = available_windows
        self._selected_titles = selected_titles or []

        self.setWindowTitle("Select Windows to Capture")
        self.setMinimumSize(400, 300)

        self._setup_ui()
        self._populate_windows()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Select one or more windows to capture.\n"
            "Only the selected windows will be captured from the screen."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Selection buttons
        selection_buttons_layout = QHBoxLayout()

        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self._select_all)
        selection_buttons_layout.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self._deselect_all)
        selection_buttons_layout.addWidget(deselect_all_button)

        selection_buttons_layout.addStretch()

        layout.addLayout(selection_buttons_layout)

        # Window list
        self._window_list = QListWidget()
        self._window_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self._window_list)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        logger.debug("WindowSelectorDialog UI setup complete")

    def _select_all(self) -> None:
        """Select all windows in the list."""
        self._window_list.selectAll()
        logger.debug("Selected all windows")

    def _deselect_all(self) -> None:
        """Deselect all windows in the list."""
        self._window_list.clearSelection()
        logger.debug("Deselected all windows")

    def _populate_windows(self) -> None:
        """Populate the window list with available windows."""
        self._window_list.clear()

        for window in self._available_windows:
            # Create display text with title and dimensions
            display_text = f"{window.title} ({window.width}×{window.height})"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, window.title)

            # Select if in selected_titles
            if window.title in self._selected_titles:
                item.setSelected(True)

            self._window_list.addItem(item)

        logger.debug(f"Populated {len(self._available_windows)} windows")

    def get_selected_titles(self) -> list[str]:
        """Get list of selected window titles.

        Returns:
            List of selected window titles
        """
        selected_items = self._window_list.selectedItems()
        titles = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        logger.debug(f"Selected {len(titles)} windows")
        return titles
