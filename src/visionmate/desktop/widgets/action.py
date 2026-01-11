"""Action container widget for session control, request input, and response display.

This module provides the ActionContainer widget that combines SessionControlWidget,
RequestWidget, and ResponseWidget in a vertical stack.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from visionmate.desktop.widgets.metrics import MetricsWidget
from visionmate.desktop.widgets.request import RequestWidget
from visionmate.desktop.widgets.response import ResponseWidget
from visionmate.desktop.widgets.session import SessionControlWidget

logger = logging.getLogger(__name__)


class ActionContainer(QWidget):
    """Container for action widgets.

    Manages:
    - Session control (Start, Stop, Reset buttons)
    - Request input (text instructions)
    - Response display (VLM responses)

    Widgets are stacked vertically from top to bottom.
    """

    # Forward signals from child widgets
    start_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()
    request_submitted = Signal(str)  # request_text
    status_message = Signal(str, int)  # message, timeout

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the ActionContainer.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing ActionContainer")

        # Widgets
        self._session_control_widget: Optional[SessionControlWidget] = None
        self._request_widget: Optional[RequestWidget] = None
        self._response_widget: Optional[ResponseWidget] = None
        self._metrics_widget: Optional[MetricsWidget] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Set fixed width for action panel
        self.setFixedWidth(320)
        self.setStyleSheet(
            """
            QWidget#actionContainer {
                background-color: #f8f8f8;
            }
            """
        )
        self.setObjectName("actionContainer")

        # Create scroll area for actions
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        # Create actions container
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(12)

        # Add Session Control widget
        self._session_control_widget = SessionControlWidget()
        actions_layout.addWidget(self._session_control_widget)

        # Add Metrics widget
        self._metrics_widget = MetricsWidget()
        actions_layout.addWidget(self._metrics_widget)

        # Add Request widget
        self._request_widget = RequestWidget()
        actions_layout.addWidget(self._request_widget)

        # Add Response widget
        self._response_widget = ResponseWidget()
        self._response_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        actions_layout.addWidget(self._response_widget, stretch=1)

        # Set actions widget in scroll area
        scroll_area.setWidget(actions_widget)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        logger.debug("ActionContainer UI setup complete")

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        if self._session_control_widget is not None:
            # Forward session control signals
            self._session_control_widget.start_requested.connect(self.start_requested)
            self._session_control_widget.stop_requested.connect(self.stop_requested)
            self._session_control_widget.reset_requested.connect(self.reset_requested)

        if self._request_widget is not None:
            # Forward request signals
            self._request_widget.request_submitted.connect(self.request_submitted)

        if self._response_widget is not None:
            # Forward response signals
            self._response_widget.status_message.connect(self.status_message)

        logger.debug("ActionContainer signals connected")

    # Expose child widget methods for external access

    def get_session_control_widget(self) -> Optional[SessionControlWidget]:
        """Get the session control widget.

        Returns:
            SessionControlWidget instance
        """
        return self._session_control_widget

    def get_request_widget(self) -> Optional[RequestWidget]:
        """Get the request widget.

        Returns:
            RequestWidget instance
        """
        return self._request_widget

    def get_response_widget(self) -> Optional[ResponseWidget]:
        """Get the response widget.

        Returns:
            ResponseWidget instance
        """
        return self._response_widget

    def get_metrics_widget(self) -> Optional[MetricsWidget]:
        """Get the metrics widget.

        Returns:
            MetricsWidget instance
        """
        return self._metrics_widget
