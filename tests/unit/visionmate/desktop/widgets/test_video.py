"""Unit tests for VideoPreviewWidget."""

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from visionmate.core.models import Resolution, VideoFrame, VideoSourceType
from visionmate.desktop.widgets import VideoPreviewWidget


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_capture():
    """Create a mock VideoCaptureInterface."""
    capture = Mock()
    capture.get_frame = Mock(return_value=None)
    capture.is_capturing = Mock(return_value=True)
    return capture


@pytest.fixture
def sample_frame():
    """Create a sample VideoFrame for testing."""
    # Create a simple RGB image (100x100 red square)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Red channel

    frame = VideoFrame(
        image=image,
        timestamp=datetime.now(timezone.utc),
        source_id="screen_1",
        source_type=VideoSourceType.SCREEN,
        resolution=Resolution(width=100, height=100),
        fps=1,
        frame_number=1,
    )
    return frame


def test_video_preview_widget_creation(qapp, mock_capture):
    """Test VideoPreviewWidget can be created."""
    widget = VideoPreviewWidget(
        source_id="screen_1",
        capture=mock_capture,
    )

    assert widget is not None
    assert widget.get_source_id() == "screen_1"


def test_video_preview_widget_signals(qapp, mock_capture):
    """Test VideoPreviewWidget emits signals correctly."""
    widget = VideoPreviewWidget(
        source_id="screen_1",
        capture=mock_capture,
    )

    # Test close signal
    close_signal_received = []

    def on_close(source_id):
        close_signal_received.append(source_id)

    widget.close_requested.connect(on_close)
    widget._on_close_clicked()

    assert len(close_signal_received) == 1
    assert close_signal_received[0] == "screen_1"

    # Test info button (now shows tooltip instead of emitting signal)
    # Just verify the button exists and can be clicked without error
    widget._on_info_clicked()  # Should not raise exception


def test_video_preview_widget_frame_update(qapp, mock_capture, sample_frame):
    """Test VideoPreviewWidget updates frames correctly."""
    # Setup mock to return a frame
    mock_capture.get_frame.return_value = sample_frame

    widget = VideoPreviewWidget(
        source_id="screen_1",
        capture=mock_capture,
    )

    # Trigger frame update
    widget._update_frame()

    # Verify get_frame was called
    mock_capture.get_frame.assert_called()

    # Verify pixmap was set (not None)
    assert widget._video_label.pixmap() is not None


def test_video_preview_widget_cleanup(qapp, mock_capture):
    """Test VideoPreviewWidget cleanup stops timer."""
    widget = VideoPreviewWidget(
        source_id="screen_1",
        capture=mock_capture,
    )

    # Verify timer is running
    assert widget._update_timer is not None
    assert widget._update_timer.isActive()

    # Cleanup
    widget.cleanup()

    # Verify timer is stopped
    assert not widget._update_timer.isActive()
