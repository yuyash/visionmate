"""Tests for UI components."""

import pytest
from PySide6.QtWidgets import QApplication

from visionmate.ui import MainWindow
from visionmate.ui.widgets import (
    AudioWaveformWidget,
    DeviceControlsWidget,
    VideoPreviewWidget,
)


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_main_window_creation(qapp):
    """Test that MainWindow can be created."""
    window = MainWindow()
    assert window is not None
    assert window.windowTitle() == "VisionMate - Real-time QA Assistant"
    assert window.minimumWidth() == 1200
    assert window.minimumHeight() == 800


def test_main_window_layouts(qapp):
    """Test that MainWindow has control and preview panel layouts."""
    window = MainWindow()

    control_layout = window.get_control_panel_layout()
    assert control_layout is not None

    preview_layout = window.get_preview_panel_layout()
    assert preview_layout is not None


def test_video_preview_creation(qapp):
    """Test that VideoPreviewWidget can be created."""
    widget = VideoPreviewWidget()
    assert widget is not None


def test_audio_waveform_creation(qapp):
    """Test that AudioWaveformWidget can be created."""
    widget = AudioWaveformWidget()
    assert widget is not None


def test_device_controls_creation(qapp):
    """Test that DeviceControlsWidget can be created."""
    widget = DeviceControlsWidget()
    assert widget is not None


def test_device_controls_signals(qapp):
    """Test that DeviceControlsWidget has required signals."""
    widget = DeviceControlsWidget()

    # Check that signals exist
    assert hasattr(widget, "video_device_changed")
    assert hasattr(widget, "audio_device_changed")


def test_device_controls_capture_state(qapp):
    """Test that DeviceControlsWidget can enable/disable controls based on capture state."""
    widget = DeviceControlsWidget()

    # Initially not capturing
    assert not widget.is_capture_active()

    # Set capture active
    widget.set_capture_active(True)
    assert widget.is_capture_active()

    # Device controls should be disabled
    assert not widget._video_device_combo.isEnabled()
    assert not widget._video_refresh_button.isEnabled()
    assert not widget._audio_device_combo.isEnabled()
    assert not widget._audio_refresh_button.isEnabled()

    # Set capture inactive
    widget.set_capture_active(False)
    assert not widget.is_capture_active()

    # Device controls should be enabled (if devices are available)
    # Note: Controls may still be disabled if no devices found
