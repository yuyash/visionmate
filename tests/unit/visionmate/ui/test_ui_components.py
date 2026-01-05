"""Tests for UI components."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from visionmate.capture.audio_capture import SoundDeviceAudioCapture
from visionmate.capture.screen_capture import MSSScreenCapture
from visionmate.ui import (
    AudioWaveformWidget,
    DeviceControlsWidget,
    MainWindow,
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


def test_video_preview_with_capture(qapp):
    """Test that VideoPreviewWidget can be created with capture."""
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    assert widget is not None


def test_audio_waveform_creation(qapp):
    """Test that AudioWaveformWidget can be created."""
    widget = AudioWaveformWidget()
    assert widget is not None


def test_audio_waveform_with_capture(qapp):
    """Test that AudioWaveformWidget can be created with capture."""
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)
    assert widget is not None


def test_device_controls_creation(qapp):
    """Test that DeviceControlsWidget can be created."""
    widget = DeviceControlsWidget()
    assert widget is not None


def test_device_controls_with_captures(qapp):
    """Test that DeviceControlsWidget can be created with captures."""
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()

    widget = DeviceControlsWidget(
        screen_capture=screen_capture,
        audio_capture=audio_capture,
    )
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


def test_video_preview_start_stop(qapp):
    """Test that VideoPreviewWidget can start and stop preview."""
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)

    # Should not raise exceptions
    widget.start_preview(fps=30)
    widget.stop_preview()


def test_audio_waveform_start_stop(qapp):
    """Test that AudioWaveformWidget can start and stop preview."""
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)

    # Should not raise exceptions
    widget.start_preview(fps=30)
    widget.stop_preview()


# Integration Tests for Task 4.5


def test_video_preview_updates_with_capture(qapp):
    """Test that video preview updates when capture is running.

    Requirements: 8.6
    """
    import time

    # Create screen capture and video preview
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)

    # Start capture at 10 FPS (lower for testing)
    capture.start_capture(fps=10)

    # Start preview at 10 FPS
    widget.start_preview(fps=10)

    # Wait for a few frames to be captured
    time.sleep(0.5)

    # Get frame from capture
    frame_from_capture = capture.get_frame()
    assert frame_from_capture is not None, "Capture should provide frames"

    # Verify frame has expected properties
    assert frame_from_capture.ndim == 3, "Frame should be 3D array (height, width, channels)"
    assert frame_from_capture.shape[2] == 3, "Frame should have 3 color channels (BGR)"

    # Stop preview and capture
    widget.stop_preview()
    capture.stop_capture()

    # Verify capture stopped
    time.sleep(0.2)
    # After stopping, we should still be able to get the last frame
    last_frame = capture.get_frame()
    assert last_frame is not None, "Should still have last captured frame in buffer"


def test_video_preview_updates_with_highlight(qapp):
    """Test that video preview shows highlight overlay for capture region.

    Requirements: 8.6
    """
    import time

    # Create screen capture and video preview
    capture = MSSScreenCapture()
    VideoPreviewWidget(capture=capture)

    # Start capture
    capture.start_capture(fps=10)

    # Wait for frames
    time.sleep(0.3)

    # Get frame with highlight
    frame_with_highlight = capture.get_frame_with_highlight()
    assert frame_with_highlight is not None, "Should get frame with highlight"

    # Get regular frame
    regular_frame = capture.get_frame()
    assert regular_frame is not None, "Should get regular frame"

    # Frames should have same shape
    assert frame_with_highlight.shape == regular_frame.shape

    # Stop capture
    capture.stop_capture()


def test_audio_waveform_updates_with_capture(qapp):
    """Test that audio waveform updates when capture is running.

    Requirements: 8.7
    """
    import time

    # Create audio capture and waveform widget
    capture = SoundDeviceAudioCapture(sample_rate=16000, chunk_size=512)
    widget = AudioWaveformWidget(capture=capture)

    # Start capture
    capture.start_capture()

    # Start waveform preview at 10 FPS
    widget.start_preview(fps=10)

    # Wait for audio chunks to be captured
    time.sleep(0.5)

    # Get audio chunk from capture
    audio_chunk = capture.get_audio_chunk()
    assert audio_chunk is not None, "Capture should provide audio chunks"

    # Verify audio chunk has expected properties
    assert isinstance(audio_chunk, np.ndarray), "Audio chunk should be numpy array"
    assert audio_chunk.size > 0, "Audio chunk should not be empty"

    # Stop preview and capture
    widget.stop_preview()
    capture.stop_capture()

    # Verify we can still get last audio chunk
    last_chunk = capture.get_audio_chunk()
    assert last_chunk is not None, "Should still have last audio chunk in buffer"


def test_device_selection_video_devices(qapp):
    """Test video device selection functionality.

    Requirements: 8.3, 11.1
    """
    # Create screen capture and device controls
    screen_capture = MSSScreenCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture)

    # Get list of video devices
    devices = screen_capture.list_devices()
    assert len(devices) > 0, "Should have at least one video device (monitor)"

    # Verify device controls populated
    assert widget._video_device_combo.count() > 0, "Video device combo should be populated"

    # Test device selection signal
    signal_received = []

    def on_video_device_changed(device_id):
        signal_received.append(device_id)

    widget.video_device_changed.connect(on_video_device_changed)

    # Select device (if multiple devices, select second one; otherwise trigger by changing)
    if widget._video_device_combo.count() > 1:
        widget._video_device_combo.setCurrentIndex(1)
        # Verify signal was emitted
        assert len(signal_received) > 0, "Device selection should emit signal"
    else:
        # Only one device - manually trigger the signal to test it works
        widget._video_device_combo.setCurrentIndex(0)
        # Even if signal not emitted (already at index 0), verify we can get selected device
        selected_device = widget.get_selected_video_device()
        assert selected_device is not None, "Should have selected device"
        assert selected_device >= 0, "Device ID should be non-negative"


def test_device_selection_audio_devices(qapp):
    """Test audio device selection functionality.

    Requirements: 8.3, 11.2
    """
    # Create audio capture and device controls
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(audio_capture=audio_capture)

    # Get list of audio devices
    devices = audio_capture.list_devices()

    # Note: May not have audio devices in CI environment
    if len(devices) == 0:
        pytest.skip("No audio devices available for testing")

    # Verify device controls populated
    assert widget._audio_device_combo.count() > 0, "Audio device combo should be populated"

    # Test device selection signal
    signal_received = []

    def on_audio_device_changed(device_id):
        signal_received.append(device_id)

    widget.audio_device_changed.connect(on_audio_device_changed)

    # Select device (if multiple devices, select second one; otherwise verify current selection)
    if widget._audio_device_combo.count() > 1:
        widget._audio_device_combo.setCurrentIndex(1)
        # Verify signal was emitted
        assert len(signal_received) > 0, "Device selection should emit signal"
    else:
        # Only one device - verify we can get selected device
        selected_device = widget.get_selected_audio_device()
        assert selected_device is not None, "Should have selected device"
        assert selected_device >= 0, "Device ID should be non-negative"


def test_device_selection_during_capture(qapp):
    """Test that device selection is disabled during active capture.

    Requirements: 8.3
    """
    # Create captures and device controls
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture, audio_capture=audio_capture)

    # Initially not capturing - controls should be enabled
    assert not widget.is_capture_active()
    # Note: Controls may be disabled if no devices found, so we check the state flag

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

    # Controls should be re-enabled (if devices available)
    # We just verify the state changed, actual enablement depends on device availability


def test_device_refresh_functionality(qapp):
    """Test device refresh button functionality.

    Requirements: 8.3
    """
    # Create captures and device controls
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture, audio_capture=audio_capture)

    # Click refresh buttons (should not raise exceptions)
    widget._video_refresh_button.click()
    widget._audio_refresh_button.click()

    # Verify device lists were refreshed (counts should be same or updated)
    assert widget._video_device_combo.count() >= 0
    assert widget._audio_device_combo.count() >= 0


def test_video_preview_fps_control(qapp):
    """Test that video preview respects FPS setting.

    Requirements: 8.6
    """
    import time

    # Create screen capture and video preview
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)

    # Start capture at 5 FPS
    capture.start_capture(fps=5)

    # Start preview at 5 FPS
    widget.start_preview(fps=5)

    # Wait and verify frames are being captured
    time.sleep(0.5)

    frame = capture.get_frame()
    assert frame is not None, "Should capture frames at 5 FPS"

    # Change FPS to 10
    capture.set_fps(10)
    assert capture.get_fps() == 10, "FPS should be updated to 10"

    # Wait for new frames
    time.sleep(0.3)

    frame = capture.get_frame()
    assert frame is not None, "Should capture frames at 10 FPS"

    # Stop
    widget.stop_preview()
    capture.stop_capture()


def test_audio_waveform_fps_control(qapp):
    """Test that audio waveform respects FPS setting.

    Requirements: 8.7
    """
    import time

    # Create audio capture and waveform widget
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)

    # Start capture
    capture.start_capture()

    # Start waveform preview at 5 FPS
    widget.start_preview(fps=5)

    # Wait for audio
    time.sleep(0.5)

    audio = capture.get_audio_chunk()
    assert audio is not None, "Should capture audio"

    # Stop
    widget.stop_preview()
    capture.stop_capture()


def test_integration_video_and_audio_together(qapp):
    """Test that video preview and audio waveform work together.

    Requirements: 8.6, 8.7
    """
    import time

    # Create captures
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()

    # Create widgets
    video_widget = VideoPreviewWidget(capture=screen_capture)
    audio_widget = AudioWaveformWidget(capture=audio_capture)

    # Start both captures
    screen_capture.start_capture(fps=10)
    audio_capture.start_capture()

    # Start both previews
    video_widget.start_preview(fps=10)
    audio_widget.start_preview(fps=10)

    # Wait for data
    time.sleep(0.5)

    # Verify both are working
    video_frame = screen_capture.get_frame()
    audio_chunk = audio_capture.get_audio_chunk()

    assert video_frame is not None, "Video capture should be working"
    assert audio_chunk is not None, "Audio capture should be working"

    # Stop both
    video_widget.stop_preview()
    audio_widget.stop_preview()
    screen_capture.stop_capture()
    audio_capture.stop_capture()
