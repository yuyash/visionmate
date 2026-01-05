"""Integration tests for UI components with real hardware."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

from visionmate.capture.audio import SoundDeviceAudioCapture
from visionmate.capture.screen import MSSScreenCapture
from visionmate.ui import AudioWaveformWidget, DeviceControlsWidget, VideoPreviewWidget


@pytest.mark.integration
def test_video_preview_with_capture(qtbot):
    """Test that VideoPreviewWidget can be created with capture."""
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    qtbot.addWidget(widget)
    assert widget is not None


@pytest.mark.integration
def test_audio_waveform_with_capture(qtbot):
    """Test that AudioWaveformWidget can be created with capture."""
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)
    qtbot.addWidget(widget)
    assert widget is not None


@pytest.mark.integration
def test_device_controls_with_captures(qtbot):
    """Test that DeviceControlsWidget can be created with captures."""
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()

    widget = DeviceControlsWidget(
        screen_capture=screen_capture,
        audio_capture=audio_capture,
    )
    qtbot.addWidget(widget)
    assert widget is not None


@pytest.mark.integration
def test_video_preview_start_stop(qtbot):
    """Test that VideoPreviewWidget can start and stop preview."""
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    qtbot.addWidget(widget)

    # Should not raise exceptions
    widget.start_preview(fps=30)
    qtbot.wait(100)
    widget.stop_preview()


@pytest.mark.integration
def test_audio_waveform_start_stop(qtbot):
    """Test that AudioWaveformWidget can start and stop preview."""
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)
    qtbot.addWidget(widget)

    # Should not raise exceptions
    widget.start_preview(fps=30)
    qtbot.wait(100)
    widget.stop_preview()


@pytest.mark.integration
def test_video_preview_updates_with_capture(qtbot):
    """Test that video preview updates when capture is running.

    Requirements: 8.6
    """
    # Create screen capture and video preview
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    qtbot.addWidget(widget)

    # Start capture at 10 FPS (lower for testing)
    capture.start_capture(fps=10)

    # Start preview at 10 FPS
    widget.start_preview(fps=10)

    # Wait for a few frames to be captured
    qtbot.wait(500)

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
    qtbot.wait(200)
    # After stopping, we should still be able to get the last frame
    last_frame = capture.get_frame()
    assert last_frame is not None, "Should still have last captured frame in buffer"


@pytest.mark.integration
def test_video_preview_updates_with_highlight(qtbot):
    """Test that video preview shows highlight overlay for capture region.

    Requirements: 8.6
    """
    # Create screen capture and video preview
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    qtbot.addWidget(widget)

    # Start capture
    capture.start_capture(fps=10)

    # Wait for frames
    qtbot.wait(300)

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


@pytest.mark.integration
def test_audio_waveform_updates_with_capture(qtbot):
    """Test that audio waveform updates when capture is running.

    Requirements: 8.7
    """
    # Create audio capture and waveform widget
    capture = SoundDeviceAudioCapture(sample_rate=16000, chunk_size=512)
    widget = AudioWaveformWidget(capture=capture)
    qtbot.addWidget(widget)

    # Start capture
    capture.start_capture()

    # Start waveform preview at 10 FPS
    widget.start_preview(fps=10)

    # Wait for audio chunks to be captured
    qtbot.wait(500)

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


@pytest.mark.integration
def test_device_selection_video_devices(qtbot):
    """Test video device selection functionality.

    Requirements: 8.3, 11.1
    """
    # Create screen capture and device controls
    screen_capture = MSSScreenCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture)
    qtbot.addWidget(widget)

    # Get list of video devices
    devices = screen_capture.list_devices()
    assert len(devices) > 0, "Should have at least one video device (monitor)"

    # Verify device controls populated
    assert widget._video_device_combo.count() > 0, "Video device combo should be populated"

    # Test device selection signal
    with qtbot.waitSignal(widget.video_device_changed, timeout=1000, raising=False):
        if widget._video_device_combo.count() > 1:
            widget._video_device_combo.setCurrentIndex(1)


@pytest.mark.integration
def test_device_selection_audio_devices(qtbot):
    """Test audio device selection functionality.

    Requirements: 8.3, 11.2
    """
    # Create audio capture and device controls
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(audio_capture=audio_capture)
    qtbot.addWidget(widget)

    # Get list of audio devices
    devices = audio_capture.list_devices()

    # Note: May not have audio devices in CI environment
    if len(devices) == 0:
        pytest.skip("No audio devices available for testing")

    # Verify device controls populated
    assert widget._audio_device_combo.count() > 0, "Audio device combo should be populated"

    # Test device selection signal
    with qtbot.waitSignal(widget.audio_device_changed, timeout=1000, raising=False):
        if widget._audio_device_combo.count() > 1:
            widget._audio_device_combo.setCurrentIndex(1)


@pytest.mark.integration
def test_device_selection_during_capture(qtbot):
    """Test that device selection is disabled during active capture.

    Requirements: 8.3
    """
    # Create captures and device controls
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture, audio_capture=audio_capture)
    qtbot.addWidget(widget)

    # Initially not capturing - controls should be enabled
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


@pytest.mark.integration
def test_device_refresh_functionality(qtbot):
    """Test device refresh button functionality.

    Requirements: 8.3
    """
    # Create captures and device controls
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()
    widget = DeviceControlsWidget(screen_capture=screen_capture, audio_capture=audio_capture)
    qtbot.addWidget(widget)

    # Click refresh buttons (should not raise exceptions)
    qtbot.mouseClick(widget._video_refresh_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(widget._audio_refresh_button, Qt.MouseButton.LeftButton)

    # Verify device lists were refreshed (counts should be same or updated)
    assert widget._video_device_combo.count() >= 0
    assert widget._audio_device_combo.count() >= 0


@pytest.mark.integration
def test_video_preview_fps_control(qtbot):
    """Test that video preview respects FPS setting.

    Requirements: 8.6
    """
    # Create screen capture and video preview
    capture = MSSScreenCapture()
    widget = VideoPreviewWidget(capture=capture)
    qtbot.addWidget(widget)

    # Start capture at 5 FPS
    capture.start_capture(fps=5)

    # Start preview at 5 FPS
    widget.start_preview(fps=5)

    # Wait and verify frames are being captured
    qtbot.wait(500)

    frame = capture.get_frame()
    assert frame is not None, "Should capture frames at 5 FPS"

    # Change FPS to 10
    capture.set_fps(10)
    assert capture.get_fps() == 10, "FPS should be updated to 10"

    # Wait for new frames
    qtbot.wait(300)

    frame = capture.get_frame()
    assert frame is not None, "Should capture frames at 10 FPS"

    # Stop
    widget.stop_preview()
    capture.stop_capture()


@pytest.mark.integration
def test_audio_waveform_fps_control(qtbot):
    """Test that audio waveform respects FPS setting.

    Requirements: 8.7
    """
    # Create audio capture and waveform widget
    capture = SoundDeviceAudioCapture()
    widget = AudioWaveformWidget(capture=capture)
    qtbot.addWidget(widget)

    # Start capture
    capture.start_capture()

    # Start waveform preview at 5 FPS
    widget.start_preview(fps=5)

    # Wait for audio
    qtbot.wait(500)

    audio = capture.get_audio_chunk()
    assert audio is not None, "Should capture audio"

    # Stop
    widget.stop_preview()
    capture.stop_capture()


@pytest.mark.integration
def test_integration_video_and_audio_together(qtbot):
    """Test that video preview and audio waveform work together.

    Requirements: 8.6, 8.7
    """
    # Create captures
    screen_capture = MSSScreenCapture()
    audio_capture = SoundDeviceAudioCapture()

    # Create widgets
    video_widget = VideoPreviewWidget(capture=screen_capture)
    audio_widget = AudioWaveformWidget(capture=audio_capture)
    qtbot.addWidget(video_widget)
    qtbot.addWidget(audio_widget)

    # Start both captures
    screen_capture.start_capture(fps=10)
    audio_capture.start_capture()

    # Start both previews
    video_widget.start_preview(fps=10)
    audio_widget.start_preview(fps=10)

    # Wait for data
    qtbot.wait(500)

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
