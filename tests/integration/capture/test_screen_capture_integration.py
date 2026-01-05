"""Integration tests for screen capture with real hardware."""

import time

import pytest

from visionmate.capture.screen import MSSScreenCapture


@pytest.mark.integration
class TestMSSScreenCaptureIntegration:
    """Integration tests for MSSScreenCapture with real display."""

    def test_initialization(self):
        """Test MSSScreenCapture initializes correctly."""
        capture = MSSScreenCapture()
        assert capture.get_fps() == 30
        assert capture.get_target_window() is None
        assert not capture._is_capturing

    def test_fps_control(self):
        """Test FPS control within valid range."""
        capture = MSSScreenCapture()

        # Test valid FPS
        capture.set_fps(15)
        assert capture.get_fps() == 15

        capture.set_fps(60)
        assert capture.get_fps() == 60

        # Test FPS clamping
        capture.set_fps(0)
        assert capture.get_fps() == 1

        capture.set_fps(100)
        assert capture.get_fps() == 60

    def test_list_devices(self):
        """Test device enumeration returns monitor list."""
        capture = MSSScreenCapture()
        devices = capture.list_devices()

        assert isinstance(devices, list)
        # Should have at least one monitor
        assert len(devices) >= 1

        # Check device structure
        for device in devices:
            assert "id" in device
            assert "name" in device
            assert "width" in device
            assert "height" in device

    def test_capture_start_stop(self):
        """Test starting and stopping capture."""
        capture = MSSScreenCapture()

        # Start capture
        capture.start_capture(fps=30)
        assert capture._is_capturing
        assert capture._capture_thread is not None

        # Give thread time to start
        time.sleep(0.1)

        # Stop capture
        capture.stop_capture()
        assert not capture._is_capturing

    def test_frame_buffering(self):
        """Test frame buffer management."""
        capture = MSSScreenCapture()

        # Start capture
        capture.start_capture(fps=10)

        # Wait for some frames to be captured
        time.sleep(0.3)

        # Get frame
        frame = capture.get_frame()
        assert frame is not None
        assert frame.ndim == 3

        # Stop capture
        capture.stop_capture()

    def test_target_window_setting(self):
        """Test setting target window."""
        capture = MSSScreenCapture()

        # Initially no target
        assert capture.get_target_window() is None

        # Set target window (will be None if window doesn't exist)
        capture.set_target_window(12345)
        # Window won't exist, so should remain None
        assert capture.get_target_window() is None

    def test_capture_region(self):
        """Test getting capture region."""
        capture = MSSScreenCapture()
        region = capture.get_capture_region()

        assert isinstance(region, tuple)
        assert len(region) == 4
