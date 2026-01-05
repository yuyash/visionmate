"""Unit tests for screen capture module."""

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from visionmate.capture.screen_capture import (
    MSSScreenCapture,
    ScreenCaptureInterface,
    UVCScreenCapture,
    WindowDetector,
    WindowInfo,
)


class TestScreenCaptureInterface:
    """Test ScreenCaptureInterface abstract base class."""

    def test_interface_is_abstract(self):
        """Test that ScreenCaptureInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            ScreenCaptureInterface()


class TestMSSScreenCapture:
    """Test MSSScreenCapture implementation."""

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

    @patch("visionmate.capture.screen_capture.mss.mss")
    def test_capture_start_stop(self, mock_mss):
        """Test starting and stopping capture."""
        # Mock MSS
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},  # Monitor 0 (all monitors)
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor 1
        ]
        mock_mss.return_value = mock_sct

        capture = MSSScreenCapture()
        capture._sct = mock_sct

        # Start capture
        capture.start_capture(fps=30)
        assert capture._is_capturing
        assert capture._capture_thread is not None

        # Give thread time to start
        time.sleep(0.1)

        # Stop capture
        capture.stop_capture()
        assert not capture._is_capturing

    @patch("visionmate.capture.screen_capture.mss.mss")
    def test_frame_buffering(self, mock_mss):
        """Test frame buffer management."""
        # Mock MSS
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},
            {"left": 0, "top": 0, "width": 100, "height": 100},
        ]

        # Create mock screenshot that returns numpy array
        def create_mock_img():
            mock_img = MagicMock()
            # Make it behave like a numpy array
            test_array = np.zeros((100, 100, 4), dtype=np.uint8)
            mock_img.__array__ = lambda *args, **kwargs: test_array
            return mock_img

        mock_sct.grab.return_value = create_mock_img()
        mock_mss.return_value = mock_sct

        capture = MSSScreenCapture()
        capture._sct = mock_sct

        # Start capture
        capture.start_capture(fps=10)

        # Wait for some frames to be captured
        time.sleep(0.3)

        # Get frame
        frame = capture.get_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)

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


class TestWindowDetector:
    """Test WindowDetector for UVC capture."""

    def test_initialization(self):
        """Test WindowDetector initializes with correct defaults."""
        detector = WindowDetector()
        assert detector.min_window_size_percent == 25.0
        assert detector.min_aspect_ratio == 0.75
        assert detector.max_aspect_ratio == 2.33
        assert detector.get_confidence() == 0.0

    def test_detect_window_with_clear_rectangle(self):
        """Test window detection with a clear rectangular region."""
        # Use more lenient thresholds for testing
        detector = WindowDetector(
            min_window_size_percent=10.0,
            rectangularity_threshold=0.5,
            confidence_threshold=0.3,
        )

        # Create test frame with clear rectangle and edges
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw white rectangle with black border (simulating a window with edges)
        cv2.rectangle(frame, (100, 100), (500, 350), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 100), (500, 350), (0, 0, 0), 3)

        region = detector.detect_window_region(frame)

        # Should detect the rectangle
        assert region is not None
        x, y, w, h = region
        # Allow generous tolerance due to edge detection
        assert x >= 80 and x <= 120
        assert y >= 80 and y <= 120
        assert w >= 350 and w <= 450
        assert h >= 200 and h <= 300

    def test_detect_window_with_no_clear_region(self):
        """Test window detection with no clear rectangular region."""
        detector = WindowDetector()

        # Create noisy frame with no clear rectangle
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        region = detector.detect_window_region(frame)

        # May or may not detect depending on noise, but should not crash
        assert region is None or isinstance(region, tuple)

    def test_detect_window_with_none_frame(self):
        """Test window detection handles None frame gracefully."""
        detector = WindowDetector()
        region = detector.detect_window_region(None)
        assert region is None

    def test_confidence_score(self):
        """Test confidence score is updated after detection."""
        detector = WindowDetector()

        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (500, 350), (255, 255, 255), -1)

        detector.detect_window_region(frame)
        confidence = detector.get_confidence()

        assert confidence >= 0.0
        assert confidence <= 1.0


class TestUVCScreenCapture:
    """Test UVCScreenCapture implementation."""

    def test_initialization(self):
        """Test UVCScreenCapture initializes correctly."""
        capture = UVCScreenCapture(device_id=0)
        assert capture.get_fps() == 30
        assert capture.get_target_window() is None
        assert not capture._is_capturing

    def test_fps_control(self):
        """Test FPS control within valid range."""
        capture = UVCScreenCapture()

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

    @patch("cv2.VideoCapture")
    def test_list_devices(self, mock_video_capture):
        """Test UVC device enumeration."""

        # Mock VideoCapture to simulate 2 devices
        def video_capture_side_effect(device_id):
            mock_cap = MagicMock()
            if device_id < 2:
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_WIDTH: 1920,
                    cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 30,
                }.get(prop, 0)
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap

        mock_video_capture.side_effect = video_capture_side_effect

        capture = UVCScreenCapture()
        devices = capture.list_devices()

        assert isinstance(devices, list)
        assert len(devices) == 2

        # Check device structure
        for device in devices:
            assert "id" in device
            assert "name" in device
            assert "width" in device
            assert "height" in device
            assert "fps" in device

    def test_list_windows_returns_empty(self):
        """Test that list_windows returns empty list for UVC."""
        capture = UVCScreenCapture()
        windows = capture.list_windows()
        assert windows == []

    def test_target_window_not_applicable(self):
        """Test that target window operations are no-ops for UVC."""
        capture = UVCScreenCapture()

        # Should not raise error
        capture.set_target_window(12345)
        assert capture.get_target_window() is None

    @patch("cv2.VideoCapture")
    def test_capture_start_stop(self, mock_video_capture):
        """Test starting and stopping UVC capture."""
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        capture = UVCScreenCapture()

        # Start capture
        capture.start_capture(fps=30)
        assert capture._is_capturing
        assert capture._capture_thread is not None

        # Give thread time to start
        time.sleep(0.1)

        # Stop capture
        capture.stop_capture()
        assert not capture._is_capturing
        mock_cap.release.assert_called()

    @patch("cv2.VideoCapture")
    def test_capture_with_device_failure(self, mock_video_capture):
        """Test capture handles device open failure."""
        # Mock VideoCapture to fail
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        capture = UVCScreenCapture()

        # Should raise error when device can't be opened
        with pytest.raises(RuntimeError):
            capture.start_capture(fps=30)

    def test_detection_confidence(self):
        """Test getting detection confidence."""
        capture = UVCScreenCapture()
        confidence = capture.get_detection_confidence()

        assert confidence >= 0.0
        assert confidence <= 1.0


class TestWindowInfo:
    """Test WindowInfo dataclass."""

    def test_window_info_creation(self):
        """Test creating WindowInfo instance."""
        window = WindowInfo(
            window_id=12345,
            title="Test Window",
            app_name="TestApp",
            bounds=(100, 100, 800, 600),
            is_self=False,
        )

        assert window.window_id == 12345
        assert window.title == "Test Window"
        assert window.app_name == "TestApp"
        assert window.bounds == (100, 100, 800, 600)
        assert window.is_self is False
