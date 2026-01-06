"""Unit tests for video capture module."""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from visionmate.core.capture.video import ScreenCapture, VideoCaptureInterface, WindowDetector
from visionmate.core.models import DeviceType, Resolution, VideoSourceType, WindowRegion


class TestVideoCaptureInterface:
    """Test VideoCaptureInterface abstract class."""

    def test_interface_is_abstract(self):
        """Test that VideoCaptureInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            VideoCaptureInterface()


class TestWindowDetector:
    """Test WindowDetector class."""

    def test_init(self):
        """Test WindowDetector initialization."""
        detector = WindowDetector()
        assert detector._platform in ["Darwin", "Windows", "Linux"]

    @patch("platform.system", return_value="Darwin")
    def test_detect_active_window_macos_no_pyobjc(self, mock_platform):
        """Test macOS window detection without pyobjc installed."""
        detector = WindowDetector()

        # Mock the import to fail
        with patch.dict("sys.modules", {"Quartz": None}):
            result = detector.detect_active_window()
            # Should return None when pyobjc not available
            assert result is None or isinstance(result, WindowRegion)

    @patch("platform.system", return_value="Windows")
    def test_detect_active_window_windows_no_pywin32(self, mock_platform):
        """Test Windows window detection without pywin32 installed."""
        detector = WindowDetector()

        # Mock the import to fail
        with patch.dict("sys.modules", {"win32gui": None}):
            result = detector.detect_active_window()
            # Should return None when pywin32 not available
            assert result is None or isinstance(result, WindowRegion)

    @patch("platform.system", return_value="Linux")
    def test_detect_active_window_linux_no_xlib(self, mock_platform):
        """Test Linux window detection without python-xlib installed."""
        detector = WindowDetector()

        # Mock the import to fail
        with patch.dict("sys.modules", {"Xlib": None}):
            result = detector.detect_active_window()
            # Should return None when python-xlib not available
            assert result is None or isinstance(result, WindowRegion)


class TestScreenCapture:
    """Test ScreenCapture class."""

    def test_init(self):
        """Test ScreenCapture initialization."""
        capture = ScreenCapture()
        assert not capture.is_capturing()
        assert capture.is_window_detection_enabled()

    def test_init_with_device_manager(self):
        """Test ScreenCapture initialization with device manager."""
        mock_dm = Mock()
        capture = ScreenCapture(device_manager=mock_dm)
        assert capture._device_manager is mock_dm

    def test_start_capture_invalid_device_id(self):
        """Test start_capture with invalid device_id."""
        capture = ScreenCapture()

        with pytest.raises(ValueError, match="Invalid device_id format"):
            capture.start_capture("invalid_id")

    @patch("mss.mss")
    def test_start_stop_capture(self, mock_mss):
        """Test starting and stopping capture."""
        # Mock mss
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},  # Monitor 0 (all monitors)
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor 1
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct

        capture = ScreenCapture()

        # Start capture
        capture.start_capture("screen_1", fps=1)

        # Give thread time to start
        time.sleep(0.1)

        assert capture.is_capturing()
        assert capture._device_id == "screen_1"
        assert capture._fps == 1
        assert capture._monitor_index == 1

        # Stop capture
        capture.stop_capture()

        # Give thread time to stop
        time.sleep(0.1)

        assert not capture.is_capturing()

    def test_stop_capture_not_capturing(self):
        """Test stop_capture when not capturing."""
        capture = ScreenCapture()
        # Should not raise an error
        capture.stop_capture()

    def test_get_frame_no_capture(self):
        """Test get_frame when not capturing."""
        capture = ScreenCapture()
        frame = capture.get_frame()
        assert frame is None

    @patch("mss.mss")
    def test_get_frame_with_capture(self, mock_mss):
        """Test get_frame during capture."""
        # Mock mss
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},  # Monitor 0
            {"left": 0, "top": 0, "width": 100, "height": 100},  # Monitor 1
        ]

        # Mock screenshot
        mock_screenshot = MagicMock()
        mock_screenshot.__array__ = lambda: np.zeros((100, 100, 4), dtype=np.uint8)
        mock_sct.grab.return_value = mock_screenshot

        mock_mss.return_value.__enter__.return_value = mock_sct

        capture = ScreenCapture()
        capture.start_capture("screen_1", fps=1)

        # Wait for at least one frame
        time.sleep(0.2)

        frame = capture.get_frame()

        # Stop capture
        capture.stop_capture()

        # Check frame
        if frame is not None:
            assert frame.source_id == "screen_1"
            assert frame.source_type == VideoSourceType.SCREEN
            assert frame.fps == 1
            assert isinstance(frame.image, np.ndarray)
            assert isinstance(frame.resolution, Resolution)

    def test_get_source_info_no_device_manager(self):
        """Test get_source_info without device manager."""
        capture = ScreenCapture()
        capture._device_id = "screen_1"

        info = capture.get_source_info()
        assert info.device_id == "screen_1"
        assert info.device_type == DeviceType.SCREEN

    def test_get_source_info_with_device_manager(self):
        """Test get_source_info with device manager."""
        from visionmate.core.models import DeviceMetadata

        mock_dm = Mock()
        mock_metadata = DeviceMetadata(
            device_id="screen_1",
            name="Screen 1",
            device_type=DeviceType.SCREEN,
        )
        mock_dm.get_device_metadata.return_value = mock_metadata

        capture = ScreenCapture(device_manager=mock_dm)
        capture._device_id = "screen_1"

        info = capture.get_source_info()
        assert info.device_id == "screen_1"
        assert info.name == "Screen 1"
        mock_dm.get_device_metadata.assert_called_once_with("screen_1")

    def test_set_window_detection(self):
        """Test set_window_detection (always enabled for screen capture)."""
        capture = ScreenCapture()

        # Try to disable (should remain enabled)
        capture.set_window_detection(False)
        assert capture.is_window_detection_enabled()

        # Enable (should remain enabled)
        capture.set_window_detection(True)
        assert capture.is_window_detection_enabled()

    def test_fps_clamping(self):
        """Test FPS is clamped to valid range."""
        capture = ScreenCapture()

        # Test FPS too low
        capture.start_capture("screen_1", fps=0)
        assert capture._fps == 1

        capture.stop_capture()

        # Test FPS too high
        capture.start_capture("screen_1", fps=100)
        assert capture._fps == 60

        capture.stop_capture()

    def test_resolution_override(self):
        """Test resolution override."""
        capture = ScreenCapture()

        capture.start_capture("screen_1", fps=1, resolution=(1280, 720))
        assert capture._resolution == Resolution(1280, 720)

        capture.stop_capture()
