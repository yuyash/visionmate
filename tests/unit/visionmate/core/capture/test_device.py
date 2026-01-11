"""Unit tests for DeviceManager using mocks.

Tests device enumeration, metadata retrieval, settings validation,
and optimal settings suggestions without accessing real hardware.
"""

from unittest.mock import MagicMock, patch

import pytest

from visionmate.core.capture import DeviceManager
from visionmate.core.models import DeviceType, Resolution


class TestDeviceManager:
    """Test suite for DeviceManager class using mocks."""

    @pytest.fixture
    def device_manager(self):
        """Create a DeviceManager instance for testing."""
        return DeviceManager()

    @patch("mss.mss")
    def test_get_screens(self, mock_mss, device_manager):
        """Test getting screens with mocked mss."""
        # Mock mss to return fake monitors
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},  # Monitor 0 is virtual "all monitors"
            {"width": 1920, "height": 1080},
            {"width": 2560, "height": 1440},
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct

        screens = device_manager.get_screens()

        # Should find 2 screens (excluding virtual monitor 0)
        assert len(screens) == 2

        # Verify first screen
        assert screens[0].device_type == DeviceType.SCREEN
        assert screens[0].device_id == "screen_1"
        assert screens[0].resolution == Resolution(1920, 1080)
        assert screens[0].fps >= 0  # May be 0 if refresh rate detection fails
        assert screens[0].is_available is True

        # Verify second screen
        assert screens[1].device_id == "screen_2"
        assert screens[1].resolution == Resolution(2560, 1440)

    @patch("cv2.VideoCapture")
    def test_get_uvc_devices(self, mock_video_capture, device_manager):
        """Test getting UVC devices with mocked OpenCV."""

        # Mock VideoCapture to return one device at index 0
        def mock_capture_factory(index):
            mock_cap = MagicMock()
            if index == 0:
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    3: 1920,  # CAP_PROP_FRAME_WIDTH
                    4: 1080,  # CAP_PROP_FRAME_HEIGHT
                    5: 30,  # CAP_PROP_FPS
                }.get(prop, 0)
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap

        mock_video_capture.side_effect = mock_capture_factory

        devices = device_manager.get_uvc_devices()

        # Should find 1 device
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.UVC
        assert devices[0].device_id == "uvc_0"
        assert devices[0].resolution == Resolution(1920, 1080)
        assert devices[0].fps == 30
        assert devices[0].is_available is True

    @patch("sounddevice.query_devices")
    def test_get_audio_devices(self, mock_query_devices, device_manager):
        """Test getting audio devices with mocked sounddevice."""
        # Mock sounddevice to return fake audio devices
        mock_query_devices.return_value = [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "Built-in Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
            {
                "name": "USB Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100,
            },
        ]

        devices = device_manager.get_audio_devices()

        # Should find 2 input devices (excluding output-only device)
        assert len(devices) == 2

        # Verify first device
        assert devices[0].device_type == DeviceType.AUDIO
        assert devices[0].device_id == "audio_0"
        assert devices[0].name == "Built-in Microphone"
        assert devices[0].sample_rate == 48000
        assert devices[0].is_available is True

        # Verify second device
        assert devices[1].device_id == "audio_2"
        assert devices[1].name == "USB Microphone"
        assert devices[1].sample_rate == 44100

    @patch("mss.mss")
    def test_get_device_metadata_screen(self, mock_mss, device_manager):
        """Test getting metadata for a screen device."""
        # Mock mss
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},
            {"width": 1920, "height": 1080},
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct

        metadata = device_manager.get_device_metadata("screen_1")

        assert metadata.device_id == "screen_1"
        assert metadata.device_type == DeviceType.SCREEN
        assert metadata.resolution == Resolution(1920, 1080)
        # FPS may be 0 if refresh rate detection fails

    @patch("sounddevice.query_devices")
    def test_get_device_metadata_audio(self, mock_query_devices, device_manager):
        """Test getting metadata for an audio device."""
        # Mock sounddevice
        mock_query_devices.return_value = [
            {
                "name": "Test Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
        ]

        metadata = device_manager.get_device_metadata("audio_0")

        assert metadata.device_id == "audio_0"
        assert metadata.device_type == DeviceType.AUDIO
        assert metadata.sample_rate == 48000
        assert len(metadata.channels) > 0

    def test_get_device_metadata_invalid_id(self, device_manager):
        """Test getting metadata with invalid device ID."""
        with pytest.raises(ValueError, match="Invalid device_id format"):
            device_manager.get_device_metadata("invalid_device_id")

    @patch("mss.mss")
    def test_get_device_metadata_not_found(self, mock_mss, device_manager):
        """Test getting metadata for non-existent device."""
        # Mock mss with only one screen
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},
            {"width": 1920, "height": 1080},
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct

        with pytest.raises(ValueError, match="Device not found"):
            device_manager.get_device_metadata("screen_999")
