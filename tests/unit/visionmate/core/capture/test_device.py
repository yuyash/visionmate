"""Unit tests for DeviceManager.

Tests device enumeration, metadata retrieval, settings validation,
and optimal settings suggestions.
"""

import pytest

from visionmate.core.capture import DeviceManager
from visionmate.core.models import DeviceType


class TestDeviceManager:
    """Test suite for DeviceManager class."""

    @pytest.fixture
    def device_manager(self):
        """Create a DeviceManager instance for testing."""
        return DeviceManager()

    def test_enumerate_screens(self, device_manager):
        """Test screen enumeration.

        Requirements: 1.7
        """
        screens = device_manager.enumerate_screens()

        # Should find at least one screen
        assert len(screens) > 0

        # Verify screen metadata structure
        for screen in screens:
            assert screen.device_type == DeviceType.SCREEN
            assert screen.device_id.startswith("screen_")
            assert len(screen.supported_resolutions) > 0
            assert len(screen.supported_fps) > 0
            assert screen.native_fps is not None
            assert screen.is_available is True

    def test_enumerate_uvc_devices(self, device_manager):
        """Test UVC device enumeration.

        Requirements: 1.8

        Note: This test may find 0 devices if no UVC devices are connected.
        """
        devices = device_manager.enumerate_uvc_devices()

        # Verify device metadata structure for any found devices
        for device in devices:
            assert device.device_type == DeviceType.UVC
            assert device.device_id.startswith("uvc_")
            assert len(device.supported_resolutions) > 0
            assert len(device.supported_fps) > 0
            assert device.native_fps is not None
            assert device.is_available is True

    def test_enumerate_audio_devices(self, device_manager):
        """Test audio device enumeration.

        Requirements: 2.7
        """
        devices = device_manager.enumerate_audio_devices()

        # Should find at least one audio device
        assert len(devices) > 0

        # Verify audio device metadata structure
        for device in devices:
            assert device.device_type == DeviceType.AUDIO
            assert device.device_id.startswith("audio_")
            assert len(device.sample_rates) > 0
            assert len(device.channels) > 0
            assert device.current_sample_rate is not None
            assert device.is_available is True

    def test_get_device_metadata_screen(self, device_manager):
        """Test getting metadata for a screen device.

        Requirements: 27.1-27.6
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        # Get metadata for first screen
        device_id = screens[0].device_id
        metadata = device_manager.get_device_metadata(device_id)

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.SCREEN
        assert len(metadata.supported_resolutions) > 0
        assert len(metadata.supported_fps) > 0
        assert metadata.native_fps is not None

    def test_get_device_metadata_audio(self, device_manager):
        """Test getting metadata for an audio device.

        Requirements: 27.1-27.6
        """
        devices = device_manager.enumerate_audio_devices()
        assert len(devices) > 0

        # Get metadata for first audio device
        device_id = devices[0].device_id
        metadata = device_manager.get_device_metadata(device_id)

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.AUDIO
        assert len(metadata.sample_rates) > 0
        assert len(metadata.channels) > 0

    def test_get_device_metadata_invalid_id(self, device_manager):
        """Test getting metadata with invalid device ID."""
        with pytest.raises(ValueError, match="Invalid device_id format"):
            device_manager.get_device_metadata("invalid_device_id")

    def test_get_device_metadata_not_found(self, device_manager):
        """Test getting metadata for non-existent device."""
        with pytest.raises(ValueError, match="Device not found"):
            device_manager.get_device_metadata("screen_999")

    def test_validate_settings_valid_resolution(self, device_manager):
        """Test validating valid resolution settings.

        Requirements: 27.8
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        device_id = screens[0].device_id
        resolution = screens[0].supported_resolutions[0].to_tuple()

        # Should validate successfully
        assert device_manager.validate_settings(device_id, resolution=resolution) is True

    def test_validate_settings_valid_fps(self, device_manager):
        """Test validating valid FPS settings.

        Requirements: 27.8
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        device_id = screens[0].device_id
        fps = screens[0].supported_fps[0]

        # Should validate successfully
        assert device_manager.validate_settings(device_id, fps=fps) is True

    def test_validate_settings_invalid_resolution(self, device_manager):
        """Test validating invalid resolution settings.

        Requirements: 27.8
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        device_id = screens[0].device_id
        # Use an unlikely resolution
        invalid_resolution = (99999, 99999)

        # Should fail validation
        assert device_manager.validate_settings(device_id, resolution=invalid_resolution) is False

    def test_validate_settings_invalid_fps(self, device_manager):
        """Test validating invalid FPS settings.

        Requirements: 27.8
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        device_id = screens[0].device_id
        # Use an invalid FPS
        invalid_fps = 999

        # Should fail validation
        assert device_manager.validate_settings(device_id, fps=invalid_fps) is False

    def test_suggest_optimal_settings_screen(self, device_manager):
        """Test suggesting optimal settings for screen device.

        Requirements: 27.9
        """
        screens = device_manager.enumerate_screens()
        assert len(screens) > 0

        device_id = screens[0].device_id
        optimal = device_manager.suggest_optimal_settings(device_id)

        # Should have video settings
        assert optimal.resolution is not None
        assert optimal.fps is not None
        assert optimal.color_format is not None
        assert len(optimal.reason) > 0

        # Should suggest 1 FPS for optimal performance
        assert optimal.fps == 1

    def test_suggest_optimal_settings_audio(self, device_manager):
        """Test suggesting optimal settings for audio device.

        Requirements: 27.9
        """
        devices = device_manager.enumerate_audio_devices()
        assert len(devices) > 0

        device_id = devices[0].device_id
        optimal = device_manager.suggest_optimal_settings(device_id)

        # Should have audio settings
        assert optimal.sample_rate is not None
        assert optimal.channels is not None
        assert len(optimal.reason) > 0

        # Should suggest 16kHz mono for speech recognition
        assert optimal.sample_rate == 16000
        assert optimal.channels == 1

    def test_suggest_optimal_settings_invalid_device(self, device_manager):
        """Test suggesting optimal settings for invalid device."""
        optimal = device_manager.suggest_optimal_settings("invalid_999")

        # Should return default settings with error message
        assert "Error" in optimal.reason
