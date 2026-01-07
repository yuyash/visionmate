"""Integration tests for DeviceManager with real hardware.

These tests scan actual devices on the host system and verify
that the DeviceManager can correctly enumerate and interact with
real hardware devices.
"""

import pytest

from visionmate.core.capture import DeviceManager
from visionmate.core.models import DeviceType


class TestDeviceManagerIntegration:
    """Integration test suite for DeviceManager with real devices."""

    @pytest.fixture
    def device_manager(self):
        """Create a DeviceManager instance for testing."""
        return DeviceManager()

    def test_get_screens_real(self, device_manager):
        """Test getting screens with real hardware.

        This test scans actual screens connected to the host system.

        Requirements: 1.7
        """
        screens = device_manager.get_screens()

        # Should find at least one screen
        assert len(screens) > 0, "No screens found on host system"

        print(f"\nFound {len(screens)} screen(s):")

        # Verify screen metadata structure
        for screen in screens:
            print(f"  - {screen.name}: {screen.current_resolution} @ {screen.native_fps}Hz")

            assert screen.device_type == DeviceType.SCREEN
            assert screen.device_id.startswith("screen_")
            assert len(screen.supported_resolutions) > 0
            assert len(screen.supported_fps) > 0
            assert screen.native_fps is not None
            assert screen.is_available is True

            # Verify resolution is valid
            assert screen.current_resolution.width > 0
            assert screen.current_resolution.height > 0

    def test_get_uvc_devices_real(self, device_manager):
        """Test getting UVC devices with real hardware.

        This test scans actual UVC devices connected to the host system.
        Note: This test may find 0 devices if no UVC devices are connected.

        Requirements: 1.8
        """
        devices = device_manager.get_uvc_devices()

        print(f"\nFound {len(devices)} UVC device(s):")

        # Verify device metadata structure for any found devices
        for device in devices:
            print(f"  - {device.name}: {device.current_resolution} @ {device.native_fps}fps")

            assert device.device_type == DeviceType.UVC
            assert device.device_id.startswith("uvc_")
            assert len(device.supported_resolutions) > 0
            assert len(device.supported_fps) > 0
            assert device.native_fps is not None
            assert device.is_available is True

            # Verify resolution is valid
            assert device.current_resolution.width > 0
            assert device.current_resolution.height > 0

    def test_get_audio_devices_real(self, device_manager):
        """Test getting audio devices with real hardware.

        This test scans actual audio devices on the host system.

        Requirements: 2.7
        """
        devices = device_manager.get_audio_devices()

        # Should find at least one audio device
        assert len(devices) > 0, "No audio devices found on host system"

        print(f"\nFound {len(devices)} audio device(s):")

        # Verify audio device metadata structure
        for device in devices:
            print(f"  - {device.name}: {device.current_sample_rate}Hz, {device.current_channels}ch")

            assert device.device_type == DeviceType.AUDIO
            assert device.device_id.startswith("audio_")
            assert len(device.sample_rates) > 0
            assert len(device.channels) > 0
            assert device.current_sample_rate is not None
            assert device.is_available is True

            # Verify sample rate is valid
            assert device.current_sample_rate > 0

    def test_get_device_metadata_screen_real(self, device_manager):
        """Test getting metadata for a real screen device.

        Requirements: 27.1-27.6
        """
        screens = device_manager.get_screens()
        assert len(screens) > 0, "No screens found on host system"

        # Get metadata for first screen
        device_id = screens[0].device_id
        metadata = device_manager.get_device_metadata(device_id)

        print(f"\nScreen metadata for {device_id}:")
        print(f"  Name: {metadata.name}")
        print(f"  Resolution: {metadata.current_resolution}")
        print(f"  Native FPS: {metadata.native_fps}")
        print(f"  Supported FPS: {metadata.supported_fps[:5]}...")  # Show first 5

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.SCREEN
        assert len(metadata.supported_resolutions) > 0
        assert len(metadata.supported_fps) > 0
        assert metadata.native_fps is not None

    def test_get_device_metadata_audio_real(self, device_manager):
        """Test getting metadata for a real audio device.

        Requirements: 27.1-27.6
        """
        devices = device_manager.get_audio_devices()
        assert len(devices) > 0, "No audio devices found on host system"

        # Get metadata for first audio device
        device_id = devices[0].device_id
        metadata = device_manager.get_device_metadata(device_id)

        print(f"\nAudio metadata for {device_id}:")
        print(f"  Name: {metadata.name}")
        print(f"  Sample Rate: {metadata.current_sample_rate}Hz")
        print(f"  Channels: {metadata.current_channels}")
        print(f"  Supported Sample Rates: {metadata.sample_rates}")

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.AUDIO
        assert len(metadata.sample_rates) > 0
        assert len(metadata.channels) > 0

    def test_validate_settings_real_screen(self, device_manager):
        """Test validating settings with real screen device.

        Requirements: 27.8
        """
        screens = device_manager.get_screens()
        assert len(screens) > 0, "No screens found on host system"

        device_id = screens[0].device_id
        resolution = screens[0].supported_resolutions[0].to_tuple()
        fps = screens[0].supported_fps[0]

        # Should validate successfully with supported settings
        assert device_manager.validate_settings(device_id, resolution=resolution) is True
        assert device_manager.validate_settings(device_id, fps=fps) is True

        # Should fail with invalid settings
        assert device_manager.validate_settings(device_id, resolution=(99999, 99999)) is False
        assert device_manager.validate_settings(device_id, fps=999) is False

    def test_suggest_optimal_settings_real_screen(self, device_manager):
        """Test suggesting optimal settings for real screen device.

        Requirements: 27.9
        """
        screens = device_manager.get_screens()
        assert len(screens) > 0, "No screens found on host system"

        device_id = screens[0].device_id
        optimal = device_manager.suggest_optimal_settings(device_id)

        print(f"\nOptimal settings for {device_id}:")
        print(f"  Resolution: {optimal.resolution}")
        print(f"  FPS: {optimal.fps}")
        print(f"  Color Format: {optimal.color_format}")
        print(f"  Reason: {optimal.reason}")

        # Should have video settings
        assert optimal.resolution is not None
        assert optimal.fps is not None
        assert optimal.color_format is not None
        assert len(optimal.reason) > 0

        # Should suggest 1 FPS for optimal performance
        assert optimal.fps == 1

    def test_suggest_optimal_settings_real_audio(self, device_manager):
        """Test suggesting optimal settings for real audio device.

        Requirements: 27.9
        """
        devices = device_manager.get_audio_devices()
        assert len(devices) > 0, "No audio devices found on host system"

        device_id = devices[0].device_id
        optimal = device_manager.suggest_optimal_settings(device_id)

        print(f"\nOptimal settings for {device_id}:")
        print(f"  Sample Rate: {optimal.sample_rate}Hz")
        print(f"  Channels: {optimal.channels}")
        print(f"  Reason: {optimal.reason}")

        # Should have audio settings
        assert optimal.sample_rate is not None
        assert optimal.channels is not None
        assert len(optimal.reason) > 0

        # Should suggest 16kHz mono for speech recognition
        assert optimal.sample_rate == 16000
        assert optimal.channels == 1
