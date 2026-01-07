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
            print(f"  - {screen.name}: {screen.resolution} @ {screen.fps}Hz")

            assert screen.device_type == DeviceType.SCREEN
            assert screen.device_id.startswith("screen_")
            assert screen.resolution is not None
            assert screen.is_available is True

            # Verify resolution is valid
            assert screen.resolution.width > 0
            assert screen.resolution.height > 0

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
            print(f"  - {device.name}: {device.resolution} @ {device.fps}Hz")

            assert device.device_type == DeviceType.UVC
            assert device.device_id.startswith("uvc_")
            assert device.resolution is not None
            assert device.is_available is True

            # Verify resolution is valid
            assert device.resolution.width > 0
            assert device.resolution.height > 0

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
            print(f"  - {device.name}: {device.sample_rate}Hz, {device.current_channels}ch")

            assert device.device_type == DeviceType.AUDIO
            assert device.device_id.startswith("audio_")
            assert device.sample_rate is not None
            assert len(device.channels) > 0
            assert device.is_available is True

            # Verify sample rate is valid
            assert device.sample_rate > 0

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
        print(f"  Resolution: {metadata.resolution}")
        print(f"  FPS: {metadata.fps}Hz")

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.SCREEN
        assert metadata.resolution is not None

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
        print(f"  Sample Rate: {metadata.sample_rate}Hz")
        print(f"  Channels: {metadata.current_channels}")

        assert metadata.device_id == device_id
        assert metadata.device_type == DeviceType.AUDIO
        assert metadata.sample_rate is not None
        assert len(metadata.channels) > 0
