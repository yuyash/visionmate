"""Property-based tests for DeviceManager.

Feature: visionmate-core
Property 12: Device Metadata Retrieval

This module contains property-based tests that validate device metadata
retrieval across all available input devices.

**Validates: Requirements 27.1-27.6**
"""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from visionmate.core.capture.device import DeviceManager
from visionmate.core.models import DeviceType, Resolution

# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def screen_monitor_strategy(draw):
    """Generate mock screen monitor data."""
    width = draw(st.integers(min_value=640, max_value=7680))
    height = draw(st.integers(min_value=480, max_value=4320))
    return {
        "width": width,
        "height": height,
        "left": 0,
        "top": 0,
    }


@st.composite
def uvc_device_strategy(draw):
    """Generate mock UVC device data."""
    # Common resolutions
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
    ]
    resolution = draw(st.sampled_from(resolutions))
    fps = draw(st.integers(min_value=15, max_value=60))

    return {
        "width": resolution[0],
        "height": resolution[1],
        "fps": fps,
        "is_opened": True,
    }


@st.composite
def audio_device_strategy(draw):
    """Generate mock audio device data."""
    sample_rates = [8000, 16000, 22050, 44100, 48000]
    sample_rate = draw(st.sampled_from(sample_rates))
    channels = draw(st.integers(min_value=1, max_value=8))

    # Generate device name with letters and numbers only
    name_base = draw(
        st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))
    )
    # Ensure non-empty name
    name = name_base.strip() or "AudioDevice"

    return {
        "name": name,
        "max_input_channels": channels,
        "default_samplerate": float(sample_rate),
    }


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestDeviceMetadataProperties:
    """Property-based tests for device metadata retrieval.

    **Property 12: Device Metadata Retrieval**

    For any available input device, the system should be able to retrieve
    metadata including supported resolutions, frame rates, color formats
    (for video) or sample rates and channels (for audio).

    **Validates: Requirements 27.1-27.6**
    """

    @settings(max_examples=100)
    @given(monitor=screen_monitor_strategy())
    def test_screen_metadata_retrieval(self, monitor):
        """Property: For any screen device, metadata should include resolution, FPS, and color formats.

        **Validates: Requirements 27.1, 27.2, 27.3, 27.4**
        """
        with patch("mss.mss") as mock_mss:
            # Setup mock
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_context)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_context.monitors = [
                {"width": 0, "height": 0, "left": 0, "top": 0},  # Virtual "all monitors"
                monitor,
            ]
            mock_mss.return_value = mock_context

            device_manager = DeviceManager()

            # Enumerate screens
            screens = device_manager.enumerate_screens()

            # Property: At least one screen should be found
            assert len(screens) > 0, "Should enumerate at least one screen"

            # Property: Each screen should have complete metadata
            for screen in screens:
                # Requirement 27.1: Capture metadata from all input devices
                assert screen.device_id is not None, "Device ID should be present"
                assert screen.name is not None, "Device name should be present"
                assert screen.device_type == DeviceType.SCREEN, "Device type should be SCREEN"

                # Requirement 27.2: Retrieve supported resolutions
                assert len(screen.supported_resolutions) > 0, "Should have supported resolutions"
                assert all(
                    isinstance(res, Resolution) for res in screen.supported_resolutions
                ), "All resolutions should be Resolution objects"
                assert all(
                    res.width > 0 and res.height > 0 for res in screen.supported_resolutions
                ), "All resolutions should have positive dimensions"

                # Requirement 27.3: Retrieve supported frame rates
                assert len(screen.supported_fps) > 0, "Should have supported FPS values"
                assert all(
                    fps > 0 for fps in screen.supported_fps
                ), "All FPS values should be positive"
                assert screen.native_fps is not None, "Native FPS should be present"
                assert screen.native_fps > 0, "Native FPS should be positive"

                # Requirement 27.4: Retrieve color format information
                assert len(screen.color_formats) > 0, "Should have color formats"
                assert all(
                    isinstance(fmt, str) for fmt in screen.color_formats
                ), "All color formats should be strings"

                # Current settings should be populated
                assert screen.current_resolution is not None, "Current resolution should be set"
                assert screen.current_fps is not None, "Current FPS should be set"

    @settings(max_examples=100)
    @given(device_data=uvc_device_strategy())
    def test_uvc_metadata_retrieval(self, device_data):
        """Property: For any UVC device, metadata should include resolution, FPS, and color formats.

        **Validates: Requirements 27.1, 27.2, 27.3, 27.4**
        """
        with patch("cv2.VideoCapture") as mock_video_capture:
            # Setup mock
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = device_data["is_opened"]
            mock_cap.get.side_effect = lambda prop: {
                3: device_data["width"],  # CAP_PROP_FRAME_WIDTH
                4: device_data["height"],  # CAP_PROP_FRAME_HEIGHT
                5: device_data["fps"],  # CAP_PROP_FPS
            }.get(prop, 0)
            mock_video_capture.return_value = mock_cap

            device_manager = DeviceManager()

            # Enumerate UVC devices
            devices = device_manager.enumerate_uvc_devices()

            # Property: If device is opened, it should have complete metadata
            if device_data["is_opened"]:
                assert len(devices) > 0, "Should enumerate at least one UVC device"

                for device in devices:
                    # Requirement 27.1: Capture metadata from all input devices
                    assert device.device_id is not None, "Device ID should be present"
                    assert device.name is not None, "Device name should be present"
                    assert device.device_type == DeviceType.UVC, "Device type should be UVC"

                    # Requirement 27.2: Retrieve supported resolutions
                    assert (
                        len(device.supported_resolutions) > 0
                    ), "Should have supported resolutions"
                    assert all(
                        isinstance(res, Resolution) for res in device.supported_resolutions
                    ), "All resolutions should be Resolution objects"

                    # Requirement 27.3: Retrieve supported frame rates
                    assert len(device.supported_fps) > 0, "Should have supported FPS values"
                    assert all(
                        fps > 0 for fps in device.supported_fps
                    ), "All FPS values should be positive"
                    assert device.native_fps is not None, "Native FPS should be present"

                    # Requirement 27.4: Retrieve color format information
                    assert len(device.color_formats) > 0, "Should have color formats"

                    # Current settings should be populated
                    assert device.current_resolution is not None, "Current resolution should be set"
                    assert device.current_fps is not None, "Current FPS should be set"

    @settings(max_examples=100)
    @given(device_info=audio_device_strategy())
    def test_audio_metadata_retrieval(self, device_info):
        """Property: For any audio device, metadata should include sample rates and channels.

        **Validates: Requirements 27.1, 27.5, 27.6**
        """
        with patch("sounddevice.query_devices") as mock_query_devices:
            # Setup mock
            mock_query_devices.return_value = [device_info]

            device_manager = DeviceManager()

            # Enumerate audio devices
            devices = device_manager.enumerate_audio_devices()

            # Property: At least one audio device should be found
            assert len(devices) > 0, "Should enumerate at least one audio device"

            for device in devices:
                # Requirement 27.1: Capture metadata from all input devices
                assert device.device_id is not None, "Device ID should be present"
                assert device.name is not None, "Device name should be present"
                assert device.device_type == DeviceType.AUDIO, "Device type should be AUDIO"

                # Requirement 27.5: Retrieve audio sample rates
                assert len(device.sample_rates) > 0, "Should have supported sample rates"
                assert all(
                    rate > 0 for rate in device.sample_rates
                ), "All sample rates should be positive"
                assert device.current_sample_rate is not None, "Current sample rate should be set"
                assert device.current_sample_rate > 0, "Current sample rate should be positive"

                # Requirement 27.6: Retrieve audio channel information
                assert len(device.channels) > 0, "Should have supported channel configurations"
                assert all(
                    ch > 0 for ch in device.channels
                ), "All channel counts should be positive"
                assert device.current_channels is not None, "Current channels should be set"
                assert device.current_channels > 0, "Current channels should be positive"

    @settings(max_examples=100)
    @given(
        monitor=screen_monitor_strategy(),
        device_data=uvc_device_strategy(),
        audio_info=audio_device_strategy(),
    )
    def test_get_device_metadata_for_any_device(
        self,
        monitor,
        device_data,
        audio_info,
    ):
        """Property: For any device ID, get_device_metadata should return complete metadata.

        **Validates: Requirements 27.1-27.6**
        """
        with (
            patch("mss.mss") as mock_mss,
            patch("cv2.VideoCapture") as mock_video_capture,
            patch("sounddevice.query_devices") as mock_query_devices,
        ):
            # Setup mocks for all device types
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_context)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_context.monitors = [
                {"width": 0, "height": 0, "left": 0, "top": 0},
                monitor,
            ]
            mock_mss.return_value = mock_context

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = device_data["is_opened"]
            mock_cap.get.side_effect = lambda prop: {
                3: device_data["width"],
                4: device_data["height"],
                5: device_data["fps"],
            }.get(prop, 0)
            mock_video_capture.return_value = mock_cap

            mock_query_devices.return_value = [audio_info]

            device_manager = DeviceManager()

            # Test screen device metadata retrieval
            screen_metadata = device_manager.get_device_metadata("screen_1")
            assert screen_metadata.device_type == DeviceType.SCREEN
            assert len(screen_metadata.supported_resolutions) > 0
            assert len(screen_metadata.supported_fps) > 0
            assert len(screen_metadata.color_formats) > 0

            # Test UVC device metadata retrieval (if device is opened)
            if device_data["is_opened"]:
                uvc_metadata = device_manager.get_device_metadata("uvc_0")
                assert uvc_metadata.device_type == DeviceType.UVC
                assert len(uvc_metadata.supported_resolutions) > 0
                assert len(uvc_metadata.supported_fps) > 0
                assert len(uvc_metadata.color_formats) > 0

            # Test audio device metadata retrieval
            audio_metadata = device_manager.get_device_metadata("audio_0")
            assert audio_metadata.device_type == DeviceType.AUDIO
            assert len(audio_metadata.sample_rates) > 0
            assert len(audio_metadata.channels) > 0

    @settings(max_examples=100)
    @given(monitor=screen_monitor_strategy())
    def test_metadata_consistency_across_calls(self, monitor):
        """Property: Multiple calls to retrieve metadata for the same device should return consistent data.

        **Validates: Requirements 27.1-27.6**
        """
        with patch("mss.mss") as mock_mss:
            # Setup mock
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_context)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_context.monitors = [
                {"width": 0, "height": 0, "left": 0, "top": 0},
                monitor,
            ]
            mock_mss.return_value = mock_context

            device_manager = DeviceManager()

            # Get metadata twice
            metadata1 = device_manager.get_device_metadata("screen_1")
            metadata2 = device_manager.get_device_metadata("screen_1")

            # Property: Metadata should be consistent across calls
            assert metadata1.device_id == metadata2.device_id
            assert metadata1.name == metadata2.name
            assert metadata1.device_type == metadata2.device_type
            assert metadata1.supported_resolutions == metadata2.supported_resolutions
            assert metadata1.supported_fps == metadata2.supported_fps
            assert metadata1.color_formats == metadata2.color_formats
            assert metadata1.current_resolution == metadata2.current_resolution
            assert metadata1.current_fps == metadata2.current_fps
            assert metadata1.native_fps == metadata2.native_fps
