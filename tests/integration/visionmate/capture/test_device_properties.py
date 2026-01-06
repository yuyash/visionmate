"""Integration property-based tests for DeviceManager with real hardware.

Feature: visionmate-core
Property 12: Device Metadata Retrieval (Integration)

This module contains property-based tests that validate device metadata
retrieval using actual hardware devices on the host system.

**Validates: Requirements 27.1-27.6**

Note: These tests require actual hardware devices to be present on the system.
They are designed to run in parallel using pytest-xdist for efficiency.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from visionmate.core.capture.device import DeviceManager
from visionmate.core.models import DeviceType, Resolution

# ============================================================================
# Hypothesis Strategies for Real Devices
# ============================================================================


@st.composite
def real_screen_device_strategy(draw, device_manager):
    """Generate strategy from actual available screens."""
    screens = device_manager.enumerate_screens()
    if not screens:
        pytest.skip("No screen devices available on this system")
    return draw(st.sampled_from(screens))


@st.composite
def real_uvc_device_strategy(draw, device_manager):
    """Generate strategy from actual available UVC devices."""
    devices = device_manager.enumerate_uvc_devices()
    if not devices:
        pytest.skip("No UVC devices available on this system")
    return draw(st.sampled_from(devices))


@st.composite
def real_audio_device_strategy(draw, device_manager):
    """Generate strategy from actual available audio devices."""
    devices = device_manager.enumerate_audio_devices()
    if not devices:
        pytest.skip("No audio devices available on this system")
    return draw(st.sampled_from(devices))


# ============================================================================
# Integration Property-Based Tests
# ============================================================================


class TestDeviceMetadataPropertiesIntegration:
    """Integration property-based tests for device metadata retrieval.

    **Property 12: Device Metadata Retrieval (Integration)**

    For any available input device on the actual system, the system should
    be able to retrieve metadata including supported resolutions, frame rates,
    color formats (for video) or sample rates and channels (for audio).

    **Validates: Requirements 27.1-27.6**
    """

    @pytest.fixture(scope="class")
    def device_manager(self):
        """Create DeviceManager instance for the test class."""
        return DeviceManager()

    @settings(
        max_examples=10,
        deadline=None,  # Real hardware can be slow
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_screen_metadata_retrieval(self, device_manager, data):
        """Property: For any real screen device, metadata should be complete and valid.

        **Validates: Requirements 27.1, 27.2, 27.3, 27.4**
        """
        screen = data.draw(real_screen_device_strategy(device_manager))

        # Property: Screen metadata should be complete
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
        assert all(fps > 0 for fps in screen.supported_fps), "All FPS values should be positive"
        assert screen.native_fps is not None, "Native FPS should be present"
        assert screen.native_fps > 0, "Native FPS should be positive"

        # Requirement 27.4: Retrieve color format information
        assert len(screen.color_formats) > 0, "Should have color formats"
        assert all(
            isinstance(fmt, str) for fmt in screen.color_formats
        ), "All color formats should be strings"

        # Current settings should be valid
        assert screen.current_resolution is not None, "Current resolution should be set"
        assert screen.current_resolution.width > 0, "Current resolution width should be positive"
        assert screen.current_resolution.height > 0, "Current resolution height should be positive"
        assert screen.current_fps is not None, "Current FPS should be set"
        assert screen.current_fps > 0, "Current FPS should be positive"

        # Property: Current settings should be in supported lists
        assert (
            screen.current_resolution in screen.supported_resolutions
        ), "Current resolution should be in supported list"
        assert screen.current_fps in screen.supported_fps, "Current FPS should be in supported list"

    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_uvc_metadata_retrieval(self, device_manager, data):
        """Property: For any real UVC device, metadata should be complete and valid.

        **Validates: Requirements 27.1, 27.2, 27.3, 27.4**
        """
        device = data.draw(real_uvc_device_strategy(device_manager))

        # Property: UVC metadata should be complete
        assert device.device_id is not None, "Device ID should be present"
        assert device.name is not None, "Device name should be present"
        assert device.device_type == DeviceType.UVC, "Device type should be UVC"

        # Requirement 27.2: Retrieve supported resolutions
        assert len(device.supported_resolutions) > 0, "Should have supported resolutions"
        assert all(
            isinstance(res, Resolution) for res in device.supported_resolutions
        ), "All resolutions should be Resolution objects"
        assert all(
            res.width > 0 and res.height > 0 for res in device.supported_resolutions
        ), "All resolutions should have positive dimensions"

        # Requirement 27.3: Retrieve supported frame rates
        assert len(device.supported_fps) > 0, "Should have supported FPS values"
        assert all(fps > 0 for fps in device.supported_fps), "All FPS values should be positive"
        assert device.native_fps is not None, "Native FPS should be present"
        assert device.native_fps > 0, "Native FPS should be positive"

        # Requirement 27.4: Retrieve color format information
        assert len(device.color_formats) > 0, "Should have color formats"
        assert all(
            isinstance(fmt, str) for fmt in device.color_formats
        ), "All color formats should be strings"

        # Current settings should be valid
        assert device.current_resolution is not None, "Current resolution should be set"
        assert device.current_resolution.width > 0, "Current resolution width should be positive"
        assert device.current_resolution.height > 0, "Current resolution height should be positive"
        assert device.current_fps is not None, "Current FPS should be set"
        assert device.current_fps > 0, "Current FPS should be positive"

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_audio_metadata_retrieval(self, device_manager, data):
        """Property: For any real audio device, metadata should be complete and valid.

        **Validates: Requirements 27.1, 27.5, 27.6**
        """
        device = data.draw(real_audio_device_strategy(device_manager))

        # Property: Audio metadata should be complete
        assert device.device_id is not None, "Device ID should be present"
        assert device.name is not None, "Device name should be present"
        assert device.device_type == DeviceType.AUDIO, "Device type should be AUDIO"

        # Requirement 27.5: Retrieve audio sample rates
        assert len(device.sample_rates) > 0, "Should have supported sample rates"
        assert all(rate > 0 for rate in device.sample_rates), "All sample rates should be positive"
        assert device.current_sample_rate is not None, "Current sample rate should be set"
        assert device.current_sample_rate > 0, "Current sample rate should be positive"

        # Requirement 27.6: Retrieve audio channel information
        assert len(device.channels) > 0, "Should have supported channel configurations"
        assert all(ch > 0 for ch in device.channels), "All channel counts should be positive"
        assert device.current_channels is not None, "Current channels should be set"
        assert device.current_channels > 0, "Current channels should be positive"

        # Property: Current settings should be in supported lists
        assert (
            device.current_sample_rate in device.sample_rates
        ), "Current sample rate should be in supported list"
        assert (
            device.current_channels in device.channels
        ), "Current channels should be in supported list"

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_device_metadata_retrieval_by_id(self, device_manager, data):
        """Property: For any real device ID, get_device_metadata should return consistent data.

        **Validates: Requirements 27.1-27.6**
        """
        # Get all available devices
        all_devices = []
        all_devices.extend(device_manager.enumerate_screens())
        all_devices.extend(device_manager.enumerate_uvc_devices())
        all_devices.extend(device_manager.enumerate_audio_devices())

        if not all_devices:
            pytest.skip("No devices available on this system")

        device = data.draw(st.sampled_from(all_devices))

        # Property: get_device_metadata should return the same data as enumeration
        retrieved_metadata = device_manager.get_device_metadata(device.device_id)

        assert retrieved_metadata.device_id == device.device_id
        assert retrieved_metadata.name == device.name
        assert retrieved_metadata.device_type == device.device_type

        # For video devices
        if device.device_type in [DeviceType.SCREEN, DeviceType.UVC]:
            assert retrieved_metadata.supported_resolutions == device.supported_resolutions
            assert retrieved_metadata.supported_fps == device.supported_fps
            assert retrieved_metadata.color_formats == device.color_formats
            assert retrieved_metadata.native_fps == device.native_fps

        # For audio devices
        if device.device_type == DeviceType.AUDIO:
            assert retrieved_metadata.sample_rates == device.sample_rates
            assert retrieved_metadata.channels == device.channels

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_device_settings_validation(self, device_manager, data):
        """Property: For any real device, validate_settings should correctly validate supported settings.

        **Validates: Requirements 27.8**
        """
        # Get all video devices (screens and UVC)
        video_devices = []
        video_devices.extend(device_manager.enumerate_screens())
        video_devices.extend(device_manager.enumerate_uvc_devices())

        if not video_devices:
            pytest.skip("No video devices available on this system")

        device = data.draw(st.sampled_from(video_devices))

        # Property: Supported settings should validate as True
        if device.supported_resolutions:
            supported_res = data.draw(st.sampled_from(device.supported_resolutions))
            assert device_manager.validate_settings(
                device.device_id, resolution=supported_res.to_tuple()
            ), "Supported resolution should validate as True"

        if device.supported_fps:
            supported_fps = data.draw(st.sampled_from(device.supported_fps))
            assert device_manager.validate_settings(
                device.device_id, fps=supported_fps
            ), "Supported FPS should validate as True"

        # Property: Unsupported settings should validate as False
        # Generate an unlikely resolution
        unlikely_resolution = (9999, 9999)
        if Resolution.from_tuple(unlikely_resolution) not in device.supported_resolutions:
            assert not device_manager.validate_settings(
                device.device_id, resolution=unlikely_resolution
            ), "Unsupported resolution should validate as False"

        # Generate an unlikely FPS
        unlikely_fps = 999
        if unlikely_fps not in device.supported_fps:
            assert not device_manager.validate_settings(
                device.device_id, fps=unlikely_fps
            ), "Unsupported FPS should validate as False"

    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_device_optimal_settings_suggestion(self, device_manager, data):
        """Property: For any real device, suggest_optimal_settings should return valid settings.

        **Validates: Requirements 27.9**
        """
        # Get all available devices
        all_devices = []
        all_devices.extend(device_manager.enumerate_screens())
        all_devices.extend(device_manager.enumerate_uvc_devices())
        all_devices.extend(device_manager.enumerate_audio_devices())

        if not all_devices:
            pytest.skip("No devices available on this system")

        device = data.draw(st.sampled_from(all_devices))

        # Property: Optimal settings should be valid and include reasoning
        optimal = device_manager.suggest_optimal_settings(device.device_id)

        assert optimal.reason is not None, "Optimal settings should include reasoning"
        assert len(optimal.reason) > 0, "Reasoning should not be empty"

        # For video devices
        if device.device_type in [DeviceType.SCREEN, DeviceType.UVC]:
            assert optimal.resolution is not None, "Video device should have optimal resolution"
            assert optimal.fps is not None, "Video device should have optimal FPS"
            assert optimal.color_format is not None, "Video device should have optimal color format"

            # Property: Optimal settings should be in supported lists
            assert (
                optimal.resolution in device.supported_resolutions
            ), "Optimal resolution should be supported"
            assert optimal.fps in device.supported_fps, "Optimal FPS should be supported"
            assert (
                optimal.color_format in device.color_formats
            ), "Optimal color format should be supported"

        # For audio devices
        if device.device_type == DeviceType.AUDIO:
            assert optimal.sample_rate is not None, "Audio device should have optimal sample rate"
            assert optimal.channels is not None, "Audio device should have optimal channels"

            # Property: Optimal settings should be in supported lists
            assert (
                optimal.sample_rate in device.sample_rates
            ), "Optimal sample rate should be supported"
            assert optimal.channels in device.channels, "Optimal channels should be supported"

    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(data=st.data())
    def test_real_device_metadata_consistency(self, device_manager, data):
        """Property: Multiple enumerations should return consistent device lists.

        **Validates: Requirements 27.1-27.6**
        """
        # Enumerate devices twice
        screens1 = device_manager.enumerate_screens()
        screens2 = device_manager.enumerate_screens()

        # Property: Device count should be consistent
        assert len(screens1) == len(
            screens2
        ), "Screen count should be consistent across enumerations"

        # Property: Device IDs should be consistent
        screen_ids1 = {screen.device_id for screen in screens1}
        screen_ids2 = {screen.device_id for screen in screens2}
        assert screen_ids1 == screen_ids2, "Screen IDs should be consistent across enumerations"

        # Do the same for audio devices
        audio1 = device_manager.enumerate_audio_devices()
        audio2 = device_manager.enumerate_audio_devices()

        assert len(audio1) == len(audio2), "Audio device count should be consistent"

        audio_ids1 = {device.device_id for device in audio1}
        audio_ids2 = {device.device_id for device in audio2}
        assert audio_ids1 == audio_ids2, "Audio device IDs should be consistent"
