"""Device Manager for enumerating and managing capture devices.

This module provides the DeviceManager class which handles device enumeration,
metadata retrieval, settings validation, and optimal settings suggestions for
all types of capture devices (screens, UVC devices, audio devices).
"""

from __future__ import annotations

import logging
import platform
from typing import List, Optional, Tuple

from visionmate.core.models import DeviceMetadata, DeviceType, OptimalSettings, Resolution

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device enumeration and metadata retrieval.

    This class provides methods to enumerate available capture devices,
    retrieve detailed metadata about devices, validate settings against
    device capabilities, and suggest optimal capture settings.
    """

    def __init__(self):
        """Initialize the DeviceManager."""
        self._platform = platform.system()
        logger.info(f"DeviceManager initialized on platform: {self._platform}")

    def enumerate_screens(self) -> List[DeviceMetadata]:
        """Enumerate available screens.

        Returns:
            List of DeviceMetadata objects for each available screen.

        Requirements: 1.7
        """
        logger.info("Enumerating screens...")
        screens: List[DeviceMetadata] = []

        try:
            import mss

            with mss.mss() as sct:
                # Monitor 0 is the "all monitors" virtual screen, skip it
                for i, monitor in enumerate(sct.monitors[1:], start=1):
                    device_id = f"screen_{i}"
                    name = f"Screen {i}"

                    # Get screen resolution
                    width = monitor["width"]
                    height = monitor["height"]
                    resolution = Resolution(width=width, height=height)

                    # Screens support any FPS up to 60 (configurable)
                    # Native FPS is typically 60 for most displays
                    supported_fps = list(range(1, 61))
                    native_fps = 60

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.SCREEN,
                        supported_resolutions=[resolution],
                        supported_fps=supported_fps,
                        color_formats=["RGB", "BGR"],
                        current_resolution=resolution,
                        current_fps=1,  # Default FPS
                        native_fps=native_fps,
                        is_available=True,
                    )

                    screens.append(metadata)
                    logger.debug(f"Found screen: {name} ({resolution})")

        except Exception as e:
            logger.error(f"Error enumerating screens: {e}", exc_info=True)

        logger.info(f"Found {len(screens)} screen(s)")
        return screens

    def enumerate_uvc_devices(self) -> List[DeviceMetadata]:
        """Enumerate UVC video devices.

        Returns:
            List of DeviceMetadata objects for each available UVC device.

        Requirements: 1.8
        """
        logger.info("Enumerating UVC devices...")
        devices: List[DeviceMetadata] = []

        try:
            import cv2

            # Try to open video capture devices (typically 0-9)
            for i in range(10):
                cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    device_id = f"uvc_{i}"
                    name = f"UVC Device {i}"

                    # Get device capabilities
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    # Common resolutions for UVC devices
                    common_resolutions = [
                        Resolution(640, 480),
                        Resolution(1280, 720),
                        Resolution(1920, 1080),
                        Resolution(3840, 2160),
                    ]

                    # Filter to resolutions that might be supported
                    current_resolution = Resolution(width, height)
                    supported_resolutions = [
                        res
                        for res in common_resolutions
                        if res.total_pixels <= current_resolution.total_pixels
                    ]
                    if current_resolution not in supported_resolutions:
                        supported_resolutions.append(current_resolution)

                    # Common FPS values
                    supported_fps = [1, 5, 10, 15, 24, 30, 60]
                    native_fps = fps if fps > 0 else 30

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.UVC,
                        supported_resolutions=supported_resolutions,
                        supported_fps=supported_fps,
                        color_formats=["BGR", "RGB", "YUV"],
                        current_resolution=current_resolution,
                        current_fps=native_fps,
                        native_fps=native_fps,
                        is_available=True,
                    )

                    devices.append(metadata)
                    logger.debug(f"Found UVC device: {name} ({current_resolution})")

                    cap.release()

        except Exception as e:
            logger.error(f"Error enumerating UVC devices: {e}", exc_info=True)

        logger.info(f"Found {len(devices)} UVC device(s)")
        return devices

    def enumerate_audio_devices(self) -> List[DeviceMetadata]:
        """Enumerate audio input devices.

        Returns:
            List of DeviceMetadata objects for each available audio device.

        Requirements: 2.7
        """
        logger.info("Enumerating audio devices...")
        devices: List[DeviceMetadata] = []

        try:
            import sounddevice as sd

            # Get list of audio devices
            device_list = sd.query_devices()

            for i, device_info in enumerate(device_list):
                # Only include input devices
                if device_info["max_input_channels"] > 0:
                    device_id = f"audio_{i}"
                    name = device_info["name"]

                    # Get device capabilities
                    max_channels = device_info["max_input_channels"]
                    default_sample_rate = int(device_info["default_samplerate"])

                    # Common sample rates
                    common_sample_rates = [8000, 16000, 22050, 44100, 48000]
                    supported_sample_rates = [
                        rate for rate in common_sample_rates if rate <= default_sample_rate
                    ]
                    if default_sample_rate not in supported_sample_rates:
                        supported_sample_rates.append(default_sample_rate)

                    # Supported channel configurations
                    supported_channels = list(range(1, max_channels + 1))

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.AUDIO,
                        sample_rates=supported_sample_rates,
                        channels=supported_channels,
                        current_sample_rate=default_sample_rate,
                        current_channels=1,  # Default to mono
                        is_available=True,
                    )

                    devices.append(metadata)
                    logger.debug(
                        f"Found audio device: {name} ({default_sample_rate}Hz, {max_channels}ch)"
                    )

        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}", exc_info=True)

        logger.info(f"Found {len(devices)} audio device(s)")
        return devices

    def get_device_metadata(self, device_id: str) -> DeviceMetadata:
        """Get detailed metadata for a specific device.

        Args:
            device_id: Device identifier (e.g., "screen_1", "uvc_0", "audio_2")

        Returns:
            DeviceMetadata object with detailed device information.

        Raises:
            ValueError: If device_id is invalid or device not found.

        Requirements: 27.1-27.6
        """
        logger.debug(f"Getting metadata for device: {device_id}")

        # Determine device type from ID
        if device_id.startswith("screen_"):
            devices = self.enumerate_screens()
        elif device_id.startswith("uvc_"):
            devices = self.enumerate_uvc_devices()
        elif device_id.startswith("audio_"):
            devices = self.enumerate_audio_devices()
        else:
            raise ValueError(f"Invalid device_id format: {device_id}")

        # Find the device
        for device in devices:
            if device.device_id == device_id:
                logger.debug(f"Found device metadata for: {device_id}")
                return device

        # Device not found
        error_msg = f"Device not found: {device_id}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def validate_settings(
        self,
        device_id: str,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
    ) -> bool:
        """Validate if settings are supported by device.

        Args:
            device_id: Device identifier
            resolution: Desired resolution as (width, height) tuple
            fps: Desired frame rate

        Returns:
            True if settings are valid, False otherwise.

        Requirements: 27.8
        """
        logger.debug(f"Validating settings for device: {device_id}")

        try:
            metadata = self.get_device_metadata(device_id)

            # Validate resolution for video devices
            if resolution is not None and metadata.device_type != DeviceType.AUDIO:
                res = Resolution.from_tuple(resolution)
                if res not in metadata.supported_resolutions:
                    logger.warning(
                        f"Resolution {resolution} not supported by {device_id}. "
                        f"Supported: {[r.to_tuple() for r in metadata.supported_resolutions]}"
                    )
                    return False

            # Validate FPS for video devices
            if fps is not None and metadata.device_type != DeviceType.AUDIO:
                if fps not in metadata.supported_fps:
                    logger.warning(
                        f"FPS {fps} not supported by {device_id}. "
                        f"Supported: {metadata.supported_fps}"
                    )
                    return False

            logger.debug(f"Settings validated successfully for {device_id}")
            return True

        except ValueError as e:
            logger.error(f"Error validating settings: {e}")
            return False

    def suggest_optimal_settings(self, device_id: str) -> OptimalSettings:
        """Suggest optimal capture settings for device.

        Args:
            device_id: Device identifier

        Returns:
            OptimalSettings object with suggested settings and reasoning.

        Requirements: 27.9
        """
        logger.debug(f"Suggesting optimal settings for device: {device_id}")

        try:
            metadata = self.get_device_metadata(device_id)

            if metadata.device_type == DeviceType.AUDIO:
                # For audio devices, suggest common speech recognition settings
                optimal = OptimalSettings(
                    sample_rate=16000,  # Standard for speech recognition
                    channels=1,  # Mono is sufficient for speech
                    reason="16kHz mono is optimal for speech recognition while minimizing bandwidth",
                )
                logger.debug(f"Suggested audio settings: {optimal}")
                return optimal

            else:
                # For video devices, balance quality and performance
                # Prefer 1080p for good quality without excessive bandwidth
                preferred_resolution = Resolution(1920, 1080)

                # Find closest supported resolution
                if preferred_resolution in metadata.supported_resolutions:
                    optimal_resolution = preferred_resolution
                else:
                    # Find closest resolution by total pixels
                    optimal_resolution = min(
                        metadata.supported_resolutions,
                        key=lambda r: abs(r.total_pixels - preferred_resolution.total_pixels),
                    )

                # Use default 1 FPS for optimal performance
                optimal_fps = 1

                # Prefer RGB color format
                optimal_color_format = (
                    "RGB" if "RGB" in metadata.color_formats else metadata.color_formats[0]
                )

                optimal = OptimalSettings(
                    resolution=optimal_resolution,
                    fps=optimal_fps,
                    color_format=optimal_color_format,
                    reason=(
                        f"Resolution {optimal_resolution} balances quality and performance. "
                        f"FPS {optimal_fps} minimizes bandwidth while maintaining responsiveness. "
                        f"Color format {optimal_color_format} is widely compatible."
                    ),
                )
                logger.debug(f"Suggested video settings: {optimal}")
                return optimal

        except ValueError as e:
            logger.error(f"Error suggesting optimal settings: {e}")
            # Return default settings
            return OptimalSettings(
                reason=f"Error retrieving device metadata: {e}. Using default settings."
            )
