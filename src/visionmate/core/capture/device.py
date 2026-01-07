"""Device Manager for enumerating and managing capture devices.

This module provides the DeviceManager class which handles device enumeration,
metadata retrieval, settings validation, and optimal settings suggestions for
all types of capture devices (screens, UVC devices, audio devices).
"""

from __future__ import annotations

import logging
import platform
from typing import List

import cv2

from visionmate.core.models import DeviceMetadata, DeviceType, Resolution

logger = logging.getLogger(__name__)

# Screen capture constants
MAX_SCREEN_FPS = 240  # Maximum supported FPS for screen capture

# UVC device enumeration constants
MAX_UVC_DEVICE_INDEX = 100  # Maximum device index to check
MAX_CONSECUTIVE_FAILURES = 5  # Stop after this many consecutive failures


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

    def get_screens(self) -> List[DeviceMetadata]:
        """Get available screens.

        Returns:
            List of DeviceMetadata objects for each available screen.
        """
        logger.info("Getting screens...")
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

                    # Get native refresh rate from the monitor
                    fps = self._get_monitor_refresh_rate(i - 1)

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.SCREEN,
                        resolution=resolution,
                        fps=fps,
                        color_format="RGB",  # mss returns RGB format
                        is_available=True,
                    )

                    screens.append(metadata)
                    logger.debug(f"Found screen: {name} ({resolution} @ {fps}Hz)")

        except Exception as e:
            logger.error(f"Error enumerating screens: {e}", exc_info=True)

        logger.info(f"Found {len(screens)} screen(s)")
        return screens

    def get_uvc_devices(self) -> List[DeviceMetadata]:
        """Get UVC video devices.

        Returns:
            List of DeviceMetadata objects for each available UVC device.
        """
        logger.info("Getting UVC devices...")
        devices: List[DeviceMetadata] = []

        try:
            consecutive_failures = 0

            # Try to open video capture devices up to MAX_UVC_DEVICE_INDEX
            # Stop early if we encounter MAX_CONSECUTIVE_FAILURES consecutive failures
            for i in range(MAX_UVC_DEVICE_INDEX):
                cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    consecutive_failures = 0  # Reset failure counter

                    device_id = f"uvc_{i}"
                    name = f"UVC Device {i}"

                    # Get device capabilities
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    device_fps = int(cap.get(cv2.CAP_PROP_FPS))

                    resolution = Resolution(width=width, height=height)
                    fps = device_fps if device_fps > 0 else 0  # 0 if unknown

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.UVC,
                        resolution=resolution,
                        fps=fps,
                        color_format="BGR",  # OpenCV returns BGR format
                        is_available=True,
                    )

                    devices.append(metadata)
                    logger.debug(f"Found UVC device: {name} ({resolution} @ {fps}Hz)")

                    cap.release()
                else:
                    consecutive_failures += 1
                    # Stop searching if we've had too many consecutive failures
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.debug(
                            f"Stopping UVC device search after {consecutive_failures} "
                            f"consecutive failures at index {i}"
                        )
                        break

        except Exception as e:
            logger.error(f"Error getting UVC devices: {e}", exc_info=True)

        logger.info(f"Found {len(devices)} UVC device(s)")
        return devices

    def get_audio_devices(self) -> List[DeviceMetadata]:
        """Get audio input devices.

        Returns:
            List of DeviceMetadata objects for each available audio device.
        """
        logger.info("Getting audio devices...")
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
                    sample_rate = int(device_info["default_samplerate"])

                    # Supported channel configurations
                    supported_channels = list(range(1, max_channels + 1))

                    metadata = DeviceMetadata(
                        device_id=device_id,
                        name=name,
                        device_type=DeviceType.AUDIO,
                        sample_rate=sample_rate,
                        channels=supported_channels,
                        current_channels=1,  # Default to mono
                        is_available=True,
                    )

                    devices.append(metadata)
                    logger.debug(f"Found audio device: {name} ({sample_rate}Hz, {max_channels}ch)")

        except Exception as e:
            logger.error(f"Error getting audio devices: {e}", exc_info=True)

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
        """
        logger.debug(f"Getting metadata for device: {device_id}")

        # Determine device type from ID
        if device_id.startswith("screen_"):
            devices = self.get_screens()
        elif device_id.startswith("uvc_"):
            devices = self.get_uvc_devices()
        elif device_id.startswith("audio_"):
            devices = self.get_audio_devices()
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

    def _get_monitor_refresh_rate(self, monitor_index: int) -> int:
        """Get the refresh rate of a monitor.

        Args:
            monitor_index: Index of the monitor (0-based)

        Returns:
            Refresh rate in Hz, defaults to 0 if unable to detect.
        """
        try:
            if self._platform == "Darwin":  # macOS
                try:
                    import Quartz  # type: ignore[import-untyped]

                    # Get all displays
                    online_displays = Quartz.CGGetOnlineDisplayList(32, None, None)[1]  # type: ignore[attr-defined]
                    if monitor_index < len(online_displays):
                        display_id = online_displays[monitor_index]
                        mode = Quartz.CGDisplayCopyDisplayMode(display_id)  # type: ignore[attr-defined]
                        if mode:
                            refresh_rate = int(Quartz.CGDisplayModeGetRefreshRate(mode))  # type: ignore[attr-defined]
                            # Some displays return 0 for default refresh rate
                            if refresh_rate > 0:
                                logger.debug(
                                    f"Monitor {monitor_index} refresh rate: {refresh_rate}Hz"
                                )
                                return refresh_rate
                except Exception as e:
                    logger.debug(f"Error getting macOS refresh rate: {e}")

            elif self._platform == "Windows":
                try:
                    import win32api  # type: ignore[import-untyped]

                    devices = win32api.EnumDisplayDevices()  # type: ignore[attr-defined]
                    if monitor_index < len(devices):
                        device = devices[monitor_index]
                        settings = win32api.EnumDisplaySettings(device.DeviceName, -1)  # type: ignore[attr-defined]
                        if settings and hasattr(settings, "DisplayFrequency"):
                            refresh_rate = settings.DisplayFrequency
                            logger.debug(f"Monitor {monitor_index} refresh rate: {refresh_rate}Hz")
                            return refresh_rate
                except Exception as e:
                    logger.debug(f"Error getting Windows refresh rate: {e}")

            elif self._platform == "Linux":
                try:
                    import subprocess

                    # Try using xrandr to get refresh rate
                    result = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        lines = result.stdout.split("\n")
                        connected_displays = []
                        for line in lines:
                            if " connected" in line and "*" in line:
                                # Extract refresh rate from lines like "1920x1080 60.00*+"
                                parts = line.split()
                                for part in parts:
                                    if "*" in part:
                                        rate_str = part.replace("*", "").replace("+", "")
                                        try:
                                            refresh_rate = int(float(rate_str))
                                            connected_displays.append(refresh_rate)
                                            break
                                        except ValueError:
                                            continue

                        if monitor_index < len(connected_displays):
                            refresh_rate = connected_displays[monitor_index]
                            logger.debug(f"Monitor {monitor_index} refresh rate: {refresh_rate}Hz")
                            return refresh_rate
                except Exception as e:
                    logger.debug(f"Error getting Linux refresh rate: {e}")

        except Exception as e:
            logger.debug(f"Unexpected error getting refresh rate: {e}")

        # Return 0 if unable to detect (unknown refresh rate)
        logger.debug(f"Unable to detect refresh rate for monitor {monitor_index}, returning 0")
        return 0
