"""Capture module for video and audio input.

This module provides device management and capture functionality for
various input sources including screens, UVC devices, RTSP streams,
and audio devices.
"""

from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.video import (
    ScreenCapture,
    VideoCaptureInterface,
    WindowDetector,
)

__all__ = [
    "DeviceManager",
    "VideoCaptureInterface",
    "ScreenCapture",
    "WindowDetector",
]
