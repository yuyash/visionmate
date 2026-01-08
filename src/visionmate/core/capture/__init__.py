"""Capture module for video and audio input.

This module provides device management and capture functionality for
various input sources including screens, UVC devices, RTSP streams,
and audio devices.
"""

from visionmate.core.capture.audio import (
    AudioCaptureInterface,
    AudioMixer,
    DeviceAudioCapture,
    RTSPAudioCapture,
    UVCAudioCapture,
)
from visionmate.core.capture.device import DeviceManager
from visionmate.core.capture.manager import CaptureManager
from visionmate.core.capture.source import VideoSourceManager
from visionmate.core.capture.stream import AudioStream, StreamManager, VideoStream
from visionmate.core.capture.video import (
    ScreenCapture,
    VideoCaptureInterface,
    WindowDetector,
)

__all__ = [
    "AudioCaptureInterface",
    "AudioMixer",
    "AudioStream",
    "CaptureManager",
    "DeviceAudioCapture",
    "DeviceManager",
    "RTSPAudioCapture",
    "ScreenCapture",
    "StreamManager",
    "UVCAudioCapture",
    "VideoCaptureInterface",
    "VideoSourceManager",
    "VideoStream",
    "WindowDetector",
]
