"""Capture layer for screen and audio input."""

from .audio import AudioCaptureInterface, AudioChunk, SoundDeviceAudioCapture
from .screen import (
    MSSScreenCapture,
    ScreenCaptureInterface,
    UVCScreenCapture,
    WindowDetector,
    WindowInfo,
)

__all__ = [
    # Screen capture
    "ScreenCaptureInterface",
    "MSSScreenCapture",
    "UVCScreenCapture",
    "WindowInfo",
    "WindowDetector",
    # Audio capture
    "AudioCaptureInterface",
    "AudioChunk",
    "SoundDeviceAudioCapture",
]
