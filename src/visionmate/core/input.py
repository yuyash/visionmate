"""Input mode definitions for the Real-time QA Assistant."""

from enum import Enum


class CaptureMethod(Enum):
    """Capture method for screen input.

    Defines which capture technology is used for screen capture.
    """

    OS_NATIVE = "os_native"  # OS-native screen capture (MSS)
    UVC_DEVICE = "uvc_device"  # UVC device capture (OpenCV)

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return {
            CaptureMethod.OS_NATIVE: "OS-Native (MSS)",
            CaptureMethod.UVC_DEVICE: "UVC Device",
        }[self]


class InputMode(Enum):
    """Input mode for capture system.

    Defines which input sources are active during capture.
    """

    VIDEO_AUDIO = "video_audio"  # Both video and audio capture active
    VIDEO_ONLY = "video_only"  # Only video capture active
    AUDIO_ONLY = "audio_only"  # Only audio capture active

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return {
            InputMode.VIDEO_AUDIO: "Video + Audio",
            InputMode.VIDEO_ONLY: "Video Only",
            InputMode.AUDIO_ONLY: "Audio Only",
        }[self]

    @property
    def has_video(self) -> bool:
        """Check if this mode includes video capture."""
        return self in (InputMode.VIDEO_AUDIO, InputMode.VIDEO_ONLY)

    @property
    def has_audio(self) -> bool:
        """Check if this mode includes audio capture."""
        return self in (InputMode.VIDEO_AUDIO, InputMode.AUDIO_ONLY)
