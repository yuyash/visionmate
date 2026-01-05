"""Input mode definitions for the Real-time QA Assistant."""

from enum import Enum


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
