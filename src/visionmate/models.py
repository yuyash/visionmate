"""Core data models and enumerations for VisionMate application.

This module defines the fundamental data structures used throughout the application,
including enums for various modes and states, and dataclasses for configuration
and metadata.
"""

from dataclasses import dataclass, field
from enum import Enum


class InputMode(Enum):
    """Defines which types of inputs are active."""

    VIDEO_AUDIO = "video_audio"
    VIDEO_ONLY = "video_only"
    AUDIO_ONLY = "audio_only"


class SourceType(Enum):
    """Defines the type of video input source."""

    SCREEN_CAPTURE = "screen_capture"
    UVC_DEVICE = "uvc_device"
    RTSP_STREAM = "rtsp_stream"


class ContentType(Enum):
    """Defines the content type for metadata."""

    GENERAL_VIDEO = "general_video"
    SCREEN_CONTENT = "screen_content"


class LayoutMode(Enum):
    """Defines how previews are arranged."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


class CaptureState(Enum):
    """Defines the state of a capture source."""

    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class SourceMetadata:
    """Metadata for an input source."""

    source_id: str
    source_type: SourceType
    content_type: ContentType
    device_name: str
    resolution: tuple[int, int]
    fps: int
    state: CaptureState
    error_message: str | None = None
    # Type-specific fields
    device_id: str | None = None  # For UVC devices
    rtsp_url: str | None = None  # For RTSP streams
    capture_region: tuple[int, int, int, int] | None = None  # For screen capture


@dataclass
class PreviewConfig:
    """Configuration for a preview widget."""

    source_id: str
    position: int  # Position in layout order
    visible: bool = True


@dataclass
class VideoSourceConfig:
    """Configuration for a video source."""

    source_type: SourceType
    content_type: ContentType
    # Type-specific config
    monitor_index: int | None = None
    device_index: int | None = None
    rtsp_url: str | None = None


@dataclass
class AudioSourceConfig:
    """Configuration for an audio source."""

    device_index: int
    sample_rate: int = 44100
    channels: int = 1


@dataclass
class AppState:
    """Application-wide state."""

    input_mode: InputMode = InputMode.VIDEO_AUDIO
    video_source_type: SourceType = SourceType.SCREEN_CAPTURE
    layout_mode: LayoutMode = LayoutMode.VERTICAL
    is_session_active: bool = False
    video_configs: dict[str, VideoSourceConfig] = field(default_factory=dict)
    audio_configs: dict[str, AudioSourceConfig] = field(default_factory=dict)
    preview_order: list[str] = field(default_factory=list)
