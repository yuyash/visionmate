"""Core data models for Visionmate application.

This module contains all the core data models including enums, dataclasses,
and type definitions used throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from babel import Locale

# ============================================================================
# Basic Value Objects
# ============================================================================


@dataclass(frozen=True)
class Resolution:
    """Video resolution."""

    width: int
    height: int

    def __str__(self) -> str:
        """String representation."""
        return f"{self.width}x{self.height}"

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to (width, height) tuple."""
        return (self.width, self.height)

    @classmethod
    def from_tuple(cls, resolution: Tuple[int, int]) -> Resolution:
        """Create from (width, height) tuple."""
        return cls(width=resolution[0], height=resolution[1])

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height


# ============================================================================
# Basic Enums
# ============================================================================


class DeviceType(Enum):
    """Type of input device."""

    SCREEN = "screen"
    UVC = "uvc"
    RTSP = "rtsp"
    AUDIO = "audio"


class SessionState(Enum):
    """State of the application session."""

    IDLE = "idle"
    ACTIVE = "active"


class InputMode(Enum):
    """Input mode configuration."""

    VIDEO_AUDIO = "video_audio"
    VIDEO_ONLY = "video_only"
    AUDIO_ONLY = "audio_only"


class VideoSourceType(Enum):
    """Type of video source."""

    SCREEN = "screen"
    UVC = "uvc"
    RTSP = "rtsp"


class AudioSourceType(Enum):
    """Type of audio source."""

    DEVICE = "device"
    UVC = "uvc"
    RTSP = "rtsp"


class VLMProvider(Enum):
    """VLM provider type."""

    OPENAI_REALTIME = "openai_realtime"
    OPENAI_COMPATIBLE = "openai_compatible"


class OpenAIRealtimeModel(Enum):
    """OpenAI Realtime API models."""

    GPT_4O_REALTIME = "gpt-4o-realtime-preview"
    GPT_4O_REALTIME_2024_12_17 = "gpt-4o-realtime-preview-2024-12-17"


class OpenAICompatibleModel(Enum):
    """OpenAI-compatible models (can be extended)."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CUSTOM = "custom"  # For custom model names


class STTProvider(Enum):
    """Speech-to-text provider."""

    WHISPER = "whisper"
    CLOUD = "cloud"


class AudioMode(Enum):
    """Audio processing mode."""

    DIRECT = "direct"  # Pass audio directly to VLM
    TEXT = "text"  # Convert audio to text first


class PreviewLayout(Enum):
    """Preview layout mode."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


# ============================================================================
# Device Metadata Models
# ============================================================================


@dataclass
class DeviceMetadata:
    """Metadata for a capture device."""

    device_id: str
    name: str
    device_type: DeviceType

    # Video metadata
    resolution: Optional[Resolution] = None  # Device's resolution
    fps: Optional[int] = None  # Device's frame rate
    color_format: Optional[str] = None  # Color format (e.g., "RGB", "BGR")

    # Audio metadata
    sample_rate: Optional[int] = None  # Device's sample rate
    channels: List[int] = field(default_factory=list)
    current_channels: Optional[int] = None

    # Additional info
    is_available: bool = True
    error_message: Optional[str] = None


# ============================================================================
# Capture Data Models
# ============================================================================


@dataclass
class WindowRegion:
    """Detected window region."""

    x: int
    y: int
    width: int
    height: int
    confidence: float  # Detection confidence (0.0-1.0)
    title: str = ""  # Window title
    window_id: Optional[int] = None  # Platform-specific window ID
    area: int = field(init=False)  # Region area in pixels

    def __post_init__(self):
        """Calculate area after initialization."""
        self.area = self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class VideoFrame:
    """Captured video frame with metadata."""

    image: np.ndarray
    timestamp: datetime
    source_id: str
    source_type: VideoSourceType
    resolution: Resolution
    fps: int
    frame_number: int

    # Active window detection
    detected_regions: List[WindowRegion] = field(default_factory=list)
    active_region: Optional[WindowRegion] = None  # Currently used region (largest by default)
    is_cropped: bool = False


@dataclass
class AudioChunk:
    """Captured audio chunk with metadata."""

    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: datetime
    source_id: str
    source_type: AudioSourceType
    chunk_number: int


# ============================================================================
# Settings Models
# ============================================================================


# ============================================================================
# Settings Models
# ============================================================================


@dataclass
class VLMSettings:
    """VLM configuration settings."""

    provider: VLMProvider
    model: str  # Model name (validated against provider's enum)
    api_key_service: str = "visionmate"  # Keyring service name
    base_url: Optional[str] = None  # For OpenAI-compatible providers

    def get_model_enum(self) -> Optional[Enum]:
        """Get the appropriate model enum based on provider."""
        if self.provider == VLMProvider.OPENAI_REALTIME:
            try:
                return OpenAIRealtimeModel(self.model)
            except ValueError:
                return None
        elif self.provider == VLMProvider.OPENAI_COMPATIBLE:
            try:
                return OpenAICompatibleModel(self.model)
            except ValueError:
                return OpenAICompatibleModel.CUSTOM
        return None


@dataclass
class STTSettings:
    """Speech-to-text configuration settings."""

    provider: STTProvider
    audio_mode: AudioMode
    language: str = "en"  # Language code for transcription


@dataclass
class LocaleSettings:
    """Locale and internationalization settings."""

    locale: Locale = field(default_factory=lambda: Locale.parse("en_US"))
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("UTC"))

    @classmethod
    def from_strings(cls, language: str, timezone_name: str) -> LocaleSettings:
        """Create LocaleSettings from string representations.

        Args:
            language: Language code (e.g., "en", "ja", "en_US", "ja_JP")
            timezone_name: IANA timezone name (e.g., "UTC", "Asia/Tokyo", "America/New_York")

        Returns:
            LocaleSettings instance
        """
        locale = Locale.parse(language) if "_" in language else Locale(language)
        timezone = ZoneInfo(timezone_name)
        return cls(locale=locale, timezone=timezone)

    @property
    def language_code(self) -> str:
        """Get language code (e.g., 'en', 'ja')."""
        return self.locale.language or "en"

    @property
    def territory_code(self) -> Optional[str]:
        """Get territory/region code (e.g., 'US', 'JP')."""
        return self.locale.territory

    @property
    def locale_string(self) -> str:
        """Get full locale string (e.g., 'en_US', 'ja_JP')."""
        return str(self.locale)

    @property
    def timezone_name(self) -> str:
        """Get timezone name (e.g., 'UTC', 'Asia/Tokyo')."""
        return str(self.timezone)


@dataclass
class PreviewLayoutSettings:
    """Preview layout configuration."""

    video_layout: PreviewLayout = PreviewLayout.HORIZONTAL
    audio_layout: PreviewLayout = PreviewLayout.HORIZONTAL


@dataclass
class VideoSourceConfig:
    """Configuration for a video source."""

    source_type: VideoSourceType
    device_id: str
    fps: int = 1
    resolution: Optional[Resolution] = None
    enable_window_detection: bool = False


@dataclass
class AudioSourceConfig:
    """Configuration for an audio source."""

    source_type: AudioSourceType
    device_id: str
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class WindowGeometry:
    """Window geometry for UI persistence."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class AppSettings:
    """Application settings."""

    # Input settings
    input_mode: InputMode = InputMode.VIDEO_AUDIO
    video_sources: List[VideoSourceConfig] = field(default_factory=list)
    audio_source: Optional[AudioSourceConfig] = None
    default_fps: int = 1

    # VLM settings
    vlm_settings: VLMSettings = field(
        default_factory=lambda: VLMSettings(
            provider=VLMProvider.OPENAI_REALTIME,
            model=OpenAIRealtimeModel.GPT_4O_REALTIME.value,
        )
    )

    # STT settings
    stt_settings: STTSettings = field(
        default_factory=lambda: STTSettings(
            provider=STTProvider.WHISPER,
            audio_mode=AudioMode.DIRECT,
        )
    )

    # Locale settings
    locale_settings: LocaleSettings = field(default_factory=LocaleSettings)

    # Preview layout settings
    preview_layout_settings: PreviewLayoutSettings = field(default_factory=PreviewLayoutSettings)

    # Window geometry
    window_geometry: Optional[WindowGeometry] = None
