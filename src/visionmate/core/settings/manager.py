"""Settings manager for Visionmate application.

This module provides the SettingsManager class for persisting and loading
application settings, as well as secure credential management using the
system keyring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import keyring
import keyring.errors

from visionmate.core.models import (
    AppSettings,
    AudioMode,
    AudioSourceConfig,
    AudioSourceType,
    InputMode,
    LocaleSettings,
    PreviewLayout,
    PreviewLayoutSettings,
    Resolution,
    STTProvider,
    STTSettings,
    VideoSourceConfig,
    VideoSourceType,
    VLMProvider,
    VLMSettings,
    WindowGeometry,
)

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings and credentials.

    Provides methods for:
    - Loading and saving settings to JSON files
    - Storing and retrieving credentials from system keyring
    - Managing settings persistence in user config directory
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize settings manager.

        Args:
            config_dir: Optional custom config directory. If None, uses platform default.
        """
        if config_dir is None:
            # Use platform-specific config directory
            config_dir = self._get_default_config_dir()

        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.config_dir / "settings.json"

        logger.info(f"Settings manager initialized with config dir: {self.config_dir}")

    def _get_default_config_dir(self) -> Path:
        """Get platform-specific default config directory.

        Returns:
            Path to config directory
        """
        import sys

        if sys.platform == "darwin":
            # macOS: ~/Library/Application Support/Visionmate
            base = Path.home() / "Library" / "Application Support"
        elif sys.platform == "win32":
            # Windows: %APPDATA%/Visionmate
            import os

            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            # Linux: ~/.config/visionmate
            import os

            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

        return base / "visionmate"

    def load_settings(self) -> AppSettings:
        """Load settings from storage.

        Returns:
            AppSettings instance with loaded settings, or default settings if file doesn't exist

        Raises:
            ValueError: If settings file is corrupted or invalid
        """
        if not self.settings_file.exists():
            logger.info("Settings file not found, using default settings")
            return AppSettings()

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info("Settings loaded successfully")
            return self._deserialize_settings(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse settings file: {e}")
            raise ValueError(f"Settings file is corrupted: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            raise ValueError(f"Failed to load settings: {e}") from e

    def save_settings(self, settings: AppSettings) -> None:
        """Save settings to storage.

        Args:
            settings: AppSettings instance to save

        Raises:
            IOError: If settings file cannot be written
        """
        try:
            data = self._serialize_settings(settings)

            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.settings_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.settings_file)

            logger.info("Settings saved successfully")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            raise IOError(f"Failed to save settings: {e}") from e

    def _serialize_settings(self, settings: AppSettings) -> Dict[str, Any]:
        """Serialize AppSettings to JSON-compatible dictionary.

        Args:
            settings: AppSettings instance

        Returns:
            Dictionary representation of settings
        """
        return {
            "input_mode": settings.input_mode.value,
            "video_sources": [
                {
                    "source_type": vs.source_type.value,
                    "device_id": vs.device_id,
                    "fps": vs.fps,
                    "resolution": (
                        {"width": vs.resolution.width, "height": vs.resolution.height}
                        if vs.resolution
                        else None
                    ),
                    "enable_window_detection": vs.enable_window_detection,
                }
                for vs in settings.video_sources
            ],
            "audio_source": (
                {
                    "source_type": settings.audio_source.source_type.value,
                    "device_id": settings.audio_source.device_id,
                    "sample_rate": settings.audio_source.sample_rate,
                    "channels": settings.audio_source.channels,
                }
                if settings.audio_source
                else None
            ),
            "default_fps": settings.default_fps,
            "vlm_settings": {
                "provider": settings.vlm_settings.provider.value,
                "model": settings.vlm_settings.model,
                "api_key_service": settings.vlm_settings.api_key_service,
                "base_url": settings.vlm_settings.base_url,
            },
            "stt_settings": {
                "provider": settings.stt_settings.provider.value,
                "audio_mode": settings.stt_settings.audio_mode.value,
                "language": settings.stt_settings.language,
            },
            "locale_settings": {
                "locale": settings.locale_settings.locale_string,
                "timezone": settings.locale_settings.timezone_name,
            },
            "preview_layout_settings": {
                "video_layout": settings.preview_layout_settings.video_layout.value,
                "audio_layout": settings.preview_layout_settings.audio_layout.value,
            },
            "window_geometry": (
                {
                    "x": settings.window_geometry.x,
                    "y": settings.window_geometry.y,
                    "width": settings.window_geometry.width,
                    "height": settings.window_geometry.height,
                }
                if settings.window_geometry
                else None
            ),
        }

    def _deserialize_settings(self, data: Dict[str, Any]) -> AppSettings:
        """Deserialize JSON dictionary to AppSettings.

        Args:
            data: Dictionary representation of settings

        Returns:
            AppSettings instance
        """
        # Parse video sources
        video_sources = []
        for vs_data in data.get("video_sources", []):
            resolution = None
            if vs_data.get("resolution"):
                res_data = vs_data["resolution"]
                resolution = Resolution(width=res_data["width"], height=res_data["height"])

            video_sources.append(
                VideoSourceConfig(
                    source_type=VideoSourceType(vs_data["source_type"]),
                    device_id=vs_data["device_id"],
                    fps=vs_data.get("fps", 1),
                    resolution=resolution,
                    enable_window_detection=vs_data.get("enable_window_detection", False),
                )
            )

        # Parse audio source
        audio_source = None
        if data.get("audio_source"):
            as_data = data["audio_source"]
            audio_source = AudioSourceConfig(
                source_type=AudioSourceType(as_data["source_type"]),
                device_id=as_data["device_id"],
                sample_rate=as_data.get("sample_rate", 16000),
                channels=as_data.get("channels", 1),
            )

        # Parse VLM settings
        vlm_data = data.get("vlm_settings", {})
        vlm_settings = VLMSettings(
            provider=VLMProvider(vlm_data.get("provider", "openai_realtime")),
            model=vlm_data.get("model", "gpt-4o-realtime-preview"),
            api_key_service=vlm_data.get("api_key_service", "visionmate"),
            base_url=vlm_data.get("base_url"),
        )

        # Parse STT settings
        stt_data = data.get("stt_settings", {})
        stt_settings = STTSettings(
            provider=STTProvider(stt_data.get("provider", "whisper")),
            audio_mode=AudioMode(stt_data.get("audio_mode", "direct")),
            language=stt_data.get("language", "en"),
        )

        # Parse locale settings
        locale_data = data.get("locale_settings", {})
        locale_settings = LocaleSettings.from_strings(
            language=locale_data.get("locale", "en_US"),
            timezone_name=locale_data.get("timezone", "UTC"),
        )

        # Parse preview layout settings
        layout_data = data.get("preview_layout_settings", {})
        preview_layout_settings = PreviewLayoutSettings(
            video_layout=PreviewLayout(layout_data.get("video_layout", "horizontal")),
            audio_layout=PreviewLayout(layout_data.get("audio_layout", "horizontal")),
        )

        # Parse window geometry
        window_geometry = None
        if data.get("window_geometry"):
            wg_data = data["window_geometry"]
            window_geometry = WindowGeometry(
                x=wg_data["x"],
                y=wg_data["y"],
                width=wg_data["width"],
                height=wg_data["height"],
            )

        return AppSettings(
            input_mode=InputMode(data.get("input_mode", "video_audio")),
            video_sources=video_sources,
            audio_source=audio_source,
            default_fps=data.get("default_fps", 1),
            vlm_settings=vlm_settings,
            stt_settings=stt_settings,
            locale_settings=locale_settings,
            preview_layout_settings=preview_layout_settings,
            window_geometry=window_geometry,
        )

    def store_credential(self, service: str, username: str, password: str) -> None:
        """Store credential in system keyring.

        Args:
            service: Service name (e.g., "visionmate_openai")
            username: Username or identifier
            password: Password or API key to store

        Raises:
            RuntimeError: If credential storage fails
        """
        try:
            keyring.set_password(service, username, password)
            logger.info(f"Credential stored for service: {service}, username: {username}")
        except Exception as e:
            logger.error(f"Failed to store credential: {e}")
            raise RuntimeError(f"Failed to store credential: {e}") from e

    def get_credential(self, service: str, username: str) -> Optional[str]:
        """Retrieve credential from system keyring.

        Args:
            service: Service name
            username: Username or identifier

        Returns:
            Password/API key if found, None otherwise

        Raises:
            RuntimeError: If credential retrieval fails
        """
        try:
            password = keyring.get_password(service, username)
            if password:
                logger.info(f"Credential retrieved for service: {service}, username: {username}")
            else:
                logger.info(f"No credential found for service: {service}, username: {username}")
            return password
        except Exception as e:
            logger.error(f"Failed to retrieve credential: {e}")
            raise RuntimeError(f"Failed to retrieve credential: {e}") from e

    def delete_credential(self, service: str, username: str) -> None:
        """Delete credential from system keyring.

        Args:
            service: Service name
            username: Username or identifier

        Raises:
            RuntimeError: If credential deletion fails
        """
        try:
            keyring.delete_password(service, username)
            logger.info(f"Credential deleted for service: {service}, username: {username}")
        except keyring.errors.PasswordDeleteError:
            logger.warning(f"No credential to delete for service: {service}, username: {username}")
        except Exception as e:
            logger.error(f"Failed to delete credential: {e}")
            raise RuntimeError(f"Failed to delete credential: {e}") from e
