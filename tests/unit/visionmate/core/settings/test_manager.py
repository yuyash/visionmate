"""Unit tests for SettingsManager."""

import json
import tempfile
from pathlib import Path

import pytest

from visionmate.core.models import (
    AppSettings,
    InputMode,
    LocaleSettings,
    Resolution,
    VideoSourceConfig,
    VideoSourceType,
    VLMProvider,
    VLMSettings,
    WindowGeometry,
)
from visionmate.core.settings import SettingsManager


class TestSettingsManager:
    """Test suite for SettingsManager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def settings_manager(self, temp_config_dir):
        """Create a SettingsManager with temporary config directory."""
        return SettingsManager(config_dir=temp_config_dir)

    def test_initialization(self, temp_config_dir):
        """Test SettingsManager initialization."""
        manager = SettingsManager(config_dir=temp_config_dir)
        assert manager.config_dir == temp_config_dir
        assert manager.settings_file == temp_config_dir / "settings.json"
        assert temp_config_dir.exists()

    def test_load_settings_default(self, settings_manager):
        """Test loading settings when file doesn't exist returns defaults."""
        settings = settings_manager.load_settings()
        assert isinstance(settings, AppSettings)
        assert settings.input_mode == InputMode.VIDEO_AUDIO
        assert settings.default_fps == 1
        assert len(settings.video_sources) == 0

    def test_save_and_load_settings(self, settings_manager):
        """Test saving and loading settings round trip."""
        # Create custom settings
        settings = AppSettings(
            input_mode=InputMode.VIDEO_ONLY,
            default_fps=5,
            window_geometry=WindowGeometry(x=100, y=200, width=1024, height=768),
        )

        # Save settings
        settings_manager.save_settings(settings)

        # Load settings
        loaded_settings = settings_manager.load_settings()

        # Verify
        assert loaded_settings.input_mode == InputMode.VIDEO_ONLY
        assert loaded_settings.default_fps == 5
        assert loaded_settings.window_geometry is not None
        assert loaded_settings.window_geometry.x == 100
        assert loaded_settings.window_geometry.y == 200
        assert loaded_settings.window_geometry.width == 1024
        assert loaded_settings.window_geometry.height == 768

    def test_save_settings_with_video_sources(self, settings_manager):
        """Test saving settings with video sources."""
        settings = AppSettings(
            video_sources=[
                VideoSourceConfig(
                    source_type=VideoSourceType.SCREEN,
                    device_id="0",
                    fps=2,
                    resolution=Resolution(width=1920, height=1080),
                    enable_window_detection=True,
                )
            ]
        )

        settings_manager.save_settings(settings)
        loaded_settings = settings_manager.load_settings()

        assert len(loaded_settings.video_sources) == 1
        vs = loaded_settings.video_sources[0]
        assert vs.source_type == VideoSourceType.SCREEN
        assert vs.device_id == "0"
        assert vs.fps == 2
        assert vs.resolution is not None
        assert vs.resolution.width == 1920
        assert vs.resolution.height == 1080
        assert vs.enable_window_detection is True

    def test_save_settings_with_vlm_settings(self, settings_manager):
        """Test saving settings with VLM configuration."""
        settings = AppSettings(
            vlm_settings=VLMSettings(
                provider=VLMProvider.OPENAI_COMPATIBLE,
                model="custom-model",
                base_url="http://localhost:8000",
            )
        )

        settings_manager.save_settings(settings)
        loaded_settings = settings_manager.load_settings()

        assert loaded_settings.vlm_settings.provider == VLMProvider.OPENAI_COMPATIBLE
        assert loaded_settings.vlm_settings.model == "custom-model"
        assert loaded_settings.vlm_settings.base_url == "http://localhost:8000"

    def test_save_settings_with_locale(self, settings_manager):
        """Test saving settings with locale configuration."""
        settings = AppSettings(
            locale_settings=LocaleSettings.from_strings(
                language="ja_JP", timezone_name="Asia/Tokyo"
            )
        )

        settings_manager.save_settings(settings)
        loaded_settings = settings_manager.load_settings()

        assert loaded_settings.locale_settings.language_code == "ja"
        assert loaded_settings.locale_settings.timezone_name == "Asia/Tokyo"

    def test_save_settings_atomic(self, settings_manager, temp_config_dir):
        """Test that settings are saved atomically."""
        settings = AppSettings(default_fps=10)
        settings_manager.save_settings(settings)

        # Verify settings file exists
        assert settings_manager.settings_file.exists()

        # Verify no temporary file remains
        temp_files = list(temp_config_dir.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_load_settings_corrupted_file(self, settings_manager):
        """Test loading settings with corrupted JSON file."""
        # Write corrupted JSON
        with open(settings_manager.settings_file, "w") as f:
            f.write("{ invalid json }")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Settings file is corrupted"):
            settings_manager.load_settings()

    def test_store_and_get_credential(self, settings_manager):
        """Test storing and retrieving credentials."""
        service = "test_service"
        username = "test_user"
        password = "test_password"

        # Store credential
        settings_manager.store_credential(service, username, password)

        # Retrieve credential
        retrieved = settings_manager.get_credential(service, username)
        assert retrieved == password

    def test_get_credential_not_found(self, settings_manager):
        """Test retrieving non-existent credential."""
        result = settings_manager.get_credential("nonexistent", "user")
        assert result is None

    def test_delete_credential(self, settings_manager):
        """Test deleting credentials."""
        service = "test_service"
        username = "test_user"
        password = "test_password"

        # Store credential
        settings_manager.store_credential(service, username, password)

        # Verify it exists
        assert settings_manager.get_credential(service, username) == password

        # Delete credential
        settings_manager.delete_credential(service, username)

        # Verify it's gone
        assert settings_manager.get_credential(service, username) is None

    def test_delete_nonexistent_credential(self, settings_manager):
        """Test deleting non-existent credential doesn't raise error."""
        # Should not raise an exception
        settings_manager.delete_credential("nonexistent", "user")

    def test_settings_file_format(self, settings_manager):
        """Test that settings file is properly formatted JSON."""
        settings = AppSettings(
            input_mode=InputMode.VIDEO_AUDIO,
            default_fps=1,
        )

        settings_manager.save_settings(settings)

        # Read and parse JSON
        with open(settings_manager.settings_file, "r") as f:
            data = json.load(f)

        # Verify structure
        assert "input_mode" in data
        assert "default_fps" in data
        assert "vlm_settings" in data
        assert "stt_settings" in data
        assert "locale_settings" in data
        assert data["input_mode"] == "video_audio"
        assert data["default_fps"] == 1
