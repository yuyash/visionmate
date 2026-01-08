"""Unit tests for SettingsDialog."""

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from visionmate.core.models import (
    AudioMode,
    STTProvider,
    VLMProvider,
)
from visionmate.core.settings import SettingsManager
from visionmate.desktop.dialogs import SettingsDialog


@pytest.fixture
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def settings_manager(tmp_path):
    """Create a SettingsManager with temporary config directory."""
    return SettingsManager(config_dir=tmp_path / "config")


def test_settings_dialog_initialization(qapp, settings_manager):
    """Test that SettingsDialog initializes correctly.

    Requirements: 15.1
    """
    # Mock keyring to avoid CI environment issues
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)

        # Verify dialog is created
        assert dialog is not None
        assert dialog.windowTitle() == "Settings"

        # Verify tabs are created
        assert dialog._tab_widget.count() == 5
        assert dialog._tab_widget.tabText(0) == "General"
        assert dialog._tab_widget.tabText(1) == "VLM"
        assert dialog._tab_widget.tabText(2) == "Audio"
        assert dialog._tab_widget.tabText(3) == "UI"
        assert dialog._tab_widget.tabText(4) == "Advanced"


def test_settings_dialog_general_tab(qapp, settings_manager):
    """Test General tab controls.

    Requirements: 15.4
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=5)

        # Verify FPS spinbox
        assert dialog._fps_spinbox.value() == 5
        assert dialog._fps_spinbox.minimum() == 1
        assert dialog._fps_spinbox.maximum() == 240


def test_settings_dialog_vlm_tab(qapp, settings_manager):
    """Test VLM tab controls.

    Requirements: 15.2, 5.5, 5.6
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)

        # Verify VLM provider combo box
        assert dialog._vlm_provider_combo.count() == 2
        assert dialog._vlm_provider_combo.itemData(0) == VLMProvider.OPENAI_REALTIME
        assert dialog._vlm_provider_combo.itemData(1) == VLMProvider.OPENAI_COMPATIBLE

        # Verify model combo box is populated
        assert dialog._vlm_model_combo.count() > 0

        # Verify API key input exists
        assert dialog._api_key_input is not None

        # Verify base URL input exists
        assert dialog._base_url_input is not None


def test_settings_dialog_audio_tab(qapp, settings_manager):
    """Test Audio tab controls.

    Requirements: 15.3, 7.5, 7.6
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)

        # Verify audio mode combo box
        assert dialog._audio_mode_combo.count() == 2
        assert dialog._audio_mode_combo.itemData(0) == AudioMode.DIRECT
        assert dialog._audio_mode_combo.itemData(1) == AudioMode.TEXT

        # Verify STT provider combo box
        assert dialog._stt_provider_combo.count() == 2
        assert dialog._stt_provider_combo.itemData(0) == STTProvider.WHISPER
        assert dialog._stt_provider_combo.itemData(1) == STTProvider.CLOUD


def test_settings_dialog_ui_tab(qapp, settings_manager):
    """Test UI tab controls.

    Requirements: 15.5
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)

        # Verify language combo box
        assert dialog._language_combo.count() >= 2

        # Verify timezone combo box
        assert dialog._timezone_combo.count() >= 5

        # Verify layout combo boxes
        assert dialog._video_layout_combo.count() == 3
        assert dialog._audio_layout_combo.count() == 3


def test_settings_dialog_vlm_provider_change(qapp, settings_manager):
    """Test VLM provider change updates model list.

    Requirements: 5.5, 5.6
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)
        dialog.show()  # Show dialog to ensure layout is active

        # Switch to VLM tab
        dialog._tab_widget.setCurrentIndex(1)  # VLM tab is index 1
        qapp.processEvents()  # Process pending events

        # Select OpenAI Realtime
        dialog._vlm_provider_combo.setCurrentIndex(0)
        dialog._on_vlm_provider_changed(0)  # Manually trigger handler
        qapp.processEvents()  # Process pending events
        assert dialog._vlm_model_combo.count() > 0
        assert not dialog._base_url_input.isVisible()

        # Select OpenAI Compatible
        dialog._vlm_provider_combo.setCurrentIndex(1)
        dialog._on_vlm_provider_changed(1)  # Manually trigger handler
        qapp.processEvents()  # Process pending events
        assert dialog._vlm_model_combo.count() > 0
        assert dialog._base_url_input.isVisible()


def test_settings_dialog_audio_mode_change(qapp, settings_manager):
    """Test audio mode change shows/hides STT provider.

    Requirements: 7.5, 7.6
    """
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=1)
        dialog.show()  # Show dialog to ensure layout is active

        # Switch to Audio tab
        dialog._tab_widget.setCurrentIndex(2)  # Audio tab is index 2
        qapp.processEvents()  # Process pending events

        # Select Direct mode
        dialog._audio_mode_combo.setCurrentIndex(0)
        dialog._on_audio_mode_changed(0)  # Manually trigger handler
        qapp.processEvents()  # Process pending events
        assert not dialog._stt_provider_combo.isVisible()

        # Select Text mode
        dialog._audio_mode_combo.setCurrentIndex(1)
        dialog._on_audio_mode_changed(1)  # Manually trigger handler
        qapp.processEvents()  # Process pending events
        assert dialog._stt_provider_combo.isVisible()


def test_settings_dialog_get_fps(qapp, settings_manager):
    """Test get_fps method for backward compatibility."""
    with patch("keyring.get_password", return_value=None):
        dialog = SettingsDialog(settings_manager=settings_manager, current_fps=10)

        assert dialog.get_fps() == 10

        # Change FPS
        dialog._fps_spinbox.setValue(30)
        assert dialog.get_fps() == 30
