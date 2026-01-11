"""Settings dialog for Visionmate application.

This module provides the comprehensive SettingsDialog for configuring all
application settings including VLM, audio, UI, and advanced options.
"""

from logging import Logger, getLogger
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from visionmate.core.models import (
    AudioMode,
    FrameSelectionStrategy,
    OpenAICompatibleModel,
    OpenAIRealtimeModel,
    PreviewLayout,
    STTProvider,
    VLMProvider,
)
from visionmate.core.settings import SettingsManager

logger: Logger = getLogger(name=__name__)


class SettingsDialog(QDialog):
    """Comprehensive settings dialog for Visionmate application.

    Provides tabbed interface for:
    - General settings (FPS, input mode)
    - VLM provider configuration
    - Audio processing configuration
    - UI preferences (language, timezone, layout)
    - Advanced settings

    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        current_fps: int = 1,  # For backward compatibility
        parent: Optional[QWidget] = None,
    ):
        """Initialize the SettingsDialog.

        Args:
            settings_manager: Settings manager instance
            current_fps: Current FPS value (for backward compatibility)
            parent: Optional parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing comprehensive SettingsDialog")

        self._settings_manager = settings_manager
        self._settings = settings_manager.load_settings()
        self._current_fps = current_fps  # For backward compatibility

        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumSize(600, 500)

        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self) -> None:
        """Setup the UI components with tabs."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Create tab widget
        self._tab_widget = QTabWidget()

        # Add tabs
        self._tab_widget.addTab(self._create_general_tab(), "General")
        self._tab_widget.addTab(self._create_vlm_tab(), "VLM")
        self._tab_widget.addTab(self._create_audio_tab(), "Audio")
        self._tab_widget.addTab(self._create_ui_tab(), "UI")
        self._tab_widget.addTab(self._create_advanced_tab(), "Advanced")

        layout.addWidget(self._tab_widget)

        # Add dialog buttons (OK and Cancel)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        logger.debug("SettingsDialog UI setup complete")

    def _create_general_tab(self) -> QWidget:
        """Create the General settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # FPS settings group
        fps_group = QGroupBox("Video Capture")
        fps_layout = QFormLayout(fps_group)

        self._fps_spinbox = QSpinBox()
        self._fps_spinbox.setMinimum(1)
        self._fps_spinbox.setMaximum(240)
        self._fps_spinbox.setValue(self._current_fps)
        self._fps_spinbox.setToolTip(
            "Default frame capture rate (1-240 fps)\nHigher values capture more frames per second"
        )

        fps_container = QWidget()
        fps_container_layout = QHBoxLayout(fps_container)
        fps_container_layout.setContentsMargins(0, 0, 0, 0)
        fps_container_layout.addWidget(self._fps_spinbox)
        fps_container_layout.addWidget(QLabel("fps"))
        fps_container_layout.addStretch()

        fps_layout.addRow("Default FPS:", fps_container)

        layout.addWidget(fps_group)

        # Add stretch to push content to top
        layout.addStretch()

        return widget

    def _create_vlm_tab(self) -> QWidget:
        """Create the VLM settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # VLM Provider group
        provider_group = QGroupBox("VLM Provider")
        provider_layout = QFormLayout(provider_group)

        # Provider selection
        self._vlm_provider_combo = QComboBox()
        self._vlm_provider_combo.addItem("OpenAI Realtime API", VLMProvider.OPENAI_REALTIME)
        self._vlm_provider_combo.addItem("OpenAI-Compatible API", VLMProvider.OPENAI_COMPATIBLE)
        self._vlm_provider_combo.currentIndexChanged.connect(self._on_vlm_provider_changed)
        provider_layout.addRow("Provider:", self._vlm_provider_combo)

        # Model selection
        self._vlm_model_combo = QComboBox()
        provider_layout.addRow("Model:", self._vlm_model_combo)

        # API Key input
        self._api_key_input = QLineEdit()
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText("Enter API key")
        self._api_key_input.setToolTip("API key will be stored securely in system keyring")
        provider_layout.addRow("API Key:", self._api_key_input)

        # Base URL input (for OpenAI-compatible)
        self._base_url_input = QLineEdit()
        self._base_url_input.setPlaceholderText("https://api.example.com/v1")
        self._base_url_input.setToolTip("Base URL for OpenAI-compatible API")
        self._base_url_label = QLabel("Base URL:")
        provider_layout.addRow(self._base_url_label, self._base_url_input)

        layout.addWidget(provider_group)

        # Add stretch to push content to top
        layout.addStretch()

        return widget

    def _create_audio_tab(self) -> QWidget:
        """Create the Audio settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # Audio Processing group
        audio_group = QGroupBox("Audio Processing")
        audio_layout = QFormLayout(audio_group)

        # Audio mode selection (for multimedia manager)
        self._mm_audio_mode_combo = QComboBox()
        self._mm_audio_mode_combo.addItem("Server-Side (Streaming)", AudioMode.SERVER_SIDE)
        self._mm_audio_mode_combo.addItem("Client-Side (Buffered)", AudioMode.CLIENT_SIDE)
        self._mm_audio_mode_combo.setToolTip(
            "Server-Side: Continuous streaming to VLM\n"
            "Client-Side: Local STT with buffered segments"
        )
        audio_layout.addRow("Audio Mode:", self._mm_audio_mode_combo)

        # Legacy audio mode selection (for backward compatibility)
        self._audio_mode_combo = QComboBox()
        self._audio_mode_combo.addItem("Direct Audio to VLM", AudioMode.DIRECT)
        self._audio_mode_combo.addItem("Convert to Text First", AudioMode.TEXT)
        self._audio_mode_combo.setToolTip(
            "Direct: Send audio directly to VLM\nText: Convert audio to text using STT first"
        )
        self._audio_mode_combo.currentIndexChanged.connect(self._on_audio_mode_changed)
        audio_layout.addRow("Legacy Mode:", self._audio_mode_combo)

        # STT Provider selection
        self._stt_provider_combo = QComboBox()
        self._stt_provider_combo.addItem("Whisper (Local)", STTProvider.WHISPER)
        self._stt_provider_combo.addItem("Cloud STT", STTProvider.CLOUD)
        self._stt_provider_combo.setToolTip("Speech-to-text provider for text conversion mode")
        self._stt_provider_label = QLabel("STT Provider:")
        audio_layout.addRow(self._stt_provider_label, self._stt_provider_combo)

        layout.addWidget(audio_group)

        # Frame Selection group
        frame_group = QGroupBox("Frame Selection")
        frame_layout = QFormLayout(frame_group)

        # Frame selection strategy
        self._frame_strategy_combo = QComboBox()
        self._frame_strategy_combo.addItem("Middle Frame", FrameSelectionStrategy.MIDDLE)
        self._frame_strategy_combo.addItem("Most Different", FrameSelectionStrategy.MOST_DIFFERENT)
        self._frame_strategy_combo.addItem("Adaptive", FrameSelectionStrategy.ADAPTIVE)
        self._frame_strategy_combo.addItem("Keyframe", FrameSelectionStrategy.KEYFRAME)
        self._frame_strategy_combo.setToolTip(
            "Middle: Select middle frame (simple, fast)\n"
            "Most Different: Select frame most different from last sent\n"
            "Adaptive: Select multiple frames if changes detected\n"
            "Keyframe: Select frames with high information content"
        )
        frame_layout.addRow("Strategy:", self._frame_strategy_combo)

        # Max frames per segment
        self._max_frames_spinbox = QSpinBox()
        self._max_frames_spinbox.setMinimum(1)
        self._max_frames_spinbox.setMaximum(10)
        self._max_frames_spinbox.setValue(3)
        self._max_frames_spinbox.setToolTip("Maximum frames to include per segment (1-10)")
        frame_layout.addRow("Max Frames/Segment:", self._max_frames_spinbox)

        layout.addWidget(frame_group)

        # Add stretch to push content to top
        layout.addStretch()

        return widget

    def _create_ui_tab(self) -> QWidget:
        """Create the UI settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # Locale group
        locale_group = QGroupBox("Locale")
        locale_layout = QFormLayout(locale_group)

        # Language selection
        self._language_combo = QComboBox()
        self._language_combo.addItem("English", "en_US")
        self._language_combo.addItem("日本語 (Japanese)", "ja_JP")
        locale_layout.addRow("Language:", self._language_combo)

        # Timezone selection
        self._timezone_combo = QComboBox()
        self._timezone_combo.addItem("UTC", "UTC")
        self._timezone_combo.addItem("America/New_York", "America/New_York")
        self._timezone_combo.addItem("America/Los_Angeles", "America/Los_Angeles")
        self._timezone_combo.addItem("Europe/London", "Europe/London")
        self._timezone_combo.addItem("Asia/Tokyo", "Asia/Tokyo")
        self._timezone_combo.addItem("Asia/Shanghai", "Asia/Shanghai")
        self._timezone_combo.setEditable(True)
        self._timezone_combo.setToolTip("IANA timezone name (e.g., America/New_York)")
        locale_layout.addRow("Timezone:", self._timezone_combo)

        layout.addWidget(locale_group)

        # Preview Layout group
        layout_group = QGroupBox("Preview Layout")
        layout_layout = QFormLayout(layout_group)

        # Video layout
        self._video_layout_combo = QComboBox()
        self._video_layout_combo.addItem("Horizontal", PreviewLayout.HORIZONTAL)
        self._video_layout_combo.addItem("Vertical", PreviewLayout.VERTICAL)
        self._video_layout_combo.addItem("Grid", PreviewLayout.GRID)
        layout_layout.addRow("Video Layout:", self._video_layout_combo)

        # Audio layout
        self._audio_layout_combo = QComboBox()
        self._audio_layout_combo.addItem("Horizontal", PreviewLayout.HORIZONTAL)
        self._audio_layout_combo.addItem("Vertical", PreviewLayout.VERTICAL)
        self._audio_layout_combo.addItem("Grid", PreviewLayout.GRID)
        layout_layout.addRow("Audio Layout:", self._audio_layout_combo)

        layout.addWidget(layout_group)

        # Add stretch to push content to top
        layout.addStretch()

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create the Advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # Advanced options group
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout(advanced_group)

        # Window detection checkbox
        self._window_detection_checkbox = QCheckBox("Enable window detection by default")
        self._window_detection_checkbox.setToolTip(
            "Automatically detect and crop to active windows"
        )
        advanced_layout.addWidget(self._window_detection_checkbox)

        layout.addWidget(advanced_group)

        # Buffer Settings group
        buffer_group = QGroupBox("Buffer Settings")
        buffer_layout = QFormLayout(buffer_group)

        # Max segment buffer size
        self._max_buffer_size_spinbox = QSpinBox()
        self._max_buffer_size_spinbox.setMinimum(10)
        self._max_buffer_size_spinbox.setMaximum(1000)
        self._max_buffer_size_spinbox.setValue(300)
        self._max_buffer_size_spinbox.setToolTip("Maximum number of segments to buffer (10-1000)")
        buffer_layout.addRow("Max Buffer Size:", self._max_buffer_size_spinbox)

        # Max buffer memory
        self._max_buffer_memory_spinbox = QSpinBox()
        self._max_buffer_memory_spinbox.setMinimum(50)
        self._max_buffer_memory_spinbox.setMaximum(2000)
        self._max_buffer_memory_spinbox.setValue(500)
        self._max_buffer_memory_spinbox.setSuffix(" MB")
        self._max_buffer_memory_spinbox.setToolTip("Maximum memory usage for buffer (50-2000 MB)")
        buffer_layout.addRow("Max Buffer Memory:", self._max_buffer_memory_spinbox)

        layout.addWidget(buffer_group)

        # Audio Detection group
        detection_group = QGroupBox("Audio Detection")
        detection_layout = QFormLayout(detection_group)

        # Energy threshold
        self._energy_threshold_spinbox = QDoubleSpinBox()
        self._energy_threshold_spinbox.setMinimum(0.001)
        self._energy_threshold_spinbox.setMaximum(1.0)
        self._energy_threshold_spinbox.setSingleStep(0.001)
        self._energy_threshold_spinbox.setDecimals(3)
        self._energy_threshold_spinbox.setValue(0.01)
        self._energy_threshold_spinbox.setToolTip(
            "Energy threshold for speech detection (0.001-1.0)"
        )
        detection_layout.addRow("Energy Threshold:", self._energy_threshold_spinbox)

        # Silence duration
        self._silence_duration_spinbox = QDoubleSpinBox()
        self._silence_duration_spinbox.setMinimum(0.5)
        self._silence_duration_spinbox.setMaximum(10.0)
        self._silence_duration_spinbox.setSingleStep(0.1)
        self._silence_duration_spinbox.setDecimals(1)
        self._silence_duration_spinbox.setValue(1.5)
        self._silence_duration_spinbox.setSuffix(" sec")
        self._silence_duration_spinbox.setToolTip(
            "Silence duration before considering speech ended (0.5-10.0 sec)"
        )
        detection_layout.addRow("Silence Duration:", self._silence_duration_spinbox)

        layout.addWidget(detection_group)

        # Network Settings group
        network_group = QGroupBox("Network Settings")
        network_layout = QFormLayout(network_group)

        # Max retry attempts
        self._max_retry_spinbox = QSpinBox()
        self._max_retry_spinbox.setMinimum(0)
        self._max_retry_spinbox.setMaximum(10)
        self._max_retry_spinbox.setValue(3)
        self._max_retry_spinbox.setToolTip("Maximum retry attempts for failed requests (0-10)")
        network_layout.addRow("Max Retry Attempts:", self._max_retry_spinbox)

        # Connection timeout
        self._connection_timeout_spinbox = QDoubleSpinBox()
        self._connection_timeout_spinbox.setMinimum(1.0)
        self._connection_timeout_spinbox.setMaximum(60.0)
        self._connection_timeout_spinbox.setSingleStep(1.0)
        self._connection_timeout_spinbox.setDecimals(1)
        self._connection_timeout_spinbox.setValue(10.0)
        self._connection_timeout_spinbox.setSuffix(" sec")
        self._connection_timeout_spinbox.setToolTip("Connection timeout (1.0-60.0 sec)")
        network_layout.addRow("Connection Timeout:", self._connection_timeout_spinbox)

        layout.addWidget(network_group)

        # Add stretch to push content to top
        layout.addStretch()

        return widget

    def _load_current_settings(self) -> None:
        """Load current settings into UI controls."""
        logger.debug("Loading current settings into UI")

        # General tab - use current_fps parameter if provided, otherwise use settings
        if self._current_fps > 0:
            self._fps_spinbox.setValue(self._current_fps)
        else:
            self._fps_spinbox.setValue(self._settings.default_fps)

        # VLM tab
        self._set_vlm_provider(self._settings.vlm_settings.provider)
        self._set_vlm_model(self._settings.vlm_settings.model)

        # Load API key from keyring
        api_key = self._settings_manager.get_credential(
            self._settings.vlm_settings.api_key_service, "api_key"
        )
        if api_key:
            self._api_key_input.setText(api_key)

        # Load base URL
        if self._settings.vlm_settings.base_url:
            self._base_url_input.setText(self._settings.vlm_settings.base_url)

        # Audio tab
        self._set_mm_audio_mode(self._settings.audio_mode)
        self._set_audio_mode(self._settings.stt_settings.audio_mode)
        self._set_stt_provider(self._settings.stt_settings.provider)
        self._set_frame_strategy(self._settings.manager_config.frame_selection_strategy)
        self._max_frames_spinbox.setValue(self._settings.manager_config.max_frames_per_segment)

        # UI tab
        self._set_language(self._settings.locale_settings.locale_string)
        self._set_timezone(self._settings.locale_settings.timezone_name)
        self._set_video_layout(self._settings.preview_layout_settings.video_layout)
        self._set_audio_layout(self._settings.preview_layout_settings.audio_layout)

        # Advanced tab
        self._max_buffer_size_spinbox.setValue(
            self._settings.manager_config.max_segment_buffer_size
        )
        self._max_buffer_memory_spinbox.setValue(self._settings.manager_config.max_buffer_memory_mb)
        self._energy_threshold_spinbox.setValue(self._settings.manager_config.energy_threshold)
        self._silence_duration_spinbox.setValue(self._settings.manager_config.silence_duration_sec)
        self._max_retry_spinbox.setValue(self._settings.manager_config.max_retry_attempts)
        self._connection_timeout_spinbox.setValue(
            self._settings.manager_config.connection_timeout_sec
        )

        logger.debug("Settings loaded successfully")

    def _set_vlm_provider(self, provider: VLMProvider) -> None:
        """Set VLM provider in combo box."""
        for i in range(self._vlm_provider_combo.count()):
            if self._vlm_provider_combo.itemData(i) == provider:
                self._vlm_provider_combo.setCurrentIndex(i)
                # Manually trigger the change handler to update visibility
                self._on_vlm_provider_changed(i)
                break

    def _set_vlm_model(self, model: str) -> None:
        """Set VLM model in combo box."""
        for i in range(self._vlm_model_combo.count()):
            if self._vlm_model_combo.itemData(i) == model:
                self._vlm_model_combo.setCurrentIndex(i)
                break

    def _set_audio_mode(self, mode: AudioMode) -> None:
        """Set audio mode in combo box."""
        for i in range(self._audio_mode_combo.count()):
            if self._audio_mode_combo.itemData(i) == mode:
                self._audio_mode_combo.setCurrentIndex(i)
                # Manually trigger the change handler to update visibility
                self._on_audio_mode_changed(i)
                break

    def _set_mm_audio_mode(self, mode: AudioMode) -> None:
        """Set multimedia manager audio mode in combo box."""
        for i in range(self._mm_audio_mode_combo.count()):
            if self._mm_audio_mode_combo.itemData(i) == mode:
                self._mm_audio_mode_combo.setCurrentIndex(i)
                break

    def _set_frame_strategy(self, strategy: FrameSelectionStrategy) -> None:
        """Set frame selection strategy in combo box."""
        for i in range(self._frame_strategy_combo.count()):
            if self._frame_strategy_combo.itemData(i) == strategy:
                self._frame_strategy_combo.setCurrentIndex(i)
                break

    def _set_stt_provider(self, provider: STTProvider) -> None:
        """Set STT provider in combo box."""
        for i in range(self._stt_provider_combo.count()):
            if self._stt_provider_combo.itemData(i) == provider:
                self._stt_provider_combo.setCurrentIndex(i)
                break

    def _set_language(self, language: str) -> None:
        """Set language in combo box."""
        for i in range(self._language_combo.count()):
            if self._language_combo.itemData(i) == language:
                self._language_combo.setCurrentIndex(i)
                break

    def _set_timezone(self, timezone: str) -> None:
        """Set timezone in combo box."""
        # Try to find in combo box
        index = self._timezone_combo.findData(timezone)
        if index >= 0:
            self._timezone_combo.setCurrentIndex(index)
        else:
            # Set as custom text
            self._timezone_combo.setCurrentText(timezone)

    def _set_video_layout(self, layout: PreviewLayout) -> None:
        """Set video layout in combo box."""
        for i in range(self._video_layout_combo.count()):
            if self._video_layout_combo.itemData(i) == layout:
                self._video_layout_combo.setCurrentIndex(i)
                break

    def _set_audio_layout(self, layout: PreviewLayout) -> None:
        """Set audio layout in combo box."""
        for i in range(self._audio_layout_combo.count()):
            if self._audio_layout_combo.itemData(i) == layout:
                self._audio_layout_combo.setCurrentIndex(i)
                break

    def _on_vlm_provider_changed(self, index: int) -> None:
        """Handle VLM provider change.

        Args:
            index: Combo box index

        """
        provider = self._vlm_provider_combo.itemData(index)

        # Update model combo box based on provider
        self._vlm_model_combo.clear()

        if provider == VLMProvider.OPENAI_REALTIME:
            # Add OpenAI Realtime models
            for model in OpenAIRealtimeModel:
                self._vlm_model_combo.addItem(model.value, model.value)

            # Hide base URL for OpenAI Realtime
            self._base_url_label.setVisible(False)
            self._base_url_input.setVisible(False)

        elif provider == VLMProvider.OPENAI_COMPATIBLE:
            # Add OpenAI-compatible models
            for model in OpenAICompatibleModel:
                if model != OpenAICompatibleModel.CUSTOM:
                    self._vlm_model_combo.addItem(model.value, model.value)

            # Show base URL for OpenAI-compatible
            self._base_url_label.setVisible(True)
            self._base_url_input.setVisible(True)

        logger.debug(f"VLM provider changed to: {provider}")

    def _on_audio_mode_changed(self, index: int) -> None:
        """Handle audio mode change.

        Args:
            index: Combo box index

        """
        mode = self._audio_mode_combo.itemData(index)

        # Show/hide STT provider based on mode
        is_text_mode = mode == AudioMode.TEXT
        self._stt_provider_label.setVisible(is_text_mode)
        self._stt_provider_combo.setVisible(is_text_mode)

        logger.debug(f"Audio mode changed to: {mode}")

    def _on_accept(self) -> None:
        """Handle OK button click - save settings."""
        logger.info("Saving settings")

        try:
            # Update settings from UI
            self._settings.default_fps = self._fps_spinbox.value()

            # VLM settings
            self._settings.vlm_settings.provider = self._vlm_provider_combo.currentData()
            self._settings.vlm_settings.model = self._vlm_model_combo.currentData()
            self._settings.vlm_settings.base_url = (
                self._base_url_input.text() if self._base_url_input.text() else None
            )

            # Store API key in keyring if provided
            api_key = self._api_key_input.text()
            if api_key:
                self._settings_manager.store_credential(
                    self._settings.vlm_settings.api_key_service, "api_key", api_key
                )
                logger.info("API key stored in keyring")

            # Audio settings
            self._settings.audio_mode = self._mm_audio_mode_combo.currentData()
            self._settings.stt_settings.audio_mode = self._audio_mode_combo.currentData()
            self._settings.stt_settings.provider = self._stt_provider_combo.currentData()

            # Manager config - frame selection
            self._settings.manager_config.frame_selection_strategy = (
                self._frame_strategy_combo.currentData()
            )
            self._settings.manager_config.max_frames_per_segment = self._max_frames_spinbox.value()

            # Manager config - buffer settings
            self._settings.manager_config.max_segment_buffer_size = (
                self._max_buffer_size_spinbox.value()
            )
            self._settings.manager_config.max_buffer_memory_mb = (
                self._max_buffer_memory_spinbox.value()
            )

            # Manager config - audio detection
            self._settings.manager_config.energy_threshold = self._energy_threshold_spinbox.value()
            self._settings.manager_config.silence_duration_sec = (
                self._silence_duration_spinbox.value()
            )

            # Manager config - network settings
            self._settings.manager_config.max_retry_attempts = self._max_retry_spinbox.value()
            self._settings.manager_config.connection_timeout_sec = (
                self._connection_timeout_spinbox.value()
            )

            # UI settings
            from visionmate.core.models import LocaleSettings

            language = self._language_combo.currentData()
            timezone = self._timezone_combo.currentText()
            self._settings.locale_settings = LocaleSettings.from_strings(language, timezone)

            self._settings.preview_layout_settings.video_layout = (
                self._video_layout_combo.currentData()
            )
            self._settings.preview_layout_settings.audio_layout = (
                self._audio_layout_combo.currentData()
            )

            # Save settings to disk
            self._settings_manager.save_settings(self._settings)

            logger.info("Settings saved successfully")
            self.accept()

        except Exception as e:
            logger.error(f"Failed to save settings: {e}", exc_info=True)
            # Show error to user (could add QMessageBox here)
            self.reject()

    def get_fps(self) -> int:
        """Get the selected FPS value (for backward compatibility).

        Returns:
            FPS value (1-240)
        """
        return self._fps_spinbox.value()

    def exec(self) -> bool:
        """Execute the dialog and return whether OK was clicked.

        Returns:
            True if OK was clicked, False if Cancel was clicked
        """
        result = super().exec()
        return result == QDialog.DialogCode.Accepted
