"""Session manager for coordinating application state and components.

This module provides the SessionManager class that coordinates all components
and manages the application session state, including capture lifecycle,
input mode configuration, and event broadcasting.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from visionmate.core.capture.audio import AudioCaptureInterface, AudioMixer
from visionmate.core.capture.manager import CaptureManager
from visionmate.core.capture.video import VideoCaptureInterface
from visionmate.core.models import (
    AudioSourceConfig,
    AudioSourceType,
    InputMode,
    SessionState,
    VideoSourceConfig,
    VideoSourceType,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages application session and coordinates components.

    This class coordinates all application components including capture,
    recognition, and UI. It manages the session state machine and broadcasts
    events to registered callbacks.

    State Machine:
        IDLE → ACTIVE (start)
        ACTIVE → IDLE (stop)
        ACTIVE → ACTIVE (reset - internal state change only)

    Requirements: 9.1-9.10, 3.1, 3.2
    """

    def __init__(self):
        """Initialize the SessionManager."""
        self._state = SessionState.IDLE
        self._input_mode = InputMode.VIDEO_AUDIO
        self._capture_manager = CaptureManager()
        self._audio_mixer = AudioMixer()
        self._callbacks: dict[str, list[Callable]] = {}
        self._video_source_configs: dict[str, VideoSourceConfig] = {}
        self._audio_source_config: Optional[AudioSourceConfig] = None
        logger.info("SessionManager initialized")

    # ========================================================================
    # State Management
    # ========================================================================

    def get_state(self) -> SessionState:
        """Get current session state.

        Returns:
            Current SessionState

        Requirements: 9.1-9.10
        """
        return self._state

    def get_input_mode(self) -> InputMode:
        """Get current input mode.

        Returns:
            Current InputMode

        Requirements: 3.1, 3.2
        """
        return self._input_mode

    def set_input_mode(self, mode: InputMode) -> None:
        """Set input mode.

        Args:
            mode: InputMode to set

        Raises:
            RuntimeError: If session is active

        Requirements: 3.1, 3.2
        """
        if self._state == SessionState.ACTIVE:
            raise RuntimeError("Cannot change input mode while session is active")

        old_mode = self._input_mode
        self._input_mode = mode
        logger.info(f"Input mode changed: {old_mode.value} → {mode.value}")
        self._broadcast_event("input_mode_changed", {"mode": mode})

    # ========================================================================
    # Session Control Operations
    # ========================================================================

    def start(self) -> None:
        """Start capture and recognition.

        This method transitions the session from IDLE to ACTIVE state,
        starts all configured video and audio sources, and begins
        processing input.

        Raises:
            RuntimeError: If session is already active or no sources configured

        Requirements: 9.1, 9.6
        """
        if self._state == SessionState.ACTIVE:
            raise RuntimeError("Session is already active")

        # Validate that we have at least one source configured
        has_video = len(self._video_source_configs) > 0
        has_audio = self._audio_source_config is not None

        if not has_video and not has_audio:
            raise RuntimeError("No input sources configured")

        # Validate input mode matches configured sources
        if self._input_mode == InputMode.VIDEO_ONLY and not has_video:
            raise RuntimeError("Video-only mode requires at least one video source")
        if self._input_mode == InputMode.AUDIO_ONLY and not has_audio:
            raise RuntimeError("Audio-only mode requires an audio source")
        if self._input_mode == InputMode.VIDEO_AUDIO and (not has_video or not has_audio):
            raise RuntimeError("Video+Audio mode requires both video and audio sources")

        logger.info("Starting session...")

        try:
            # Start video sources if needed
            if self._input_mode in (InputMode.VIDEO_AUDIO, InputMode.VIDEO_ONLY):
                for source_id, _config in self._video_source_configs.items():
                    capture = self._capture_manager.get_video_source(source_id)
                    if capture and not capture.is_capturing():
                        logger.info(f"Video source {source_id} already started")
                    else:
                        logger.warning(f"Video source {source_id} not found or not started")

            # Start audio sources if needed
            if self._input_mode in (InputMode.VIDEO_AUDIO, InputMode.AUDIO_ONLY):
                if self._audio_source_config:
                    # Audio sources should already be started when added
                    logger.info("Audio sources ready")

            # Transition to ACTIVE state
            self._state = SessionState.ACTIVE
            logger.info("Session started successfully")
            self._broadcast_event("state_changed", {"state": SessionState.ACTIVE})

        except Exception as e:
            logger.error(f"Failed to start session: {e}", exc_info=True)
            # Clean up on failure
            self._cleanup_sources()
            raise RuntimeError(f"Failed to start session: {e}") from e

    def stop(self) -> None:
        """Stop capture and recognition.

        This method transitions the session from ACTIVE to IDLE state,
        stops all video and audio capture, and stops processing input.

        Requirements: 9.2, 9.7
        """
        if self._state == SessionState.IDLE:
            logger.warning("Session is already stopped")
            return

        logger.info("Stopping session...")

        try:
            # Stop all video sources
            self._capture_manager.stop_all_video_sources()

            # Stop all audio sources
            self._capture_manager.stop_all_audio_sources()

            # Transition to IDLE state
            self._state = SessionState.IDLE
            logger.info("Session stopped successfully")
            self._broadcast_event("state_changed", {"state": SessionState.IDLE})

        except Exception as e:
            logger.error(f"Error stopping session: {e}", exc_info=True)
            # Force state transition even on error
            self._state = SessionState.IDLE
            self._broadcast_event("state_changed", {"state": SessionState.IDLE})
            raise

    def reset(self) -> None:
        """Reset question understanding (keep capturing).

        This method resets the recognition engine's question understanding
        state while keeping capture active. It notifies the VLM of a topic
        change and restarts question segmentation.

        Raises:
            RuntimeError: If session is not active

        Requirements: 9.3, 9.8
        """
        if self._state != SessionState.ACTIVE:
            raise RuntimeError("Cannot reset - session is not active")

        logger.info("Resetting session (topic change)...")

        try:
            # TODO: Notify VLM of topic change (will be implemented in VLM integration)
            # TODO: Reset question segmenter (will be implemented in recognition module)

            logger.info("Session reset successfully")
            self._broadcast_event("session_reset", {})

        except Exception as e:
            logger.error(f"Error resetting session: {e}", exc_info=True)
            raise

    # ========================================================================
    # Video Source Management
    # ========================================================================

    def add_video_source(
        self,
        source_type: VideoSourceType,
        device_id: str,
        fps: int = 1,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> str:
        """Add a video source.

        Args:
            source_type: Type of video source
            device_id: Device identifier
            fps: Frame rate (default: 1)
            resolution: Optional resolution override
            enable_window_detection: Enable window detection (default: False)

        Returns:
            Source ID

        Raises:
            RuntimeError: If session is active
            ValueError: If source cannot be created

        Requirements: 9.4, 1.6
        """
        if self._state == SessionState.ACTIVE:
            raise RuntimeError("Cannot add video source while session is active")

        # Generate source ID
        source_id = f"{source_type.value}_{device_id}"

        # Check if source already exists
        if source_id in self._video_source_configs:
            raise ValueError(f"Video source already exists: {source_id}")

        # Create source configuration
        from visionmate.core.models import Resolution

        config = VideoSourceConfig(
            source_type=source_type,
            device_id=device_id,
            fps=fps,
            resolution=Resolution.from_tuple(resolution) if resolution else None,
            enable_window_detection=enable_window_detection,
        )

        # Create and start the appropriate capture implementation
        capture: Optional[VideoCaptureInterface] = None

        try:
            if source_type == VideoSourceType.SCREEN:
                from visionmate.core.capture.video import ScreenCapture

                capture = ScreenCapture()
                capture.start_capture(
                    device_id=device_id,
                    fps=fps,
                    resolution=resolution,
                    enable_window_detection=True,  # Always enabled for screen
                )

            elif source_type == VideoSourceType.UVC:
                from visionmate.core.capture.video import UVCCapture

                capture = UVCCapture()
                capture.start_capture(
                    device_id=device_id,
                    fps=fps,
                    resolution=resolution,
                    enable_window_detection=enable_window_detection,
                )

            elif source_type == VideoSourceType.RTSP:
                from visionmate.core.capture.video import RTSPCapture

                capture = RTSPCapture()
                capture.start_capture(
                    device_id=device_id,
                    fps=fps,
                    resolution=resolution,
                    enable_window_detection=enable_window_detection,
                )

            else:
                raise ValueError(f"Unsupported video source type: {source_type}")

            # Add to capture manager
            self._capture_manager.add_video_source(source_id, capture)

            # Store configuration
            self._video_source_configs[source_id] = config

            logger.info(f"Added video source: {source_id} ({source_type.value})")
            self._broadcast_event("video_source_added", {"source_id": source_id, "config": config})

            return source_id

        except Exception as e:
            logger.error(f"Failed to add video source: {e}", exc_info=True)
            # Clean up on failure
            if capture:
                try:
                    capture.stop_capture()
                except Exception:
                    pass
            raise ValueError(f"Failed to add video source: {e}") from e

    def remove_video_source(self, source_id: str) -> None:
        """Remove a video source.

        Args:
            source_id: Source identifier

        Raises:
            RuntimeError: If session is active
            KeyError: If source does not exist

        Requirements: 9.4, 1.6
        """
        if self._state == SessionState.ACTIVE:
            raise RuntimeError("Cannot remove video source while session is active")

        if source_id not in self._video_source_configs:
            raise KeyError(f"Video source not found: {source_id}")

        # Remove from capture manager
        self._capture_manager.remove_video_source(source_id)

        # Remove configuration
        del self._video_source_configs[source_id]

        logger.info(f"Removed video source: {source_id}")
        self._broadcast_event("video_source_removed", {"source_id": source_id})

    def get_video_source_ids(self) -> list[str]:
        """Get list of all video source IDs.

        Returns:
            List of source IDs

        Requirements: 1.6
        """
        return list(self._video_source_configs.keys())

    def get_video_source_count(self) -> int:
        """Get number of video sources.

        Returns:
            Number of video sources

        Requirements: 1.6
        """
        return len(self._video_source_configs)

    # ========================================================================
    # Audio Source Management
    # ========================================================================

    def set_audio_source(
        self,
        source_type: AudioSourceType,
        device_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """Set audio source.

        Args:
            source_type: Type of audio source
            device_id: Device identifier
            sample_rate: Sample rate (default: 16000)
            channels: Number of channels (default: 1)

        Returns:
            Source ID

        Raises:
            RuntimeError: If session is active
            ValueError: If source cannot be created

        Requirements: 9.4, 2.1-2.3
        """
        if self._state == SessionState.ACTIVE:
            raise RuntimeError("Cannot set audio source while session is active")

        # Generate source ID
        source_id = f"{source_type.value}_{device_id}"

        # Create source configuration
        config = AudioSourceConfig(
            source_type=source_type,
            device_id=device_id,
            sample_rate=sample_rate,
            channels=channels,
        )

        # Create and start the appropriate capture implementation
        capture: Optional[AudioCaptureInterface] = None

        try:
            if source_type == AudioSourceType.DEVICE:
                from visionmate.core.capture.audio import DeviceAudioCapture

                capture = DeviceAudioCapture()
                capture.start_capture(
                    device_id=device_id,
                    sample_rate=sample_rate,
                    channels=channels,
                )

            elif source_type == AudioSourceType.UVC:
                from visionmate.core.capture.audio import UVCAudioCapture

                capture = UVCAudioCapture()
                capture.start_capture(
                    device_id=device_id,
                    sample_rate=sample_rate,
                    channels=channels,
                )

            elif source_type == AudioSourceType.RTSP:
                from visionmate.core.capture.audio import RTSPAudioCapture

                capture = RTSPAudioCapture()
                capture.start_capture(
                    device_id=device_id,
                    sample_rate=sample_rate,
                    channels=channels,
                )

            else:
                raise ValueError(f"Unsupported audio source type: {source_type}")

            # Remove existing audio source if any
            if self._audio_source_config:
                old_source_id = f"{self._audio_source_config.source_type.value}_{self._audio_source_config.device_id}"
                try:
                    self._capture_manager.remove_audio_source(old_source_id)
                except Exception as e:
                    logger.warning(f"Failed to remove old audio source: {e}")

            # Add to capture manager
            self._capture_manager.add_audio_source(source_id, capture)

            # Add to audio mixer
            self._audio_mixer.add_source(capture)

            # Store configuration
            self._audio_source_config = config

            logger.info(f"Set audio source: {source_id} ({source_type.value})")
            self._broadcast_event("audio_source_set", {"source_id": source_id, "config": config})

            return source_id

        except Exception as e:
            logger.error(f"Failed to set audio source: {e}", exc_info=True)
            # Clean up on failure
            if capture:
                try:
                    capture.stop_capture()
                except Exception:
                    pass
            raise ValueError(f"Failed to set audio source: {e}") from e

    def get_audio_source_id(self) -> Optional[str]:
        """Get audio source ID.

        Returns:
            Source ID, or None if no audio source configured

        Requirements: 2.1-2.3
        """
        if self._audio_source_config:
            return f"{self._audio_source_config.source_type.value}_{self._audio_source_config.device_id}"
        return None

    def has_audio_source(self) -> bool:
        """Check if audio source is configured.

        Returns:
            True if audio source is configured

        Requirements: 2.4
        """
        return self._audio_source_config is not None

    # ========================================================================
    # Event Broadcasting
    # ========================================================================

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for events.

        Args:
            event: Event name
            callback: Callback function

        Supported events:
            - state_changed: Session state changed (data: {"state": SessionState})
            - input_mode_changed: Input mode changed (data: {"mode": InputMode})
            - video_source_added: Video source added (data: {"source_id": str, "config": VideoSourceConfig})
            - video_source_removed: Video source removed (data: {"source_id": str})
            - audio_source_set: Audio source set (data: {"source_id": str, "config": AudioSourceConfig})
            - session_reset: Session reset (data: {})
            - error_occurred: Error occurred (data: {"error": str})

        Requirements: 9.1-9.10
        """
        if event not in self._callbacks:
            self._callbacks[event] = []

        self._callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")

    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Unregister callback for events.

        Args:
            event: Event name
            callback: Callback function to remove
        """
        if event in self._callbacks:
            try:
                self._callbacks[event].remove(callback)
                logger.debug(f"Unregistered callback for event: {event}")
            except ValueError:
                logger.warning(f"Callback not found for event: {event}")

    def _broadcast_event(self, event: str, data: dict) -> None:
        """Broadcast event to all registered callbacks.

        Args:
            event: Event name
            data: Event data
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event}: {e}", exc_info=True)

    # ========================================================================
    # Component Access
    # ========================================================================

    def get_capture_manager(self) -> CaptureManager:
        """Get the capture manager.

        Returns:
            CaptureManager instance
        """
        return self._capture_manager

    def get_audio_mixer(self) -> AudioMixer:
        """Get the audio mixer.

        Returns:
            AudioMixer instance
        """
        return self._audio_mixer

    # ========================================================================
    # Cleanup
    # ========================================================================

    def _cleanup_sources(self) -> None:
        """Clean up all sources on error."""
        try:
            self._capture_manager.stop_all_video_sources()
            self._capture_manager.stop_all_audio_sources()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation
        """
        return (
            f"SessionManager(state={self._state.value}, "
            f"input_mode={self._input_mode.value}, "
            f"video_sources={len(self._video_source_configs)}, "
            f"audio_source={'yes' if self._audio_source_config else 'no'})"
        )
