"""Session manager for coordinating capture and recognition."""

import logging
from enum import Enum
from typing import Callable, Dict, List, Optional

from visionmate.capture.audio import AudioCaptureInterface
from visionmate.capture.screen import ScreenCaptureInterface, WindowInfo
from visionmate.core.input import CaptureMethod, InputMode

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session state for the application.

    Defines the current operational state of the session.
    """

    IDLE = "idle"  # No capture or recognition active
    CAPTURING = "capturing"  # Capture active, recognition not started
    RECOGNIZING = "recognizing"  # Both capture and recognition active

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return {
            SessionState.IDLE: "Idle",
            SessionState.CAPTURING: "Capturing",
            SessionState.RECOGNIZING: "Recognizing",
        }[self]


class SessionManager:
    """Manages capture sessions and coordinates input modes.

    The SessionManager is responsible for:
    - Managing session state machine (IDLE, CAPTURING, RECOGNIZING)
    - Managing capture lifecycle (start, stop)
    - Managing recognition lifecycle (start, stop, reset)
    - Coordinating input mode switching
    - Broadcasting events to UI components
    """

    def __init__(
        self,
        screen_capture: ScreenCaptureInterface,
        audio_capture: AudioCaptureInterface,
        uvc_capture: Optional[ScreenCaptureInterface] = None,
    ):
        """Initialize session manager.

        Args:
            screen_capture: Screen capture interface (OS-native)
            audio_capture: Audio capture interface
            uvc_capture: Optional UVC capture interface
        """
        self._os_native_capture = screen_capture
        self._uvc_capture = uvc_capture
        self._audio_capture = audio_capture
        self._capture_method = CaptureMethod.OS_NATIVE
        self._input_mode = InputMode.VIDEO_AUDIO
        self._session_state = SessionState.IDLE
        self._capture_fps = 30
        self._callbacks: Dict[str, List[Callable]] = {}

    def start_capture(self, fps: int = 30, window_id: Optional[int] = None) -> None:
        """Start capturing input at specified FPS.

        Transitions from IDLE to CAPTURING state.

        Args:
            fps: Frame rate (1-60)
            window_id: Optional window ID to capture specific window

        Raises:
            RuntimeError: If not in IDLE state
        """
        if self._session_state != SessionState.IDLE:
            logger.warning(f"Cannot start capture from {self._session_state} state")
            raise RuntimeError(f"Cannot start capture from {self._session_state} state")

        self._capture_fps = fps

        try:
            # Get the appropriate screen capture based on method
            screen_capture = self._get_active_screen_capture()

            # Start video capture if mode includes video
            if self._input_mode.has_video:
                logger.info(f"Starting video capture at {fps} FPS using {self._capture_method}")
                screen_capture.start_capture(fps=fps, window_id=window_id)

            # Start audio capture if mode includes audio
            if self._input_mode.has_audio:
                logger.info("Starting audio capture")
                self._audio_capture.start_capture()

            # Transition to CAPTURING state
            self._session_state = SessionState.CAPTURING
            self._emit_event("capture_started")
            logger.info(f"Capture started in {self._input_mode} mode, state: {self._session_state}")

        except Exception as e:
            logger.error(f"Error starting capture: {e}", exc_info=True)
            # Clean up on error
            self._cleanup_capture()
            raise

    def stop_capture(self) -> None:
        """Stop capturing input.

        Transitions from CAPTURING or RECOGNIZING to IDLE state.
        If in RECOGNIZING state, stops recognition first.

        Raises:
            RuntimeError: If in IDLE state
        """
        if self._session_state == SessionState.IDLE:
            logger.warning("Cannot stop capture from IDLE state")
            return

        try:
            # If recognizing, stop recognition first
            if self._session_state == SessionState.RECOGNIZING:
                logger.info("Stopping recognition before stopping capture")
                self._stop_recognition_internal()

            # Stop capture
            self._cleanup_capture()

            # Transition to IDLE state
            self._session_state = SessionState.IDLE
            self._emit_event("capture_stopped")
            logger.info(f"Capture stopped, state: {self._session_state}")

        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)

    def _cleanup_capture(self) -> None:
        """Internal method to clean up capture resources."""
        # Get the appropriate screen capture based on method
        screen_capture = self._get_active_screen_capture()

        # Stop video capture
        if self._input_mode.has_video:
            logger.info("Stopping video capture")
            screen_capture.stop_capture()

        # Stop audio capture
        if self._input_mode.has_audio:
            logger.info("Stopping audio capture")
            self._audio_capture.stop_capture()

    def start_recognition(self) -> None:
        """Start VLM recognition.

        Transitions from CAPTURING to RECOGNIZING state.

        Raises:
            RuntimeError: If not in CAPTURING state
        """
        if self._session_state != SessionState.CAPTURING:
            logger.warning(f"Cannot start recognition from {self._session_state} state")
            raise RuntimeError(f"Cannot start recognition from {self._session_state} state")

        try:
            # TODO: Initialize recognition components (VLM, STT, Question Segmenter)
            # This will be implemented in later tasks

            # Transition to RECOGNIZING state
            self._session_state = SessionState.RECOGNIZING
            self._emit_event("recognition_started")
            logger.info(f"Recognition started, state: {self._session_state}")

        except Exception as e:
            logger.error(f"Error starting recognition: {e}", exc_info=True)
            raise

    def stop_recognition(self) -> None:
        """Stop VLM recognition while keeping capture active.

        Transitions from RECOGNIZING to CAPTURING state.

        Raises:
            RuntimeError: If not in RECOGNIZING state
        """
        if self._session_state != SessionState.RECOGNIZING:
            logger.warning(f"Cannot stop recognition from {self._session_state} state")
            raise RuntimeError(f"Cannot stop recognition from {self._session_state} state")

        try:
            self._stop_recognition_internal()

            # Transition to CAPTURING state
            self._session_state = SessionState.CAPTURING
            self._emit_event("recognition_stopped")
            logger.info(f"Recognition stopped, state: {self._session_state}")

        except Exception as e:
            logger.error(f"Error stopping recognition: {e}", exc_info=True)

    def _stop_recognition_internal(self) -> None:
        """Internal method to stop recognition without state transition."""
        # TODO: Stop recognition components (VLM, STT, Question Segmenter)
        # This will be implemented in later tasks
        logger.debug("Stopping recognition components")

    def reset(self) -> None:
        """Reset question understanding and clear context.

        Can only be called in RECOGNIZING state.
        Remains in RECOGNIZING state after reset.

        Raises:
            RuntimeError: If not in RECOGNIZING state
        """
        if self._session_state != SessionState.RECOGNIZING:
            logger.warning(f"Cannot reset from {self._session_state} state")
            raise RuntimeError(f"Cannot reset from {self._session_state} state")

        try:
            # TODO: Reset recognition components (clear question context, history)
            # This will be implemented in later tasks

            self._emit_event("session_reset")
            logger.info("Session reset, cleared question context")

        except Exception as e:
            logger.error(f"Error resetting session: {e}", exc_info=True)
            raise

    def get_session_state(self) -> SessionState:
        """Get current session state.

        Returns:
            Current session state
        """
        return self._session_state

    def set_input_mode(self, mode: InputMode) -> None:
        """Set input mode (video+audio, video-only, audio-only).

        If capture is active, this will restart capture with the new mode
        while preserving session state.

        Args:
            mode: New input mode
        """
        if mode == self._input_mode:
            logger.debug(f"Input mode already set to {mode}")
            return

        old_mode = self._input_mode
        was_capturing = self._session_state != SessionState.IDLE
        was_recognizing = self._session_state == SessionState.RECOGNIZING

        logger.info(f"Switching input mode from {old_mode} to {mode}")

        # Stop current capture if active
        if was_capturing:
            if was_recognizing:
                self._stop_recognition_internal()
            self._cleanup_capture()

        # Update mode
        self._input_mode = mode

        # Restart capture if it was active
        if was_capturing:
            try:
                screen_capture = self._get_active_screen_capture()
                if self._input_mode.has_video:
                    screen_capture.start_capture(fps=self._capture_fps)
                if self._input_mode.has_audio:
                    self._audio_capture.start_capture()

                # Restart recognition if it was active
                if was_recognizing:
                    # TODO: Restart recognition components
                    pass

            except Exception as e:
                logger.error(f"Error restarting capture after mode change: {e}", exc_info=True)
                # Reset to IDLE on error
                self._session_state = SessionState.IDLE

        self._emit_event("input_mode_changed", mode=mode)
        logger.info(f"Input mode changed to {mode}")

    def get_input_mode(self) -> InputMode:
        """Get current input mode.

        Returns:
            Current input mode
        """
        return self._input_mode

    def is_capturing(self) -> bool:
        """Check if capture is currently active.

        Returns:
            True if in CAPTURING or RECOGNIZING state, False otherwise
        """
        return self._session_state in (SessionState.CAPTURING, SessionState.RECOGNIZING)

    def get_capture_fps(self) -> int:
        """Get current capture frame rate.

        Returns:
            Current FPS setting
        """
        return self._capture_fps

    def set_capture_fps(self, fps: int) -> None:
        """Set capture frame rate (1-60 FPS).

        If capture is active, this will update the FPS dynamically.

        Args:
            fps: Frame rate between 1 and 60
        """
        fps = max(1, min(60, fps))

        if fps == self._capture_fps:
            return

        self._capture_fps = fps

        # Update FPS if capturing
        if self.is_capturing() and self._input_mode.has_video:
            screen_capture = self._get_active_screen_capture()
            screen_capture.set_fps(fps)

        logger.info(f"Capture FPS set to {fps}")

    def set_capture_method(self, method: CaptureMethod) -> None:
        """Set capture method (OS-native or UVC).

        If capture is active, this will restart capture with the new method
        while preserving session state.

        Args:
            method: New capture method
        """
        if method == self._capture_method:
            logger.debug(f"Capture method already set to {method}")
            return

        # Check if requested method is available
        if method == CaptureMethod.UVC_DEVICE and self._uvc_capture is None:
            logger.error("UVC capture not available")
            raise ValueError("UVC capture not available")

        old_method = self._capture_method
        was_capturing = self._session_state != SessionState.IDLE
        was_recognizing = self._session_state == SessionState.RECOGNIZING

        logger.info(f"Switching capture method from {old_method} to {method}")

        # Stop current capture if active
        if was_capturing:
            if was_recognizing:
                self._stop_recognition_internal()
            self._cleanup_capture()

        # Update method
        self._capture_method = method

        # Restart capture if it was active
        if was_capturing:
            try:
                screen_capture = self._get_active_screen_capture()
                if self._input_mode.has_video:
                    screen_capture.start_capture(fps=self._capture_fps)
                if self._input_mode.has_audio:
                    self._audio_capture.start_capture()

                # Restart recognition if it was active
                if was_recognizing:
                    # TODO: Restart recognition components
                    pass

            except Exception as e:
                logger.error(f"Error restarting capture after method change: {e}", exc_info=True)
                # Reset to IDLE on error
                self._session_state = SessionState.IDLE

        self._emit_event("capture_method_changed", method=method)
        logger.info(f"Capture method changed to {method}")

    def get_capture_method(self) -> CaptureMethod:
        """Get current capture method.

        Returns:
            Current capture method
        """
        return self._capture_method

    def get_active_screen_capture(self) -> ScreenCaptureInterface:
        """Get the currently active screen capture interface.

        Returns:
            Active screen capture interface
        """
        return self._get_active_screen_capture()

    def list_windows(self) -> List[WindowInfo]:
        """List all capturable windows.

        Returns:
            List of WindowInfo objects
        """
        screen_capture = self._get_active_screen_capture()
        return screen_capture.list_windows()

    def set_target_window(self, window_id: Optional[int]) -> None:
        """Set target window for capture.

        Args:
            window_id: Window ID to capture, or None for full screen
        """
        screen_capture = self._get_active_screen_capture()
        screen_capture.set_target_window(window_id)
        logger.info(f"Target window set to {window_id}")

    def get_target_window(self) -> Optional[WindowInfo]:
        """Get currently targeted window.

        Returns:
            WindowInfo for current target or None if capturing full screen
        """
        screen_capture = self._get_active_screen_capture()
        return screen_capture.get_target_window()

    def _get_active_screen_capture(self) -> ScreenCaptureInterface:
        """Get the screen capture interface based on current method.

        Returns:
            Screen capture interface for current method
        """
        if self._capture_method == CaptureMethod.OS_NATIVE:
            return self._os_native_capture
        elif self._capture_method == CaptureMethod.UVC_DEVICE:
            if self._uvc_capture is None:
                raise ValueError("UVC capture not available")
            return self._uvc_capture
        else:
            raise ValueError(f"Unknown capture method: {self._capture_method}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for events.

        Args:
            event: Event name (e.g., "capture_started", "input_mode_changed")
            callback: Callback function to invoke when event occurs
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
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"Unregistered callback for event: {event}")

    def _emit_event(self, event: str, **kwargs) -> None:
        """Emit event to registered callbacks.

        Args:
            event: Event name
            **kwargs: Event data to pass to callbacks
        """
        if event not in self._callbacks:
            return

        for callback in self._callbacks[event]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Error in callback for event {event}: {e}", exc_info=True)
