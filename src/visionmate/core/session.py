"""Session manager for coordinating capture and recognition."""

import logging
from typing import Callable, Dict, List, Optional

from visionmate.capture.audio import AudioCaptureInterface
from visionmate.capture.screen import ScreenCaptureInterface
from visionmate.core.input import CaptureMethod, InputMode

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages capture sessions and coordinates input modes.

    The SessionManager is responsible for:
    - Managing capture lifecycle (start, stop)
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
        self._is_capturing = False
        self._capture_fps = 30
        self._callbacks: Dict[str, List[Callable]] = {}

    def start_capture(self, fps: int = 30, window_id: Optional[int] = None) -> None:
        """Start capturing input at specified FPS.

        Args:
            fps: Frame rate (1-60)
            window_id: Optional window ID to capture specific window
        """
        if self._is_capturing:
            logger.warning("Capture already active, stopping first")
            self.stop_capture()

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

            self._is_capturing = True
            self._emit_event("capture_started")
            logger.info(f"Capture started in {self._input_mode} mode")

        except Exception as e:
            logger.error(f"Error starting capture: {e}", exc_info=True)
            # Clean up on error
            self.stop_capture()
            raise

    def stop_capture(self) -> None:
        """Stop capturing input."""
        if not self._is_capturing:
            return

        try:
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

            self._is_capturing = False
            self._emit_event("capture_stopped")
            logger.info("Capture stopped")

        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)

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
        was_capturing = self._is_capturing

        logger.info(f"Switching input mode from {old_mode} to {mode}")

        # Stop current capture if active
        if was_capturing:
            self.stop_capture()

        # Update mode
        self._input_mode = mode

        # Restart capture if it was active
        if was_capturing:
            self.start_capture(fps=self._capture_fps)

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
            True if capturing, False otherwise
        """
        return self._is_capturing

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
        if self._is_capturing and self._input_mode.has_video:
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
        was_capturing = self._is_capturing

        logger.info(f"Switching capture method from {old_method} to {method}")

        # Stop current capture if active
        if was_capturing:
            self.stop_capture()

        # Update method
        self._capture_method = method

        # Restart capture if it was active
        if was_capturing:
            self.start_capture(fps=self._capture_fps)

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
