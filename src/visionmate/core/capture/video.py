"""Video capture module for capturing video from various sources.

This module provides abstract interfaces and concrete implementations for
capturing video from screens, UVC devices, and RTSP streams. It includes
support for active window detection and frame cropping.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from visionmate.core.models import (
    DeviceMetadata,
    Resolution,
    VideoFrame,
    VideoSourceType,
    WindowRegion,
)

logger = logging.getLogger(__name__)


class VideoCaptureInterface(ABC):
    """Abstract interface for video capture.

    This interface defines the contract for all video capture implementations,
    supporting screen capture, UVC devices, and RTSP streams with optional
    active window detection and cropping.
    """

    @abstractmethod
    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video.

        Args:
            device_id: Device identifier
            fps: Frame rate (1-60)
            resolution: Optional resolution override as (width, height)
            enable_window_detection: Enable active window detection and cropping

        Requirements: 1.1, 1.2, 1.3, 28.1-28.3
        """
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.1, 1.2, 1.3
        """
        pass

    @abstractmethod
    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.1, 1.2, 1.3, 28.5
        """
        pass

    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.1, 1.2, 1.3
        """
        pass

    @abstractmethod
    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        pass

    @abstractmethod
    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable active window detection.

        Args:
            enabled: True to enable window detection, False to disable

        Requirements: 28.6, 28.8
        """
        pass

    @abstractmethod
    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True if window detection is enabled, False otherwise.

        Requirements: 28.6, 28.8
        """
        pass


class WindowDetector:
    """Detects window regions in frames using platform-specific APIs.

    For screen capture, uses platform-specific APIs to detect the active window.
    For UVC/RTSP, uses computer vision techniques (future implementation).
    """

    def __init__(self):
        """Initialize the WindowDetector."""
        self._platform = platform.system()
        self._platform_available = self._check_platform_libraries()
        logger.info(
            f"WindowDetector initialized on platform: {self._platform}, "
            f"libraries available: {self._platform_available}"
        )

    def _check_platform_libraries(self) -> bool:
        """Check if platform-specific libraries are available.

        Returns:
            True if required libraries are installed, False otherwise.
        """
        try:
            if self._platform == "Darwin":
                import Quartz  # noqa: F401

                return True
            elif self._platform == "Windows":
                import win32gui  # noqa: F401

                return True
            elif self._platform == "Linux":
                import Xlib  # noqa: F401

                return True
            return False
        except ImportError:
            return False

    def detect_active_window(self) -> Optional[WindowRegion]:
        """Detect the active window region using platform-specific APIs.

        This method is used for screen capture to detect the currently active
        window and return its bounds.

        Returns:
            WindowRegion object for the active window, or None if detection fails.

        Requirements: 28.1, 28.4, 28.7
        """
        try:
            if self._platform == "Darwin":
                return self._detect_active_window_macos()
            elif self._platform == "Windows":
                return self._detect_active_window_windows()
            elif self._platform == "Linux":
                return self._detect_active_window_linux()
            else:
                logger.warning(f"Unsupported platform for window detection: {self._platform}")
                return None
        except Exception as e:
            logger.error(f"Error detecting active window: {e}", exc_info=True)
            return None

    def _detect_active_window_macos(self) -> Optional[WindowRegion]:
        """Detect active window on macOS using Quartz API.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowBounds,
                kCGWindowLayer,
                kCGWindowListOptionOnScreenOnly,
            )

            # Get list of on-screen windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )

            # Find the frontmost window (layer 0)
            for window in window_list:
                if window.get(kCGWindowLayer, -1) == 0:
                    bounds = window.get(kCGWindowBounds, {})
                    if bounds:
                        x = int(bounds.get("X", 0))
                        y = int(bounds.get("Y", 0))
                        width = int(bounds.get("Width", 0))
                        height = int(bounds.get("Height", 0))

                        if width > 0 and height > 0:
                            region = WindowRegion(
                                x=x, y=y, width=width, height=height, confidence=1.0
                            )
                            logger.debug(f"Detected active window on macOS: {region.to_tuple()}")
                            return region

            logger.debug("No active window found on macOS")
            return None

        except ImportError as e:
            logger.warning(
                f"pyobjc-framework-Quartz not installed: {e}. "
                "Install with: uv add 'pyobjc-framework-Quartz>=10.3.1'"
            )
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on macOS: {e}", exc_info=True)
            return None

    def _detect_active_window_windows(self) -> Optional[WindowRegion]:
        """Detect active window on Windows using Win32 API.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            import win32gui

            # Get the foreground window
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                width = right - x
                height = bottom - y

                if width > 0 and height > 0:
                    region = WindowRegion(x=x, y=y, width=width, height=height, confidence=1.0)
                    logger.debug(f"Detected active window on Windows: {region.to_tuple()}")
                    return region

            logger.debug("No active window found on Windows")
            return None

        except ImportError as e:
            logger.warning(f"pywin32 not installed: {e}. Install with: uv add 'pywin32>=308'")
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on Windows: {e}", exc_info=True)
            return None

    def _detect_active_window_linux(self) -> Optional[WindowRegion]:
        """Detect active window on Linux using X11.

        Returns:
            WindowRegion object for the active window, or None if detection fails.
        """
        try:
            from Xlib import X, display

            d = display.Display()
            root = d.screen().root

            # Get the active window
            active_window = root.get_full_property(
                d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
            )

            if active_window and active_window.value:
                window_id = active_window.value[0]
                window = d.create_resource_object("window", window_id)

                # Get window geometry
                geom = window.get_geometry()
                x, y = geom.x, geom.y
                width, height = geom.width, geom.height

                # Translate coordinates to root window
                coords = window.translate_coords(root, 0, 0)
                x, y = coords.x, coords.y

                if width > 0 and height > 0:
                    region = WindowRegion(x=x, y=y, width=width, height=height, confidence=1.0)
                    logger.debug(f"Detected active window on Linux: {region.to_tuple()}")
                    return region

            logger.debug("No active window found on Linux")
            return None

        except ImportError as e:
            logger.warning(
                f"python-xlib not installed: {e}. Install with: uv add 'python-xlib>=0.33'"
            )
            return None
        except Exception as e:
            logger.error(f"Error detecting active window on Linux: {e}", exc_info=True)
            return None


class ScreenCapture(VideoCaptureInterface):
    """OS-native screen capture using mss library.

    Features:
    - Cross-platform screen capture using mss
    - Configurable FPS with precise timing control
    - Frame buffering for smooth playback
    - Automatic window detection enabled for screen capture
    - Platform-specific active window detection and cropping

    Requirements: 1.1, 1.4, 4.1, 4.2, 28.1, 28.7
    """

    def __init__(self, device_manager=None):
        """Initialize the ScreenCapture.

        Args:
            device_manager: Optional DeviceManager instance for metadata retrieval
        """
        self._device_manager = device_manager
        self._device_id: Optional[str] = None
        self._fps: int = 1
        self._resolution: Optional[Resolution] = None
        self._window_detection_enabled: bool = True  # Auto-enabled for screen capture
        self._capturing: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(
            maxlen=5
        )  # Buffer last 5 frames. # TODO: The buffer size should be parameterized.
        self._frame_number: int = 0
        self._window_detector = WindowDetector()
        self._monitor_index: int = 0
        self._lock = threading.Lock()

        logger.info("ScreenCapture initialized")

    def start_capture(
        self,
        device_id: str,
        fps: int = 30,
        resolution: Optional[tuple[int, int]] = None,
        enable_window_detection: bool = False,
    ) -> None:
        """Start capturing video from screen.

        Args:
            device_id: Device identifier (e.g., "screen_1")
            fps: Frame rate (1-60)
            resolution: Optional resolution override (not used for screen capture)
            enable_window_detection: Ignored - always enabled for screen capture

        Requirements: 1.1, 1.4, 4.1, 4.2, 28.1, 28.7
        """
        if self._capturing:
            logger.warning("Capture already in progress")
            return

        # Parse device_id to get monitor index
        try:
            self._monitor_index = int(device_id.split("_")[1])
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid device_id format: {device_id}")
            raise ValueError(f"Invalid device_id format: {device_id}") from e

        self._device_id = device_id
        self._fps = max(
            1, min(fps, 60)
        )  # Clamp FPS to 1-60. # TODO: 60fps limit should be placed in the configuration.
        self._resolution = Resolution.from_tuple(resolution) if resolution else None
        self._window_detection_enabled = True  # Always enabled for screen capture
        self._frame_number = 0
        self._frame_buffer.clear()

        # Start capture thread
        self._capturing = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        logger.info(
            f"Started screen capture: device={device_id}, fps={self._fps}, window_detection=enabled"
        )

    def stop_capture(self) -> None:
        """Stop capturing video.

        Requirements: 1.1
        """
        if not self._capturing:
            logger.warning("Capture not in progress")
            return

        self._capturing = False

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        logger.info(f"Stopped screen capture: device={self._device_id}")

    def get_frame(self) -> Optional[VideoFrame]:
        """Get the latest captured frame (cropped if window detection enabled).

        Returns:
            VideoFrame object with the latest frame, or None if no frame available.

        Requirements: 1.1, 28.5
        """
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
            return None

    def is_capturing(self) -> bool:
        """Check if currently capturing.

        Returns:
            True if capture is active, False otherwise.

        Requirements: 1.1
        """
        return self._capturing

    def get_source_info(self) -> DeviceMetadata:
        """Get source device metadata.

        Returns:
            DeviceMetadata object with device information.

        Requirements: 27.1-27.6
        """
        if self._device_manager and self._device_id:
            return self._device_manager.get_device_metadata(self._device_id)
        else:
            # Return minimal metadata if device_manager not available
            from visionmate.core.models import DeviceType

            return DeviceMetadata(
                device_id=self._device_id or "unknown",
                name="Screen",
                device_type=DeviceType.SCREEN,
            )

    def set_window_detection(self, enabled: bool) -> None:
        """Enable or disable active window detection.

        Note: For screen capture, window detection is always enabled.

        Args:
            enabled: Ignored - always enabled for screen capture

        Requirements: 28.6, 28.7
        """
        if not enabled:
            logger.warning(
                "Window detection cannot be disabled for screen capture - "
                "it is automatically enabled"
            )
        self._window_detection_enabled = True

    def is_window_detection_enabled(self) -> bool:
        """Check if window detection is enabled.

        Returns:
            True (always enabled for screen capture)

        Requirements: 28.6, 28.7
        """
        return True

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread.

        Captures frames at the specified FPS rate, detects active window,
        and crops frames to the window region.
        """
        import mss

        frame_interval = 1.0 / self._fps
        next_capture_time = time.time()

        try:
            with mss.mss() as sct:
                # Get monitor (add 1 because monitor 0 is "all monitors")
                monitor = sct.monitors[self._monitor_index]

                logger.debug(f"Capturing from monitor: {monitor}")

                while self._capturing:
                    current_time = time.time()

                    # Wait until next capture time
                    if current_time < next_capture_time:
                        time.sleep(0.001)  # Sleep 1ms to avoid busy waiting
                        continue

                    # Capture frame
                    try:
                        # Capture full screen
                        screenshot = sct.grab(monitor)
                        img = np.array(screenshot)

                        # Convert BGRA to RGB
                        img = img[:, :, :3]  # Drop alpha channel
                        img = img[:, :, ::-1]  # BGR to RGB

                        # Get current timestamp
                        timestamp = datetime.now(timezone.utc)

                        # Detect active window
                        detected_regions = []
                        active_region = None
                        is_cropped = False

                        if self._window_detection_enabled:
                            window_region = self._window_detector.detect_active_window()
                            if window_region:
                                detected_regions = [window_region]
                                active_region = window_region

                                # Crop image to window region
                                # Adjust coordinates relative to monitor
                                x = max(0, window_region.x - monitor["left"])
                                y = max(0, window_region.y - monitor["top"])
                                x2 = min(img.shape[1], x + window_region.width)
                                y2 = min(img.shape[0], y + window_region.height)

                                if x < x2 and y < y2:
                                    img = img[y:y2, x:x2]
                                    is_cropped = True
                                    logger.debug(
                                        f"Cropped frame to window region: "
                                        f"{window_region.to_tuple()}"
                                    )

                        # Get resolution
                        height, width = img.shape[:2]
                        resolution = Resolution(width=width, height=height)

                        # Ensure device_id is set (should always be true in capture loop)
                        if self._device_id is None:
                            logger.error("Device ID is None during capture")
                            break

                        # Create VideoFrame
                        frame = VideoFrame(
                            image=img,
                            timestamp=timestamp,
                            source_id=self._device_id,
                            source_type=VideoSourceType.SCREEN,
                            resolution=resolution,
                            fps=self._fps,
                            frame_number=self._frame_number,
                            detected_regions=detected_regions,
                            active_region=active_region,
                            is_cropped=is_cropped,
                        )

                        # Add to buffer
                        with self._lock:
                            self._frame_buffer.append(frame)

                        self._frame_number += 1

                        # Schedule next capture
                        next_capture_time += frame_interval

                        # If we're falling behind, reset timing
                        if next_capture_time < current_time:
                            next_capture_time = current_time + frame_interval

                    except Exception as e:
                        logger.error(f"Error capturing frame: {e}", exc_info=True)
                        time.sleep(frame_interval)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
        finally:
            self._capturing = False
            logger.debug("Capture loop ended")
