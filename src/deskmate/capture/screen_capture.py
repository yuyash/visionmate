"""Screen capture interface and implementations."""

import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import mss.tools
import numpy as np


@dataclass
class WindowInfo:
    """Information about a capturable window."""

    window_id: int
    title: str
    app_name: str
    bounds: Tuple[int, int, int, int]  # (x, y, width, height)
    is_self: bool  # True if this is the capture app itself


class ScreenCaptureInterface(ABC):
    """Abstract interface for screen capture implementations."""

    @abstractmethod
    def start_capture(self, fps: int = 30, window_id: Optional[int] = None) -> None:
        """Start capturing screen frames at specified FPS.

        Args:
            fps: Frame rate (1-60)
            window_id: Optional window ID to capture specific window only
                      If None, captures entire screen
        """
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing screen frames."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame.

        Returns:
            Frame as numpy array (BGR format) or None if no frame available
        """
        pass

    @abstractmethod
    def get_frame_with_highlight(self) -> Optional[np.ndarray]:
        """Get frame with capture region highlighted in orange.

        Returns:
            Frame with orange overlay or None if no frame available
        """
        pass

    @abstractmethod
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available capture sources.

        Returns:
            List of device information dictionaries
        """
        pass

    @abstractmethod
    def list_windows(self) -> List[WindowInfo]:
        """List all capturable windows (excluding self).

        Returns:
            List of WindowInfo objects
        """
        pass

    @abstractmethod
    def set_target_window(self, window_id: Optional[int]) -> None:
        """Set target window for capture.

        Args:
            window_id: Window ID to capture, or None for full screen
        """
        pass

    @abstractmethod
    def get_target_window(self) -> Optional[WindowInfo]:
        """Get currently targeted window info.

        Returns:
            WindowInfo for current target or None if capturing full screen
        """
        pass

    @abstractmethod
    def set_fps(self, fps: int) -> None:
        """Set capture frame rate (1-60 FPS).

        Args:
            fps: Frame rate between 1 and 60
        """
        pass

    @abstractmethod
    def get_fps(self) -> int:
        """Get current capture frame rate.

        Returns:
            Current FPS setting
        """
        pass

    @abstractmethod
    def get_capture_region(self) -> Tuple[int, int, int, int]:
        """Get current capture region bounds.

        Returns:
            Tuple of (x, y, width, height)
        """
        pass


class MSSScreenCapture(ScreenCaptureInterface):
    """OS-native screen capture using MSS library."""

    def __init__(self):
        """Initialize MSS screen capture."""
        self._sct = mss.mss()
        self._fps = 30
        self._target_window_id: Optional[int] = None
        self._target_window_info: Optional[WindowInfo] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame_buffer: deque = deque(maxlen=10)  # Buffer last 10 frames
        self._is_capturing = False
        self._capture_region: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def start_capture(self, fps: int = 30, window_id: Optional[int] = None) -> None:
        """Start capturing screen frames at specified FPS."""
        if self._is_capturing:
            self.stop_capture()

        self._fps = max(1, min(60, fps))
        self._target_window_id = window_id
        self._update_target_window_info()
        self._stop_event.clear()
        self._is_capturing = True

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop_capture(self) -> None:
        """Stop capturing screen frames."""
        if not self._is_capturing:
            return

        self._is_capturing = False
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self._frame_lock:
            if not self._frame_buffer:
                return None
            return self._frame_buffer[-1].copy()

    def get_frame_with_highlight(self) -> Optional[np.ndarray]:
        """Get frame with capture region highlighted in orange."""
        frame = self.get_frame()
        if frame is None:
            return None

        # If capturing full screen, no highlight needed
        if self._target_window_id is None:
            return frame

        # Create orange overlay (RGB: 255, 165, 0, 50% opacity)
        overlay = frame.copy()
        x, y, w, h = self._capture_region

        # Draw orange rectangle border (5 pixels thick)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 5)

        # Blend with original frame (50% opacity)
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        return result

    def list_devices(self) -> List[Dict[str, Any]]:
        """List available capture sources (monitors)."""
        devices = []
        for i, monitor in enumerate(
            self._sct.monitors[1:], start=1
        ):  # Skip monitor 0 (all monitors)
            devices.append(
                {
                    "id": i,
                    "name": f"Monitor {i}",
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "left": monitor["left"],
                    "top": monitor["top"],
                }
            )
        return devices

    def list_windows(self) -> List[WindowInfo]:
        """List all capturable windows (excluding self)."""
        system = platform.system()

        if system == "Darwin":  # macOS
            return self._list_windows_macos()
        elif system == "Windows":
            return self._list_windows_windows()
        else:
            # Unsupported platform
            return []

    def set_target_window(self, window_id: Optional[int]) -> None:
        """Set target window for capture."""
        self._target_window_id = window_id
        self._update_target_window_info()

    def get_target_window(self) -> Optional[WindowInfo]:
        """Get currently targeted window info."""
        return self._target_window_info

    def set_fps(self, fps: int) -> None:
        """Set capture frame rate (1-60 FPS)."""
        self._fps = max(1, min(60, fps))

    def get_fps(self) -> int:
        """Get current capture frame rate."""
        return self._fps

    def get_capture_region(self) -> Tuple[int, int, int, int]:
        """Get current capture region bounds."""
        return self._capture_region

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self._fps
        next_frame_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()

            if current_time >= next_frame_time:
                # Capture frame
                frame = self._capture_frame()

                if frame is not None:
                    with self._frame_lock:
                        self._frame_buffer.append(frame)

                # Calculate next frame time with drift compensation
                next_frame_time += frame_interval
                if next_frame_time < current_time:
                    # If we're behind, reset to current time
                    next_frame_time = current_time + frame_interval
            else:
                # Sleep until next frame time
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))  # Sleep max 1ms at a time for responsiveness

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        try:
            # Determine capture region
            if self._target_window_id is not None and self._target_window_info is not None:
                # Capture specific window region
                x, y, w, h = self._target_window_info.bounds
                monitor = {"left": x, "top": y, "width": w, "height": h}
                self._capture_region = (x, y, w, h)
            else:
                # Capture entire primary monitor
                monitor = self._sct.monitors[1]  # Primary monitor
                self._capture_region = (
                    monitor["left"],
                    monitor["top"],
                    monitor["width"],
                    monitor["height"],
                )

            # Capture screenshot
            sct_img = self._sct.grab(monitor)

            # Convert to numpy array (BGR format for OpenCV compatibility)
            frame = np.array(sct_img)
            frame = frame[:, :, :3]  # Remove alpha channel
            frame = frame[:, :, ::-1]  # Convert BGRA to BGR

            return frame

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def _update_target_window_info(self) -> None:
        """Update target window info based on current window_id."""
        if self._target_window_id is None:
            self._target_window_info = None
            return

        windows = self.list_windows()
        for window in windows:
            if window.window_id == self._target_window_id:
                self._target_window_info = window
                return

        # Window not found
        self._target_window_info = None

    def _list_windows_macos(self) -> List[WindowInfo]:
        """List windows on macOS using Quartz."""
        try:
            from Quartz import (  # type: ignore[import-not-found]
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )

            # Get current process ID to exclude self
            current_pid = os.getpid()

            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )

            windows = []
            for window in window_list:
                # Skip windows without bounds or name
                if "kCGWindowBounds" not in window or "kCGWindowName" not in window:
                    continue

                # Skip windows from current process
                window_pid = window.get("kCGWindowOwnerPID", 0)
                if window_pid == current_pid:
                    continue

                # Extract window info
                bounds = window["kCGWindowBounds"]
                window_id = window.get("kCGWindowNumber", 0)
                title = window.get("kCGWindowName", "")
                app_name = window.get("kCGWindowOwnerName", "")

                # Skip windows with no title or very small size
                if not title or bounds["Width"] < 100 or bounds["Height"] < 100:
                    continue

                windows.append(
                    WindowInfo(
                        window_id=window_id,
                        title=title,
                        app_name=app_name,
                        bounds=(
                            int(bounds["X"]),
                            int(bounds["Y"]),
                            int(bounds["Width"]),
                            int(bounds["Height"]),
                        ),
                        is_self=False,
                    )
                )

            return windows

        except ImportError:
            print("pyobjc-framework-Quartz not installed, window enumeration unavailable on macOS")
            return []
        except Exception as e:
            print(f"Error listing windows on macOS: {e}")
            return []

    def _list_windows_windows(self) -> List[WindowInfo]:
        """List windows on Windows using win32gui."""
        try:
            import win32gui  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]

            # Get current process ID to exclude self
            current_pid = os.getpid()

            windows = []

            def enum_callback(hwnd, _):
                # Skip invisible windows
                if not win32gui.IsWindowVisible(hwnd):
                    return

                # Get window title
                title = win32gui.GetWindowText(hwnd)
                if not title:
                    return

                # Get window process ID
                _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
                if window_pid == current_pid:
                    return

                # Get window bounds
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, right, bottom = rect
                    width = right - x
                    height = bottom - y

                    # Skip very small windows
                    if width < 100 or height < 100:
                        return

                    windows.append(
                        WindowInfo(
                            window_id=hwnd,
                            title=title,
                            app_name=title,  # Windows doesn't easily provide app name
                            bounds=(x, y, width, height),
                            is_self=False,
                        )
                    )
                except Exception:
                    pass

            win32gui.EnumWindows(enum_callback, None)
            return windows

        except ImportError:
            print("pywin32 not installed, window enumeration unavailable on Windows")
            return []
        except Exception as e:
            print(f"Error listing windows on Windows: {e}")
            return []


class WindowDetector:
    """Detect active window region in UVC video feed."""

    def __init__(
        self,
        min_window_size_percent: float = 25.0,
        min_aspect_ratio: float = 0.75,
        max_aspect_ratio: float = 2.33,
        rectangularity_threshold: float = 0.8,
        confidence_threshold: float = 0.7,
    ):
        """Initialize window detector.

        Args:
            min_window_size_percent: Minimum window size as percentage of frame
            min_aspect_ratio: Minimum aspect ratio (width/height)
            max_aspect_ratio: Maximum aspect ratio (width/height)
            rectangularity_threshold: How close to rectangle (0.0-1.0)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
        """
        self.min_window_size_percent = min_window_size_percent
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.rectangularity_threshold = rectangularity_threshold
        self.confidence_threshold = confidence_threshold
        self._last_confidence = 0.0

    def detect_window_region(
        self, frame: Optional[np.ndarray]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect largest rectangular region in frame.

        Args:
            frame: Input frame as numpy array or None

        Returns:
            (x, y, width, height) of detected window, or None if not found
        """
        if frame is None or frame.size == 0:
            return None

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                self._last_confidence = 0.0
                return None

            # Frame dimensions
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            min_area = frame_area * (self.min_window_size_percent / 100.0)

            # Find best candidate
            best_contour = None
            best_score = 0.0

            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter by minimum size
                if area < min_area:
                    continue

                # Filter by aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue

                # Calculate rectangularity (how close to rectangle)
                contour_area = cv2.contourArea(contour)
                rectangularity = contour_area / area if area > 0 else 0

                if rectangularity < self.rectangularity_threshold:
                    continue

                # Calculate score (larger area and better rectangularity = higher score)
                score = (area / frame_area) * rectangularity

                if score > best_score:
                    best_score = score
                    best_contour = (x, y, w, h)

            # Check if we found a good candidate
            if best_contour is not None and best_score >= self.confidence_threshold:
                self._last_confidence = best_score
                return best_contour
            else:
                self._last_confidence = best_score
                return None

        except Exception as e:
            print(f"Error detecting window region: {e}")
            self._last_confidence = 0.0
            return None

    def get_confidence(self) -> float:
        """Get confidence score for last detection (0.0-1.0)."""
        return self._last_confidence


class UVCScreenCapture(ScreenCaptureInterface):
    """UVC device screen capture using OpenCV."""

    def __init__(self, device_id: int = 0):
        """Initialize UVC screen capture.

        Args:
            device_id: Video device ID (default 0)
        """
        self._device_id = device_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 30
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame_buffer: deque = deque(maxlen=10)
        self._is_capturing = False
        self._window_detector = WindowDetector()
        self._detected_region: Optional[Tuple[int, int, int, int]] = None
        self._capture_region: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def start_capture(self, fps: int = 30, window_id: Optional[int] = None) -> None:
        """Start capturing from UVC device at specified FPS.

        Note: window_id parameter is ignored for UVC capture (uses automatic detection)
        """
        if self._is_capturing:
            self.stop_capture()

        self._fps = max(1, min(60, fps))

        # Open video capture device
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video device {self._device_id}")

        # Set capture FPS
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._stop_event.clear()
        self._is_capturing = True

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop_capture(self) -> None:
        """Stop capturing from UVC device."""
        if not self._is_capturing:
            return

        self._is_capturing = False
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self._frame_lock:
            if not self._frame_buffer:
                return None
            return self._frame_buffer[-1].copy()

    def get_frame_with_highlight(self) -> Optional[np.ndarray]:
        """Get frame with detected window region highlighted in orange."""
        frame = self.get_frame()
        if frame is None or self._detected_region is None:
            return frame

        # Create orange overlay
        overlay = frame.copy()
        x, y, w, h = self._detected_region

        # Draw orange rectangle border (5 pixels thick)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 5)

        # Blend with original frame (50% opacity)
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        return result

    def list_devices(self) -> List[Dict[str, Any]]:
        """List available UVC devices."""
        devices = []
        max_devices_to_check = 20  # Check up to 20 devices
        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive failures

        # Try to open devices until we hit consecutive failures
        for i in range(max_devices_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get device properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                devices.append(
                    {
                        "id": i,
                        "name": f"UVC Device {i}",
                        "width": width,
                        "height": height,
                        "fps": fps,
                    }
                )
                cap.release()
                consecutive_failures = 0  # Reset counter on success
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # Stop searching after multiple consecutive failures
                    break

        return devices

    def list_windows(self) -> List[WindowInfo]:
        """List windows (not applicable for UVC devices).

        Returns:
            Empty list (window enumeration not supported for UVC)
        """
        return []

    def set_target_window(self, window_id: Optional[int]) -> None:
        """Set target window (not applicable for UVC devices).

        Note: UVC capture uses automatic window detection instead
        """
        pass

    def get_target_window(self) -> Optional[WindowInfo]:
        """Get target window (not applicable for UVC devices).

        Returns:
            None (window targeting not supported for UVC)
        """
        return None

    def set_fps(self, fps: int) -> None:
        """Set capture frame rate (1-60 FPS)."""
        self._fps = max(1, min(60, fps))
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def get_fps(self) -> int:
        """Get current capture frame rate."""
        return self._fps

    def get_capture_region(self) -> Tuple[int, int, int, int]:
        """Get current capture region bounds."""
        return self._capture_region

    def get_detection_confidence(self) -> float:
        """Get confidence score for window detection (0.0-1.0)."""
        return self._window_detector.get_confidence()

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self._fps
        next_frame_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()

            if current_time >= next_frame_time:
                # Capture frame
                frame = self._capture_frame()

                if frame is not None:
                    with self._frame_lock:
                        self._frame_buffer.append(frame)

                # Calculate next frame time with drift compensation
                next_frame_time += frame_interval
                if next_frame_time < current_time:
                    next_frame_time = current_time + frame_interval
            else:
                # Sleep until next frame time
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from UVC device."""
        if not self._cap or not self._cap.isOpened():
            return None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

            # Detect window region
            detected_region = self._window_detector.detect_window_region(frame)

            if detected_region is not None:
                # Crop to detected window
                x, y, w, h = detected_region
                self._detected_region = detected_region
                self._capture_region = (x, y, w, h)
                cropped_frame = frame[y : y + h, x : x + w]
                return cropped_frame
            else:
                # Use full frame if no window detected
                self._detected_region = None
                h, w = frame.shape[:2]
                self._capture_region = (0, 0, w, h)
                return frame

        except Exception as e:
            print(f"Error capturing frame from UVC device: {e}")
            return None
